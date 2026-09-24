from __future__ import annotations

"""Aggregate the frozen final 901-930 evaluation matrix.

No model selection or hyperparameter tuning is performed here. The script only
implements the reporting and paired-bootstrap analysis pre-registered in
FINAL_EXPERIMENT_PROTOCOL_20260924.md.
"""

from pathlib import Path
import argparse
import csv
import json
import math
from collections import Counter, defaultdict

import numpy as np

METRICS = [
    "mean_truck_completion_delay",
    "mean_storage_completion_delay",
    "rehandling_moves",
    "proactive_moves",
    "extra_yc_minutes_per_retrieval",
    "total_yc_moves",
    "mean_yc_utilization",
    "max_yc_queue",
    "objective_proxy",
]
LEARNED = ["proposed_marl","marl_noresource","single_rule"]
BASELINES = ["heuristic","single_rule","marl_noresource"]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_eval_records(root: Path) -> list[dict]:
    records=[]
    for mp in sorted(root.rglob("eval_manifest.json")):
        m=json.loads(mp.read_text(encoding="utf-8"))
        cp=mp.with_name("eval_901_930.csv")
        if not cp.exists(): raise FileNotFoundError(cp)
        records.append({"manifest":m,"rows":read_csv(cp),"manifest_path":str(mp)})
    return records


def find_training_manifests(root: Path | None) -> list[dict]:
    if root is None: return []
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.rglob("train_manifest.json"))]


def mean_rows(rows: list[dict], metric: str) -> float:
    return float(np.mean([float(r[metric]) for r in rows]))


def bootstrap_ci(diffs: np.ndarray, n: int, seed: int) -> tuple[float,float,float]:
    diffs=np.asarray(diffs,dtype=np.float64)
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(diffs),size=(n,len(diffs)))
    boot=diffs[idx].mean(axis=1)
    lo,hi=np.quantile(boot,[0.025,0.975])
    return float(diffs.mean()),float(lo),float(hi)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--eval-root",required=True)
    ap.add_argument("--training-root")
    ap.add_argument("--config",default="configs/final_experiment_20260924.json")
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    cfg=json.loads(Path(a.config).read_text(encoding="utf-8"))
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    records=find_eval_records(Path(a.eval_root))
    train_manifests=find_training_manifests(Path(a.training_root) if a.training_root else None)

    known_arms=set(LEARNED)|{"heuristic"}
    unknown=[r["manifest"].get("arm") for r in records if r["manifest"].get("arm") not in known_arms]
    if unknown: raise RuntimeError({"unexpected evaluation arms":unknown})
    learned_records=[r for r in records if r["manifest"]["arm"] in LEARNED]
    heur=[r for r in records if r["manifest"]["arm"]=="heuristic"]
    expected_pairs={(arm,seed) for arm in LEARNED for seed in cfg["training_seeds"]}
    learned_keys=[(r["manifest"]["arm"],int(r["manifest"]["training_seed"])) for r in learned_records]
    counts=Counter(learned_keys)
    duplicates=sorted(k for k,v in counts.items() if v!=1)
    got_pairs=set(learned_keys)
    if len(learned_records)!=len(expected_pairs) or got_pairs!=expected_pairs or duplicates:
        raise RuntimeError({
            "missing":sorted(expected_pairs-got_pairs),
            "extra":sorted(got_pairs-expected_pairs),
            "duplicate_or_repeated":duplicates,
            "record_count":len(learned_records),
        })
    if len(heur)!=1: raise RuntimeError(f"Expected one heuristic evaluation, got {len(heur)}")

    train_by_key={}
    if train_manifests:
        train_keys=[(m["arm"],int(m["training_seed"])) for m in train_manifests]
        train_counts=Counter(train_keys)
        if len(train_manifests)!=len(expected_pairs) or set(train_keys)!=expected_pairs or any(v!=1 for v in train_counts.values()):
            raise RuntimeError("training manifest matrix is incomplete or duplicated")
        train_by_key={k:m for k,m in zip(train_keys,train_manifests)}

    final_scenarios=set(range(cfg["evaluation"]["scenarios_start"],cfg["evaluation"]["scenarios_end"]+1))
    repeats=int(cfg["evaluation"]["stochastic_repeats"])
    expected_eval_pairs={(s,s*100+r) for s in final_scenarios for r in range(repeats)}
    by_arm_seed={}
    for r in learned_records:
        m=r["manifest"];key=(m["arm"],int(m["training_seed"]))
        if key in by_arm_seed: raise RuntimeError(("duplicate learned evaluation",key))
        rows=r["rows"]
        got_eval_pairs=[(int(x["seed"]),int(x["policy_seed"])) for x in rows]
        if len(got_eval_pairs)!=len(expected_eval_pairs) or len(set(got_eval_pairs))!=len(got_eval_pairs):
            raise RuntimeError(("duplicate or missing stochastic repeats",key))
        if set(got_eval_pairs)!=expected_eval_pairs:
            raise RuntimeError(("scenario/policy-seed matrix mismatch",key))
        if train_by_key:
            tm=train_by_key[key]
            if m.get("checkpoint_sha256")!=tm.get("final_checkpoint_sha256"):
                raise RuntimeError(("evaluation checkpoint does not match training checkpoint",key))
            if m.get("config_sha256")!=tm.get("config_sha256"):
                raise RuntimeError(("evaluation/training config hash mismatch",key))
            if m.get("git_commit")!=tm.get("git_commit"):
                raise RuntimeError(("evaluation/training commit mismatch",key))
        by_arm_seed[key]=rows
    heuristic_rows=heur[0]["rows"]
    heuristic_scenarios=[int(x["seed"]) for x in heuristic_rows]
    if len(heuristic_scenarios)!=len(final_scenarios) or len(set(heuristic_scenarios))!=len(heuristic_scenarios):
        raise RuntimeError("heuristic scenarios are missing or duplicated")
    if set(heuristic_scenarios)!=final_scenarios:
        raise RuntimeError("heuristic scenario set mismatch")

    cell={}
    for (arm,seed),rows in by_arm_seed.items():
        groups=defaultdict(list)
        for row in rows: groups[int(row["seed"])].append(row)
        for scenario,rr in groups.items():
            if len(rr)!=cfg["evaluation"]["stochastic_repeats"]:
                raise RuntimeError((arm,seed,scenario,len(rr)))
            cell[(arm,seed,scenario)]={m:mean_rows(rr,m) for m in METRICS}

    arm_stats={}
    scenario_means={}
    for arm in LEARNED:
        seed_means={}
        for seed in cfg["training_seeds"]:
            seed_means[str(seed)]={
                m:float(np.mean([cell[(arm,seed,s)][m] for s in sorted(final_scenarios)]))
                for m in METRICS
            }
        overall={}
        for m in METRICS:
            vals=np.asarray([seed_means[str(seed)][m] for seed in cfg["training_seeds"]],dtype=np.float64)
            overall[m]={
                "mean":float(vals.mean()),
                "sd_across_training_seeds":float(vals.std(ddof=1))
            }
        arm_stats[arm]={"seed_means":seed_means,"overall":overall}
        scenario_means[arm]={
            str(s):{
                m:float(np.mean([cell[(arm,seed,s)][m] for seed in cfg["training_seeds"]]))
                for m in METRICS
            }
            for s in sorted(final_scenarios)
        }

    scenario_means["heuristic"]={
        str(s):{
            m:float(next(float(r[m]) for r in heuristic_rows if int(r["seed"])==s))
            for m in METRICS
        }
        for s in sorted(final_scenarios)
    }
    heuristic_overall={
        m:float(np.mean([scenario_means["heuristic"][str(s)][m] for s in sorted(final_scenarios)]))
        for m in METRICS
    }

    comparisons={}
    paired_rows=[]
    nboot=int(cfg["evaluation"]["bootstrap_resamples"])
    bseed=int(cfg["evaluation"]["bootstrap_rng_seed"])
    for base in BASELINES:
        comp={}
        for m in METRICS:
            diffs=np.asarray([
                scenario_means["proposed_marl"][str(s)][m]-scenario_means[base][str(s)][m]
                for s in sorted(final_scenarios)
            ],dtype=np.float64)
            mean_diff,lo,hi=bootstrap_ci(diffs,nboot,bseed)
            comp[m]={
                "mean_difference_proposed_minus_baseline":mean_diff,
                "ci95_low":lo,"ci95_high":hi
            }
            for s,d in zip(sorted(final_scenarios),diffs):
                paired_rows.append({
                    "baseline":base,"metric":m,"scenario":s,
                    "difference_proposed_minus_baseline":float(d)
                })
        comparisons[base]=comp

    summary={
        "protocol_file":cfg["protocol_file"],
        "final_scenarios":[cfg["evaluation"]["scenarios_start"],cfg["evaluation"]["scenarios_end"]],
        "stochastic_repeats":cfg["evaluation"]["stochastic_repeats"],
        "training_seeds":cfg["training_seeds"],
        "arm_statistics":arm_stats,
        "heuristic_overall":heuristic_overall,
        "scenario_means":scenario_means,
        "paired_bootstrap":comparisons,
        "bootstrap_resamples":nboot,
        "bootstrap_rng_seed":bseed,
        "training_manifests":train_manifests,
        "interpretation_note":"Observed final-test results only; no post-hoc success threshold or model selection is applied."
    }
    if not all(
        math.isfinite(float(summary["arm_statistics"][arm]["overall"]["objective_proxy"]["mean"]))
        for arm in LEARNED
    ):
        raise RuntimeError("non-finite aggregate")

    (out/"final_summary.json").write_text(
        json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8"
    )
    with (out/"scenario_paired_differences.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(
            f,fieldnames=["baseline","metric","scenario","difference_proposed_minus_baseline"]
        )
        w.writeheader();w.writerows(paired_rows)

    lines=[
        "# ACK2026 final experiment summary","",
        "Final bank: scenarios 901-930. Results are reported as observed; no post-hoc success gate is applied.","",
        "## Objective J"
    ]
    for arm in LEARNED:
        x=arm_stats[arm]["overall"]["objective_proxy"]
        lines.append(
            f"- {arm}: {x['mean']:.6f} (SD across training seeds {x['sd_across_training_seeds']:.6f})"
        )
    lines.append(f"- heuristic: {heuristic_overall['objective_proxy']:.6f}")
    lines.extend(["","## Proposed minus baseline paired bootstrap for J"])
    for base in BASELINES:
        x=comparisons[base]["objective_proxy"]
        lines.append(
            f"- {base}: mean {x['mean_difference_proposed_minus_baseline']:.6f}, "
            f"95% CI [{x['ci95_low']:.6f}, {x['ci95_high']:.6f}]"
        )
    (out/"FINAL_SUMMARY.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"status":"ok","records":len(records),"output":str(out)}),flush=True)


if __name__=="__main__":
    main()
