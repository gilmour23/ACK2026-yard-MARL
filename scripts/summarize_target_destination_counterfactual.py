from __future__ import annotations

"""Aggregate target-vs-destination counterfactual decomposition results."""

from pathlib import Path
import argparse
import json

import numpy as np


def load_all(root: Path, name: str) -> list:
    out=[]
    for p in sorted(root.rglob(name)):
        obj=json.loads(p.read_text(encoding="utf-8"))
        out.append((p,obj))
    return out


def q(vals, p):
    return float(np.quantile(np.asarray(vals,dtype=np.float64),p))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)

    summaries=load_all(root,"summary.json")
    states_files=load_all(root,"states.json")
    if len(summaries)!=3 or len(states_files)!=3:
        raise RuntimeError(("expected three seed artifacts",len(summaries),len(states_files)))

    by_seed={}
    all_states=[]
    for _,s in summaries:
        seed=int(s["training_seed"])
        if seed in by_seed: raise RuntimeError(("duplicate seed",seed))
        by_seed[seed]=s
    if set(by_seed)!={51,52,53}:
        raise RuntimeError(("seed set",sorted(by_seed)))

    for _,rows in states_files:
        all_states.extend(rows)
    if not all_states:
        raise RuntimeError("no state rows")

    top1_reg=[float(r["top1_target_regret"]) for r in all_states]
    top3_reg=[float(r["top3_target_regret"]) for r in all_states]
    top5_reg=[float(r["top5_target_regret"]) for r in all_states]
    target_range=[float(r["target_J_range"]) for r in all_states]
    dest_reg=[float(r["mean_destination_regret"]) for r in all_states]
    dest_range=[float(r["mean_destination_J_range"]) for r in all_states]
    best_rank=[int(r["global_best_target_rank"]) for r in all_states]

    pooled={
        "source_training_seeds":[51,52,53],
        "valid_probe_states":len(all_states),
        "top1_best_target_coverage":float(np.mean([x==1 for x in best_rank])),
        "top3_best_target_coverage":float(np.mean([x<=3 for x in best_rank])),
        "top5_best_target_coverage":float(np.mean([x<=5 for x in best_rank])),
        "best_target_rank_median":float(np.median(best_rank)),
        "best_target_rank_p90":q(best_rank,0.90),
        "mean_top1_target_regret":float(np.mean(top1_reg)),
        "median_top1_target_regret":float(np.median(top1_reg)),
        "p90_top1_target_regret":q(top1_reg,0.90),
        "mean_top3_target_regret":float(np.mean(top3_reg)),
        "mean_top5_target_regret":float(np.mean(top5_reg)),
        "mean_target_J_range":float(np.mean(target_range)),
        "median_target_J_range":float(np.median(target_range)),
        "mean_destination_regret":float(np.mean(dest_reg)),
        "median_destination_regret":float(np.median(dest_reg)),
        "p90_destination_regret":q(dest_reg,0.90),
        "mean_destination_J_range":float(np.mean(dest_range)),
        "median_destination_J_range":float(np.median(dest_range)),
        "heuristic_destination_best_fraction_state_mean":float(np.mean([
            float(r["heuristic_destination_best_fraction"]) for r in all_states
        ])),
        "mean_eligible_targets":float(np.mean([int(r["eligible_target_count"]) for r in all_states])),
        "target_to_destination_regret_ratio":(
            float(np.mean(top1_reg))/float(np.mean(dest_reg))
            if float(np.mean(dest_reg))>1e-12 else None
        ),
    }

    result={
        "per_seed":{str(k):v for k,v in sorted(by_seed.items())},
        "pooled":pooled,
        "interpretation_guardrails":[
            "This is a no-training mechanism diagnostic, not a final performance comparison.",
            "Target regret is exhaustive over eligible targets under each target's heuristic destination.",
            "Destination regret is exhaustive over feasible destinations only for heuristic top-5 targets.",
            "Architecture selection should consider both pooled and seed-specific patterns, not one source model."
        ],
    }
    (out/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")

    lines=[
        "# Target–Destination counterfactual decomposition",
        "",
        f"Valid probe states: {pooled['valid_probe_states']}",
        "",
        "## Target choice",
        f"- Best-target coverage by current heuristic ranking: top1 {pooled['top1_best_target_coverage']:.3f}, top3 {pooled['top3_best_target_coverage']:.3f}, top5 {pooled['top5_best_target_coverage']:.3f}",
        f"- Mean top1 target regret: {pooled['mean_top1_target_regret']:.4f} J",
        f"- Mean top3 target regret: {pooled['mean_top3_target_regret']:.4f} J",
        f"- Mean top5 target regret: {pooled['mean_top5_target_regret']:.4f} J",
        f"- Mean target J range: {pooled['mean_target_J_range']:.4f}",
        "",
        "## Destination choice",
        f"- Mean heuristic-destination regret: {pooled['mean_destination_regret']:.4f} J",
        f"- Mean destination J range: {pooled['mean_destination_J_range']:.4f}",
        f"- Heuristic destination best fraction (state mean): {pooled['heuristic_destination_best_fraction_state_mean']:.3f}",
        "",
        "## Reading",
        "- If top1 regret is negligible and destination regret is negligible, keep top1 resolver and make the operation gate candidate-aware.",
        "- If top3/top5 recover most target regret while destination regret remains small, learn Target within Top-K and retain heuristic Destination.",
        "- If destination regret is also material, use a small Top-K Target×Destination candidate policy rather than fixing Destination.",
        ""
    ]
    (out/"SUMMARY.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(pooled,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
