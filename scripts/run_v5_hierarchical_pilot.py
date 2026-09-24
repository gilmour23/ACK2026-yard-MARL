from __future__ import annotations

"""Pre-registered V5 hierarchical 6k bounded pilot runner."""

from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict

import numpy as np
import torch

from pretrain_yc_bc import train_bc
from train_yc_marl import ResourcePPOConfig, ResourceCooperativeModel, train_resource_marl
from evaluate_yc_policies import evaluate
from yc_marl_env import ResourceMARLYardEnv
from v5.train_hierarchical_marl import HierarchicalPPOConfig, train_hierarchical_marl
from v5.evaluate_hierarchical import evaluate_hierarchical, load_hierarchical_model


METRICS=(
    "mean_truck_completion_delay",
    "mean_storage_completion_delay",
    "rehandling_moves",
    "proactive_moves",
    "total_yc_moves",
    "extra_yc_minutes_per_retrieval",
    "mean_yc_utilization",
    "max_yc_queue",
    "objective_proxy",
)


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit()->str:
    try:
        return subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except Exception:
        return "unknown"


def load_config(path:Path)->dict:
    cfg=json.loads(Path(path).read_text(encoding="utf-8"))
    validate_config(cfg)
    return cfg


def validate_config(cfg:dict)->None:
    assert cfg["arms"]==["control_rule","treatment_v5"]
    assert cfg["training_seeds"]==[61,62,63]
    assert cfg["bc"]=={
        "start_seed":1000,
        "scenario_count":8,
        "epochs":3,
        "batch_size":128,
        "learning_rate":0.0005,
        "rng_seed":20260924,
    }
    tr=cfg["training"]
    expected={
        "decision_threshold":6000,
        "rollout_steps":512,
        "min_storage_transitions":64,
        "min_yc_transitions":128,
        "max_rollout_multiplier":4,
        "actor_critic_epochs":2,
        "critic_extra_epochs":18,
        "minibatch_size":256,
        "learning_rate":0.0003,
        "clip_coef":0.2,
        "max_grad_norm":0.5,
        "gamma":1.0,
        "gae_lambda":1.0,
        "episode_complete_rollout":True,
        "enable_proactive":True,
        "arrival_rate_per_hour":20.0,
        "include_resource_state":True,
        "truck_wait_weight":1.0,
        "storage_wait_weight":1.0,
        "extra_move_weight":0.1,
        "risk_shaping_weight":0.0,
        "yc_queue_shaping_weight":0.0,
        "use_action_q_critic":False,
        "control_rule_resolve_proactive_pair":True,
        "treatment_rule_resolve_proactive_pair":False,
    }
    for k,v in expected.items():
        assert tr[k]==v,(k,tr[k],v)
    ev=cfg["evaluation"]
    assert (ev["scenarios_start"],ev["scenarios_end"],ev["stochastic_repeats"])==(1201,1210,3)
    assert ev["policy_seed_rule"]=="scenario*100+repeat_index_0_based"
    assert ev["bootstrap_resamples"]==10000
    assert ev["bootstrap_rng_seed"]==20260924
    gate=cfg["viability_gate"]
    assert gate=={
        "mechanism_entropy_drop":0.01,
        "mechanism_seed_count":2,
        "performance_seed_count":2,
        "scenario_paired_mean_must_be_negative":True,
    }


def finite_tree(x)->bool:
    if isinstance(x,dict):
        return all(finite_tree(v) for v in x.values())
    if isinstance(x,(list,tuple)):
        return all(finite_tree(v) for v in x)
    if isinstance(x,(str,bool,int,np.integer)) or x is None:
        return True
    if isinstance(x,(float,np.floating)):
        return math.isfinite(float(x))
    return True


def build_shared_bc(cfg:dict,out:Path,config_path:Path)->dict:
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    bc=cfg["bc"]; tr=cfg["training"]
    path=out/"bc_marl_resource_storage_only.pt"
    metrics=train_bc(
        "marl",path,
        seeds=int(bc["scenario_count"]),
        epochs=int(bc["epochs"]),
        batch_size=int(bc["batch_size"]),
        lr=float(bc["learning_rate"]),
        arrival_rate_per_hour=float(tr["arrival_rate_per_hour"]),
        include_resource_state=True,
        enable_proactive=True,
        seed=int(bc["rng_seed"]),
        hidden=128,
    )
    if not finite_tree(metrics):
        raise RuntimeError("non-finite BC metrics")
    manifest={
        "phase":"bc",
        "git_commit":git_commit(),
        "config_sha256":sha256(config_path),
        "checkpoint":str(path),
        "checkpoint_sha256":sha256(path),
        "metrics":metrics,
    }
    (out/"bc_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    return manifest


def strict_control_load(checkpoint:Path)->None:
    ck=torch.load(Path(checkpoint),map_location="cpu",weights_only=False)
    env=ResourceMARLYardEnv(
        seed=1,arrival_rate_per_hour=20.0,
        include_resource_state=True,rule_resolve_proactive_pair=True,
    )
    env.reset()
    hidden=int(ck.get("hidden",ck.get("config",{}).get("hidden",128)))
    model=ResourceCooperativeModel(
        env.global_obs_dim,env.storage_obs_dim,env.yc_obs_dim,hidden=hidden
    )
    model.load_state_dict(ck["state_dict"],strict=True)


def read_csv(path:Path)->list[dict]:
    with Path(path).open(newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate_updates(updates:list[dict],threshold:int)->None:
    if not updates:
        raise RuntimeError("no PPO updates")
    if int(updates[-1]["global_step"])<threshold:
        raise RuntimeError(("threshold not reached",updates[-1]["global_step"],threshold))
    if not all(bool(u.get("rollout_ended_at_terminal",False)) for u in updates):
        raise RuntimeError("non-terminal rollout in episode-complete pilot")
    if any(float(u.get("critic_extra_actor_max_abs_delta",0.0))!=0.0 for u in updates):
        raise RuntimeError("actor changed in critic-only epochs")
    for u in updates:
        for k,v in u.items():
            if isinstance(v,(float,np.floating)) and not math.isfinite(float(v)):
                # MC EV can be NaN only if target variance is degenerate; this
                # should not occur in this pilot.
                raise RuntimeError(("non-finite update value",k,v))


def mechanism_summary(updates:list[dict])->dict:
    def mean_window(key,rows):
        vals=[float(r[key]) for r in rows if key in r]
        return float(np.mean(vals)) if vals else float("nan")
    early=updates[:min(2,len(updates))]
    late=updates[-min(2,len(updates)):]
    te=mean_window("yc_target_entropy_norm",early)
    tl=mean_window("yc_target_entropy_norm",late)
    de=mean_window("yc_destination_entropy_norm",early)
    dl=mean_window("yc_destination_entropy_norm",late)
    return {
        "target_entropy_early":te,
        "target_entropy_late":tl,
        "target_entropy_drop":te-tl,
        "destination_entropy_early":de,
        "destination_entropy_late":dl,
        "destination_entropy_drop":de-dl,
        "late_p_proactive":mean_window("yc_proactive_probability",late),
        "late_feasible_targets":mean_window("yc_feasible_targets",late),
        "late_feasible_pairs":mean_window("yc_feasible_pairs",late),
    }


def validate_eval_rows(rows:list[dict],cfg:dict)->None:
    ev=cfg["evaluation"]
    expected={
        (s,s*100+r)
        for s in range(int(ev["scenarios_start"]),int(ev["scenarios_end"])+1)
        for r in range(int(ev["stochastic_repeats"]))
    }
    got=[(int(x["seed"]),int(x["policy_seed"])) for x in rows]
    if len(got)!=len(expected) or len(set(got))!=len(got) or set(got)!=expected:
        raise RuntimeError("evaluation scenario/policy-seed matrix mismatch")
    for row in rows:
        if row["evaluation_mode"]!="stochastic":
            raise RuntimeError("pilot evaluation must be stochastic")
        for m in METRICS:
            if not math.isfinite(float(row[m])):
                raise RuntimeError(("non-finite eval metric",m,row[m]))


def train_eval_one(
    cfg:dict,
    arm:str,
    seed:int,
    bc_path:Path,
    bc_manifest_path:Path,
    out:Path,
    config_path:Path,
)->dict:
    if arm not in cfg["arms"] or seed not in cfg["training_seeds"]:
        raise ValueError((arm,seed))
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    bc_manifest=json.loads(Path(bc_manifest_path).read_text(encoding="utf-8"))
    bc_hash=sha256(bc_path)
    if bc_hash!=bc_manifest["checkpoint_sha256"]:
        raise RuntimeError("shared BC hash mismatch")

    tr=cfg["training"]
    run_dir=out/"train"
    if arm=="control_rule":
        rc=ResourcePPOConfig(
            total_steps=int(tr["decision_threshold"]),
            rollout_steps=int(tr["rollout_steps"]),
            min_storage_transitions=int(tr["min_storage_transitions"]),
            min_yc_transitions=int(tr["min_yc_transitions"]),
            max_rollout_multiplier=int(tr["max_rollout_multiplier"]),
            update_epochs=int(tr["actor_critic_epochs"]),
            critic_extra_epochs=int(tr["critic_extra_epochs"]),
            minibatch_size=int(tr["minibatch_size"]),
            gamma=float(tr["gamma"]),
            gae_lambda=float(tr["gae_lambda"]),
            clip_coef=float(tr["clip_coef"]),
            learning_rate=float(tr["learning_rate"]),
            max_grad_norm=float(tr["max_grad_norm"]),
            seed=int(seed),
            arrival_rate_per_hour=float(tr["arrival_rate_per_hour"]),
            include_resource_state=True,
            truck_wait_weight=float(tr["truck_wait_weight"]),
            storage_wait_weight=float(tr["storage_wait_weight"]),
            extra_move_weight=float(tr["extra_move_weight"]),
            risk_shaping_weight=float(tr["risk_shaping_weight"]),
            yc_queue_shaping_weight=float(tr["yc_queue_shaping_weight"]),
            enable_proactive=True,
            episode_complete_rollout=True,
            use_action_q_critic=False,
            rule_resolve_proactive_pair=True,
        )
        _,updates=train_resource_marl(rc,run_dir,Path(bc_path))
        checkpoint=run_dir/"resource_marl_final.pt"
        strict_control_load(checkpoint)
        eval_csv=out/"eval.csv"
        evaluate(
            eval_csv,
            "marl",
            checkpoint=checkpoint,
            seeds=range(1201,1211),
            repeats=3,
            arrival_rate_per_hour=20.0,
            enable_proactive=True,
            include_resource_state=True,
            yc_move_time=2.0,
            stochastic=True,
            rule_resolve_proactive_pair=True,
        )
        mech=None
        episodes=read_csv(run_dir/"resource_marl_train_episodes.csv")
    else:
        hc=HierarchicalPPOConfig(
            total_steps=int(tr["decision_threshold"]),
            rollout_steps=int(tr["rollout_steps"]),
            min_storage_transitions=int(tr["min_storage_transitions"]),
            min_yc_transitions=int(tr["min_yc_transitions"]),
            max_rollout_multiplier=int(tr["max_rollout_multiplier"]),
            update_epochs=int(tr["actor_critic_epochs"]),
            critic_extra_epochs=int(tr["critic_extra_epochs"]),
            minibatch_size=int(tr["minibatch_size"]),
            gamma=float(tr["gamma"]),
            gae_lambda=float(tr["gae_lambda"]),
            clip_coef=float(tr["clip_coef"]),
            learning_rate=float(tr["learning_rate"]),
            max_grad_norm=float(tr["max_grad_norm"]),
            seed=int(seed),
            arrival_rate_per_hour=float(tr["arrival_rate_per_hour"]),
            include_resource_state=True,
            truck_wait_weight=float(tr["truck_wait_weight"]),
            storage_wait_weight=float(tr["storage_wait_weight"]),
            extra_move_weight=float(tr["extra_move_weight"]),
            risk_shaping_weight=float(tr["risk_shaping_weight"]),
            yc_queue_shaping_weight=float(tr["yc_queue_shaping_weight"]),
            enable_proactive=True,
            episode_complete_rollout=True,
        )
        _,updates=train_hierarchical_marl(hc,run_dir,Path(bc_path))
        checkpoint=run_dir/"hierarchical_marl_final.pt"
        load_hierarchical_model(checkpoint)
        eval_csv=out/"eval.csv"
        evaluate_hierarchical(
            eval_csv,checkpoint,
            seeds=range(1201,1211),
            repeats=3,
            arrival_rate_per_hour=20.0,
            include_resource_state=True,
            stochastic=True,
        )
        mech=mechanism_summary(updates)
        episodes=read_csv(run_dir/"hierarchical_marl_train_episodes.csv")

    validate_updates(updates,int(tr["decision_threshold"]))
    eval_rows=read_csv(eval_csv)
    validate_eval_rows(eval_rows,cfg)
    scenario_sequence=[int(r["seed"]) for r in episodes]

    manifest={
        "phase":"train_eval",
        "arm":arm,
        "training_seed":int(seed),
        "git_commit":git_commit(),
        "protocol_file":cfg["protocol_file"],
        "config_sha256":sha256(config_path),
        "shared_bc_sha256":bc_hash,
        "checkpoint":str(checkpoint),
        "checkpoint_sha256":sha256(checkpoint),
        "actual_decisions":int(updates[-1]["global_step"]),
        "ppo_updates":len(updates),
        "terminal_all":True,
        "critic_extra_actor_max_abs_delta":max(
            float(u.get("critic_extra_actor_max_abs_delta",0.0)) for u in updates
        ),
        "training_scenario_sequence":scenario_sequence,
        "mechanism":mech,
        "evaluation_rows":len(eval_rows),
        "evaluation_scenarios":[1201,1210],
        "evaluation_repeats":3,
    }
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    return manifest


def bootstrap_ci(diffs:np.ndarray,n:int,seed:int)->tuple[float,float,float]:
    diffs=np.asarray(diffs,dtype=np.float64)
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(diffs),size=(n,len(diffs)))
    boot=diffs[idx].mean(axis=1)
    lo,hi=np.quantile(boot,[0.025,0.975])
    return float(diffs.mean()),float(lo),float(hi)


def aggregate(cfg:dict,root:Path,out:Path)->dict:
    root=Path(root); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    manifests=[]
    for p in root.rglob("manifest.json"):
        m=json.loads(p.read_text(encoding="utf-8"))
        if m.get("phase")=="train_eval":
            manifests.append((p,m))
    expected={(a,s) for a in cfg["arms"] for s in cfg["training_seeds"]}
    keys=[(m["arm"],int(m["training_seed"])) for _,m in manifests]
    if len(keys)!=len(expected) or set(keys)!=expected or len(set(keys))!=len(keys):
        raise RuntimeError(("pilot manifest matrix mismatch",keys))
    bc_hashes={m["shared_bc_sha256"] for _,m in manifests}
    if len(bc_hashes)!=1:
        raise RuntimeError(("BC was not shared exactly",bc_hashes))

    by_key={}
    seed_scenario={}
    for p,m in manifests:
        rows=read_csv(p.parent/"eval.csv")
        validate_eval_rows(rows,cfg)
        key=(m["arm"],int(m["training_seed"]))
        by_key[key]=rows
        seed_scenario[key]=m["training_scenario_sequence"]

    # Verify both arms saw the same common training-scenario prefix per seed.
    training_parity={}
    for seed in cfg["training_seeds"]:
        c=seed_scenario[("control_rule",seed)]
        v=seed_scenario[("treatment_v5",seed)]
        n=min(len(c),len(v))
        common=(c[:n]==v[:n])
        if not common:
            raise RuntimeError(("training scenario prefix mismatch",seed))
        training_parity[str(seed)]={
            "common_prefix_length":n,
            "control_episodes":len(c),
            "v5_episodes":len(v),
            "tail_difference":len(v)-len(c),
        }

    # Average repeats within arm/seed/scenario.
    av={}
    for key,rows in by_key.items():
        buckets=defaultdict(list)
        for r in rows:
            buckets[int(r["seed"])].append(r)
        for scenario,reps in buckets.items():
            if len(reps)!=3:
                raise RuntimeError(("repeat count",key,scenario,len(reps)))
            av[(key[0],key[1],scenario)]={
                m:float(np.mean([float(x[m]) for x in reps])) for m in METRICS
            }

    seed_means={}
    for arm in cfg["arms"]:
        seed_means[arm]={}
        for seed in cfg["training_seeds"]:
            vals=[av[(arm,seed,s)]["objective_proxy"] for s in range(1201,1211)]
            seed_means[arm][str(seed)]=float(np.mean(vals))

    seed_diffs={
        str(seed):seed_means["treatment_v5"][str(seed)]-seed_means["control_rule"][str(seed)]
        for seed in cfg["training_seeds"]
    }
    seed_improve_count=sum(v<0 for v in seed_diffs.values())

    scenario_diffs=[]
    scenario_rows=[]
    for scenario in range(1201,1211):
        c=float(np.mean([
            av[("control_rule",seed,scenario)]["objective_proxy"]
            for seed in cfg["training_seeds"]
        ]))
        v=float(np.mean([
            av[("treatment_v5",seed,scenario)]["objective_proxy"]
            for seed in cfg["training_seeds"]
        ]))
        d=v-c
        scenario_diffs.append(d)
        scenario_rows.append({
            "scenario":scenario,
            "control_rule_J":c,
            "treatment_v5_J":v,
            "v5_minus_control":d,
        })
    mean_diff,lo,hi=bootstrap_ci(
        np.asarray(scenario_diffs),
        int(cfg["evaluation"]["bootstrap_resamples"]),
        int(cfg["evaluation"]["bootstrap_rng_seed"]),
    )

    mechanism={}
    mech_pass_count=0
    threshold=float(cfg["viability_gate"]["mechanism_entropy_drop"])
    for seed in cfg["training_seeds"]:
        m=next(
            x for _,x in manifests
            if x["arm"]=="treatment_v5" and int(x["training_seed"])==seed
        )["mechanism"]
        passed=bool(
            float(m["target_entropy_drop"])>=threshold
            or float(m["destination_entropy_drop"])>=threshold
        )
        mech_pass_count+=int(passed)
        mechanism[str(seed)]={**m,"passed":passed}

    mechanism_gate=mech_pass_count>=int(cfg["viability_gate"]["mechanism_seed_count"])
    performance_seed_gate=seed_improve_count>=int(cfg["viability_gate"]["performance_seed_count"])
    scenario_gate=mean_diff<0.0
    performance_gate=performance_seed_gate and scenario_gate
    all_gates=mechanism_gate and performance_gate

    summary={
        "shared_bc_sha256":next(iter(bc_hashes)),
        "training_parity":training_parity,
        "seed_mean_J":seed_means,
        "seed_v5_minus_control":seed_diffs,
        "seed_improve_count":seed_improve_count,
        "scenario_paired":{
            "mean_v5_minus_control":mean_diff,
            "bootstrap_95":[lo,hi],
            "n_scenarios":10,
        },
        "mechanism":mechanism,
        "gates":{
            "mechanism_pass_count":mech_pass_count,
            "mechanism_gate":mechanism_gate,
            "performance_seed_gate":performance_seed_gate,
            "scenario_mean_gate":scenario_gate,
            "performance_gate":performance_gate,
            "all_viability_gates_pass":all_gates,
        },
        "decision":(
            "eligible_for_longer_bounded_confirmation"
            if all_gates
            else "do_not_extend_without_diagnosis"
        ),
        "note":"Passing does not authorize a 30k run.",
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")

    with (out/"scenario_differences.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(scenario_rows[0].keys()))
        w.writeheader();w.writerows(scenario_rows)

    lines=[
        "# V5 hierarchical 6k bounded pilot",
        "",
        f"Shared BC SHA-256: `{summary['shared_bc_sha256']}`",
        "",
        "## Seed mean J",
        "",
        "| Seed | Control rule | V5 hierarchical | V5-Control |",
        "|---:|---:|---:|---:|",
    ]
    for seed in cfg["training_seeds"]:
        lines.append(
            f"| {seed} | {seed_means['control_rule'][str(seed)]:.4f} | "
            f"{seed_means['treatment_v5'][str(seed)]:.4f} | "
            f"{seed_diffs[str(seed)]:+.4f} |"
        )
    lines += [
        "",
        "## Scenario-paired",
        "",
        f"- Mean V5-Control J: **{mean_diff:+.4f}**",
        f"- Scenario bootstrap 95% CI: **[{lo:+.4f}, {hi:+.4f}]**",
        "",
        "## Mechanism",
    ]
    for seed in cfg["training_seeds"]:
        m=mechanism[str(seed)]
        lines.append(
            f"- seed {seed}: target entropy drop {m['target_entropy_drop']:+.4f}, "
            f"destination entropy drop {m['destination_entropy_drop']:+.4f}, "
            f"gate={'PASS' if m['passed'] else 'FAIL'}"
        )
    lines += [
        "",
        "## Gates",
        f"- mechanism: **{'PASS' if mechanism_gate else 'FAIL'}** ({mech_pass_count}/3 seeds)",
        f"- performance seed gate: **{'PASS' if performance_seed_gate else 'FAIL'}** ({seed_improve_count}/3 seeds)",
        f"- scenario mean gate: **{'PASS' if scenario_gate else 'FAIL'}**",
        f"- overall V5 viability: **{'PASS' if all_gates else 'FAIL'}**",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        "Passing this bounded pilot does not authorize a 30k run.",
    ]
    (out/"SUMMARY.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(summary,sort_keys=True),flush=True)
    return summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True)
    sub=ap.add_subparsers(dest="cmd",required=True)

    p=sub.add_parser("build-bc")
    p.add_argument("--out",required=True)

    p=sub.add_parser("one")
    p.add_argument("--arm",required=True,choices=["control_rule","treatment_v5"])
    p.add_argument("--seed",required=True,type=int)
    p.add_argument("--bc",required=True)
    p.add_argument("--bc-manifest",required=True)
    p.add_argument("--out",required=True)

    p=sub.add_parser("aggregate")
    p.add_argument("--root",required=True)
    p.add_argument("--out",required=True)

    a=ap.parse_args()
    config_path=Path(a.config)
    cfg=load_config(config_path)

    if a.cmd=="build-bc":
        build_shared_bc(cfg,Path(a.out),config_path)
    elif a.cmd=="one":
        train_eval_one(
            cfg,a.arm,int(a.seed),Path(a.bc),Path(a.bc_manifest),
            Path(a.out),config_path,
        )
    else:
        aggregate(cfg,Path(a.root),Path(a.out))


if __name__=="__main__":
    main()
