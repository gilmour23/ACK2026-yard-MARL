from __future__ import annotations

"""Pre-registered ACK2026 final experiment runner.

This file deliberately separates training from final-bank evaluation.
The final 901-930 bank is only opened by the evaluation commands after all
training jobs have completed successfully in the manual GitHub Actions workflow.
"""

from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import subprocess
from typing import Any

import numpy as np
import torch

from pretrain_yc_bc import train_bc
from train_yc_marl import ResourcePPOConfig, ResourceCooperativeModel, train_resource_marl
from train_yc_single import SinglePPOConfig, CentralizedSingleModel, train_single_ppo
from evaluate_yc_policies import evaluate
from validate_yc_marl import heuristic_action
from yc_marl_env import (
    ResourceMARLYardEnv, STORAGE_AGENT, YC_PROACTIVE_BASE
)

METRICS = (
    "mean_truck_completion_delay",
    "mean_storage_completion_delay",
    "rehandling_moves",
    "proactive_moves",
    "extra_yc_minutes_per_retrieval",
    "total_yc_moves",
    "mean_yc_utilization",
    "max_yc_queue",
    "objective_proxy",
)


def load_config(path: Path) -> dict:
    cfg=json.loads(Path(path).read_text(encoding="utf-8"))
    validate_plan(cfg)
    return cfg


def validate_plan(cfg: dict) -> None:
    assert cfg["learned_arms"] == ["proposed_marl","marl_noresource","single_rule"]
    assert cfg["training_seeds"] == [51,52,53]
    bc=cfg["bc"]
    assert bc == {
        "start_seed":1000,"scenario_count":8,"epochs":3,"batch_size":128,
        "learning_rate":0.0005,"rng_seed":20260924,
    }
    tr=cfg["training"]
    required={
        "decision_threshold":30000,"rollout_steps":512,
        "min_storage_transitions":64,"min_yc_transitions":128,
        "max_rollout_multiplier":4,"actor_critic_epochs":2,
        "critic_extra_epochs":18,"minibatch_size":256,
        "learning_rate":0.0003,"clip_coef":0.2,"max_grad_norm":0.5,
        "gamma":1.0,"gae_lambda":1.0,"episode_complete_rollout":True,
        "use_action_q_critic":False,"rule_resolve_proactive_pair":True,
        "enable_proactive":True,"arrival_rate_per_hour":20.0,
        "truck_wait_weight":1.0,"storage_wait_weight":1.0,
        "extra_move_weight":0.1,"risk_shaping_weight":0.0,
        "yc_queue_shaping_weight":0.0,
    }
    for k,v in required.items():
        assert tr[k] == v, (k,tr[k],v)
    ev=cfg["evaluation"]
    assert (ev["scenarios_start"],ev["scenarios_end"],ev["stochastic_repeats"]) == (901,930,3)
    assert ev["policy_seed_rule"] == "scenario*100+repeat_index_0_based"
    assert ev["bootstrap_resamples"] == 10000
    assert ev["bootstrap_rng_seed"] == 20260924
    final=set(range(ev["scenarios_start"],ev["scenarios_end"]+1))
    for lo,hi in cfg["diagnostic_banks_already_consumed"]:
        assert final.isdisjoint(range(lo,hi+1))
    assert cfg["historical_12k_checkpoint_used_for_final_init"] is False


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except Exception:
        return "unknown"


def finite_tree(x: Any) -> bool:
    if isinstance(x,dict): return all(finite_tree(v) for v in x.values())
    if isinstance(x,(list,tuple)): return all(finite_tree(v) for v in x)
    if isinstance(x,(str,bool,int,np.integer)) or x is None: return True
    if isinstance(x,(float,np.floating)): return math.isfinite(float(x))
    return True


def arm_spec(arm: str) -> tuple[str,bool]:
    if arm=="proposed_marl": return "marl",True
    if arm=="marl_noresource": return "marl",False
    if arm=="single_rule": return "single",True
    raise ValueError(arm)


def verify_rule_support(include_resource_state: bool) -> dict:
    """Non-final synthetic support check; does not touch scenarios 901-930."""
    env=ResourceMARLYardEnv(
        seed=777,arrival_rate_per_hour=20.0,
        include_resource_state=include_resource_state,
        rule_resolve_proactive_pair=True,
    )
    env.reset();done=False;steps=viol=0;max_support=0
    while not done and steps<100000:
        mask=env.action_mask()
        if env.active_agent()!=STORAGE_AGENT:
            n=int(np.asarray(mask[YC_PROACTIVE_BASE:],dtype=np.bool_).sum())
            max_support=max(max_support,n)
            viol += int(n>1)
        action=heuristic_action(env)
        _,_,done,_,_=env.step(action);steps+=1
    if not done: raise RuntimeError("rule-support preflight did not terminate")
    if viol or max_support>1: raise RuntimeError((viol,max_support))
    return {"scenario":777,"decisions":steps,"max_proactive_support":max_support,"violations":viol}


def strict_checkpoint_load(kind: str, checkpoint: Path, include_resource_state: bool) -> None:
    ck=torch.load(Path(checkpoint),map_location="cpu",weights_only=False)
    env=ResourceMARLYardEnv(seed=1,arrival_rate_per_hour=20.0,include_resource_state=include_resource_state)
    env.reset()
    hidden=int(ck.get("hidden",ck.get("config",{}).get("hidden",128)))
    if kind=="marl":
        model=ResourceCooperativeModel(env.global_obs_dim,env.storage_obs_dim,env.yc_obs_dim,hidden=hidden)
    else:
        model=CentralizedSingleModel(env.global_obs_dim,env.yc_obs_dim,hidden=hidden)
    model.load_state_dict(ck.get("state_dict",ck),strict=True)


def build_bc(cfg: dict, arm: str, out: Path) -> Path:
    kind,include_resource=arm_spec(arm)
    bc=cfg["bc"]; tr=cfg["training"]
    path=out/f"bc_{arm}_storage_only.pt"
    metrics=train_bc(
        kind,path,
        seeds=int(bc["scenario_count"]),
        epochs=int(bc["epochs"]),
        batch_size=int(bc["batch_size"]),
        lr=float(bc["learning_rate"]),
        arrival_rate_per_hour=float(tr["arrival_rate_per_hour"]),
        include_resource_state=include_resource,
        enable_proactive=True,
        seed=int(bc["rng_seed"]),
        hidden=128,
    )
    if not finite_tree(metrics): raise RuntimeError("non-finite BC metrics")
    return path


def train_one(cfg: dict, arm: str, seed: int, out: Path, config_path: Path) -> dict:
    if arm not in cfg["learned_arms"] or seed not in cfg["training_seeds"]:
        raise ValueError((arm,seed))
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    kind,include_resource=arm_spec(arm)
    support=verify_rule_support(include_resource)
    bc_path=build_bc(cfg,arm,out)
    tr=cfg["training"]
    run_dir=out/"train"

    if kind=="marl":
        rc=ResourcePPOConfig(
            total_steps=int(tr["decision_threshold"]),
            rollout_steps=int(tr["rollout_steps"]),
            min_storage_transitions=int(tr["min_storage_transitions"]),
            min_yc_transitions=int(tr["min_yc_transitions"]),
            max_rollout_multiplier=int(tr["max_rollout_multiplier"]),
            update_epochs=int(tr["actor_critic_epochs"]),
            critic_extra_epochs=int(tr["critic_extra_epochs"]),
            minibatch_size=int(tr["minibatch_size"]),
            gamma=float(tr["gamma"]),gae_lambda=float(tr["gae_lambda"]),
            clip_coef=float(tr["clip_coef"]),learning_rate=float(tr["learning_rate"]),
            max_grad_norm=float(tr["max_grad_norm"]),seed=int(seed),
            arrival_rate_per_hour=float(tr["arrival_rate_per_hour"]),
            include_resource_state=include_resource,
            truck_wait_weight=float(tr["truck_wait_weight"]),
            storage_wait_weight=float(tr["storage_wait_weight"]),
            extra_move_weight=float(tr["extra_move_weight"]),
            risk_shaping_weight=float(tr["risk_shaping_weight"]),
            yc_queue_shaping_weight=float(tr["yc_queue_shaping_weight"]),
            enable_proactive=True,episode_complete_rollout=True,
            use_action_q_critic=False,rule_resolve_proactive_pair=True,
        )
        _,updates=train_resource_marl(rc,run_dir,bc_path)
        ck=run_dir/"resource_marl_final.pt"
    else:
        sc=SinglePPOConfig(
            total_steps=int(tr["decision_threshold"]),
            rollout_steps=int(tr["rollout_steps"]),
            min_storage_transitions=int(tr["min_storage_transitions"]),
            min_yc_transitions=int(tr["min_yc_transitions"]),
            max_rollout_multiplier=int(tr["max_rollout_multiplier"]),
            update_epochs=int(tr["actor_critic_epochs"]),
            critic_extra_epochs=int(tr["critic_extra_epochs"]),
            minibatch_size=int(tr["minibatch_size"]),
            gamma=float(tr["gamma"]),gae_lambda=float(tr["gae_lambda"]),
            clip_coef=float(tr["clip_coef"]),learning_rate=float(tr["learning_rate"]),
            max_grad_norm=float(tr["max_grad_norm"]),seed=int(seed),
            arrival_rate_per_hour=float(tr["arrival_rate_per_hour"]),
            truck_wait_weight=float(tr["truck_wait_weight"]),
            storage_wait_weight=float(tr["storage_wait_weight"]),
            extra_move_weight=float(tr["extra_move_weight"]),
            risk_shaping_weight=float(tr["risk_shaping_weight"]),
            yc_queue_shaping_weight=float(tr["yc_queue_shaping_weight"]),
            enable_proactive=True,episode_complete_rollout=True,
            rule_resolve_proactive_pair=True,
        )
        _,updates=train_single_ppo(sc,run_dir,bc_path)
        ck=run_dir/"single_ppo_final.pt"

    if not updates: raise RuntimeError("no PPO updates")
    if int(updates[-1]["global_step"]) < int(tr["decision_threshold"]):
        raise RuntimeError("decision threshold not reached")
    if not all(bool(u.get("rollout_ended_at_terminal",False)) for u in updates):
        raise RuntimeError("episode-complete integrity failure")
    if any(float(u.get("critic_extra_actor_max_abs_delta",0.0)) != 0.0 for u in updates):
        raise RuntimeError("actor changed during critic-only epochs")
    if not all(finite_tree(u) for u in updates):
        raise RuntimeError("non-finite update record")
    if sum(int(u.get("critic_extra_steps",0)) for u in updates) <= 0:
        raise RuntimeError("value20 extra critic steps missing")
    strict_checkpoint_load(kind,ck,include_resource)

    manifest={
        "phase":"train","arm":arm,"kind":kind,"training_seed":seed,
        "git_commit":git_commit(),
        "protocol_file":cfg["protocol_file"],
        "config_path":str(config_path),
        "config_sha256":sha256(config_path),
        "bc_checkpoint":str(bc_path),"bc_sha256":sha256(bc_path),
        "final_checkpoint":str(ck),"final_checkpoint_sha256":sha256(ck),
        "actual_decisions":int(updates[-1]["global_step"]),
        "ppo_updates":len(updates),
        "terminal_all":True,
        "critic_extra_actor_max_abs_delta":max(float(u.get("critic_extra_actor_max_abs_delta",0.0)) for u in updates),
        "critic_extra_steps_total":sum(int(u.get("critic_extra_steps",0)) for u in updates),
        "rule_support_preflight":support,
        "include_resource_state":include_resource,
        "rule_resolve_proactive_pair":True,
        "episode_complete_rollout":True,
        "use_action_q_critic":False,
        "historical_12k_init_used":False,
    }
    if not finite_tree(manifest): raise RuntimeError("non-finite manifest")
    (out/"train_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(manifest,sort_keys=True),flush=True)
    return manifest


def read_eval_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate_learned_eval_rows(rows: list[dict], ev: dict, include_resource: bool) -> None:
    start=int(ev["scenarios_start"]); end=int(ev["scenarios_end"])
    repeats=int(ev["stochastic_repeats"])
    expected_pairs={(scenario, scenario*100+r) for scenario in range(start,end+1) for r in range(repeats)}
    got_pairs=[(int(row["seed"]),int(row["policy_seed"])) for row in rows]
    if len(got_pairs)!=len(expected_pairs):
        raise RuntimeError(("evaluation row count mismatch",len(got_pairs),len(expected_pairs)))
    if len(set(got_pairs))!=len(got_pairs):
        raise RuntimeError("duplicate scenario/policy_seed rows")
    if set(got_pairs)!=expected_pairs:
        raise RuntimeError("missing or unexpected scenario/policy_seed rows")
    for row in rows:
        scenario=int(row["seed"])
        if not (start <= scenario <= end): raise RuntimeError("final-bank contamination")
        if row["evaluation_mode"]!="stochastic": raise RuntimeError("non-stochastic learned evaluation")
        if row["rule_resolve_proactive_pair"].lower()!="true": raise RuntimeError("rule resolver disabled")
        if row["include_resource_state"].lower()!=str(bool(include_resource)).lower():
            raise RuntimeError("resource-state flag mismatch")
        if row["enable_proactive"].lower()!="true": raise RuntimeError("proactive support disabled")
        for m in METRICS:
            if not math.isfinite(float(row[m])): raise RuntimeError(("non-finite",m))


def evaluate_learned(cfg: dict, arm: str, seed: int, checkpoint: Path, out: Path, config_path: Path) -> dict:
    if arm not in cfg["learned_arms"] or seed not in cfg["training_seeds"]:
        raise ValueError((arm,seed))
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    kind,include_resource=arm_spec(arm)
    strict_checkpoint_load(kind,checkpoint,include_resource)
    ev=cfg["evaluation"]
    csv_path=out/"eval_901_930.csv"
    summary=evaluate(
        csv_path,kind,checkpoint=Path(checkpoint),
        seeds=range(int(ev["scenarios_start"]),int(ev["scenarios_end"])+1),
        repeats=int(ev["stochastic_repeats"]),
        arrival_rate_per_hour=float(cfg["training"]["arrival_rate_per_hour"]),
        enable_proactive=True,include_resource_state=include_resource,
        stochastic=True,rule_resolve_proactive_pair=True,
    )
    rows=read_eval_csv(csv_path)
    validate_learned_eval_rows(rows,ev,include_resource)
    manifest={
        "phase":"evaluation","arm":arm,"kind":kind,"training_seed":seed,
        "git_commit":git_commit(),"protocol_file":cfg["protocol_file"],
        "config_sha256":sha256(config_path),"checkpoint_sha256":sha256(checkpoint),
        "scenarios":[int(ev["scenarios_start"]),int(ev["scenarios_end"])],
        "repeats":int(ev["stochastic_repeats"]),
        "policy_seed_rule":ev["policy_seed_rule"],
        "rows":len(rows),"summary":summary,
        "include_resource_state":include_resource,
        "rule_resolve_proactive_pair":True,
    }
    (out/"eval_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"arm":arm,"seed":seed,"J":summary["objective_proxy"],"rows":len(rows)}),flush=True)
    return manifest


def evaluate_heuristic(cfg: dict, out: Path, config_path: Path) -> dict:
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    ev=cfg["evaluation"];csv_path=out/"eval_901_930.csv"
    summary=evaluate(
        csv_path,"heuristic",
        seeds=range(int(ev["scenarios_start"]),int(ev["scenarios_end"])+1),
        repeats=1,arrival_rate_per_hour=float(cfg["training"]["arrival_rate_per_hour"]),
        stochastic=False,
    )
    rows=read_eval_csv(csv_path)
    expected_scenarios=set(range(int(ev["scenarios_start"]),int(ev["scenarios_end"])+1))
    got_scenarios=[int(row["seed"]) for row in rows]
    if len(got_scenarios)!=len(expected_scenarios) or len(set(got_scenarios))!=len(got_scenarios):
        raise RuntimeError("heuristic scenario rows are missing or duplicated")
    if set(got_scenarios)!=expected_scenarios:
        raise RuntimeError("heuristic scenario set mismatch")
    for row in rows:
        if row["method"]!="heuristic": raise RuntimeError("heuristic method label mismatch")
        if row["evaluation_mode"]!="flat_greedy": raise RuntimeError("heuristic evaluation mode mismatch")
        for m in METRICS:
            if not math.isfinite(float(row[m])): raise RuntimeError(("non-finite",m))
    manifest={
        "phase":"evaluation","arm":"heuristic","kind":"heuristic","training_seed":None,
        "git_commit":git_commit(),"protocol_file":cfg["protocol_file"],
        "config_sha256":sha256(config_path),"scenarios":[901,930],
        "repeats":1,"rows":len(rows),"summary":summary,
    }
    (out/"eval_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"arm":"heuristic","J":summary["objective_proxy"],"rows":len(rows)}),flush=True)
    return manifest


def print_plan(cfg: dict, config_path: Path) -> None:
    plan={
        "git_commit":git_commit(),"config_sha256":sha256(config_path),
        "learned_runs":[{"arm":a,"seed":s} for a in cfg["learned_arms"] for s in cfg["training_seeds"]],
        "n_learned_training_runs":len(cfg["learned_arms"])*len(cfg["training_seeds"]),
        "bc_scenarios":list(range(cfg["bc"]["start_seed"],cfg["bc"]["start_seed"]+cfg["bc"]["scenario_count"])),
        "final_scenarios":list(range(cfg["evaluation"]["scenarios_start"],cfg["evaluation"]["scenarios_end"]+1)),
        "learned_eval_episodes_per_arm":len(cfg["training_seeds"])*30*cfg["evaluation"]["stochastic_repeats"],
        "historical_12k_init_used":False,
        "manual_execution_required":True,
    }
    print(json.dumps(plan,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/final_experiment_20260924.json")
    sub=ap.add_subparsers(dest="command",required=True)
    sub.add_parser("plan")
    p=sub.add_parser("train");p.add_argument("--arm",required=True);p.add_argument("--seed",type=int,required=True);p.add_argument("--out",required=True)
    p=sub.add_parser("eval-learned");p.add_argument("--arm",required=True);p.add_argument("--seed",type=int,required=True);p.add_argument("--checkpoint",required=True);p.add_argument("--out",required=True)
    p=sub.add_parser("eval-heuristic");p.add_argument("--out",required=True)
    a=ap.parse_args();cp=Path(a.config);cfg=load_config(cp)
    if a.command=="plan": print_plan(cfg,cp)
    elif a.command=="train": train_one(cfg,a.arm,a.seed,Path(a.out),cp)
    elif a.command=="eval-learned": evaluate_learned(cfg,a.arm,a.seed,Path(a.checkpoint),Path(a.out),cp)
    else: evaluate_heuristic(cfg,Path(a.out),cp)


if __name__=="__main__":
    main()
