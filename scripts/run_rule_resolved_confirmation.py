from __future__ import annotations

"""6k bounded confirmation for the rule-resolved proactive formulation."""

from pathlib import Path
import argparse, json, math
import numpy as np

from train_yc_marl import ResourcePPOConfig, train_resource_marl
from run_value_budget_pilot import CANONICAL_CHECKPOINT_SHA256, sha256
from run_rule_resolved_pilot import evaluate, finite_tree


def run(checkpoint: Path, out: Path, arm: str, seed: int, steps: int=6000):
    checkpoint=Path(checkpoint); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    actual_sha=sha256(checkpoint)
    if actual_sha != CANONICAL_CHECKPOINT_SHA256:
        raise RuntimeError(f"checkpoint SHA mismatch: {actual_sha}")
    if arm not in {"control_flat","treatment_rule"}:
        raise ValueError(arm)
    rule=arm=="treatment_rule"

    cfg=ResourcePPOConfig(
        total_steps=int(steps),
        rollout_steps=512,
        min_storage_transitions=64,
        min_yc_transitions=128,
        max_rollout_multiplier=4,
        update_epochs=2,
        critic_extra_epochs=18,
        minibatch_size=256,
        gamma=1.0,
        gae_lambda=1.0,
        learning_rate=3e-4,
        max_grad_norm=0.5,
        seed=int(seed),
        arrival_rate_per_hour=20.0,
        episode_complete_rollout=True,
        use_action_q_critic=False,
        rule_resolve_proactive_pair=bool(rule),
    )
    _,updates=train_resource_marl(cfg,out/"native",checkpoint)
    if not updates:
        raise RuntimeError("no PPO updates")
    if not all(bool(r.get("rollout_ended_at_terminal",False)) for r in updates):
        raise RuntimeError("non-terminal training rollout")
    if any(float(r.get("critic_extra_actor_max_abs_delta",0.0)) != 0.0 for r in updates):
        raise RuntimeError("actor changed during critic-only extra epochs")

    ev=evaluate(
        out/"native"/"resource_marl_final.pt",
        out/"eval_871_880_stochastic.csv",
        rule_resolved=rule,
        scenarios=range(871,881),
        repeats=3,
    )
    train={
        "arm":arm,
        "seed":int(seed),
        "checkpoint_sha256":actual_sha,
        "actual_decisions":int(updates[-1]["global_step"]),
        "ppo_updates":int(len(updates)),
        "critic_extra_steps":int(sum(int(r.get("critic_extra_steps",0)) for r in updates)),
        "post_update_value_ev_last":float(updates[-1]["post_update_value_ev"]),
        "post_update_value_rmse_last":float(updates[-1]["post_update_value_rmse"]),
        "rule_resolve_proactive_pair":bool(rule),
        "yc_proactive_probability_mean":float(
            np.mean([float(r["yc_proactive_probability"]) for r in updates])
        ),
        "yc_sampled_proactive_rate_mean":float(
            np.mean([float(r["yc_sampled_proactive_rate"]) for r in updates])
        ),
        "rollout_decisions":[int(r["rollout_decisions"]) for r in updates],
        "rollout_terminal":[bool(r["rollout_ended_at_terminal"]) for r in updates],
    }
    summary={
        "protocol":"audits/RULE_RESOLVED_CONFIRMATION_PROTOCOL_20260923.md",
        "training":train,
        "evaluation":ev,
    }
    if not finite_tree(summary):
        raise RuntimeError("non-finite summary metric")
    (out/"rule_resolved_confirmation.json").write_text(
        json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8"
    )
    print(json.dumps({
        "arm":arm,
        "seed":int(seed),
        "decisions":train["actual_decisions"],
        "J":ev["policy"]["objective_proxy"],
        "truck":ev["policy"]["mean_truck_completion_delay"],
        "storage":ev["policy"]["mean_storage_completion_delay"],
        "rehandling":ev["policy"]["rehandling_moves"],
        "proactive":ev["policy"]["proactive_moves"],
        "p_proactive":ev["yc"]["p_proactive_mean"],
        "support_max":ev["yc"]["proactive_support_max"],
        "value_ev":ev["critic"]["ev"],
    }),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--arm",choices=["control_flat","treatment_rule"],required=True)
    ap.add_argument("--seed",type=int,choices=[31,32,33],required=True)
    ap.add_argument("--steps",type=int,default=6000)
    a=ap.parse_args()
    run(Path(a.checkpoint),Path(a.out),a.arm,a.seed,a.steps)
