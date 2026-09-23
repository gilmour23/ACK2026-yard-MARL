from __future__ import annotations

"""Pre-registered 2k Q-credit pilot.

Control: value20 episode-complete PPO.
Treatment: identical, with the existing action-conditioned YC Q-critic enabled.

One arm/seed is run per invocation so GitHub Actions can execute the six jobs in
parallel. Evaluation uses scenarios 801-810, three stochastic repeats.
"""

from pathlib import Path
import argparse, json, math
import numpy as np

from train_yc_marl import ResourcePPOConfig, train_resource_marl
from run_value_budget_pilot import (
    CANONICAL_CHECKPOINT_SHA256,
    sha256,
    evaluate_with_critic,
)


def finite_tree(x):
    if isinstance(x, dict):
        return all(finite_tree(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return all(finite_tree(v) for v in x)
    if isinstance(x, (int, np.integer, bool, str)) or x is None:
        return True
    if isinstance(x, (float, np.floating)):
        return math.isfinite(float(x))
    return True


def run(checkpoint: Path, out: Path, arm: str, seed: int, steps: int = 2000,
        eval_start: int = 801, eval_count: int = 10, repeats: int = 3):
    checkpoint=Path(checkpoint); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    actual_sha=sha256(checkpoint)
    if actual_sha != CANONICAL_CHECKPOINT_SHA256:
        raise RuntimeError(f"checkpoint SHA mismatch: {actual_sha}")
    if arm not in {"control_value20","treatment_qcredit"}:
        raise ValueError(arm)

    use_q = arm == "treatment_qcredit"
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
        use_action_q_critic=use_q,
        q_learning_rate=3e-4,
        q_update_epochs=3,
        q_adv_blend=0.5,
    )

    model,updates=train_resource_marl(cfg,out/"native",checkpoint)
    if not updates:
        raise RuntimeError("no PPO updates")
    if not all(bool(r.get("rollout_ended_at_terminal",False)) for r in updates):
        raise RuntimeError("non-terminal rollout found")
    if any(float(r.get("critic_extra_actor_max_abs_delta",0.0)) != 0.0 for r in updates):
        raise RuntimeError("actor changed during critic-only extra epochs")

    eval_seeds=range(int(eval_start),int(eval_start)+int(eval_count))
    ev=evaluate_with_critic(
        out/"native"/"resource_marl_final.pt",
        out/"eval_801_810_stochastic.csv",
        eval_seeds,
        repeats,
    )

    train={
        "arm":arm,
        "seed":int(seed),
        "checkpoint_sha256":actual_sha,
        "actual_decisions":int(updates[-1]["global_step"]),
        "ppo_updates":int(len(updates)),
        "episode_complete":True,
        "critic_extra_epochs":18,
        "use_action_q_critic":bool(use_q),
        "q_update_epochs":3,
        "q_adv_blend":0.5,
        "critic_extra_steps":int(sum(int(r.get("critic_extra_steps",0)) for r in updates)),
        "post_update_value_ev_last":float(updates[-1]["post_update_value_ev"]),
        "post_update_value_rmse_last":float(updates[-1]["post_update_value_rmse"]),
        "yc_q_loss_mean":float(np.mean([float(r.get("yc_q_loss",0.0)) for r in updates])),
        "yc_q_loss_last":float(updates[-1].get("yc_q_loss",0.0)),
        "yc_q_adv_std_mean":float(np.mean([float(r.get("yc_q_adv_std",0.0)) for r in updates])),
        "yc_pair_entropy_norm_mean":float(np.mean([float(r["yc_pair_entropy_norm"]) for r in updates])),
        "yc_proactive_probability_mean":float(np.mean([float(r["yc_proactive_probability"]) for r in updates])),
        "yc_sampled_proactive_rate_mean":float(np.mean([float(r["yc_sampled_proactive_rate"]) for r in updates])),
        "rollout_decisions":[int(r["rollout_decisions"]) for r in updates],
        "rollout_terminal":[bool(r["rollout_ended_at_terminal"]) for r in updates],
    }
    summary={
        "protocol":"audits/Q_CREDIT_PILOT_PROTOCOL_20260923.md",
        "training":train,
        "evaluation":ev,
    }
    if not finite_tree(summary):
        raise RuntimeError("non-finite summary metric")
    (out/"q_credit_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({
        "arm":arm,
        "seed":int(seed),
        "decisions":train["actual_decisions"],
        "J":ev["policy"]["objective_proxy"],
        "pair_tv":ev["pair"]["conditional_pair_tv_uniform_mean"],
        "q_loss":train["yc_q_loss_mean"],
        "value_ev":ev["critic"]["ev"],
    }),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--arm",choices=["control_value20","treatment_qcredit"],required=True)
    ap.add_argument("--seed",type=int,choices=[21,22,23],required=True)
    ap.add_argument("--steps",type=int,default=2000)
    ap.add_argument("--eval-start",type=int,default=801)
    ap.add_argument("--eval-count",type=int,default=10)
    ap.add_argument("--repeats",type=int,default=3)
    a=ap.parse_args()
    run(Path(a.checkpoint),Path(a.out),a.arm,a.seed,a.steps,a.eval_start,a.eval_count,a.repeats)
