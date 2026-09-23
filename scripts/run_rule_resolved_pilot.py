from __future__ import annotations

"""Pre-registered rule-resolved proactive structural pilot.

One arm/seed per invocation for parallel GitHub Actions execution.
Control keeps the current flat Target×Destination action support.
Treatment exposes one deterministic proactive pair chosen by the existing
ETA/blocker target priority and information-aware relocation heuristic.
"""

from pathlib import Path
import argparse, csv, json, math
import numpy as np
import torch

from train_yc_marl import (
    ResourcePPOConfig,
    train_resource_marl,
    masked_distribution,
    structured_yc_entropy,
    YC_PROACTIVE_BASE,
)
from evaluate_yc_policies import _load_model
from run_value_budget_pilot import CANONICAL_CHECKPOINT_SHA256, sha256
from yc_marl_env import ResourceMARLYardEnv, STORAGE_AGENT


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


def metric_summary(pred, target):
    p=np.asarray(pred,dtype=np.float64)
    y=np.asarray(target,dtype=np.float64)
    err=p-y
    var=float(np.var(y))
    return {
        "n":int(len(y)),
        "ev":float("nan") if var < 1e-12 else float(1.0-np.var(err)/var),
        "rmse":float(np.sqrt(np.mean(err*err))),
        "bias":float(np.mean(err)),
        "prediction_std":float(np.std(p)),
        "target_std":float(np.std(y)),
    }


def evaluate(checkpoint: Path, out_csv: Path, rule_resolved: bool,
             scenarios=range(851,861), repeats: int=3):
    model=_load_model("marl",checkpoint,20.0)
    model.eval()
    rows=[]
    all_pred=[]
    all_mc=[]
    ppro=[]
    proactive_support=[]
    support_violations=0

    for scenario in scenarios:
        for repeat in range(repeats):
            env=ResourceMARLYardEnv(
                seed=int(scenario),
                arrival_rate_per_hour=20.0,
                rule_resolve_proactive_pair=bool(rule_resolved),
            )
            env.reset()
            generator=torch.Generator(device="cpu").manual_seed(int(scenario)*100+repeat)
            rewards=[]; preds=[]; done=False; steps=0
            while not done and steps < 100000:
                role=0 if env.active_agent()==STORAGE_AGENT else 1
                mask_np=env.action_mask()
                mask=torch.as_tensor(mask_np,dtype=torch.bool)
                gobs=torch.as_tensor(env.critic_observation(),dtype=torch.float32)
                with torch.inference_mode():
                    pred=float(model.value(gobs).item())
                    aobs=torch.as_tensor(env.actor_observation(),dtype=torch.float32)
                    logits=model.storage_logits(aobs) if role==0 else model.yc_logits(aobs,mask)
                    dist=masked_distribution(logits,mask)
                    if role==1:
                        _,_,p=structured_yc_entropy(dist,mask)
                        ppro.append(float(p.item()))
                        support=int(mask[YC_PROACTIVE_BASE:].sum().item())
                        proactive_support.append(support)
                        if rule_resolved and support > 1:
                            support_violations += 1
                    action=int(torch.multinomial(dist.probs,1,generator=generator).item())
                _,reward,done,_,info=env.step(action)
                preds.append(pred); rewards.append(float(reward)); steps+=1
            if not done:
                raise RuntimeError(f"episode did not terminate: scenario={scenario} repeat={repeat}")
            mc=np.zeros(len(rewards),dtype=np.float64); running=0.0
            for i in range(len(rewards)-1,-1,-1):
                running=float(rewards[i])+running
                mc[i]=running
            all_pred.extend(preds); all_mc.extend(mc.tolist())
            k=info["kpis"]
            rows.append({
                "scenario":int(scenario),
                "repeat":int(repeat),
                "policy_seed":int(scenario)*100+repeat,
                "decisions":int(steps),
                "objective_proxy":float(
                    k["mean_truck_completion_delay"]
                    + k["mean_storage_completion_delay"]
                    + 0.1*k["extra_yc_minutes_per_retrieval"]
                ),
                **k,
            })

    if rule_resolved and support_violations:
        raise RuntimeError(f"rule-resolved support violation count={support_violations}")

    out_csv=Path(out_csv); out_csv.parent.mkdir(parents=True,exist_ok=True)
    with out_csv.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for row in rows for k in row}))
        w.writeheader(); w.writerows(rows)

    policy={
        key:float(np.mean([r[key] for r in rows]))
        for key in (
            "mean_truck_completion_delay",
            "mean_storage_completion_delay",
            "rehandling_moves",
            "proactive_moves",
            "extra_yc_minutes_per_retrieval",
            "mean_yc_utilization",
            "max_yc_queue",
            "objective_proxy",
        )
    }
    summary={
        "episodes":len(rows),
        "scenarios":list(map(int,scenarios)),
        "repeats":int(repeats),
        "rule_resolved_proactive_pair":bool(rule_resolved),
        "critic":metric_summary(all_pred,all_mc),
        "policy":policy,
        "yc":{
            "p_proactive_mean":float(np.mean(ppro)) if ppro else 0.0,
            "proactive_support_mean":float(np.mean(proactive_support)) if proactive_support else 0.0,
            "proactive_support_max":int(max(proactive_support)) if proactive_support else 0,
            "support_violations":int(support_violations),
        },
    }
    out_csv.with_suffix(".json").write_text(
        json.dumps(summary,indent=2,allow_nan=True)+"\n",encoding="utf-8"
    )
    return summary


def run(checkpoint: Path, out: Path, arm: str, seed: int, steps: int=2000):
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
        out/"eval_851_860_stochastic.csv",
        rule_resolved=rule,
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
        "protocol":"audits/RULE_RESOLVED_PROACTIVE_PILOT_PROTOCOL_20260923.md",
        "training":train,
        "evaluation":ev,
    }
    if not finite_tree(summary):
        raise RuntimeError("non-finite summary metric")
    (out/"rule_resolved_summary.json").write_text(
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
    ap.add_argument("--seed",type=int,choices=[21,22,23],required=True)
    ap.add_argument("--steps",type=int,default=2000)
    a=ap.parse_args()
    run(Path(a.checkpoint),Path(a.out),a.arm,a.seed,a.steps)
