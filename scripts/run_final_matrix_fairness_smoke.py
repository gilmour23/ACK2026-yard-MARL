from __future__ import annotations

"""Implementation-parity smoke test for the final learned-policy matrix."""

from pathlib import Path
import argparse, json, math
import numpy as np

from pretrain_yc_bc import train_bc
from train_yc_marl import ResourcePPOConfig, train_resource_marl
from train_yc_single import SinglePPOConfig, train_single_ppo
from evaluate_yc_policies import evaluate


def finite_tree(x):
    if isinstance(x,dict): return all(finite_tree(v) for v in x.values())
    if isinstance(x,(list,tuple)): return all(finite_tree(v) for v in x)
    if isinstance(x,(int,np.integer,bool,str)) or x is None: return True
    if isinstance(x,(float,np.floating)): return math.isfinite(float(x))
    return True


def run(out:Path,arm:str,seed:int):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if arm not in {"proposed_marl","marl_noresource","single_rule"}:
        raise ValueError(arm)

    kind="single" if arm=="single_rule" else "marl"
    include_resource = arm!="marl_noresource"
    bc=out/f"bc_{kind}_storage_only.pt"
    bc_metrics=train_bc(
        kind,bc,seeds=4,epochs=2,batch_size=128,lr=5e-4,
        arrival_rate_per_hour=20.0,
        include_resource_state=include_resource,
        enable_proactive=True,
        seed=20260924,
        hidden=128,
    )

    if kind=="marl":
        cfg=ResourcePPOConfig(
            total_steps=2000,rollout_steps=512,min_storage_transitions=64,min_yc_transitions=128,
            max_rollout_multiplier=4,update_epochs=2,critic_extra_epochs=18,minibatch_size=256,
            gamma=1.0,gae_lambda=1.0,learning_rate=3e-4,max_grad_norm=0.5,seed=int(seed),
            arrival_rate_per_hour=20.0,include_resource_state=include_resource,
            episode_complete_rollout=True,use_action_q_critic=False,rule_resolve_proactive_pair=True,
        )
        _,updates=train_resource_marl(cfg,out/"train",bc)
        ck=out/"train"/"resource_marl_final.pt"
        method="marl"
    else:
        cfg=SinglePPOConfig(
            total_steps=2000,rollout_steps=512,min_storage_transitions=64,min_yc_transitions=128,
            max_rollout_multiplier=4,update_epochs=2,critic_extra_epochs=18,minibatch_size=256,
            gamma=1.0,gae_lambda=1.0,learning_rate=3e-4,max_grad_norm=0.5,seed=int(seed),
            arrival_rate_per_hour=20.0,episode_complete_rollout=True,rule_resolve_proactive_pair=True,
        )
        _,updates=train_single_ppo(cfg,out/"train",bc)
        ck=out/"train"/"single_ppo_final.pt"
        method="single"

    if not updates: raise RuntimeError("no updates")
    if not all(bool(r.get("rollout_ended_at_terminal",False)) for r in updates):
        raise RuntimeError("non-terminal rollout")
    if any(float(r.get("critic_extra_actor_max_abs_delta",0.0))!=0.0 for r in updates):
        raise RuntimeError("actor changed in value-only extra epochs")

    ev=evaluate(
        out/"eval_881_885.csv",method,checkpoint=ck,seeds=range(881,886),repeats=2,
        arrival_rate_per_hour=20.0,enable_proactive=True,include_resource_state=include_resource,
        yc_move_time=2.0,stochastic=True,rule_resolve_proactive_pair=True,
    )
    summary={
        "arm":arm,"seed":int(seed),"bc":bc_metrics,
        "training":{
            "actual_decisions":int(updates[-1]["global_step"]),
            "ppo_updates":len(updates),
            "terminal_all":all(bool(r.get("rollout_ended_at_terminal",False)) for r in updates),
            "critic_extra_steps":int(sum(int(r.get("critic_extra_steps",0)) for r in updates)),
            "post_value_ev_last":float(updates[-1].get("post_update_value_ev",float("nan"))),
            "post_value_rmse_last":float(updates[-1].get("post_update_value_rmse",float("nan"))),
        },
        "evaluation":ev,
    }
    if not finite_tree(summary): raise RuntimeError("non-finite summary")
    (out/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"arm":arm,"seed":seed,"J":ev["objective_proxy"],"terminal":summary["training"]["terminal_all"]}),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True)
    ap.add_argument("--arm",choices=["proposed_marl","marl_noresource","single_rule"],required=True)
    ap.add_argument("--seed",type=int,choices=[41,42],required=True)
    a=ap.parse_args();run(Path(a.out),a.arm,a.seed)
