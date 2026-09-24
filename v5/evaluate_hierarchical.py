from __future__ import annotations

"""Evaluation helper for the isolated V5 hierarchical Proposed policy."""

from pathlib import Path
import csv
import json
from typing import Iterable

import numpy as np
import torch

from train_yc_marl import masked_distribution
from yc_marl_env import ResourceMARLYardEnv, STORAGE_AGENT
from v5.hierarchical_yc import (
    HierarchicalResourceCooperativeModel,
    build_hierarchical_yc_input,
    heuristic_ranks_for_selected_pair,
)


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


def load_hierarchical_model(checkpoint:Path)->HierarchicalResourceCooperativeModel:
    ck=torch.load(Path(checkpoint),map_location="cpu",weights_only=False)
    if ck.get("architecture_version")!="v5_hierarchical_operation_target_destination":
        raise RuntimeError(("not a V5 hierarchical checkpoint",ck.get("architecture_version")))
    cfg=ck.get("config",{})
    env=ResourceMARLYardEnv(
        seed=1,
        arrival_rate_per_hour=float(cfg.get("arrival_rate_per_hour",20.0)),
        include_resource_state=bool(cfg.get("include_resource_state",True)),
        rule_resolve_proactive_pair=False,
    )
    env.reset()
    model=HierarchicalResourceCooperativeModel(
        env.global_obs_dim,
        env.storage_obs_dim,
        hidden=int(cfg.get("hidden",128)),
        candidate_hidden=int(cfg.get("candidate_hidden",32)),
        proactive_bias=float(cfg.get("yc_proactive_init_bias",-2.197224577)),
    )
    model.load_state_dict(ck["state_dict"],strict=True)
    model.eval()
    return model


def run_hierarchical_episode(
    seed:int,
    model:HierarchicalResourceCooperativeModel,
    *,
    arrival_rate_per_hour:float=20.0,
    include_resource_state:bool=True,
    stochastic:bool=True,
    policy_seed:int=0,
)->dict:
    env=ResourceMARLYardEnv(
        seed=int(seed),
        arrival_rate_per_hour=float(arrival_rate_per_hour),
        include_resource_state=bool(include_resource_state),
        enable_proactive=True,
        rule_resolve_proactive_pair=False,
    )
    env.reset()
    rng=torch.Generator(device="cpu").manual_seed(int(policy_seed))
    done=False
    steps=0

    ppro=[]
    op_entropy=[]
    target_entropy=[]
    dest_entropy=[]
    selected_target_rank=[]
    selected_dest_rank=[]
    feasible_targets=[]
    feasible_pairs=[]

    while not done and steps<100000:
        flat_mask=torch.as_tensor(env.action_mask(),dtype=torch.bool)
        if env.active_agent()==STORAGE_AGENT:
            obs=torch.as_tensor(env.actor_observation(),dtype=torch.float32)
            with torch.inference_mode():
                dist=masked_distribution(model.storage_logits(obs),flat_mask)
            if stochastic:
                action=int(torch.multinomial(dist.probs,1,generator=rng).item())
            else:
                action=int(dist.probs.argmax().item())
        else:
            block=env.active_block()
            if block is None:
                raise RuntimeError("YC decision missing block")
            obs_np,pair_np=build_hierarchical_yc_input(
                env,block,resource_state=include_resource_state
            )
            obs=torch.as_tensor(obs_np,dtype=torch.float32)
            pair=torch.as_tensor(pair_np,dtype=torch.bool)
            sample=model.yc_actor.sample(
                obs,pair,flat_mask,generator=rng,stochastic=stochastic
            )
            action=sample.flat_action
            ppro.append(sample.p_proactive)
            op_entropy.append(sample.operation_entropy)
            target_entropy.append(sample.target_entropy_norm)
            dest_entropy.append(sample.destination_entropy_norm)
            feasible_targets.append(float(pair_np.any(axis=1).sum()))
            feasible_pairs.append(float(pair_np.sum()))
            if sample.operation==1:
                tr,dr=heuristic_ranks_for_selected_pair(
                    env,block,sample.target_position,sample.destination
                )
                selected_target_rank.append(float(tr))
                selected_dest_rank.append(float(dr))

        _,_,done,_,info=env.step(action)
        steps+=1

    if not done:
        raise RuntimeError(("V5 evaluation episode did not terminate",seed,policy_seed))
    k=info["kpis"]
    objective=float(
        k["mean_truck_completion_delay"]
        +k["mean_storage_completion_delay"]
        +0.10*k["extra_yc_minutes_per_retrieval"]
    )
    return {
        "seed":int(seed),
        "method":"v5_hierarchical_marl",
        "evaluation_mode":"stochastic" if stochastic else "hierarchical_greedy",
        "policy_seed":int(policy_seed),
        "arrival_rate_per_hour":float(arrival_rate_per_hour),
        "include_resource_state":bool(include_resource_state),
        "rule_resolve_proactive_pair":False,
        "decisions":int(steps),
        "objective_proxy":objective,
        "yc_proactive_probability":float(np.mean(ppro)) if ppro else 0.0,
        "yc_operation_entropy":float(np.mean(op_entropy)) if op_entropy else 0.0,
        "yc_target_entropy_norm":float(np.mean(target_entropy)) if target_entropy else 0.0,
        "yc_destination_entropy_norm":float(np.mean(dest_entropy)) if dest_entropy else 0.0,
        "yc_feasible_targets":float(np.mean(feasible_targets)) if feasible_targets else 0.0,
        "yc_feasible_pairs":float(np.mean(feasible_pairs)) if feasible_pairs else 0.0,
        "yc_selected_target_heuristic_rank":float(np.mean(selected_target_rank)) if selected_target_rank else 0.0,
        "yc_selected_destination_heuristic_rank":float(np.mean(selected_dest_rank)) if selected_dest_rank else 0.0,
        **k,
    }


def evaluate_hierarchical(
    out_csv:Path,
    checkpoint:Path,
    *,
    seeds:Iterable[int],
    repeats:int=3,
    arrival_rate_per_hour:float=20.0,
    include_resource_state:bool=True,
    stochastic:bool=True,
)->dict:
    model=load_hierarchical_model(Path(checkpoint))
    rows=[]
    for seed in seeds:
        nrep=int(repeats) if stochastic else 1
        for r in range(nrep):
            rows.append(
                run_hierarchical_episode(
                    int(seed),model,
                    arrival_rate_per_hour=arrival_rate_per_hour,
                    include_resource_state=include_resource_state,
                    stochastic=stochastic,
                    policy_seed=int(seed)*100+r,
                )
            )
    if not rows:
        raise RuntimeError("no V5 evaluation rows")

    out_csv=Path(out_csv)
    out_csv.parent.mkdir(parents=True,exist_ok=True)
    with out_csv.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    summary={
        key:float(np.mean([float(x[key]) for x in rows]))
        for key in METRICS
    }
    summary.update({
        "arrival_rate_per_hour":float(arrival_rate_per_hour),
        "n_scenarios":len({int(x["seed"]) for x in rows}),
        "evaluation_mode":"stochastic" if stochastic else "hierarchical_greedy",
        "repeats":int(repeats if stochastic else 1),
        "rows":len(rows),
        "architecture_version":"v5_hierarchical_operation_target_destination",
    })
    out_csv.with_suffix(".json").write_text(
        json.dumps(summary,indent=2)+"\n",encoding="utf-8"
    )
    return summary
