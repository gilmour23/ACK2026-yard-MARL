from __future__ import annotations

"""Collect exhaustive branch-credit counterfactual labels for frozen V5 checkpoints.

No policy parameter is updated.  Each probe state is copied, one proactive
Target×Destination action is forced, and the frozen V5 policy continues in
hierarchical greedy mode to terminal.
"""

from pathlib import Path
import argparse
import copy
import json
import math

import numpy as np
import torch
from torch.distributions import Categorical

from train_yc_marl import masked_distribution
from yc_marl_env import (
    ResourceMARLYardEnv,
    STORAGE_AGENT,
    YC_PROACTIVE_BASE,
    encode_proactive_action,
)
from v5.evaluate_hierarchical import load_hierarchical_model
from v5.hierarchical_yc import (
    build_hierarchical_yc_input,
    split_hierarchical_obs,
)


def objective(k:dict)->float:
    return float(
        k["mean_truck_completion_delay"]
        +k["mean_storage_completion_delay"]
        +0.10*k["extra_yc_minutes_per_retrieval"]
    )


def policy_step_stochastic(env,model,rng:torch.Generator)->int:
    flat_mask=torch.as_tensor(env.action_mask(),dtype=torch.bool)
    if env.active_agent()==STORAGE_AGENT:
        obs=torch.as_tensor(env.actor_observation(),dtype=torch.float32)
        with torch.inference_mode():
            dist=masked_distribution(model.storage_logits(obs),flat_mask)
        return int(torch.multinomial(dist.probs,1,generator=rng).item())
    block=env.active_block()
    if block is None:
        raise RuntimeError("YC decision missing block")
    obs_np,pair_np=build_hierarchical_yc_input(env,block)
    obs=torch.as_tensor(obs_np,dtype=torch.float32)
    pair=torch.as_tensor(pair_np,dtype=torch.bool)
    with torch.inference_mode():
        sample=model.yc_actor.sample(
            obs,pair,flat_mask,generator=rng,stochastic=True
        )
    return int(sample.flat_action)


def policy_step_greedy(env,model)->int:
    flat_mask=torch.as_tensor(env.action_mask(),dtype=torch.bool)
    if env.active_agent()==STORAGE_AGENT:
        obs=torch.as_tensor(env.actor_observation(),dtype=torch.float32)
        with torch.inference_mode():
            dist=masked_distribution(model.storage_logits(obs),flat_mask)
        return int(dist.probs.argmax().item())
    block=env.active_block()
    if block is None:
        raise RuntimeError("YC decision missing block")
    obs_np,pair_np=build_hierarchical_yc_input(env,block)
    obs=torch.as_tensor(obs_np,dtype=torch.float32)
    pair=torch.as_tensor(pair_np,dtype=torch.bool)
    with torch.inference_mode():
        sample=model.yc_actor.sample(
            obs,pair,flat_mask,generator=None,stochastic=False
        )
    return int(sample.flat_action)


def find_probe(model,scenario:int,training_seed:int,min_targets:int=3):
    env=ResourceMARLYardEnv(
        seed=scenario,
        arrival_rate_per_hour=20.0,
        include_resource_state=True,
        rule_resolve_proactive_pair=False,
    )
    env.reset()
    policy_seed=int(scenario*100)
    rng=torch.Generator(device="cpu").manual_seed(policy_seed)
    done=False
    decisions=0
    while not done and decisions<100000:
        if env.active_agent()!=STORAGE_AGENT:
            block=env.active_block()
            if block is None:
                raise RuntimeError("YC decision missing block")
            obs,pair_mask=build_hierarchical_yc_input(env,block)
            if int(pair_mask.any(axis=1).sum())>=min_targets:
                return copy.deepcopy(env),{
                    "scenario":int(scenario),
                    "training_seed":int(training_seed),
                    "policy_collection_seed":policy_seed,
                    "decision_index":int(decisions),
                    "block":int(block),
                    "eligible_targets":int(pair_mask.any(axis=1).sum()),
                    "feasible_pairs":int(pair_mask.sum()),
                }
        action=policy_step_stochastic(env,model,rng)
        _,_,done,_,_=env.step(action)
        decisions+=1
    return None,{
        "scenario":int(scenario),
        "training_seed":int(training_seed),
        "policy_collection_seed":policy_seed,
        "status":"no_probe",
    }


def destination_heuristic_ranks(env:ResourceMARLYardEnv,block:int,target_pos:int)->dict[int,int]:
    sim=env.sim
    assert sim is not None
    cid=sim._cid_at_target_position(block,int(target_pos))
    if cid is None:
        raise RuntimeError(("missing target",target_pos))
    c=sim.containers[cid]
    if c.stack is None:
        raise RuntimeError(("target has no source stack",cid))
    source=int(c.stack)
    moving_id=sim.yard.top(block,source)
    if moving_id is None or moving_id==cid:
        raise RuntimeError(("no blocker",cid))
    moving=sim.containers[moving_id]
    scored=[]
    for dest in sim.yard.feasible_relocation_stacks(block,source):
        stack=sim.yard.stacks[block][dest]
        inversion=sum(
            1 for lower_id in stack
            if sim.containers[lower_id].eta < moving.eta
        )
        height=len(stack)/sim.yard.max_tier
        scored.append((float(inversion+0.35*height),int(dest)))
    scored.sort(key=lambda x:(x[0],x[1]))
    return {dest:i+1 for i,(_,dest) in enumerate(scored)}


def force_pair_continue(snapshot,model,target_pos:int,dest:int)->dict:
    env=copy.deepcopy(snapshot)
    mask=np.asarray(env.action_mask(),dtype=np.bool_)
    flat=encode_proactive_action(int(target_pos),int(dest))
    if flat>=len(mask) or not bool(mask[flat]):
        raise RuntimeError(("forced pair invalid",target_pos,dest))
    _,_,done,_,info=env.step(flat)
    decisions=1
    while not done and decisions<100000:
        action=policy_step_greedy(env,model)
        _,_,done,_,info=env.step(action)
        decisions+=1
    if not done:
        raise RuntimeError("greedy continuation did not terminate")
    k=info["kpis"]
    return {
        "J":objective(k),
        "continuation_decisions":int(decisions),
        "mean_truck_completion_delay":float(k["mean_truck_completion_delay"]),
        "mean_storage_completion_delay":float(k["mean_storage_completion_delay"]),
        "rehandling_moves":float(k["rehandling_moves"]),
        "extra_yc_minutes_per_retrieval":float(k["extra_yc_minutes_per_retrieval"]),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--training-seed",required=True,type=int)
    ap.add_argument("--scenario-start",type=int,default=1221)
    ap.add_argument("--scenario-end",type=int,default=1230)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    seed=int(a.training_seed)
    if seed not in {61,62,63}:
        raise ValueError(seed)

    scenario_start=int(a.scenario_start)
    scenario_end=int(a.scenario_end)
    if not (1221 <= scenario_start <= scenario_end <= 1230):
        raise ValueError((scenario_start,scenario_end))

    model=load_hierarchical_model(Path(a.checkpoint))
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)

    pair_rows=[]
    target_rows=[]
    states=[]
    no_probe=[]

    for scenario in range(scenario_start,scenario_end+1):
        snapshot,meta=find_probe(model,scenario,seed,min_targets=3)
        if snapshot is None:
            no_probe.append(meta)
            continue
        block=int(meta["block"])
        obs_np,pair_np=build_hierarchical_yc_input(snapshot,block)
        context_t,target_t,dest_t=split_hierarchical_obs(
            torch.as_tensor(obs_np,dtype=torch.float32)
        )
        context=context_t.detach().cpu().numpy().astype(float)
        target_features=target_t.detach().cpu().numpy().astype(float)
        dest_features=dest_t.detach().cpu().numpy().astype(float)

        with torch.inference_mode():
            _,_,dest_logits,_=model.yc_actor.components(
                torch.as_tensor(obs_np,dtype=torch.float32),
                torch.as_tensor(pair_np,dtype=torch.bool),
            )
        dest_logits=dest_logits.detach().cpu()
        sim=snapshot.sim
        assert sim is not None
        candidates=list(sim.proactive_candidates(block))
        target_rank_by_pos={}
        for rank,cid in enumerate(candidates,start=1):
            pos=int(sim._target_position(cid))
            target_rank_by_pos[pos]=int(rank)

        state_pair_rows=[]
        state_target_rows=[]
        for target_pos in np.flatnonzero(pair_np.any(axis=1)):
            t=int(target_pos)
            dmask=torch.as_tensor(pair_np[t],dtype=torch.bool)
            d_dist=Categorical(
                logits=dest_logits[t].masked_fill(
                    ~dmask,torch.finfo(dest_logits.dtype).min
                )
            )
            d_probs=d_dist.probs.detach().cpu().numpy().astype(float)
            heur_d_rank=destination_heuristic_ranks(snapshot,block,t)
            t_pair_rows=[]
            for dest in np.flatnonzero(pair_np[t]):
                d=int(dest)
                result=force_pair_continue(snapshot,model,t,d)
                row={
                    **meta,
                    "target_position":t,
                    "destination":d,
                    "target_heuristic_rank":int(target_rank_by_pos[t]),
                    "destination_heuristic_rank":int(heur_d_rank[d]),
                    "destination_policy_probability":float(d_probs[d]),
                    "x_destination":np.concatenate([
                        context,target_features[t],dest_features[d]
                    ]).astype(float).tolist(),
                    "Q_destination_label":float(-result["J"]),
                    **result,
                }
                pair_rows.append(row)
                state_pair_rows.append(row)
                t_pair_rows.append(row)

            probs=np.asarray(
                [r["destination_policy_probability"] for r in t_pair_rows],
                dtype=np.float64,
            )
            probs=probs/probs.sum()
            qvals=np.asarray(
                [r["Q_destination_label"] for r in t_pair_rows],
                dtype=np.float64,
            )
            tdests=np.asarray(
                [dest_features[int(r["destination"])] for r in t_pair_rows],
                dtype=np.float64,
            )
            weighted_mean=(probs[:,None]*tdests).sum(axis=0)
            dmax=tdests.max(axis=0)
            q_target=float((probs*qvals).sum())
            tr={
                **meta,
                "target_position":t,
                "target_heuristic_rank":int(target_rank_by_pos[t]),
                "feasible_destinations":len(t_pair_rows),
                "x_target":np.concatenate([
                    context,target_features[t],weighted_mean,dmax
                ]).astype(float).tolist(),
                "Q_target_label":q_target,
                "expected_J_under_destination_policy":float(-q_target),
            }
            target_rows.append(tr)
            state_target_rows.append(tr)

        states.append({
            **meta,
            "pair_rows":len(state_pair_rows),
            "target_rows":len(state_target_rows),
        })

    if not states:
        raise RuntimeError("no valid probe states")
    for row in pair_rows:
        if not math.isfinite(float(row["J"])):
            raise RuntimeError("non-finite pair label")
    for row in target_rows:
        if not math.isfinite(float(row["Q_target_label"])):
            raise RuntimeError("non-finite target label")

    summary={
        "training_seed":seed,
        "scenario_start":scenario_start,
        "scenario_end":scenario_end,
        "valid_probe_states":len(states),
        "no_probe_states":len(no_probe),
        "destination_rows":len(pair_rows),
        "target_rows":len(target_rows),
        "mean_pairs_per_state":float(np.mean([x["pair_rows"] for x in states])),
        "mean_targets_per_state":float(np.mean([x["target_rows"] for x in states])),
        "continuation_mode":"frozen_v5_hierarchical_greedy_after_forced_pair",
        "policy_probe_seed_rule":"scenario*100",
    }
    (out/"summary.json").write_text(
        json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8"
    )
    (out/"states.json").write_text(
        json.dumps(states,indent=2,allow_nan=False)+"\n",encoding="utf-8"
    )
    (out/"destination_rows.json").write_text(
        json.dumps(pair_rows,allow_nan=False)+"\n",encoding="utf-8"
    )
    (out/"target_rows.json").write_text(
        json.dumps(target_rows,allow_nan=False)+"\n",encoding="utf-8"
    )
    (out/"no_probe.json").write_text(
        json.dumps(no_probe,indent=2,allow_nan=False)+"\n",encoding="utf-8"
    )
    print(json.dumps(summary,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
