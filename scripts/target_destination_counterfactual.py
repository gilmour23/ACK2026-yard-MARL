from __future__ import annotations

"""Target-vs-destination counterfactual decomposition for the frozen final model.

No training is performed.  A final Proposed MARL checkpoint is used only to
generate a representative rule-resolved state trajectory.  At one YC state per
scenario, alternative proactive targets and destinations are forced from deep
copies of the exact same state; all later decisions follow the deterministic
information-aware heuristic.
"""

from pathlib import Path
import argparse
import copy
import hashlib
import json
import math

import numpy as np
import torch

from train_yc_marl import ResourceCooperativeModel, masked_distribution
from validate_yc_marl import heuristic_action
from yc_marl_env import (
    ResourceMARLYardEnv,
    STORAGE_AGENT,
    YC_PROACTIVE_BASE,
    encode_proactive_action,
)

torch.set_num_threads(1)

KPI_KEYS = (
    "mean_truck_completion_delay",
    "mean_storage_completion_delay",
    "rehandling_moves",
    "proactive_moves",
    "extra_yc_minutes_per_retrieval",
    "total_yc_moves",
    "mean_yc_utilization",
    "max_yc_queue",
)


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def objective(k: dict) -> float:
    return float(
        k["mean_truck_completion_delay"]
        + k["mean_storage_completion_delay"]
        + 0.10*k["extra_yc_minutes_per_retrieval"]
    )


def load_model(checkpoint: Path) -> ResourceCooperativeModel:
    ck=torch.load(checkpoint,map_location="cpu",weights_only=False)
    env=ResourceMARLYardEnv(
        seed=1,arrival_rate_per_hour=20.0,
        include_resource_state=True,rule_resolve_proactive_pair=True,
    )
    env.reset()
    hidden=int(ck.get("hidden",ck.get("config",{}).get("hidden",128)))
    model=ResourceCooperativeModel(
        env.global_obs_dim,env.storage_obs_dim,env.yc_obs_dim,hidden=hidden
    )
    model.load_state_dict(ck.get("state_dict",ck),strict=True)
    model.eval()
    return model


def policy_action(env: ResourceMARLYardEnv, model: ResourceCooperativeModel, rng: torch.Generator) -> int:
    role=0 if env.active_agent()==STORAGE_AGENT else 1
    obs=torch.as_tensor(env.actor_observation(),dtype=torch.float32)
    mask=torch.as_tensor(env.action_mask(),dtype=torch.bool)
    with torch.inference_mode():
        logits=model.storage_logits(obs) if role==0 else model.yc_logits(obs,mask)
        dist=masked_distribution(logits,mask)
    return int(torch.multinomial(dist.probs,1,generator=rng).item())


def target_descriptor(env: ResourceMARLYardEnv, block: int, cid: str, rank: int) -> dict:
    sim=env.sim
    assert sim is not None
    c=sim.containers[cid]
    assert c.stack is not None
    source=int(c.stack)
    moving_id=sim.yard.top(block,source)
    if moving_id is None or moving_id==cid:
        raise RuntimeError(("invalid proactive target",cid))
    feasible=list(sim.yard.feasible_relocation_stacks(block,source))
    if not feasible:
        raise RuntimeError(("no destination",cid))
    heur_dest=int(sim.policy.choose_relocation_destination(sim,moving_id,block,source))
    return {
        "target_cid":cid,
        "heuristic_rank":int(rank),
        "target_pos":int(sim._target_position(cid)),
        "source_stack":source,
        "moving_cid":moving_id,
        "heuristic_destination":heur_dest,
        "feasible_destinations":[int(x) for x in feasible],
        "eta_lead":float(max(0.0,c.eta-sim.now)),
        "eta_advance":float(max(0.0,-c.eta_delta)),
        "blockers":int(sim.yard.blockers_above(cid,sim.containers)),
        "capacity_slack":float(sim.target_capacity_slack(cid)),
    }


def force_pair_and_continue(snapshot: ResourceMARLYardEnv, target_pos: int, dest: int) -> dict:
    env=copy.deepcopy(snapshot)
    assert env.sim is not None
    # Expand action support only for the forced counterfactual first action.
    # Physical dynamics/reward are unchanged.
    env.sim.rule_resolve_proactive_pair=False
    action=encode_proactive_action(int(target_pos),int(dest))
    mask=env.action_mask()
    if action>=len(mask) or not bool(mask[action]):
        raise RuntimeError(("forced pair is not physically feasible",target_pos,dest))
    _,_,done,_,info=env.step(action)
    decisions=1
    while not done and decisions<100000:
        a=heuristic_action(env)
        _,_,done,_,info=env.step(a)
        decisions+=1
    if not done:
        raise RuntimeError("counterfactual continuation did not terminate")
    k=info["kpis"]
    row={"J":objective(k),"continuation_decisions":decisions}
    row.update({key:float(k[key]) for key in KPI_KEYS})
    return row


def find_probe(
    model: ResourceCooperativeModel,
    scenario: int,
    training_seed: int,
    min_targets: int,
) -> tuple[ResourceMARLYardEnv | None, dict]:
    env=ResourceMARLYardEnv(
        seed=scenario,arrival_rate_per_hour=20.0,
        include_resource_state=True,rule_resolve_proactive_pair=True,
    )
    env.reset()
    policy_seed=int(scenario*1000+training_seed)
    rng=torch.Generator(device="cpu").manual_seed(policy_seed)
    steps=0
    done=False
    while not done and steps<100000:
        if env.active_agent()!=STORAGE_AGENT:
            block=env.active_block()
            assert block is not None and env.sim is not None
            candidates=env.sim.proactive_candidates(block)
            if len(candidates)>=min_targets:
                return copy.deepcopy(env),{
                    "scenario":scenario,
                    "training_seed":training_seed,
                    "policy_collection_seed":policy_seed,
                    "decision_index":steps,
                    "block":int(block),
                    "eligible_target_count":len(candidates),
                }
        action=policy_action(env,model,rng)
        _,_,done,_,_=env.step(action)
        steps+=1
    return None,{
        "scenario":scenario,
        "training_seed":training_seed,
        "policy_collection_seed":policy_seed,
        "decision_index":None,
        "block":None,
        "eligible_target_count":0,
        "status":"no_probe",
    }


def summarize_state(target_rows: list[dict], dest_rows: list[dict], meta: dict, destination_top_k: int) -> dict:
    if not target_rows:
        raise RuntimeError("empty target rows")
    by_rank={int(r["heuristic_rank"]):r for r in target_rows}
    global_best=min(target_rows,key=lambda r:r["J"])
    top1=by_rank[1]
    def best_within(k: int) -> dict:
        return min((r for r in target_rows if int(r["heuristic_rank"])<=k),key=lambda r:r["J"])
    top3=best_within(min(3,len(target_rows)))
    top5=best_within(min(5,len(target_rows)))
    jvals=np.asarray([r["J"] for r in target_rows],dtype=np.float64)

    dest_groups={}
    for r in dest_rows:
        dest_groups.setdefault(int(r["heuristic_rank"]),[]).append(r)
    dest_summaries=[]
    for rank,rows in sorted(dest_groups.items()):
        best=min(rows,key=lambda r:r["J"])
        heur=[r for r in rows if bool(r["is_heuristic_destination"])]
        if len(heur)!=1:
            raise RuntimeError(("heuristic destination cardinality",rank,len(heur)))
        h=heur[0]
        vals=np.asarray([r["J"] for r in rows],dtype=np.float64)
        dest_summaries.append({
            "heuristic_rank":rank,
            "target_cid":h["target_cid"],
            "destination_count":len(rows),
            "best_destination":int(best["destination"]),
            "heuristic_destination":int(h["destination"]),
            "heuristic_destination_is_best":bool(int(best["destination"])==int(h["destination"])),
            "heuristic_destination_regret":float(h["J"]-best["J"]),
            "destination_J_range":float(vals.max()-vals.min()),
        })

    return {
        **meta,
        "status":"ok",
        "global_best_target_rank":int(global_best["heuristic_rank"]),
        "global_best_target_cid":global_best["target_cid"],
        "top1_contains_global_best":bool(int(global_best["heuristic_rank"])==1),
        "top3_contains_global_best":bool(int(global_best["heuristic_rank"])<=3),
        "top5_contains_global_best":bool(int(global_best["heuristic_rank"])<=5),
        "top1_target_regret":float(top1["J"]-global_best["J"]),
        "top3_target_regret":float(top3["J"]-global_best["J"]),
        "top5_target_regret":float(top5["J"]-global_best["J"]),
        "target_J_range":float(jvals.max()-jvals.min()),
        "destination_targets_evaluated":len(dest_summaries),
        "destination_top_k":int(destination_top_k),
        "mean_destination_regret":float(np.mean([x["heuristic_destination_regret"] for x in dest_summaries])),
        "median_destination_regret":float(np.median([x["heuristic_destination_regret"] for x in dest_summaries])),
        "mean_destination_J_range":float(np.mean([x["destination_J_range"] for x in dest_summaries])),
        "median_destination_J_range":float(np.median([x["destination_J_range"] for x in dest_summaries])),
        "heuristic_destination_best_fraction":float(np.mean([x["heuristic_destination_is_best"] for x in dest_summaries])),
        "destination_summaries":dest_summaries,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--training-seed",type=int,required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--scenarios",nargs="+",type=int,default=list(range(1101,1111)))
    ap.add_argument("--min-targets",type=int,default=3)
    ap.add_argument("--destination-top-k",type=int,default=5)
    a=ap.parse_args()

    if a.training_seed not in {51,52,53}:
        raise ValueError("diagnostic is frozen to final Proposed seeds 51/52/53")
    if list(a.scenarios)!=list(range(1101,1111)):
        raise ValueError("diagnostic scenarios are frozen to 1101-1110")

    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    checkpoint=Path(a.checkpoint)
    model=load_model(checkpoint)

    state_rows=[]
    target_rows=[]
    destination_rows=[]
    no_probe=[]

    for scenario in a.scenarios:
        snapshot,meta=find_probe(model,int(scenario),a.training_seed,a.min_targets)
        if snapshot is None:
            no_probe.append(meta)
            continue
        assert snapshot.sim is not None
        block=int(meta["block"])
        candidates=list(snapshot.sim.proactive_candidates(block))
        descriptors=[
            target_descriptor(snapshot,block,cid,rank)
            for rank,cid in enumerate(candidates,start=1)
        ]

        # Exact target comparison under each target's heuristic destination.
        state_target_rows=[]
        for desc in descriptors:
            res=force_pair_and_continue(
                snapshot,desc["target_pos"],desc["heuristic_destination"]
            )
            row={**meta,**desc,**res}
            state_target_rows.append(row)
            target_rows.append(row)

        # Exact destination enumeration for heuristic top-K targets.
        state_dest_rows=[]
        for desc in descriptors[:min(a.destination_top_k,len(descriptors))]:
            for dest in desc["feasible_destinations"]:
                res=force_pair_and_continue(snapshot,desc["target_pos"],dest)
                row={
                    **meta,
                    "target_cid":desc["target_cid"],
                    "heuristic_rank":desc["heuristic_rank"],
                    "target_pos":desc["target_pos"],
                    "source_stack":desc["source_stack"],
                    "moving_cid":desc["moving_cid"],
                    "destination":int(dest),
                    "heuristic_destination":desc["heuristic_destination"],
                    "is_heuristic_destination":bool(int(dest)==int(desc["heuristic_destination"])),
                    **res,
                }
                state_dest_rows.append(row)
                destination_rows.append(row)

        state_rows.append(
            summarize_state(
                state_target_rows,state_dest_rows,meta,a.destination_top_k
            )
        )

    if not state_rows:
        raise RuntimeError("no valid counterfactual probe states")

    summary={
        "training_seed":a.training_seed,
        "checkpoint":str(checkpoint),
        "checkpoint_sha256":sha256(checkpoint),
        "scenarios":list(map(int,a.scenarios)),
        "valid_probe_states":len(state_rows),
        "no_probe_states":len(no_probe),
        "mean_eligible_targets":float(np.mean([r["eligible_target_count"] for r in state_rows])),
        "top1_best_target_coverage":float(np.mean([r["top1_contains_global_best"] for r in state_rows])),
        "top3_best_target_coverage":float(np.mean([r["top3_contains_global_best"] for r in state_rows])),
        "top5_best_target_coverage":float(np.mean([r["top5_contains_global_best"] for r in state_rows])),
        "mean_top1_target_regret":float(np.mean([r["top1_target_regret"] for r in state_rows])),
        "mean_top3_target_regret":float(np.mean([r["top3_target_regret"] for r in state_rows])),
        "mean_top5_target_regret":float(np.mean([r["top5_target_regret"] for r in state_rows])),
        "mean_target_J_range":float(np.mean([r["target_J_range"] for r in state_rows])),
        "mean_destination_regret":float(np.mean([r["mean_destination_regret"] for r in state_rows])),
        "mean_destination_J_range":float(np.mean([r["mean_destination_J_range"] for r in state_rows])),
        "heuristic_destination_best_fraction":float(np.mean([r["heuristic_destination_best_fraction"] for r in state_rows])),
        "target_counterfactual_branches":len(target_rows),
        "destination_counterfactual_branches":len(destination_rows),
        "note":"Target comparison is exhaustive over eligible targets with heuristic destination. Destination comparison is exhaustive over feasible destinations for heuristic top-K targets.",
    }

    for obj in (summary,state_rows,target_rows,destination_rows,no_probe):
        if isinstance(obj,dict):
            vals=[v for v in obj.values() if isinstance(v,float)]
            if any(not math.isfinite(v) for v in vals):
                raise RuntimeError("non-finite summary")

    (out/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (out/"states.json").write_text(json.dumps(state_rows,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (out/"targets.json").write_text(json.dumps(target_rows,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (out/"destinations.json").write_text(json.dumps(destination_rows,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (out/"no_probe.json").write_text(json.dumps(no_probe,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(summary,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
