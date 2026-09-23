"""Counterfactual Target×Destination action-sensitivity diagnostic.

No training is performed.  The canonical actor is frozen.  At sampled YC states
with feasible proactive pairs, the environment is deep-copied and several
feasible Target×Destination actions are forced one at a time.  Each branch then
continues to terminal under the same frozen stochastic policy and common policy
random-number stream.

Purpose: measure whether different proactive pairs have enough within-state
return separation to be learnable, and whether the current actor score aligns
with that action-specific return variation.

The sampled candidate set is not exhaustive; reported return spread is therefore
a lower-bound style diagnostic for the full feasible action set.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch

from train_yc_marl import ResourceCooperativeModel, masked_distribution
from yc_marl_env import ResourceMARLYardEnv, STORAGE_AGENT, YC_PROACTIVE_BASE

torch.set_num_threads(1)

FEATURE_NAMES = [
    "feasible", "target_eta", "eta_advance", "blockers", "capacity_slack",
    "moving_blocker_eta", "dest_height", "dest_inversion", "dest_nearest_eta",
]


def corr(x, y):
    x=np.asarray(x,dtype=np.float64); y=np.asarray(y,dtype=np.float64)
    if len(x)<3 or np.std(x)<1e-12 or np.std(y)<1e-12:
        return float("nan")
    return float(np.corrcoef(x,y)[0,1])


def dump(path, obj):
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, allow_nan=True)+"\n", encoding="utf-8")


def load_model(checkpoint: Path):
    ck=torch.load(checkpoint,map_location="cpu",weights_only=False)
    env=ResourceMARLYardEnv(seed=1,arrival_rate_per_hour=20.0); env.reset()
    hidden=int(ck.get("hidden",ck.get("config",{}).get("hidden",128)))
    model=ResourceCooperativeModel(env.global_obs_dim,env.storage_obs_dim,env.yc_obs_dim,hidden=hidden)
    model.load_state_dict(ck.get("state_dict",ck),strict=True)
    model.eval()
    return model


def policy_action(env, model, rng):
    role=0 if env.active_agent()==STORAGE_AGENT else 1
    obs=torch.as_tensor(env.actor_observation(),dtype=torch.float32)
    mask=torch.as_tensor(env.action_mask(),dtype=torch.bool)
    with torch.inference_mode():
        logits=model.storage_logits(obs) if role==0 else model.yc_logits(obs,mask)
        dist=masked_distribution(logits,mask)
    return int(torch.multinomial(dist.probs,1,generator=rng).item())


def pair_state_record(env, model):
    if env.active_agent()==STORAGE_AGENT:
        return None
    obs_np=env.actor_observation().copy()
    mask_np=env.action_mask().copy()
    feasible=np.flatnonzero(mask_np[YC_PROACTIVE_BASE:])
    if len(feasible)<2:
        return None
    obs=torch.as_tensor(obs_np,dtype=torch.float32)
    mask=torch.as_tensor(mask_np,dtype=torch.bool)
    with torch.inference_mode():
        logits=model.yc_logits(obs,mask)
        dist=masked_distribution(logits,mask)
        _,raw=model.yc_actor.raw_components(obs)
        probs=dist.probs.cpu().numpy()
        raw=raw.cpu().numpy()
    pair_raw=obs_np[model.yc_actor.CONTEXT_DIM:].reshape(model.yc_actor.PAIR_COUNT,model.yc_actor.PAIR_FEAT_DIM)
    p_pro=float(probs[YC_PROACTIVE_BASE:].sum())
    cond=np.zeros_like(raw,dtype=np.float64)
    if p_pro>0:
        cond[feasible]=probs[YC_PROACTIVE_BASE:][feasible]/p_pro
    return dict(
        obs=obs_np, mask=mask_np, feasible=feasible, raw=raw,
        pair_features=pair_raw, conditional=cond, p_proactive=p_pro,
    )


def choose_candidates(rec, n_random, seed):
    feasible=np.asarray(rec["feasible"],dtype=int)
    raw=np.asarray(rec["raw"],dtype=np.float64)
    chosen=[]
    top=int(feasible[np.argmax(raw[feasible])])
    bottom=int(feasible[np.argmin(raw[feasible])])
    chosen.extend([top,bottom])
    pool=np.array([x for x in feasible if x not in {top,bottom}],dtype=int)
    rng=np.random.default_rng(seed)
    if len(pool):
        take=min(int(n_random),len(pool))
        chosen.extend(map(int,rng.choice(pool,size=take,replace=False)))
    # Stable de-duplication.
    out=[]
    for x in chosen:
        if x not in out: out.append(x)
    return out


def continue_branch(snapshot, model, forced_action, policy_seed):
    env=copy.deepcopy(snapshot)
    rng=torch.Generator(device="cpu").manual_seed(int(policy_seed))
    future_return=0.0
    decisions=0
    _,reward,done,_,info=env.step(int(forced_action))
    future_return += float(reward); decisions += 1
    while not done and decisions<100000:
        action=policy_action(env,model,rng)
        _,reward,done,_,info=env.step(action)
        future_return += float(reward); decisions += 1
    if not done:
        raise RuntimeError("counterfactual branch did not terminate")
    k=info["kpis"]
    objective=float(k["mean_truck_completion_delay"]+k["mean_storage_completion_delay"]+0.10*k["extra_yc_minutes_per_retrieval"])
    return dict(
        future_return=future_return,
        final_objective=objective,
        continuation_decisions=decisions,
        truck_delay=float(k["mean_truck_completion_delay"]),
        storage_delay=float(k["mean_storage_completion_delay"]),
        rehandling=float(k["rehandling_moves"]),
        proactive=float(k["proactive_moves"]),
        extra_yc=float(k["extra_yc_minutes_per_retrieval"]),
    )


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--scenarios",nargs="+",type=int,default=[801,802,803,804])
    ap.add_argument("--states-per-scenario",type=int,default=3)
    ap.add_argument("--min-state-gap",type=int,default=80)
    ap.add_argument("--random-pairs",type=int,default=10)
    ap.add_argument("--repeats",type=int,default=2)
    args=ap.parse_args()

    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    model=load_model(Path(args.checkpoint))

    probes=[]
    for scenario in args.scenarios:
        env=ResourceMARLYardEnv(seed=scenario,arrival_rate_per_hour=20.0)
        env.reset()
        collect_rng=torch.Generator(device="cpu").manual_seed(int(scenario*100+77))
        steps=0; last_probe=-10**9; found=0; done=False
        while not done and steps<100000 and found<args.states_per_scenario:
            rec=pair_state_record(env,model)
            if rec is not None and steps-last_probe>=args.min_state_gap:
                snapshot=copy.deepcopy(env)
                probes.append(dict(
                    scenario=int(scenario), state_index=int(found), decision_index=int(steps),
                    snapshot=snapshot, rec=rec,
                ))
                last_probe=steps; found+=1
            action=policy_action(env,model,collect_rng)
            _,_,done,_,_=env.step(action); steps+=1
        if found<args.states_per_scenario:
            raise RuntimeError(f"scenario {scenario}: only {found} probe states found")

    candidate_rows=[]
    state_rows=[]
    for sid,p in enumerate(probes):
        rec=p["rec"]
        candidates=choose_candidates(rec,args.random_pairs,seed=p["scenario"]*1000+p["state_index"])
        per_candidate=[]
        for local_idx in candidates:
            action=YC_PROACTIVE_BASE+int(local_idx)
            vals=[]
            for rep in range(args.repeats):
                common_seed=p["scenario"]*1_000_000+p["state_index"]*10_000+rep
                vals.append(continue_branch(p["snapshot"],model,action,common_seed))
            mean_return=float(np.mean([v["future_return"] for v in vals]))
            mean_obj=float(np.mean([v["final_objective"] for v in vals]))
            feat=np.asarray(rec["pair_features"][local_idx],dtype=np.float64)
            row=dict(
                probe_id=sid,scenario=p["scenario"],state_index=p["state_index"],
                decision_index=p["decision_index"],local_pair_index=int(local_idx),action=int(action),
                actor_raw_score=float(rec["raw"][local_idx]),
                actor_conditional_prob=float(rec["conditional"][local_idx]),
                p_proactive=float(rec["p_proactive"]),
                mean_future_return=mean_return,mean_final_objective=mean_obj,
                return_std_repeat=float(np.std([v["future_return"] for v in vals])),
                objective_std_repeat=float(np.std([v["final_objective"] for v in vals])),
                repeats=vals,
            )
            for j,name in enumerate(FEATURE_NAMES):
                row[name]=float(feat[j])
            candidate_rows.append(row); per_candidate.append(row)

        returns=np.asarray([r["mean_future_return"] for r in per_candidate],dtype=np.float64)
        scores=np.asarray([r["actor_raw_score"] for r in per_candidate],dtype=np.float64)
        probs=np.asarray([r["actor_conditional_prob"] for r in per_candidate],dtype=np.float64)
        best_i=int(np.argmax(returns)); worst_i=int(np.argmin(returns)); top_i=int(np.argmax(scores))
        state_rows.append(dict(
            probe_id=sid,scenario=p["scenario"],state_index=p["state_index"],
            decision_index=p["decision_index"],feasible_pair_count=int(len(rec["feasible"])),
            tested_pair_count=len(per_candidate),p_proactive=float(rec["p_proactive"]),
            return_mean=float(returns.mean()),return_std=float(returns.std()),
            return_range=float(returns.max()-returns.min()),
            objective_range=float(max(r["mean_final_objective"] for r in per_candidate)-min(r["mean_final_objective"] for r in per_candidate)),
            actor_score_return_corr=corr(scores,returns),
            actor_prob_return_corr=corr(probs,returns),
            top_score_regret=float(returns[best_i]-returns[top_i]),
            best_pair_index=int(per_candidate[best_i]["local_pair_index"]),
            worst_pair_index=int(per_candidate[worst_i]["local_pair_index"]),
            top_actor_pair_index=int(per_candidate[top_i]["local_pair_index"]),
        ))

    # Within-state centered pooled correlations remove the dominant state/time level.
    ret_center=[]; score_center=[]; prob_center=[]
    feat_center={n:[] for n in FEATURE_NAMES}
    for s in state_rows:
        rr=[r for r in candidate_rows if r["probe_id"]==s["probe_id"]]
        rv=np.asarray([r["mean_future_return"] for r in rr],dtype=np.float64)
        sv=np.asarray([r["actor_raw_score"] for r in rr],dtype=np.float64)
        pv=np.asarray([r["actor_conditional_prob"] for r in rr],dtype=np.float64)
        ret_center.extend(rv-rv.mean()); score_center.extend(sv-sv.mean()); prob_center.extend(pv-pv.mean())
        for n in FEATURE_NAMES:
            fv=np.asarray([r[n] for r in rr],dtype=np.float64)
            feat_center[n].extend(fv-fv.mean())

    state_means=np.asarray([s["return_mean"] for s in state_rows],dtype=np.float64)
    within_sds=np.asarray([s["return_std"] for s in state_rows],dtype=np.float64)
    summary=dict(
        checkpoint=str(args.checkpoint),
        scenarios=list(map(int,args.scenarios)),
        probe_states=len(state_rows),
        states_per_scenario=args.states_per_scenario,
        random_pairs=args.random_pairs,
        repeats=args.repeats,
        total_counterfactual_branches=sum(len(r["repeats"]) for r in candidate_rows),
        mean_tested_pairs_per_state=float(np.mean([s["tested_pair_count"] for s in state_rows])),
        mean_return_range=float(np.mean([s["return_range"] for s in state_rows])),
        median_return_range=float(np.median([s["return_range"] for s in state_rows])),
        mean_within_state_return_std=float(within_sds.mean()),
        between_state_return_mean_std=float(state_means.std()),
        within_to_between_std_ratio=float(within_sds.mean()/state_means.std()) if state_means.std()>1e-12 else float("nan"),
        mean_abs_actor_score_return_corr=float(np.nanmean(np.abs([s["actor_score_return_corr"] for s in state_rows]))),
        mean_actor_score_return_corr=float(np.nanmean([s["actor_score_return_corr"] for s in state_rows])),
        pooled_within_state_actor_score_return_corr=corr(score_center,ret_center),
        pooled_within_state_actor_prob_return_corr=corr(prob_center,ret_center),
        pooled_within_state_feature_return_corr={n:corr(feat_center[n],ret_center) for n in FEATURE_NAMES},
        mean_top_score_regret=float(np.mean([s["top_score_regret"] for s in state_rows])),
        median_top_score_regret=float(np.median([s["top_score_regret"] for s in state_rows])),
        note="Candidate set is sampled, not exhaustive; return-range estimates do not upper-bound full feasible-pair sensitivity.",
    )

    # Strip non-serializable snapshots before writing.
    dump(out/"summary.json",summary)
    dump(out/"states.json",state_rows)
    dump(out/"candidates.json",candidate_rows)
    print(json.dumps(summary,allow_nan=True),flush=True)


if __name__=="__main__":
    main()
