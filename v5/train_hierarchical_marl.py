from __future__ import annotations

"""Bounded V5 hierarchical cooperative-MARL trainer.

This trainer is intentionally isolated from the frozen V4/final path.  It keeps
Storage, centralized value learning, reward, PPO clipping, and episode-complete
rollouts unchanged while replacing the YC actor with:

    Default/Proactive -> Target -> Destination

The simulator still receives the legacy encoded flat YC action.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from train_yc_marl import (
    ScenarioSampler,
    completed_episode_mc_diagnostics,
    compute_gae,
    load_compatible_state_dict,
    masked_distribution,
    rollout_ready,
    set_global_seeds,
)
from yc_marl_env import (
    ResourceMARLYardEnv,
    STORAGE_AGENT,
    YC_AGENT_PREFIX,
)
from v5.hierarchical_yc import (
    HIER_YC_OBS_DIM,
    HierarchicalResourceCooperativeModel,
    build_hierarchical_yc_input,
    heuristic_ranks_for_selected_pair,
)


@dataclass
class HierarchicalPPOConfig:
    total_steps:int=6000
    rollout_steps:int=512
    min_storage_transitions:int=64
    min_yc_transitions:int=128
    max_rollout_multiplier:int=4
    update_epochs:int=2
    critic_extra_epochs:int=18
    minibatch_size:int=256
    gamma:float=1.0
    gae_lambda:float=1.0
    clip_coef:float=0.20
    ent_coef:float=0.01
    yc_op_ent_coef:float=0.001
    yc_target_ent_coef:float=0.0001
    yc_destination_ent_coef:float=0.0001
    yc_proactive_init_bias:float=-2.197224577
    vf_coef:float=0.5
    learning_rate:float=3e-4
    max_grad_norm:float=0.5
    hidden:int=128
    candidate_hidden:int=32
    seed:int=61
    arrival_rate_per_hour:float=20.0
    include_resource_state:bool=True
    truck_wait_weight:float=1.0
    storage_wait_weight:float=1.0
    extra_move_weight:float=0.10
    risk_shaping_weight:float=0.0
    yc_queue_shaping_weight:float=0.0
    enable_proactive:bool=True
    yc_move_time:float=2.0
    episode_complete_rollout:bool=True


def _write_csv(path:Path,rows:List[dict])->None:
    if not rows:
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=sorted({k for row in rows for k in row})
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _load_storage_bc(
    model:HierarchicalResourceCooperativeModel,
    checkpoint_path:Path,
    include_resource_state:bool,
)->dict:
    ck=torch.load(Path(checkpoint_path),map_location="cpu",weights_only=False)
    if not bool(ck.get("storage_only",False)):
        raise RuntimeError("V5 pilot accepts Storage-only BC checkpoints only")
    if ck.get("kind")!="marl":
        raise RuntimeError(("V5 Proposed requires MARL Storage BC",ck.get("kind")))
    if bool(ck.get("include_resource_state",True))!=bool(include_resource_state):
        raise RuntimeError("Storage BC resource-state setting does not match V5 arm")
    state=ck.get("state_dict",ck)
    state={
        k:v for k,v in state.items()
        if k.startswith("storage_actor.") or k.startswith("critic.")
    }
    missing,skipped=load_compatible_state_dict(model,state)
    if skipped:
        raise RuntimeError(("unexpected incompatible Storage-BC tensors",skipped))
    storage_keys=[k for k in state if k.startswith("storage_actor.")]
    if not storage_keys:
        raise RuntimeError("Storage BC contains no storage_actor tensors")
    return {
        "bc_metrics":ck.get("metrics",{}),
        "loaded_storage_tensors":len(storage_keys),
        "missing_v5_keys":missing,
    }


def _finite_record(row:dict)->bool:
    for v in row.values():
        if isinstance(v,(float,np.floating)) and not np.isfinite(float(v)):
            return False
    return True


def train_hierarchical_marl(
    config:HierarchicalPPOConfig,
    out_dir:Path,
    init_checkpoint:Optional[Path]=None,
)->Tuple[HierarchicalResourceCooperativeModel,List[dict]]:
    if not config.episode_complete_rollout:
        raise ValueError("V5 first pilot is frozen to episode-complete rollouts")
    if config.total_steps<=0:
        raise ValueError("total_steps must be positive")

    set_global_seeds(config.seed)
    device=torch.device("cpu")
    sampler=ScenarioSampler(config.seed+1000)
    env=ResourceMARLYardEnv(
        seed=sampler.next(),
        arrival_rate_per_hour=config.arrival_rate_per_hour,
        include_resource_state=config.include_resource_state,
        truck_wait_weight=config.truck_wait_weight,
        storage_wait_weight=config.storage_wait_weight,
        extra_move_weight=config.extra_move_weight,
        risk_shaping_weight=config.risk_shaping_weight,
        yc_queue_shaping_weight=config.yc_queue_shaping_weight,
        enable_proactive=config.enable_proactive,
        yc_move_time=config.yc_move_time,
        # V5 must see the complete physically feasible pair support.
        rule_resolve_proactive_pair=False,
    )
    env.reset()

    model=HierarchicalResourceCooperativeModel(
        env.global_obs_dim,
        env.storage_obs_dim,
        hidden=config.hidden,
        candidate_hidden=config.candidate_hidden,
        proactive_bias=config.yc_proactive_init_bias,
    ).to(device)

    bc_meta=None
    if init_checkpoint is not None:
        bc_meta=_load_storage_bc(
            model,Path(init_checkpoint),config.include_resource_state
        )

    optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    critic_extra_rng=np.random.default_rng(config.seed+700_001)

    global_step=0
    update_idx=0
    episode_records:List[dict]=[]
    update_records:List[dict]=[]

    while global_step<config.total_steps:
        global_obs_buf=[]
        actor_obs_buf=[]
        action_mask_buf=[]
        pair_mask_buf=[]
        roles_buf=[]
        flat_actions_buf=[]
        operation_buf=[]
        target_buf=[]
        destination_buf=[]
        old_logp_buf=[]
        rewards_buf=[]
        team_buf=[]
        storage_local_buf=[]
        yc_local_buf=[]
        dones_buf=[]
        values_buf=[]
        next_values_buf=[]

        yc_op_entropy_buf=[]
        yc_target_entropy_buf=[]
        yc_dest_entropy_buf=[]
        yc_proactive_prob_buf=[]
        yc_feasible_targets_buf=[]
        yc_feasible_pairs_buf=[]
        yc_selected_proactive_buf=[]
        yc_selected_target_rank=[]
        yc_selected_dest_rank=[]

        remaining=max(config.total_steps-global_step,1)
        base=min(config.rollout_steps,remaining)
        max_n=max(base,config.rollout_steps*config.max_rollout_multiplier)
        storage_n=0
        yc_n=0
        rollout_target_reached=False

        while True:
            active=env.active_agent()
            role=0 if active==STORAGE_AGENT else 1
            if role==1 and not active.startswith(YC_AGENT_PREFIX):
                raise RuntimeError(("invalid active agent",active))

            gobs=env.critic_observation().copy()
            gobs_t=torch.as_tensor(gobs,dtype=torch.float32,device=device)
            flat_mask=np.asarray(env.action_mask(),dtype=np.bool_)

            with torch.no_grad():
                value_t=model.value(gobs_t)
                if role==0:
                    aobs=env.actor_observation().copy()
                    mask=flat_mask.copy()
                    aobs_t=torch.as_tensor(aobs,dtype=torch.float32,device=device)
                    mask_t=torch.as_tensor(mask,dtype=torch.bool,device=device)
                    logits=model.storage_logits(aobs_t)
                    dist=masked_distribution(logits,mask_t)
                    action_t=dist.sample()
                    flat_action=int(action_t.item())
                    logp=float(dist.log_prob(action_t).item())
                    pair_mask=None
                    op=0
                    tpos=-1
                    dest=-1
                else:
                    block=env.active_block()
                    if block is None:
                        raise RuntimeError("YC decision missing block")
                    aobs,pair_mask=build_hierarchical_yc_input(
                        env,block,resource_state=config.include_resource_state
                    )
                    aobs_t=torch.as_tensor(aobs,dtype=torch.float32,device=device)
                    pair_t=torch.as_tensor(pair_mask,dtype=torch.bool,device=device)
                    flat_mask_t=torch.as_tensor(flat_mask,dtype=torch.bool,device=device)
                    sample=model.yc_actor.sample(aobs_t,pair_t,flat_mask_t)
                    flat_action=sample.flat_action
                    logp=sample.log_prob
                    op=sample.operation
                    tpos=sample.target_position
                    dest=sample.destination

                    yc_op_entropy_buf.append(sample.operation_entropy)
                    yc_target_entropy_buf.append(sample.target_entropy_norm)
                    yc_dest_entropy_buf.append(sample.destination_entropy_norm)
                    yc_proactive_prob_buf.append(sample.p_proactive)
                    yc_feasible_targets_buf.append(float(pair_mask.any(axis=1).sum()))
                    yc_feasible_pairs_buf.append(float(pair_mask.sum()))
                    yc_selected_proactive_buf.append(float(op==1))
                    if op==1:
                        trank,drank=heuristic_ranks_for_selected_pair(
                            env,block,tpos,dest
                        )
                        yc_selected_target_rank.append(float(trank))
                        yc_selected_dest_rank.append(float(drank))

            _,reward,done,_,info=env.step(flat_action)
            with torch.no_grad():
                next_value=(
                    0.0 if done else float(
                        model.value(
                            torch.as_tensor(
                                env.critic_observation(),
                                dtype=torch.float32,
                                device=device,
                            )
                        ).item()
                    )
                )

            global_obs_buf.append(gobs)
            actor_obs_buf.append(aobs)
            action_mask_buf.append(flat_mask)
            pair_mask_buf.append(None if pair_mask is None else pair_mask.copy())
            roles_buf.append(role)
            flat_actions_buf.append(flat_action)
            operation_buf.append(op)
            target_buf.append(tpos)
            destination_buf.append(dest)
            old_logp_buf.append(logp)
            rewards_buf.append(float(reward))
            comp=info.get("reward_components",{})
            team_buf.append(float(comp.get("team",reward)))
            storage_local_buf.append(float(comp.get("storage_local",0.0)))
            yc_local_buf.append(float(comp.get("yc_local",0.0)))
            dones_buf.append(float(done))
            values_buf.append(float(value_t.item()))
            next_values_buf.append(float(next_value))

            storage_n+=int(role==0)
            yc_n+=int(role==1)
            global_step+=1
            if rollout_ready(
                len(flat_actions_buf),storage_n,yc_n,base,max_n,config
            ):
                rollout_target_reached=True

            if done:
                episode_records.append({
                    "global_step":global_step,
                    "seed":env.seed,
                    "arrival_rate_per_hour":env.arrival_rate_per_hour,
                    "episode_return":info["episode_return"],
                    **info["kpis"],
                })
                env.reset(seed=sampler.next())
                if rollout_target_reached:
                    break

        rewards=np.asarray(rewards_buf,np.float32)
        values=np.asarray(values_buf,np.float32)
        next_values=np.asarray(next_values_buf,np.float32)
        dones=np.asarray(dones_buf,np.float32)
        mc_diag=completed_episode_mc_diagnostics(rewards,dones,values)
        adv,ret=compute_gae(
            rewards,values,next_values,dones,config.gamma,config.gae_lambda
        )
        adv=(adv-adv.mean())/(adv.std()+1e-8)

        old_logp_t=torch.as_tensor(old_logp_buf,dtype=torch.float32,device=device)
        adv_t=torch.as_tensor(adv,dtype=torch.float32,device=device)
        ret_t=torch.as_tensor(ret,dtype=torch.float32,device=device)
        global_obs_t=torch.as_tensor(
            np.asarray(global_obs_buf),dtype=torch.float32,device=device
        )
        n=len(flat_actions_buf)

        for _ in range(config.update_epochs):
            order=np.random.permutation(n)
            for start in range(0,n,config.minibatch_size):
                ids=order[start:start+config.minibatch_size]
                if len(ids)==0:
                    continue
                mb=torch.as_tensor(ids,dtype=torch.long,device=device)
                pg_terms=[]
                ent_terms=[]

                storage_ids=[i for i in ids if roles_buf[i]==0]
                if storage_ids:
                    rid=torch.as_tensor(storage_ids,dtype=torch.long,device=device)
                    obs=torch.as_tensor(
                        np.stack([actor_obs_buf[i] for i in storage_ids]),
                        dtype=torch.float32,
                        device=device,
                    )
                    masks=torch.as_tensor(
                        np.stack([action_mask_buf[i] for i in storage_ids]),
                        dtype=torch.bool,
                        device=device,
                    )
                    acts=torch.as_tensor(
                        [flat_actions_buf[i] for i in storage_ids],
                        dtype=torch.long,
                        device=device,
                    )
                    dist=masked_distribution(model.storage_logits(obs),masks)
                    new_logp=dist.log_prob(acts)
                    ratio=(new_logp-old_logp_t[rid]).exp()
                    pg1=-adv_t[rid]*ratio
                    pg2=-adv_t[rid]*torch.clamp(
                        ratio,1-config.clip_coef,1+config.clip_coef
                    )
                    pg_terms.append(torch.maximum(pg1,pg2).mean())
                    ent_terms.append(config.ent_coef*dist.entropy().mean())

                yc_ids=[i for i in ids if roles_buf[i]==1]
                if yc_ids:
                    rid=torch.as_tensor(yc_ids,dtype=torch.long,device=device)
                    obs=torch.as_tensor(
                        np.stack([actor_obs_buf[i] for i in yc_ids]),
                        dtype=torch.float32,
                        device=device,
                    )
                    pm=torch.as_tensor(
                        np.stack([pair_mask_buf[i] for i in yc_ids]),
                        dtype=torch.bool,
                        device=device,
                    )
                    op=torch.as_tensor(
                        [operation_buf[i] for i in yc_ids],
                        dtype=torch.long,
                        device=device,
                    )
                    tp=torch.as_tensor(
                        [target_buf[i] for i in yc_ids],
                        dtype=torch.long,
                        device=device,
                    )
                    ds=torch.as_tensor(
                        [destination_buf[i] for i in yc_ids],
                        dtype=torch.long,
                        device=device,
                    )
                    ev=model.yc_actor.evaluate_actions(obs,pm,op,tp,ds)
                    new_logp=ev["log_prob"]
                    ratio=(new_logp-old_logp_t[rid]).exp()
                    pg1=-adv_t[rid]*ratio
                    pg2=-adv_t[rid]*torch.clamp(
                        ratio,1-config.clip_coef,1+config.clip_coef
                    )
                    pg_terms.append(torch.maximum(pg1,pg2).mean())
                    branch_bonus=(
                        config.yc_op_ent_coef*ev["operation_entropy"]
                        +ev["p_proactive"]*(
                            config.yc_target_ent_coef*ev["target_entropy_norm"]
                            +config.yc_destination_ent_coef*ev["destination_entropy_norm"]
                        )
                    )
                    ent_terms.append(branch_bonus.mean())

                if not pg_terms:
                    raise RuntimeError("minibatch contains no policy transitions")
                pg_loss=torch.stack(pg_terms).mean()
                entropy_bonus=torch.stack(ent_terms).mean()
                value=model.value(global_obs_t[mb])
                v_loss=0.5*(value-ret_t[mb]).pow(2).mean()
                loss=pg_loss+config.vf_coef*v_loss-entropy_bonus
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),config.max_grad_norm)
                optimizer.step()

        critic_extra_steps=0
        critic_extra_actor_max_abs_delta=0.0
        if config.critic_extra_epochs>0:
            actor_before={
                name:p.detach().clone()
                for name,p in model.named_parameters()
                if name.startswith(("storage_actor.","yc_actor."))
            }
            for _ in range(config.critic_extra_epochs):
                order_c=critic_extra_rng.permutation(n)
                for start in range(0,n,config.minibatch_size):
                    ids=order_c[start:start+config.minibatch_size]
                    if len(ids)==0:
                        continue
                    mb=torch.as_tensor(ids,dtype=torch.long,device=device)
                    value=model.value(global_obs_t[mb])
                    v_loss=0.5*(value-ret_t[mb]).pow(2).mean()
                    optimizer.zero_grad(set_to_none=True)
                    (config.vf_coef*v_loss).backward()
                    nn.utils.clip_grad_norm_(model.parameters(),config.max_grad_norm)
                    optimizer.step()
                    critic_extra_steps+=1
            with torch.no_grad():
                for name,p in model.named_parameters():
                    if name in actor_before:
                        critic_extra_actor_max_abs_delta=max(
                            critic_extra_actor_max_abs_delta,
                            float((p-actor_before[name]).abs().max().item()),
                        )
            if critic_extra_actor_max_abs_delta!=0.0:
                raise RuntimeError((
                    "actor changed during critic-only epochs",
                    critic_extra_actor_max_abs_delta,
                ))

        with torch.no_grad():
            post_v=model.value(global_obs_t).detach().cpu().numpy().astype(np.float64)
        target_v=np.asarray(ret,dtype=np.float64)
        target_var=float(np.var(target_v))
        post_value_ev=(
            float("nan") if target_var<1e-12
            else float(1.0-np.var(target_v-post_v)/target_var)
        )
        post_value_rmse=float(np.sqrt(np.mean((target_v-post_v)**2)))

        update_idx+=1
        row={
            "update":update_idx,
            "global_step":global_step,
            "storage_decisions":storage_n,
            "yc_decisions":yc_n,
            "mean_reward":float(np.mean(rewards)),
            "mean_team_reward":float(np.mean(team_buf)),
            "mean_storage_local":float(np.mean(storage_local_buf)),
            "mean_yc_local":float(np.mean(yc_local_buf)),
            "yc_operation_entropy":float(np.mean(yc_op_entropy_buf)) if yc_op_entropy_buf else 0.0,
            "yc_target_entropy_norm":float(np.mean(yc_target_entropy_buf)) if yc_target_entropy_buf else 0.0,
            "yc_destination_entropy_norm":float(np.mean(yc_dest_entropy_buf)) if yc_dest_entropy_buf else 0.0,
            "yc_proactive_probability":float(np.mean(yc_proactive_prob_buf)) if yc_proactive_prob_buf else 0.0,
            "yc_feasible_targets":float(np.mean(yc_feasible_targets_buf)) if yc_feasible_targets_buf else 0.0,
            "yc_feasible_pairs":float(np.mean(yc_feasible_pairs_buf)) if yc_feasible_pairs_buf else 0.0,
            "yc_sampled_proactive_rate":float(np.mean(yc_selected_proactive_buf)) if yc_selected_proactive_buf else 0.0,
            "yc_selected_target_heuristic_rank":float(np.mean(yc_selected_target_rank)) if yc_selected_target_rank else 0.0,
            "yc_selected_destination_heuristic_rank":float(np.mean(yc_selected_dest_rank)) if yc_selected_dest_rank else 0.0,
            "rollout_mode":"episode_complete",
            "rollout_decisions":int(len(flat_actions_buf)),
            "rollout_ended_at_terminal":bool(dones_buf and dones_buf[-1]>0.5),
            **mc_diag,
            "post_update_value_ev":post_value_ev,
            "post_update_value_rmse":post_value_rmse,
            "critic_extra_epochs":int(config.critic_extra_epochs),
            "critic_extra_steps":int(critic_extra_steps),
            "critic_extra_actor_max_abs_delta":float(critic_extra_actor_max_abs_delta),
        }
        if not _finite_record({
            k:v for k,v in row.items()
            if k not in {"mc_value_ev"} or np.isfinite(v)
        }):
            raise RuntimeError(("non-finite update record",row))
        update_records.append(row)

    out_dir=Path(out_dir)
    out_dir.mkdir(parents=True,exist_ok=True)
    payload={
        "architecture_version":"v5_hierarchical_operation_target_destination",
        "state_dict":model.state_dict(),
        "config":asdict(config),
        "global_obs_dim":env.global_obs_dim,
        "storage_obs_dim":env.storage_obs_dim,
        "hierarchical_yc_obs_dim":HIER_YC_OBS_DIM,
        "init_checkpoint":str(init_checkpoint) if init_checkpoint is not None else None,
        "bc_meta":bc_meta,
        "run_global_step":int(global_step),
        "update_count":int(update_idx),
        "current_env_seed":int(env.seed),
        "optimizer_state_dict":optimizer.state_dict(),
        "python_random_state":random.getstate(),
        "numpy_random_state":np.random.get_state(),
        "torch_rng_state":torch.get_rng_state(),
        "scenario_sampler_state":sampler.rng.getstate(),
    }
    torch.save(payload,out_dir/"hierarchical_marl_final.pt")
    _write_csv(out_dir/"hierarchical_marl_train_episodes.csv",episode_records)
    _write_csv(out_dir/"hierarchical_marl_updates.csv",update_records)
    (out_dir/"hierarchical_marl_config.json").write_text(
        json.dumps(asdict(config),indent=2)+"\n",encoding="utf-8"
    )
    return model,update_records


if __name__=="__main__":
    raise RuntimeError(
        "Direct V5 training is disabled. Use a pre-registered bounded pilot runner."
    )
