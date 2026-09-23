"""Read-only actor pair-learning signal audit for the value-budget pilot.

This script does not change the trainer. It monkey-patches diagnostics around the
existing train_resource_marl path, records actor-gradient and pair-distribution
movement, and restores all patched functions afterwards.

No random numbers are consumed by the instrumentation.
"""
from __future__ import annotations
import argparse, inspect, json, math, hashlib
from pathlib import Path
import numpy as np
import torch

import train_yc_marl as tr
from train_yc_marl import ResourcePPOConfig, YC_PROACTIVE_BASE
from yc_marl_env import ResourceMARLYardEnv

torch.set_num_threads(1)

def dump(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,allow_nan=True)+"\n",encoding="utf-8")

def norm(params):
    vals=[p.grad.detach().double().square().sum() for p in params if p.grad is not None]
    return float(torch.stack(vals).sum().sqrt()) if vals else 0.0

def corr(x,y):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    if len(x)<3 or np.std(x)<1e-12 or np.std(y)<1e-12:return float("nan")
    return float(np.corrcoef(x,y)[0,1])

def find_frame():
    frame=inspect.currentframe().f_back
    while frame and frame.f_code is not tr.train_resource_marl.__code__:
        frame=frame.f_back
    if frame is None: raise RuntimeError("train_resource_marl frame not found")
    return frame.f_locals

def pair_distribution_metrics(model,obs_list,mask_list):
    if not obs_list:
        return dict(states=0)
    hs=[];tvu=[];pp=[];rawstd=[];conds=[]
    with torch.inference_mode():
        for obs_np,mask_np in zip(obs_list,mask_list):
            obs=torch.as_tensor(obs_np,dtype=torch.float32)
            mask=torch.as_tensor(mask_np,dtype=torch.bool)
            logits=model.yc_logits(obs,mask)
            dist=tr.masked_distribution(logits,mask)
            _,h,p=tr.structured_yc_entropy(dist,mask)
            p=float(p.item());pp.append(p)
            feas=mask[YC_PROACTIVE_BASE:];n=int(feas.sum().item())
            _,raw=model.yc_actor.raw_components(obs)
            if n>1 and p>1e-12:
                cond=(dist.probs[YC_PROACTIVE_BASE:][feas]/p).double()
                uni=torch.full_like(cond,1.0/n)
                hs.append(float(h.item()))
                tvu.append(float(0.5*torch.abs(cond-uni).sum().item()))
                rawstd.append(float(raw[feas].std(unbiased=False).item()))
                conds.append(cond.cpu().numpy())
            else:
                conds.append(None)
    return dict(states=len(obs_list),p_proactive_mean=float(np.mean(pp)),
        pair_entropy_norm_mean=float(np.mean(hs)) if hs else float("nan"),
        pair_tv_uniform_mean=float(np.mean(tvu)) if tvu else float("nan"),
        pair_raw_score_std_mean=float(np.mean(rawstd)) if rawstd else float("nan"),
        conditional=conds)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--seed",type=int,required=True)
    ap.add_argument("--critic-extra-epochs",type=int,choices=[0,18],required=True)
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)

    original_gae=tr.compute_gae
    original_clip=torch.nn.utils.clip_grad_norm_
    original_adam=torch.optim.Adam

    rollout_data=[]
    grad_rows=[]
    active_rollout={"idx":-1,"joint_steps":0,"expected_joint_steps":0,"probe":None,"proactive":None}

    def audited_gae(rewards,values,next_values,dones,gamma,lam):
        adv,ret=original_gae(rewards,values,next_values,dones,gamma,lam)
        loc=find_frame();model=loc["model"];n=len(loc["actions_buf"])
        idx=len(rollout_data)
        roles=np.asarray(loc["roles_buf"]);actions=np.asarray(loc["actions_buf"])
        actor_obs=loc["actor_obs_buf"];masks=loc["masks_buf"]
        pro_idx=np.flatnonzero((roles==1)&(actions>=YC_PROACTIVE_BASE))
        yc_probe=[i for i in range(n) if roles[i]==1 and np.asarray(masks[i][YC_PROACTIVE_BASE:]).sum()>1][:64]
        probe_obs=[actor_obs[i].copy() for i in yc_probe]
        probe_masks=[masks[i].copy() for i in yc_probe]
        pre_probe=pair_distribution_metrics(model,probe_obs,probe_masks)

        pro={}
        if len(pro_idx):
            obs=np.stack([actor_obs[i] for i in pro_idx])
            act=actions[pro_idx]-YC_PROACTIVE_BASE
            with torch.inference_mode():
                _,raw=model.yc_actor.raw_components(torch.as_tensor(obs,dtype=torch.float32))
                sel=raw.gather(1,torch.as_tensor(act,dtype=torch.long).unsqueeze(1)).squeeze(1).cpu().numpy()
            feats=np.stack([
                actor_obs[i][model.yc_actor.CONTEXT_DIM:].reshape(model.yc_actor.PAIR_COUNT,model.yc_actor.PAIR_FEAT_DIM)[actions[i]-YC_PROACTIVE_BASE]
                for i in pro_idx
            ])
            raw_adv=np.asarray(adv)[pro_idx]
            returns=np.asarray(ret)[pro_idx]
            norm_adv=(np.asarray(adv)-np.asarray(adv).mean())/(np.asarray(adv).std()+1e-8)
            pro=dict(count=int(len(pro_idx)),obs=obs,actions=act,selected_score_pre=sel,
                raw_adv=raw_adv,normalized_adv=norm_adv[pro_idx],returns=returns,features=feats)
        else:
            pro=dict(count=0)

        rec=dict(rollout=idx+1,n=n,terminal=bool(np.asarray(dones)[-1]>0.5),
            critic_ev_before=float(1-np.var(np.asarray(ret)-np.asarray(values))/np.var(np.asarray(ret))),
            proactive_count=int(len(pro_idx)),probe_pre={k:v for k,v in pre_probe.items() if k!="conditional"})
        rollout_data.append(rec)
        active_rollout.update(idx=idx,joint_steps=0,
            expected_joint_steps=math.ceil(n/loc["config"].minibatch_size)*loc["config"].update_epochs,
            probe=(probe_obs,probe_masks,pre_probe["conditional"]),proactive=pro)
        return adv,ret

    def audited_clip(params,max_norm,*args,**kwargs):
        ps=list(params);loc=find_frame();model=loc["model"]
        named=list(model.named_parameters())
        op=[p for n,p in named if n.startswith("yc_actor.operation_head.")]
        pair=[p for n,p in named if n.startswith(("yc_actor.context_pair_proj.","yc_actor.pair_encoder.","yc_actor.pair_scorer."))]
        scorer=[p for n,p in named if n.startswith("yc_actor.pair_scorer.")]
        ctx=[p for n,p in named if n.startswith("yc_actor.context_encoder.")]
        critic=list(model.critic.parameters())
        actor=[p for n,p in named if n.startswith(("storage_actor.","yc_actor."))]
        is_joint=any(p.grad is not None for p in actor)
        if is_joint:
            ids=np.asarray(loc.get("ids",[]),dtype=int)
            roles=np.asarray(loc["roles_buf"]);actions=np.asarray(loc["actions_buf"])
            grad_rows.append(dict(
                rollout=active_rollout["idx"]+1,
                joint_step=active_rollout["joint_steps"]+1,
                minibatch_n=int(len(ids)),
                yc_n=int(np.sum(roles[ids]==1)) if len(ids) else 0,
                proactive_n=int(np.sum((roles[ids]==1)&(actions[ids]>=YC_PROACTIVE_BASE))) if len(ids) else 0,
                pg_loss=float(loc["pg_loss"].detach().item()),
                value_half_mse=float(loc["v_loss"].detach().item()),
                entropy_bonus=float(loc["entropy_bonus"].detach().item()),
                operation_head_grad=norm(op),
                pair_path_grad=norm(pair),
                pair_scorer_grad=norm(scorer),
                yc_context_encoder_grad=norm(ctx),
                critic_grad=norm(critic),
                actor_grad=norm(actor),
            ))
        return original_clip(ps,max_norm,*args,**kwargs)

    class AuditedAdam(original_adam):
        def step(self,closure=None):
            loc=find_frame();model=loc["model"]
            actors=[p for n,p in model.named_parameters() if n.startswith(("storage_actor.","yc_actor."))]
            is_joint=any(p.grad is not None for p in actors)
            result=super().step(closure)
            if is_joint:
                active_rollout["joint_steps"]+=1
                if active_rollout["joint_steps"]==active_rollout["expected_joint_steps"]:
                    ridx=active_rollout["idx"];rec=rollout_data[ridx]
                    probe_obs,probe_masks,pre_conds=active_rollout["probe"]
                    post=pair_distribution_metrics(model,probe_obs,probe_masks)
                    kl=[];move_tv=[]
                    for old,new,mask_np in zip(pre_conds,post["conditional"],probe_masks):
                        if old is None or new is None:continue
                        old=np.asarray(old,dtype=np.float64);new=np.asarray(new,dtype=np.float64)
                        kl.append(float(np.sum(old*np.log((old+1e-30)/(new+1e-30)))))
                        move_tv.append(float(0.5*np.abs(old-new).sum()))
                    rec["probe_post"]={k:v for k,v in post.items() if k!="conditional"}
                    rec["conditional_pair_kl_pre_post_mean"]=float(np.mean(kl)) if kl else float("nan")
                    rec["conditional_pair_tv_pre_post_mean"]=float(np.mean(move_tv)) if move_tv else float("nan")
                    pro=active_rollout["proactive"]
                    if pro.get("count",0):
                        obs=torch.as_tensor(pro["obs"],dtype=torch.float32)
                        act=torch.as_tensor(pro["actions"],dtype=torch.long)
                        with torch.inference_mode():
                            _,raw=model.yc_actor.raw_components(obs)
                            sel=raw.gather(1,act.unsqueeze(1)).squeeze(1).cpu().numpy()
                        delta=sel-pro["selected_score_pre"]
                        feat=pro["features"]
                        names=["feasible","target_eta","eta_advance","blockers","capacity_slack","moving_blocker_eta","dest_height","dest_inversion","dest_nearest_eta"]
                        rec["proactive_signal"]=dict(
                            count=int(pro["count"]),
                            raw_adv_mean=float(np.mean(pro["raw_adv"])),raw_adv_std=float(np.std(pro["raw_adv"])),
                            raw_adv_positive_fraction=float(np.mean(pro["raw_adv"]>0)),
                            normalized_adv_mean=float(np.mean(pro["normalized_adv"])),normalized_adv_std=float(np.std(pro["normalized_adv"])),
                            selected_pair_score_delta_mean=float(np.mean(delta)),selected_pair_score_delta_std=float(np.std(delta)),
                            corr_score_delta_raw_adv=corr(delta,pro["raw_adv"]),
                            corr_score_delta_return=corr(delta,pro["returns"]),
                            corr_raw_adv_by_feature={n:corr(pro["raw_adv"],feat[:,j]) for j,n in enumerate(names)},
                            corr_return_by_feature={n:corr(pro["returns"],feat[:,j]) for j,n in enumerate(names)},
                        )
            return result

    tr.compute_gae=audited_gae
    torch.nn.utils.clip_grad_norm_=audited_clip
    torch.optim.Adam=AuditedAdam
    try:
        cfg=ResourcePPOConfig(total_steps=2000,rollout_steps=512,update_epochs=2,
            critic_extra_epochs=a.critic_extra_epochs,minibatch_size=256,gamma=1.0,gae_lambda=1.0,
            learning_rate=3e-4,max_grad_norm=0.5,seed=a.seed,arrival_rate_per_hour=20.0,
            episode_complete_rollout=True)
        model,updates=tr.train_resource_marl(cfg,out/"native",Path(a.checkpoint))
    finally:
        tr.compute_gae=original_gae
        torch.nn.utils.clip_grad_norm_=original_clip
        torch.optim.Adam=original_adam

    # Summarize gradient rows by rollout.
    for rec in rollout_data:
        gs=[g for g in grad_rows if g["rollout"]==rec["rollout"]]
        if gs:
            rec["gradients"]={k:float(np.mean([g[k] for g in gs])) for k in [
                "operation_head_grad","pair_path_grad","pair_scorer_grad",
                "yc_context_encoder_grad","critic_grad","actor_grad"]}
            rec["gradient_minibatches"]=len(gs)

    dump(out/"actor_pair_signal.json",dict(
        seed=a.seed,critic_extra_epochs=a.critic_extra_epochs,
        actual_decisions=int(updates[-1]["global_step"]),ppo_updates=len(updates),
        rollout_diagnostics=rollout_data,gradient_minibatches=grad_rows,
        native_updates=updates,
    ))
    print(json.dumps(dict(seed=a.seed,extra=a.critic_extra_epochs,
        decisions=int(updates[-1]["global_step"]),rollouts=len(rollout_data),
        proactive=[r["proactive_count"] for r in rollout_data])),flush=True)

if __name__=="__main__":
    main()
