from __future__ import annotations

"""Pre-registered value-budget pilot following the Astra critic diagnostic.

Control: episode-complete PPO with the canonical 2 joint actor+critic epochs.
Treatment: identical training plus 18 critic-only epochs on each rollout, for
20 total value exposures.  No actor/reward/environment/learning-rate/clipping
change is introduced.

Validation is fixed to scenarios 701-710, 3 stochastic repeats per scenario.
This is a diagnostic bank, not a final unseen test set.
"""

from pathlib import Path
import argparse, csv, json, hashlib
import numpy as np
import torch

from train_yc_marl import (
    ResourcePPOConfig, train_resource_marl, masked_distribution,
    structured_yc_entropy, YC_PROACTIVE_BASE,
)
from evaluate_yc_policies import _load_model
from yc_marl_env import ResourceMARLYardEnv, STORAGE_AGENT


CANONICAL_CHECKPOINT_SHA256 = "2bbd3a2e795a215d3fcbd654d58fad154d4298b55bb50ce92f54ad50b82ddb59"


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def _metrics(pred, target):
    p=np.asarray(pred,dtype=np.float64); y=np.asarray(target,dtype=np.float64)
    e=p-y; var=float(np.var(y))
    return {
        "n":int(len(y)),
        "ev":float("nan") if var < 1e-12 else float(1.0-np.var(e)/var),
        "rmse":float(np.sqrt(np.mean(e*e))),
        "bias":float(np.mean(e)),
        "prediction_mean":float(np.mean(p)),
        "prediction_std":float(np.std(p)),
        "target_mean":float(np.mean(y)),
        "target_std":float(np.std(y)),
    }


def evaluate_with_critic(checkpoint: Path, out_csv: Path, seeds, repeats: int = 3):
    model=_load_model("marl",checkpoint,20.0)
    model.eval()
    episode_rows=[]
    all_pred=[]; all_mc=[]
    pair_h=[]; pair_tv=[]; ppro=[]
    eta_expected=[]; eta_uniform=[]
    dest_height_expected=[]; dest_height_uniform=[]
    dest_inv_expected=[]; dest_inv_uniform=[]

    for seed in seeds:
        for repeat in range(repeats):
            env=ResourceMARLYardEnv(seed=int(seed),arrival_rate_per_hour=20.0)
            env.reset()
            generator=torch.Generator(device="cpu").manual_seed(int(seed)*100+repeat)
            rewards=[]; preds=[]; done=False; steps=0
            while not done and steps < 100000:
                role=0 if env.active_agent()==STORAGE_AGENT else 1
                mask_np=env.action_mask()
                mask=torch.as_tensor(mask_np,dtype=torch.bool)
                gobs=torch.as_tensor(env.critic_observation(),dtype=torch.float32)
                with torch.inference_mode():
                    pred=float(model.value(gobs).item())
                    aobs_np=env.actor_observation()
                    aobs=torch.as_tensor(aobs_np,dtype=torch.float32)
                    logits=model.storage_logits(aobs) if role==0 else model.yc_logits(aobs,mask)
                    dist=masked_distribution(logits,mask)
                    if role==1:
                        op_h,h,p=structured_yc_entropy(dist,mask)
                        p=float(p.item()); ppro.append(p)
                        feasible=mask[YC_PROACTIVE_BASE:]
                        n=int(feasible.sum().item())
                        if p > 1e-12 and n > 1:
                            cond=(dist.probs[YC_PROACTIVE_BASE:][feasible]/p).double()
                            uni=torch.full_like(cond,1.0/n)
                            pair_h.append(float(h.item()))
                            pair_tv.append(float(0.5*torch.abs(cond-uni).sum().item()))
                            pair_raw=torch.as_tensor(
                                aobs_np[env.YC_CONTEXT_DIM:].reshape(env.YC_PAIR_COUNT,env.YC_PAIR_FEAT_DIM),
                                dtype=torch.float64
                            )[feasible]
                            eta=pair_raw[:,1]*180.0
                            dh=pair_raw[:,6]*4.0
                            di=pair_raw[:,7]*4.0
                            eta_expected.append(float((cond*eta).sum().item()))
                            eta_uniform.append(float(eta.mean().item()))
                            dest_height_expected.append(float((cond*dh).sum().item()))
                            dest_height_uniform.append(float(dh.mean().item()))
                            dest_inv_expected.append(float((cond*di).sum().item()))
                            dest_inv_uniform.append(float(di.mean().item()))
                    action=int(torch.multinomial(dist.probs,1,generator=generator).item())
                _,reward,done,_,info=env.step(action)
                preds.append(pred); rewards.append(float(reward)); steps+=1
            if not done:
                raise RuntimeError(f"episode did not terminate: seed={seed} repeat={repeat}")
            mc=np.zeros(len(rewards),dtype=np.float64); running=0.0
            for i in range(len(rewards)-1,-1,-1):
                running=float(rewards[i])+running; mc[i]=running
            all_pred.extend(preds); all_mc.extend(mc.tolist())
            k=info["kpis"]
            episode_rows.append({
                "seed":int(seed),"repeat":int(repeat),"policy_seed":int(seed)*100+repeat,
                "decisions":steps,
                "objective_proxy":float(k["mean_truck_completion_delay"]+k["mean_storage_completion_delay"]+0.1*k["extra_yc_minutes_per_retrieval"]),
                **k,
            })

    out_csv=Path(out_csv); out_csv.parent.mkdir(parents=True,exist_ok=True)
    with out_csv.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for r in episode_rows for k in r}))
        w.writeheader(); w.writerows(episode_rows)

    critic=_metrics(all_pred,all_mc)
    mean=lambda xs: float(np.mean(xs)) if xs else float("nan")
    policy={
        key:float(np.mean([r[key] for r in episode_rows]))
        for key in (
            "mean_truck_completion_delay","mean_storage_completion_delay",
            "rehandling_moves","proactive_moves","extra_yc_minutes_per_retrieval",
            "mean_yc_utilization","max_yc_queue","objective_proxy",
        )
    }
    pair={
        "states":int(len(pair_tv)),
        "p_proactive_mean":mean(ppro),
        "conditional_pair_entropy_norm_mean":mean(pair_h),
        "conditional_pair_tv_uniform_mean":mean(pair_tv),
        "target_eta_expected_min":mean(eta_expected),
        "target_eta_uniform_min":mean(eta_uniform),
        "target_eta_expected_minus_uniform_min":mean([a-b for a,b in zip(eta_expected,eta_uniform)]),
        "destination_height_expected":mean(dest_height_expected),
        "destination_height_uniform":mean(dest_height_uniform),
        "destination_inversion_expected":mean(dest_inv_expected),
        "destination_inversion_uniform":mean(dest_inv_uniform),
    }
    summary={"episodes":len(episode_rows),"scenarios":list(map(int,seeds)),"repeats":int(repeats),"critic":critic,"policy":policy,"pair":pair}
    out_csv.with_suffix(".json").write_text(json.dumps(summary,indent=2,allow_nan=True),encoding="utf-8")
    return summary


def run(checkpoint: Path, out: Path, steps: int = 2000, train_seeds=(21,22,23), eval_seeds=range(701,711), repeats: int = 3):
    checkpoint=Path(checkpoint); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    actual_sha=sha256(checkpoint)
    if actual_sha != CANONICAL_CHECKPOINT_SHA256:
        raise RuntimeError(f"checkpoint SHA mismatch: {actual_sha}")

    protocol={
        "checkpoint_sha256":actual_sha,
        "training_seeds":list(map(int,train_seeds)),
        "validation_scenarios":list(map(int,eval_seeds)),
        "repeats":int(repeats),
        "steps_threshold":int(steps),
        "control":{"update_epochs":2,"critic_extra_epochs":0},
        "treatment":{"update_epochs":2,"critic_extra_epochs":18},
        "fixed":{"episode_complete_rollout":True,"rollout_steps":512,"minibatch_size":256,"learning_rate":3e-4,"max_grad_norm":0.5},
    }
    (out/"PROTOCOL.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    results={}

    for label,extra in (("control_value2",0),("treatment_value20",18)):
        results[label]={}
        for seed in train_seeds:
            run_dir=out/label/f"seed{seed}"
            cfg=ResourcePPOConfig(
                total_steps=int(steps),
                rollout_steps=512,
                min_storage_transitions=64,
                min_yc_transitions=128,
                max_rollout_multiplier=4,
                update_epochs=2,
                critic_extra_epochs=int(extra),
                minibatch_size=256,
                gamma=1.0,
                gae_lambda=1.0,
                learning_rate=3e-4,
                max_grad_norm=0.5,
                seed=int(seed),
                arrival_rate_per_hour=20.0,
                episode_complete_rollout=True,
            )
            _,updates=train_resource_marl(cfg,run_dir,checkpoint)
            if not updates:
                raise RuntimeError(f"no updates: {label} seed={seed}")
            if any(float(r.get("critic_extra_actor_max_abs_delta",0.0)) != 0.0 for r in updates):
                raise RuntimeError(f"actor changed in extra critic epochs: {label} seed={seed}")
            train={
                "actual_decisions":int(updates[-1]["global_step"]),
                "ppo_updates":int(len(updates)),
                "critic_extra_steps":int(sum(int(r.get("critic_extra_steps",0)) for r in updates)),
                "post_update_value_ev_last":float(updates[-1]["post_update_value_ev"]),
                "post_update_value_rmse_last":float(updates[-1]["post_update_value_rmse"]),
                "mc_value_ev_mean":float(np.nanmean([float(r["mc_value_ev"]) for r in updates])),
                "mc_value_rmse_mean":float(np.nanmean([float(r["mc_value_rmse"]) for r in updates])),
                "yc_pair_entropy_norm_mean":float(np.mean([float(r["yc_pair_entropy_norm"]) for r in updates])),
                "yc_proactive_probability_mean":float(np.mean([float(r["yc_proactive_probability"]) for r in updates])),
                "yc_sampled_proactive_rate_mean":float(np.mean([float(r["yc_sampled_proactive_rate"]) for r in updates])),
            }
            ev=evaluate_with_critic(run_dir/"resource_marl_final.pt",run_dir/"eval_701_710_stochastic.csv",eval_seeds,repeats)
            results[label][str(seed)]={"training":train,"evaluation":ev}
            (out/"pilot_summary.partial.json").write_text(json.dumps(results,indent=2,allow_nan=True),encoding="utf-8")

    (out/"pilot_summary.json").write_text(json.dumps(results,indent=2,allow_nan=True),encoding="utf-8")
    return results


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--out",default="value_budget_pilot")
    ap.add_argument("--steps",type=int,default=2000)
    ap.add_argument("--train-seeds",type=int,nargs="+",default=[21,22,23])
    ap.add_argument("--eval-start",type=int,default=701)
    ap.add_argument("--eval-count",type=int,default=10)
    ap.add_argument("--repeats",type=int,default=3)
    a=ap.parse_args()
    run(Path(a.checkpoint),Path(a.out),a.steps,tuple(a.train_seeds),range(a.eval_start,a.eval_start+a.eval_count),a.repeats)
