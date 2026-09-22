from __future__ import annotations

"""Reproducible V4 experiment runner using one fixed operating condition."""

from pathlib import Path
import argparse, json

from pretrain_yc_bc import train_bc
from train_yc_marl import ResourcePPOConfig, train_resource_marl
from train_yc_single import SinglePPOConfig, train_single_ppo
from evaluate_yc_policies import evaluate

RATE=20.0


def run(root:Path,quick:bool=False,allow_full_30k:bool=False):
    if not quick and not allow_full_30k:
        raise RuntimeError(
            "Full 30k training is currently No-Go. "
            "Re-run with --allow-full-30k only after an explicit audit Go decision."
        )
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    bc_seeds=2 if quick else 8;bc_epochs=1 if quick else 3;steps=2000 if quick else 30000;train_seeds=[1] if quick else [1,2,3];eval_seeds=range(201,206) if quick else range(201,231)
    bc_m=root/'bc_marl_storage_only.pt';bc_s=root/'bc_single_storage_only.pt'
    train_bc('marl',bc_m,seeds=bc_seeds,epochs=bc_epochs,arrival_rate_per_hour=RATE)
    train_bc('single',bc_s,seeds=bc_seeds,epochs=bc_epochs,arrival_rate_per_hour=RATE)
    learned=[]
    for seed in train_seeds:
        d=root/f'marl_seed{seed}';cfg=ResourcePPOConfig(total_steps=steps,rollout_steps=512,update_epochs=2,minibatch_size=256,seed=seed,arrival_rate_per_hour=RATE);train_resource_marl(cfg,d,bc_m);learned.append(('marl',d/'resource_marl_final.pt',True,True,f'Proposed_MARL_s{seed}'))
        d=root/f'marl_noresource_seed{seed}';cfg=ResourcePPOConfig(total_steps=steps,rollout_steps=512,update_epochs=2,minibatch_size=256,seed=seed,arrival_rate_per_hour=RATE,include_resource_state=False);train_resource_marl(cfg,d,bc_m);learned.append(('marl',d/'resource_marl_final.pt',True,False,f'MARL_noResource_s{seed}'))
        d=root/f'marl_nopro_seed{seed}';cfg=ResourcePPOConfig(total_steps=steps,rollout_steps=512,update_epochs=2,minibatch_size=256,seed=seed,arrival_rate_per_hour=RATE,enable_proactive=False);train_resource_marl(cfg,d,bc_m);learned.append(('marl',d/'resource_marl_final.pt',False,True,f'MARL_noProactive_s{seed}'))
        d=root/f'central_seed{seed}';cfgs=SinglePPOConfig(total_steps=steps,rollout_steps=512,update_epochs=2,minibatch_size=256,seed=seed,arrival_rate_per_hour=RATE);train_single_ppo(cfgs,d,bc_s);learned.append(('single',d/'single_ppo_final.pt',True,True,f'Centralized_PPO_s{seed}'))
    summaries={'Heuristic':evaluate(root/'eval_heuristic.csv','heuristic',seeds=eval_seeds,arrival_rate_per_hour=RATE,stochastic=False)}
    eval_repeats=2 if quick else 3
    for kind,ck,pro,res,name in learned:
        # PPO is trained as a stochastic policy.  Stochastic evaluation is the primary
        # protocol; flat argmax is retained only as a separate deployment diagnostic.
        summaries[name+'_stochastic']=evaluate(root/f'eval_{name}_stochastic.csv',kind,checkpoint=ck,seeds=eval_seeds,repeats=eval_repeats,arrival_rate_per_hour=RATE,enable_proactive=pro,include_resource_state=res,stochastic=True)
        summaries[name+'_flat_greedy']=evaluate(root/f'eval_{name}_flat_greedy.csv',kind,checkpoint=ck,seeds=eval_seeds,arrival_rate_per_hour=RATE,enable_proactive=pro,include_resource_state=res,stochastic=False)
    (root/'V4_summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    return summaries

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='v4_run');ap.add_argument('--quick',action='store_true');ap.add_argument('--allow-full-30k',action='store_true');a=ap.parse_args();run(Path(a.out),a.quick,a.allow_full_30k)
