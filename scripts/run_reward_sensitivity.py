from __future__ import annotations
from pathlib import Path
from train_yc_marl import ResourcePPOConfig,train_resource_marl
from evaluate_yc_policies import evaluate

RATE=20.0


def run(root:Path,bc_checkpoint:Path,steps:int=10000,seed:int=1):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    rows=[]
    for wy in (0.0,0.05,0.10,0.25):
        d=root/f'wy_{wy:.2f}';cfg=ResourcePPOConfig(total_steps=steps,seed=seed,arrival_rate_per_hour=RATE,extra_move_weight=wy);train_resource_marl(cfg,d,bc_checkpoint);s=evaluate(d/'eval.csv','marl',d/'resource_marl_final.pt',seeds=range(301,311),arrival_rate_per_hour=RATE,stochastic=False);rows.append(('wY',wy,s))
    # Truck priority should be run only after selecting wY from held-out trade-offs.
    return rows
