from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

from yc_marl_env import (
    ResourceMARLYardEnv, STORAGE_AGENT, YC_MANDATORY, YC_IDLE,
    YC_PROACTIVE_BASE, encode_proactive_action,
)


def _heuristic_proactive_action(env: ResourceMARLYardEnv, block: int, mask: np.ndarray):
    sim=env.sim; assert sim is not None
    # Information-aware baseline: target priority = ETA urgency, blockers,
    # unexpected ETA advance; destination = transparent relocation heuristic.
    for cid in sim.proactive_candidates(block):
        c=sim.containers[cid]
        if c.stack is None: continue
        moving_id=sim.yard.top(block,c.stack)
        if moving_id is None or moving_id==cid: continue
        try:
            dest=sim.policy.choose_relocation_destination(sim,moving_id,block,c.stack)
        except RuntimeError:
            continue
        action=encode_proactive_action(sim._target_position(cid),dest)
        if bool(mask[action]): return action
    return None


def heuristic_action(env: ResourceMARLYardEnv)->int:
    sim=env.sim; assert sim is not None and sim.pending_decision is not None
    if env.active_agent()==STORAGE_AGENT:
        d=sim.pending_decision; b,s=sim.policy.choose_storage(sim,d.cid,d.candidates or [])
        return b*sim.yard.stacks_per_block+s
    block=env.active_block(); assert block is not None; mask=env.action_mask()
    # Baseline uses mandatory-first operation. Proactive work is an idle-capacity
    # pre-marshalling rule, while learned policies may explicitly trade the two.
    if bool(mask[YC_MANDATORY]): return YC_MANDATORY
    pa=_heuristic_proactive_action(env,block,mask)
    if pa is not None: return pa
    if bool(mask[YC_IDLE]): return YC_IDLE
    raise RuntimeError('No valid heuristic YC action')


def run_episode(seed:int,arrival_rate_per_hour:float=20.0)->dict:
    env=ResourceMARLYardEnv(seed=seed,arrival_rate_per_hour=arrival_rate_per_hour)
    _,info=env.reset();done=False;steps=0
    while not done and steps<100000:
        action=heuristic_action(env);_,_,done,_,info=env.step(action);steps+=1
    if not done: raise RuntimeError(f'Episode did not terminate: {seed=}')
    k=info['kpis'];assert k['unretrieved_containers']==0.0;assert k['unstored_inbound_containers']==0.0
    for b in range(env.n_blocks):
        assert not env.sim.ycs[b].busy and len(env.sim.ycs[b].queue)==0
        for s in range(env.stacks_per_block): assert len(env.sim.yard.stacks[b][s])<=env.max_tier
    return {'seed':seed,'arrival_rate_per_hour':arrival_rate_per_hour,'decisions':steps,**k}


def main():
    rate=20.0;rows=[run_episode(seed,rate) for seed in range(101,111)]
    out=Path(__file__).with_name('yc_marl_validation_10seeds.csv')
    with out.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print({k:float(np.mean([r[k] for r in rows])) for k in ('mean_truck_completion_delay','mean_storage_completion_delay','rehandling_moves','proactive_moves','total_yc_moves','mean_yc_utilization','max_yc_queue')})
    print(out)

if __name__=='__main__':main()
