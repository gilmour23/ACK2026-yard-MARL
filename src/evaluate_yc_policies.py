from __future__ import annotations

from pathlib import Path
import csv,json
from typing import Optional
import numpy as np
import torch

from train_yc_marl import ResourceCooperativeModel,masked_distribution
from train_yc_single import CentralizedSingleModel
from validate_yc_marl import heuristic_action
from yc_marl_env import ResourceMARLYardEnv,STORAGE_AGENT


def _load_model(kind:str,checkpoint:Path,arrival_rate_per_hour:float=20.0):
    ck=torch.load(Path(checkpoint),map_location='cpu',weights_only=False);env=ResourceMARLYardEnv(seed=1,arrival_rate_per_hour=arrival_rate_per_hour);env.reset();hidden=int(ck.get('hidden',ck.get('config',{}).get('hidden',128)))
    if kind=='marl': model=ResourceCooperativeModel(env.global_obs_dim,env.storage_obs_dim,env.yc_obs_dim,hidden=hidden)
    elif kind=='single': model=CentralizedSingleModel(env.global_obs_dim,env.yc_obs_dim,hidden=hidden)
    else: raise ValueError(kind)
    model.load_state_dict(ck.get('state_dict',ck),strict=False);model.eval();return model


def run_episode(seed:int,method:str,model=None,arrival_rate_per_hour:float=20.0,enable_proactive:bool=True,include_resource_state:bool=True,yc_move_time:float=2.0,stochastic:bool=False,policy_seed:int=0)->dict:
    env=ResourceMARLYardEnv(seed=seed,arrival_rate_per_hour=arrival_rate_per_hour,enable_proactive=enable_proactive,include_resource_state=include_resource_state,yc_move_time=yc_move_time);env.reset();rng=torch.Generator(device='cpu').manual_seed(int(policy_seed));done=False;steps=0
    while not done and steps<100000:
        if method=='heuristic': action=heuristic_action(env)
        else:
            role=0 if env.active_agent()==STORAGE_AGENT else 1;mask=torch.as_tensor(env.action_mask(),dtype=torch.bool)
            with torch.inference_mode():
                if method=='marl': obs=torch.as_tensor(env.actor_observation(),dtype=torch.float32);logits=model.storage_logits(obs) if role==0 else model.yc_logits(obs,mask)
                elif method=='single': obs=torch.as_tensor(env.centralized_actor_observation(),dtype=torch.float32);logits=model.storage_logits(obs) if role==0 else model.yc_logits(obs,mask)
                else: raise ValueError(method)
                dist=masked_distribution(logits,mask)
            action=int(torch.multinomial(dist.probs,1,generator=rng).item()) if stochastic else int(dist.probs.argmax().item())
        _,_,done,_,info=env.step(action);steps+=1
    if not done: raise RuntimeError(f'episode did not terminate {seed=} {method=}')
    k=info['kpis'];objective=(k['mean_truck_completion_delay']+k['mean_storage_completion_delay']+0.10*k['extra_yc_minutes_per_retrieval'])
    return {'seed':seed,'method':method,'evaluation_mode':'stochastic' if stochastic else 'flat_greedy','policy_seed':policy_seed,'arrival_rate_per_hour':arrival_rate_per_hour,'enable_proactive':enable_proactive,'include_resource_state':include_resource_state,'yc_move_time':yc_move_time,'decisions':steps,'objective_proxy':objective,**k}


def evaluate(out_csv:Path,method:str,checkpoint:Optional[Path]=None,seeds=range(101,131),repeats:int=1,arrival_rate_per_hour:float=20.0,enable_proactive:bool=True,include_resource_state:bool=True,yc_move_time:float=2.0,stochastic:bool=False):
    model=None if method=='heuristic' else _load_model(method,checkpoint,arrival_rate_per_hour);rows=[]
    for seed in seeds:
        reps=1 if method=='heuristic' or not stochastic else repeats
        for r in range(reps): rows.append(run_episode(seed,method,model,arrival_rate_per_hour,enable_proactive,include_resource_state,yc_move_time,stochastic,seed*100+r))
    out_csv=Path(out_csv);out_csv.parent.mkdir(parents=True,exist_ok=True)
    with out_csv.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    summary={key:float(np.mean([x[key] for x in rows])) for key in ('mean_truck_completion_delay','mean_storage_completion_delay','rehandling_moves','proactive_moves','total_yc_moves','extra_yc_minutes_per_retrieval','mean_yc_utilization','max_yc_queue','objective_proxy')};summary['arrival_rate_per_hour']=arrival_rate_per_hour;summary['n_scenarios']=len(set(x['seed'] for x in rows));summary['evaluation_mode']='stochastic' if stochastic else 'flat_greedy';summary['repeats']=int(repeats if stochastic and method!='heuristic' else 1);out_csv.with_suffix('.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');return summary
