from __future__ import annotations

"""Storage-only behavior-cloning warm start for the final V4 environment.

YC policies are deliberately not imitated; they start from the conservative
scratch initialization used by the PPO trainers.
"""

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn

from train_yc_marl import ResourceCooperativeModel
from train_yc_single import CentralizedSingleModel
from validate_yc_marl import heuristic_action
from yc_marl_env import ResourceMARLYardEnv, STORAGE_AGENT


@dataclass
class StorageDataset:
    local_obs:list
    global_obs:list
    masks:list
    actions:list


def collect_storage_dataset(seeds:int=8,start_seed:int=1000,arrival_rate_per_hour:float=20.0,include_resource_state:bool=True)->StorageDataset:
    ds=StorageDataset([],[],[],[])
    for j in range(seeds):
        env=ResourceMARLYardEnv(seed=start_seed+j,arrival_rate_per_hour=arrival_rate_per_hour,include_resource_state=include_resource_state)
        env.reset();done=False
        while not done:
            if env.active_agent()==STORAGE_AGENT:
                ds.local_obs.append(env.actor_observation().copy());ds.global_obs.append(env.critic_observation().copy());ds.masks.append(env.action_mask().copy());ds.actions.append(heuristic_action(env))
                action=ds.actions[-1]
            else:
                action=heuristic_action(env)
            _,_,done,_,_=env.step(action)
    return ds


def train_bc(kind:str,out_path:Path,seeds:int=8,epochs:int=3,batch_size:int=128,lr:float=5e-4,arrival_rate_per_hour:float=20.0,include_resource_state:bool=True,enable_proactive:bool=True,seed:int=1,hidden:int=128)->dict:
    ds=collect_storage_dataset(seeds=seeds,arrival_rate_per_hour=arrival_rate_per_hour,include_resource_state=include_resource_state)
    env=ResourceMARLYardEnv(seed=1,arrival_rate_per_hour=arrival_rate_per_hour,include_resource_state=include_resource_state,enable_proactive=enable_proactive);env.reset()
    if kind=='marl': model:nn.Module=ResourceCooperativeModel(env.global_obs_dim,env.storage_obs_dim,env.yc_obs_dim,hidden=hidden)
    elif kind=='single': model=CentralizedSingleModel(env.global_obs_dim,env.yc_obs_dim,hidden=hidden)
    else: raise ValueError(kind)
    opt=torch.optim.Adam(model.parameters(),lr=lr);idx=np.arange(len(ds.actions));rng=np.random.default_rng(seed)
    for _ in range(epochs):
        rng.shuffle(idx)
        for st in range(0,len(idx),batch_size):
            ids=idx[st:st+batch_size]
            obs_np=np.stack([ds.local_obs[i] if kind=='marl' else ds.global_obs[i] for i in ids])
            logits=model.storage_logits(torch.as_tensor(obs_np,dtype=torch.float32));masks=torch.as_tensor(np.stack([ds.masks[i] for i in ids]),dtype=torch.bool);acts=torch.as_tensor([ds.actions[i] for i in ids],dtype=torch.long)
            logits=logits.masked_fill(~masks,torch.finfo(logits.dtype).min);loss=nn.functional.cross_entropy(logits,acts)
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
    ok=0
    with torch.no_grad():
        for i in idx:
            obs=ds.local_obs[i] if kind=='marl' else ds.global_obs[i];logits=model.storage_logits(torch.as_tensor(obs,dtype=torch.float32));mask=torch.as_tensor(ds.masks[i],dtype=torch.bool);a=int(logits.masked_fill(~mask,-1e30).argmax().item());ok+=int(a==ds.actions[i])
    metrics={'storage_accuracy_train':ok/max(len(ds.actions),1),'storage_samples':len(ds.actions),'yc_samples':0,'arrival_rate_per_hour':arrival_rate_per_hour}
    out_path=Path(out_path);out_path.parent.mkdir(parents=True,exist_ok=True);torch.save({'state_dict':model.state_dict(),'kind':kind,'hidden':hidden,'storage_only':True,'include_resource_state':include_resource_state,'arrival_rate_per_hour':arrival_rate_per_hour,'metrics':metrics},out_path);out_path.with_suffix('.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8');return metrics

if __name__=='__main__':
    root=Path(__file__).parent;print('MARL',train_bc('marl',root/'bc_marl_storage_only.pt'));print('SINGLE',train_bc('single',root/'bc_single_storage_only.pt'))
