from __future__ import annotations

"""Centralized PPO baseline using the same V4 flat action interface as MARL."""

from dataclasses import asdict, dataclass
from pathlib import Path
import csv, json, random
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from v4_networks import SymmetricYardEncoder, PermutationInvariantCritic, mlp
from train_yc_marl import (ScenarioSampler, compute_gae, masked_distribution, rollout_ready, set_global_seeds,
    group_normalized_flat_yc_logits, structured_yc_entropy, load_compatible_state_dict)
from yc_marl_env import (
    N_BLOCKS, STACKS_PER_BLOCK, YC_ACTION_DIM, YC_PAIR_COUNT, YC_PROACTIVE_BASE,
    ResourceMARLYardEnv, STORAGE_AGENT, YC_AGENT_PREFIX,
)


class CentralizedSingleModel(nn.Module):
    CONTEXT_DIM=8; BLOCK_FEAT_DIM=8; SLOT_FEAT_DIM=4
    N_BLOCKS=N_BLOCKS; STACKS_PER_BLOCK=STACKS_PER_BLOCK; N_SLOTS=N_BLOCKS*STACKS_PER_BLOCK

    def __init__(self, global_obs_dim:int, yc_obs_dim:int, hidden:int=128, pair_hidden:int=16):
        super().__init__()
        expected=self.CONTEXT_DIM+self.N_BLOCKS*self.BLOCK_FEAT_DIM+self.N_SLOTS*self.SLOT_FEAT_DIM
        if global_obs_dim!=expected: raise ValueError(f"Expected global obs dim {expected}, got {global_obs_dim}")
        self.global_obs_dim=global_obs_dim; self.yc_obs_dim=yc_obs_dim
        self.actor_encoder=SymmetricYardEncoder(self.CONTEXT_DIM,self.N_BLOCKS,self.STACKS_PER_BLOCK,self.BLOCK_FEAT_DIM,self.SLOT_FEAT_DIM,hidden)
        self.storage_scorer=nn.Sequential(nn.Linear(3*hidden,hidden),nn.Tanh(),nn.Linear(hidden,1))
        self.local_context_encoder=mlp(ResourceMARLYardEnv.YC_CONTEXT_DIM,hidden,hidden)
        self.global_pair_proj=nn.Linear(hidden,pair_hidden); self.local_pair_proj=nn.Linear(hidden,pair_hidden)
        self.pair_encoder=nn.Sequential(nn.Linear(ResourceMARLYardEnv.YC_PAIR_FEAT_DIM,pair_hidden),nn.Tanh(),nn.Linear(pair_hidden,pair_hidden),nn.Tanh())
        self.pair_scorer=nn.Linear(pair_hidden,1)
        self.yc_base_head=nn.Sequential(nn.Linear(2*hidden,hidden),nn.Tanh(),nn.Linear(hidden,3))
        self.critic=PermutationInvariantCritic(self.CONTEXT_DIM,self.N_BLOCKS,self.STACKS_PER_BLOCK,self.BLOCK_FEAT_DIM,self.SLOT_FEAT_DIM,hidden)

    def _global_encode(self,obs):
        squeeze=obs.ndim==1
        if squeeze: obs=obs.unsqueeze(0)
        g,b,s=self.actor_encoder(obs)
        return obs,g,b,s,squeeze

    def storage_logits(self,obs):
        raw,g,b,s,squeeze=self._global_encode(obs)
        bf=b.unsqueeze(2).expand(-1,-1,self.STACKS_PER_BLOCK,-1)
        gf=g[:,None,None,:].expand(-1,self.N_BLOCKS,self.STACKS_PER_BLOCK,-1)
        logits=self.storage_scorer(torch.cat([gf,bf,s],dim=-1)).squeeze(-1).reshape(-1,self.N_SLOTS)
        return logits.squeeze(0) if squeeze else logits

    def yc_logits(self,central_obs,mask=None):
        squeeze=central_obs.ndim==1
        if squeeze:
            central_obs=central_obs.unsqueeze(0)
            if mask is not None and mask.ndim==1: mask=mask.unsqueeze(0)
        graw=central_obs[:,:self.global_obs_dim]
        yraw=central_obs[:,self.global_obs_dim:]
        _,g,_,_,_=self._global_encode(graw)
        cdim=ResourceMARLYardEnv.YC_CONTEXT_DIM; pf=ResourceMARLYardEnv.YC_PAIR_FEAT_DIM
        local_ctx=yraw[:,:cdim]; pairs=yraw[:,cdim:].reshape(-1,YC_PAIR_COUNT,pf)
        lc=self.local_context_encoder(local_ctx)
        operation_logits=self.yc_base_head(torch.cat([g,lc],dim=-1))
        ctx=(self.global_pair_proj(g)+self.local_pair_proj(lc)).unsqueeze(1)
        pe=self.pair_encoder(pairs); pair_scores=self.pair_scorer(torch.tanh(pe+ctx)).squeeze(-1)
        logits=group_normalized_flat_yc_logits(operation_logits,pair_scores,mask)
        if logits.shape[-1]!=YC_ACTION_DIM: raise RuntimeError(logits.shape)
        return logits.squeeze(0) if squeeze else logits

    def value(self,global_obs): return self.critic(global_obs)


def initialize_conservative_yc_head(model:CentralizedSingleModel, proactive_bias:float=-2.197224577, pair_init_std:float=0.05)->None:
    """Match the MARL GroupNorm YC parameterization for a fair centralized baseline."""
    base=model.yc_base_head[-1]; pair=model.pair_scorer
    with torch.no_grad():
        base.weight.zero_(); base.bias.zero_(); base.bias[2]=float(proactive_bias)
        nn.init.normal_(pair.weight,mean=0.0,std=float(pair_init_std)); pair.bias.zero_()


@dataclass
class SinglePPOConfig:
    total_steps:int=30000; rollout_steps:int=512; min_storage_transitions:int=64; min_yc_transitions:int=128
    max_rollout_multiplier:int=4; update_epochs:int=2; minibatch_size:int=256; gamma:float=1.0; gae_lambda:float=1.0
    clip_coef:float=0.20; ent_coef:float=0.01; yc_op_ent_coef:float=0.001; yc_pair_ent_coef:float=0.0001; yc_proactive_init_bias:float=-2.197224577; yc_pair_init_std:float=0.05
    vf_coef:float=0.5; learning_rate:float=3e-4; max_grad_norm:float=0.5; hidden:int=128; seed:int=1
    arrival_rate_per_hour:float=20.0; truck_wait_weight:float=1.0; storage_wait_weight:float=1.0; extra_move_weight:float=0.10
    risk_shaping_weight:float=0.0; yc_queue_shaping_weight:float=0.0; enable_proactive:bool=True; yc_move_time:float=2.0


def _write_csv(path:Path,rows:List[dict])->None:
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True); fields=sorted({k for r in rows for k in r})
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def train_single_ppo(config:SinglePPOConfig,out_dir:Path,init_checkpoint:Optional[Path]=None)->Tuple[CentralizedSingleModel,List[dict]]:
    set_global_seeds(config.seed); device=torch.device('cpu'); sampler=ScenarioSampler(config.seed+1000)
    env=ResourceMARLYardEnv(seed=sampler.next(),arrival_rate_per_hour=config.arrival_rate_per_hour,
        truck_wait_weight=config.truck_wait_weight,storage_wait_weight=config.storage_wait_weight,extra_move_weight=config.extra_move_weight,
        risk_shaping_weight=config.risk_shaping_weight,yc_queue_shaping_weight=config.yc_queue_shaping_weight,
        enable_proactive=config.enable_proactive,yc_move_time=config.yc_move_time)
    env.reset(); model=CentralizedSingleModel(env.global_obs_dim,env.yc_obs_dim,hidden=config.hidden).to(device)
    if init_checkpoint is not None:
        ck=torch.load(Path(init_checkpoint),map_location=device,weights_only=False); state=ck.get('state_dict',ck)
        if bool(ck.get('storage_only',False)):
            state={k:v for k,v in state.items() if not k.startswith(('yc_base_head.','pair_encoder.','pair_scorer.','local_context_encoder.','global_pair_proj.','local_pair_proj.'))}
        load_compatible_state_dict(model,state)
        if bool(ck.get('storage_only',False)): initialize_conservative_yc_head(model,config.yc_proactive_init_bias,config.yc_pair_init_std)
    else: initialize_conservative_yc_head(model,config.yc_proactive_init_bias,config.yc_pair_init_std)
    opt=torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    global_step=update_idx=0; episodes=[]; updates=[]
    while global_step<config.total_steps:
        actor_buf=[]; critic_buf=[]; masks_buf=[]; roles=[]; actions=[]; oldlog=[]; rewards=[]; dones=[]; values=[]; nextvals=[]; teams=[]
        storage_n=yc_n=0; remaining=max(config.total_steps-global_step,1); base=min(config.rollout_steps,remaining); max_n=max(base,config.rollout_steps*config.max_rollout_multiplier)
        while not rollout_ready(len(actions),storage_n,yc_n,base,max_n,config):
            active=env.active_agent(); role=0 if active==STORAGE_AGENT else 1
            if role==1 and not active.startswith(YC_AGENT_PREFIX): raise RuntimeError(active)
            aobs=env.centralized_actor_observation(); gobs=env.critic_observation(); mask=env.action_mask()
            at=torch.as_tensor(aobs,dtype=torch.float32); gt=torch.as_tensor(gobs,dtype=torch.float32); mt=torch.as_tensor(mask,dtype=torch.bool)
            with torch.no_grad():
                logits=model.storage_logits(at) if role==0 else model.yc_logits(at,mt); dist=masked_distribution(logits,mt); act=dist.sample(); lp=dist.log_prob(act); val=model.value(gt)
            _,rew,done,_,info=env.step(int(act.item()))
            with torch.no_grad(): nv=0.0 if done else float(model.value(torch.as_tensor(env.critic_observation(),dtype=torch.float32)).item())
            actor_buf.append(aobs.copy());critic_buf.append(gobs.copy());masks_buf.append(mask.copy());roles.append(role);actions.append(int(act.item()));oldlog.append(float(lp.item()));rewards.append(float(rew));dones.append(float(done));values.append(float(val.item()));nextvals.append(nv);teams.append(float(info['reward_components']['team']))
            storage_n+=int(role==0);yc_n+=int(role==1);global_step+=1
            if done:
                episodes.append({'global_step':global_step,'seed':env.seed,'arrival_rate_per_hour':env.arrival_rate_per_hour,'episode_return':info['episode_return'],**info['kpis']});env.reset(seed=sampler.next())
        rewards_np=np.asarray(rewards,np.float32); vals=np.asarray(values,np.float32); nvals=np.asarray(nextvals,np.float32); dn=np.asarray(dones,np.float32)
        adv,ret=compute_gae(rewards_np,vals,nvals,dn,config.gamma,config.gae_lambda);adv=(adv-adv.mean())/(adv.std()+1e-8)
        actions_t=torch.as_tensor(actions,dtype=torch.long);old_t=torch.as_tensor(oldlog,dtype=torch.float32);adv_t=torch.as_tensor(adv);ret_t=torch.as_tensor(ret);critic_t=torch.as_tensor(np.asarray(critic_buf),dtype=torch.float32)
        n=len(actions)
        for _ in range(config.update_epochs):
            order=np.random.permutation(n)
            for st in range(0,n,config.minibatch_size):
                ids=order[st:st+config.minibatch_size]
                if len(ids)==0:continue
                pg=[];ent=[]
                for role in (0,1):
                    rid=[i for i in ids if roles[i]==role]
                    if not rid:continue
                    obs=torch.as_tensor(np.stack([actor_buf[i] for i in rid]),dtype=torch.float32); masks=torch.as_tensor(np.stack([masks_buf[i] for i in rid]),dtype=torch.bool)
                    logits=model.storage_logits(obs) if role==0 else model.yc_logits(obs,masks);dist=masked_distribution(logits,masks);ri=torch.as_tensor(rid,dtype=torch.long)
                    nl=dist.log_prob(actions_t[ri]);ratio=(nl-old_t[ri]).exp();p1=-adv_t[ri]*ratio;p2=-adv_t[ri]*torch.clamp(ratio,1-config.clip_coef,1+config.clip_coef);pg.append(torch.maximum(p1,p2).mean())
                    if role==0:
                        ent.append(config.ent_coef*dist.entropy().mean())
                    else:
                        op_h,pair_h,_=structured_yc_entropy(dist,masks)
                        ent.append(config.yc_op_ent_coef*op_h.mean()+config.yc_pair_ent_coef*pair_h.mean())
                mb=torch.as_tensor(ids,dtype=torch.long);v=model.value(critic_t[mb]);vl=.5*(v-ret_t[mb]).pow(2).mean();loss=torch.stack(pg).mean()+config.vf_coef*vl-torch.stack(ent).mean()
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),config.max_grad_norm);opt.step()
        update_idx+=1;updates.append({'update':update_idx,'global_step':global_step,'storage_decisions':storage_n,'yc_decisions':yc_n,'mean_reward':float(np.mean(rewards_np)),'mean_team_reward':float(np.mean(teams))})
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':model.state_dict(),'config':asdict(config),'global_obs_dim':env.global_obs_dim,'yc_obs_dim':env.yc_obs_dim,'storage_only':False,
        'init_checkpoint':str(init_checkpoint) if init_checkpoint is not None else None,
        'continuation_mode':'weights_only_warm_start' if init_checkpoint is not None else 'scratch',
        'run_global_step':int(global_step),'update_count':int(update_idx),'current_env_seed':int(env.seed),
        'optimizer_state_dict':opt.state_dict(),'python_random_state':random.getstate(),
        'numpy_random_state':np.random.get_state(),'torch_rng_state':torch.get_rng_state(),
        'scenario_sampler_state':sampler.rng.getstate()},out_dir/'single_ppo_final.pt')
    _write_csv(out_dir/'single_ppo_train_episodes.csv',episodes);_write_csv(out_dir/'single_ppo_updates.csv',updates);(out_dir/'single_ppo_config.json').write_text(json.dumps(asdict(config),indent=2),encoding='utf-8')
    return model,updates

if __name__=='__main__': train_single_ppo(SinglePPOConfig(),Path('single_ppo_run'))
