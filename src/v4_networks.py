from __future__ import annotations
import torch
from torch import nn


def mlp(in_dim:int,hidden:int,out_dim:int)->nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,out_dim))


class SymmetricYardEncoder(nn.Module):
    def __init__(self,context_dim,n_blocks,stacks_per_block,block_feat_dim,slot_feat_dim,hidden):
        super().__init__();self.context_dim=context_dim;self.n_blocks=n_blocks;self.stacks_per_block=stacks_per_block;self.block_feat_dim=block_feat_dim;self.slot_feat_dim=slot_feat_dim;self.n_slots=n_blocks*stacks_per_block
        self.expected_dim=context_dim+n_blocks*block_feat_dim+self.n_slots*slot_feat_dim
        self.context_encoder=nn.Sequential(nn.Linear(context_dim,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh())
        self.slot_encoder=nn.Sequential(nn.Linear(slot_feat_dim,hidden),nn.Tanh())
        self.block_encoder=nn.Sequential(nn.Linear(block_feat_dim+2*hidden,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh())
        self.global_fusion=nn.Sequential(nn.Linear(3*hidden,hidden),nn.Tanh())
    def forward(self,obs):
        if obs.ndim!=2: raise ValueError('SymmetricYardEncoder expects batched tensor')
        if obs.shape[-1]!=self.expected_dim: raise ValueError(f'Expected {self.expected_dim}, got {obs.shape[-1]}')
        b0=self.context_dim;b1=b0+self.n_blocks*self.block_feat_dim
        context=obs[:,:b0];blocks=obs[:,b0:b1].reshape(-1,self.n_blocks,self.block_feat_dim);slots=obs[:,b1:].reshape(-1,self.n_blocks,self.stacks_per_block,self.slot_feat_dim)
        se=self.slot_encoder(slots);sm=se.mean(dim=2);sx=se.max(dim=2).values;be=self.block_encoder(torch.cat([blocks,sm,sx],dim=-1));bm=be.mean(dim=1);bx=be.max(dim=1).values;ce=self.context_encoder(context);ge=self.global_fusion(torch.cat([ce,bm,bx],dim=-1));return ge,be,se


class PermutationInvariantCritic(nn.Module):
    def __init__(self,context_dim,n_blocks,stacks_per_block,block_feat_dim,slot_feat_dim,hidden):
        super().__init__();self.encoder=SymmetricYardEncoder(context_dim,n_blocks,stacks_per_block,block_feat_dim,slot_feat_dim,hidden);self.head=nn.Linear(hidden,1)
    def forward(self,obs):
        squeeze=obs.ndim==1
        if squeeze:obs=obs.unsqueeze(0)
        z,_,_=self.encoder(obs);v=self.head(z).squeeze(-1);return v.squeeze(0) if squeeze else v


class SymmetricStorageActor(nn.Module):
    def __init__(self,context_dim,n_blocks,stacks_per_block,block_feat_dim,slot_feat_dim,hidden):
        super().__init__();self.n_blocks=n_blocks;self.stacks_per_block=stacks_per_block;self.n_slots=n_blocks*stacks_per_block;self.encoder=SymmetricYardEncoder(context_dim,n_blocks,stacks_per_block,block_feat_dim,slot_feat_dim,hidden);self.scorer=nn.Sequential(nn.Linear(3*hidden,hidden),nn.Tanh(),nn.Linear(hidden,1))
    def forward(self,obs):
        squeeze=obs.ndim==1
        if squeeze:obs=obs.unsqueeze(0)
        g,b,s=self.encoder(obs);bf=b.unsqueeze(2).expand(-1,-1,self.stacks_per_block,-1);gf=g[:,None,None,:].expand(-1,self.n_blocks,self.stacks_per_block,-1);logits=self.scorer(torch.cat([gf,bf,s],dim=-1)).squeeze(-1).reshape(-1,self.n_slots);return logits.squeeze(0) if squeeze else logits
