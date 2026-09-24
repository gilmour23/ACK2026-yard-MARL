from __future__ import annotations

"""Factorized V5 YC actor: Operation -> Target -> Destination.

The simulator keeps its existing flat YC action encoding.  This module changes
only the actor representation and probability factorization.

V5 YC input:
- 18-d block context;
- 100 target-position feature rows (8 features each);
- 25 destination-stack feature rows (10 features each);
- a physical 100x25 feasibility mask supplied separately.

The dense legacy 2,500x9 pair-feature observation is not used by this actor.
"""

from dataclasses import dataclass
import math
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from v4_networks import SymmetricStorageActor, PermutationInvariantCritic, mlp
from yc_marl_env import (
    N_BLOCKS,
    STACKS_PER_BLOCK,
    MAX_TIER,
    PROACTIVE_HORIZON_MIN,
    ResourceMARLYardEnv,
    YC_AGENT_PREFIX,
    YC_MANDATORY,
    YC_IDLE,
    YC_PROACTIVE_BASE,
    YC_TARGET_POSITIONS,
    YC_DEST_STACKS,
    encode_proactive_action,
)


CONTEXT_DIM = ResourceMARLYardEnv.YC_CONTEXT_DIM
TARGET_COUNT = YC_TARGET_POSITIONS
DEST_COUNT = YC_DEST_STACKS
TARGET_FEAT_DIM = 8
DEST_FEAT_DIM = 10
HIER_YC_OBS_DIM = CONTEXT_DIM + TARGET_COUNT * TARGET_FEAT_DIM + DEST_COUNT * DEST_FEAT_DIM


def _clip(x: float, lo: float, hi: float) -> float:
    return float(np.clip(float(x), lo, hi))


def _tier_eta_features(env: ResourceMARLYardEnv, block: int, stack_idx: int) -> tuple[list[float], list[float]]:
    sim=env.sim
    assert sim is not None
    stack=sim.yard.stacks[block][stack_idx]
    eta=[]
    occ=[]
    for tier in range(sim.yard.max_tier):
        if tier < len(stack):
            cid=stack[tier]
            lead=(sim.containers[cid].eta-sim.now)/1440.0
            eta.append(_clip(lead,-1.0,2.0))
            occ.append(1.0)
        else:
            eta.append(0.0)
            occ.append(0.0)
    return eta,occ


def build_hierarchical_yc_input(
    env: ResourceMARLYardEnv,
    block: int,
    resource_state: bool | None = None,
) -> tuple[np.ndarray,np.ndarray]:
    """Build compact fixed V5 observation + exact physical pair mask.

    Must be called at the currently active YC decision.  The returned pair mask
    is checked against the simulator's existing flat proactive-action support.
    """
    if env.sim is None or env.active_agent() == "done":
        raise RuntimeError("Environment is not at an active decision")
    if not env.active_agent().startswith(YC_AGENT_PREFIX) or env.active_block()!=block:
        raise RuntimeError(("V5 YC observation requested for inactive block",block,env.active_agent()))

    sim=env.sim
    resource=env.include_resource_state if resource_state is None else bool(resource_state)
    context=env._yc_context_observation(block,resource_state=resource).astype(np.float32,copy=False)

    flat_mask=np.asarray(env.action_mask(),dtype=np.bool_)
    pair_mask=flat_mask[YC_PROACTIVE_BASE:].reshape(TARGET_COUNT,DEST_COUNT).copy()
    target_mask=pair_mask.any(axis=1)

    target_feats=np.zeros((TARGET_COUNT,TARGET_FEAT_DIM),dtype=np.float32)
    for target_pos in np.flatnonzero(target_mask):
        cid=sim._cid_at_target_position(block,int(target_pos))
        if cid is None or not sim._proactive_eligible(cid):
            raise RuntimeError(("pair mask references invalid proactive target",block,int(target_pos),cid))
        c=sim.containers[cid]
        if c.stack is None:
            raise RuntimeError(("target without source stack",cid))
        source=int(c.stack)
        moving_id=sim.yard.top(block,source)
        if moving_id is None or moving_id==cid:
            raise RuntimeError(("target has no movable blocker",cid))
        moving=sim.containers[moving_id]
        blockers=sim.yard.blockers_above(cid,sim.containers)
        lead=max(0.0,c.eta-sim.now)
        advance=max(0.0,-c.eta_delta)
        slack=sim.target_capacity_slack(cid) if resource else 0.0
        moving_lead=moving.eta-sim.now
        tier=int(target_pos)%sim.yard.max_tier
        feasible_n=int(pair_mask[int(target_pos)].sum())
        target_feats[int(target_pos)]=np.asarray([
            _clip(lead/PROACTIVE_HORIZON_MIN,0.0,2.0),
            _clip(advance/60.0,0.0,2.0),
            _clip(blockers/sim.yard.max_tier,0.0,1.0),
            _clip(slack/PROACTIVE_HORIZON_MIN,-2.0,2.0) if resource else 0.0,
            _clip(moving_lead/1440.0,-1.0,2.0),
            _clip(len(sim.yard.stacks[block][source])/sim.yard.max_tier,0.0,1.0),
            _clip(tier/max(sim.yard.max_tier-1,1),0.0,1.0),
            _clip(feasible_n/max(sim.yard.stacks_per_block-1,1),0.0,1.0),
        ],dtype=np.float32)

    dest_feats=np.zeros((DEST_COUNT,DEST_FEAT_DIM),dtype=np.float32)
    for dest in range(DEST_COUNT):
        stack=sim.yard.stacks[block][dest]
        reserved=sim.yard.reserved[block][dest]
        eta,occ=_tier_eta_features(env,block,dest)
        dest_feats[dest]=np.asarray([
            _clip((len(stack)+reserved)/sim.yard.max_tier,0.0,1.0),
            _clip(reserved/sim.yard.max_tier,0.0,1.0) if resource else 0.0,
            *eta,
            *occ,
        ],dtype=np.float32)

    obs=np.concatenate([context,target_feats.reshape(-1),dest_feats.reshape(-1)]).astype(np.float32,copy=False)
    if obs.shape!=(HIER_YC_OBS_DIM,):
        raise RuntimeError(("V5 YC observation shape",obs.shape,HIER_YC_OBS_DIM))

    # Guard against accidental divergence between the hierarchical candidate set
    # and the simulator's accepted proactive flat actions.
    rebuilt=np.zeros_like(pair_mask)
    for target_pos in np.flatnonzero(target_mask):
        cid=sim._cid_at_target_position(block,int(target_pos))
        assert cid is not None
        c=sim.containers[cid]
        assert c.stack is not None
        for dest in sim.yard.feasible_relocation_stacks(block,int(c.stack)):
            rebuilt[int(target_pos),int(dest)]=True
    if not np.array_equal(rebuilt,pair_mask):
        raise RuntimeError("V5 pair mask does not match simulator physical support")
    return obs,pair_mask


def split_hierarchical_obs(obs: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    squeeze=obs.ndim==1
    if squeeze:
        obs=obs.unsqueeze(0)
    if obs.shape[-1]!=HIER_YC_OBS_DIM:
        raise ValueError(("unexpected V5 YC obs dim",obs.shape[-1],HIER_YC_OBS_DIM))
    p=0
    context=obs[:,p:p+CONTEXT_DIM]; p+=CONTEXT_DIM
    targets=obs[:,p:p+TARGET_COUNT*TARGET_FEAT_DIM].reshape(-1,TARGET_COUNT,TARGET_FEAT_DIM)
    p+=TARGET_COUNT*TARGET_FEAT_DIM
    destinations=obs[:,p:].reshape(-1,DEST_COUNT,DEST_FEAT_DIM)
    if squeeze:
        return context.squeeze(0),targets.squeeze(0),destinations.squeeze(0)
    return context,targets,destinations


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    w=mask.to(x.dtype).unsqueeze(-1)
    return (x*w).sum(dim=dim)/w.sum(dim=dim).clamp_min(1.0)


def _masked_max(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    neg=torch.finfo(x.dtype).min
    y=x.masked_fill(~mask.unsqueeze(-1),neg)
    out=y.max(dim=dim).values
    any_mask=mask.any(dim=dim)
    return torch.where(any_mask.unsqueeze(-1),out,torch.zeros_like(out))


def _masked_scalar_stats(x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
    w=mask.to(x.dtype)
    mean=(x*w).sum(dim=(-2,-1))/w.sum(dim=(-2,-1)).clamp_min(1.0)
    neg=torch.finfo(x.dtype).min
    mx=x.masked_fill(~mask,neg).amax(dim=(-2,-1))
    any_mask=mask.any(dim=(-2,-1))
    mx=torch.where(any_mask,mx,torch.zeros_like(mx))
    return mean,mx


def _safe_mask(mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Ensure every categorical row has one valid dummy entry.

    Dummy rows are later multiplied by zero probability/validity and therefore
    never contribute to a sampled valid proactive branch.
    """
    out=mask.clone()
    flat=out.reshape(-1,out.shape[dim])
    empty=~flat.any(dim=-1)
    if bool(empty.any()):
        flat[empty,0]=True
    return flat.reshape_as(out)


def _normalized_entropy(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    safe=_safe_mask(mask)
    dist=Categorical(logits=logits.masked_fill(~safe,torch.finfo(logits.dtype).min))
    h=dist.entropy()
    n=mask.sum(dim=-1).to(logits.dtype)
    denom=torch.log(n.clamp_min(2.0))
    return torch.where(n>1.0,h/denom,torch.zeros_like(h))


@dataclass
class HierarchicalSample:
    flat_action:int
    operation:int
    target_position:int
    destination:int
    log_prob:float
    p_proactive:float
    operation_entropy:float
    target_entropy_norm:float
    destination_entropy_norm:float


class HierarchicalYCActor(nn.Module):
    """Shared YC actor with Operation -> Target -> Destination factorization."""

    def __init__(self, hidden:int=128, candidate_hidden:int=32, proactive_bias:float=-2.197224577):
        super().__init__()
        self.hidden=hidden
        self.candidate_hidden=candidate_hidden
        self.context_encoder=mlp(CONTEXT_DIM,hidden,hidden)
        self.target_encoder=nn.Sequential(
            nn.Linear(TARGET_FEAT_DIM,candidate_hidden),nn.Tanh(),
            nn.Linear(candidate_hidden,candidate_hidden),nn.Tanh(),
        )
        self.destination_encoder=nn.Sequential(
            nn.Linear(DEST_FEAT_DIM,candidate_hidden),nn.Tanh(),
            nn.Linear(candidate_hidden,candidate_hidden),nn.Tanh(),
        )
        self.target_context=nn.Linear(hidden,candidate_hidden)
        self.target_scorer=nn.Linear(candidate_hidden,1)

        self.target_query=nn.Linear(candidate_hidden,candidate_hidden,bias=False)
        self.destination_key=nn.Linear(candidate_hidden,candidate_hidden,bias=False)
        self.pair_context=nn.Linear(hidden,candidate_hidden,bias=False)
        self.destination_bias=nn.Linear(candidate_hidden,1,bias=False)
        # Zero scale => exactly uniform conditional Destination policy at init,
        # while the scale itself receives a gradient immediately.
        self.pair_scale=nn.Parameter(torch.tensor(0.0,dtype=torch.float32))

        op_in=hidden+4*candidate_hidden+2
        self.operation_head=nn.Sequential(
            nn.Linear(op_in,hidden),nn.Tanh(),nn.Linear(hidden,2)
        )
        self.initialize_policy_heads(proactive_bias)

    def initialize_policy_heads(self, proactive_bias:float=-2.197224577) -> None:
        with torch.no_grad():
            nn.init.zeros_(self.target_scorer.weight)
            nn.init.zeros_(self.target_scorer.bias)
            nn.init.zeros_(self.destination_bias.weight)
            self.pair_scale.zero_()
            op_last=self.operation_head[-1]
            if not isinstance(op_last,nn.Linear):
                raise TypeError("operation head final layer must be Linear")
            nn.init.zeros_(op_last.weight)
            nn.init.zeros_(op_last.bias)
            op_last.bias[1]=float(proactive_bias)

    def components(
        self,
        obs: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        squeeze=obs.ndim==1
        if squeeze:
            obs=obs.unsqueeze(0)
            pair_mask=pair_mask.unsqueeze(0)
        pair_mask=pair_mask.bool()
        if pair_mask.shape[-2:]!=(TARGET_COUNT,DEST_COUNT):
            raise ValueError(("pair mask shape",pair_mask.shape))
        context_raw,target_raw,dest_raw=split_hierarchical_obs(obs)
        ctx=self.context_encoder(context_raw)
        te=self.target_encoder(target_raw)
        de=self.destination_encoder(dest_raw)
        target_mask=pair_mask.any(dim=-1)
        dest_mask=pair_mask.any(dim=-2)

        target_scores=self.target_scorer(
            torch.tanh(te+self.target_context(ctx).unsqueeze(1))
        ).squeeze(-1)

        q=self.target_query(te)+self.pair_context(ctx).unsqueeze(1)
        k=self.destination_key(de)
        dot=torch.einsum("bth,bdh->btd",q,k)/math.sqrt(float(self.candidate_hidden))
        db=self.destination_bias(de).squeeze(-1).unsqueeze(1)
        pair_scores=self.pair_scale*dot+db

        tmean=_masked_mean(te,target_mask,1)
        tmax=_masked_max(te,target_mask,1)
        dmean=_masked_mean(de,dest_mask,1)
        dmax=_masked_max(de,dest_mask,1)
        pmean,pmax=_masked_scalar_stats(pair_scores,pair_mask)
        op_features=torch.cat([ctx,tmean,tmax,dmean,dmax,pmean.unsqueeze(-1),pmax.unsqueeze(-1)],dim=-1)
        operation_logits=self.operation_head(op_features)

        if squeeze:
            return (
                operation_logits.squeeze(0),
                target_scores.squeeze(0),
                pair_scores.squeeze(0),
                target_mask.squeeze(0),
            )
        return operation_logits,target_scores,pair_scores,target_mask

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        pair_mask: torch.Tensor,
        operation: torch.Tensor,
        target_position: torch.Tensor,
        destination: torch.Tensor,
    ) -> Dict[str,torch.Tensor]:
        squeeze=obs.ndim==1
        if squeeze:
            obs=obs.unsqueeze(0);pair_mask=pair_mask.unsqueeze(0)
            operation=operation.reshape(1);target_position=target_position.reshape(1);destination=destination.reshape(1)
        pair_mask=pair_mask.bool()
        op_logits,target_logits,dest_logits,target_mask=self.components(obs,pair_mask)
        if op_logits.ndim==1:
            op_logits=op_logits.unsqueeze(0);target_logits=target_logits.unsqueeze(0)
            dest_logits=dest_logits.unsqueeze(0);target_mask=target_mask.unsqueeze(0)
        has_target=target_mask.any(dim=-1)
        op_mask=torch.stack([torch.ones_like(has_target),has_target],dim=-1)
        op_dist=Categorical(logits=op_logits.masked_fill(~op_mask,torch.finfo(op_logits.dtype).min))
        op_logp=op_dist.log_prob(operation.long())

        safe_t=_safe_mask(target_mask)
        target_dist=Categorical(logits=target_logits.masked_fill(~safe_t,torch.finfo(target_logits.dtype).min))
        t_idx=target_position.long().clamp(min=0,max=TARGET_COUNT-1)
        d_idx=destination.long().clamp(min=0,max=DEST_COUNT-1)
        target_logp=target_dist.log_prob(t_idx)

        safe_pair=_safe_mask(pair_mask)
        dest_log_probs=torch.log_softmax(
            dest_logits.masked_fill(~safe_pair,torch.finfo(dest_logits.dtype).min),dim=-1
        )
        batch=torch.arange(obs.shape[0],device=obs.device)
        dest_logp=dest_log_probs[batch,t_idx,d_idx]
        proactive=(operation.long()==1)
        joint_logp=op_logp+proactive.to(op_logp.dtype)*(target_logp+dest_logp)

        op_entropy=op_dist.entropy()
        target_entropy_norm=_normalized_entropy(target_logits,target_mask)
        dest_entropy_per_target=_normalized_entropy(dest_logits,pair_mask)
        target_probs=target_dist.probs*target_mask.to(target_logits.dtype)
        dest_entropy_norm=(target_probs*dest_entropy_per_target).sum(dim=-1)
        p_proactive=op_dist.probs[:,1]

        out={
            "log_prob":joint_logp,
            "operation_entropy":op_entropy,
            "target_entropy_norm":target_entropy_norm,
            "destination_entropy_norm":dest_entropy_norm,
            "p_proactive":p_proactive,
        }
        if squeeze:
            return {k:v.squeeze(0) for k,v in out.items()}
        return out

    @torch.no_grad()
    def sample(
        self,
        obs: torch.Tensor,
        pair_mask: torch.Tensor,
        flat_action_mask: torch.Tensor,
    ) -> HierarchicalSample:
        if obs.ndim!=1 or pair_mask.ndim!=2:
            raise ValueError("sample expects one YC state")
        op_logits,target_logits,dest_logits,target_mask=self.components(obs,pair_mask)
        has_target=bool(target_mask.any())
        op_mask=torch.tensor([True,has_target],dtype=torch.bool,device=obs.device)
        op_dist=Categorical(logits=op_logits.masked_fill(~op_mask,torch.finfo(op_logits.dtype).min))
        op=int(op_dist.sample().item())

        tpos=0;dest=0
        if op==0:
            if bool(flat_action_mask[YC_MANDATORY]):
                flat=YC_MANDATORY
            elif bool(flat_action_mask[YC_IDLE]):
                flat=YC_IDLE
            else:
                raise RuntimeError("Default branch has no simulator action")
        else:
            tdist=Categorical(logits=target_logits.masked_fill(~target_mask,torch.finfo(target_logits.dtype).min))
            tpos=int(tdist.sample().item())
            dmask=pair_mask[tpos]
            ddist=Categorical(logits=dest_logits[tpos].masked_fill(~dmask,torch.finfo(dest_logits.dtype).min))
            dest=int(ddist.sample().item())
            flat=encode_proactive_action(tpos,dest)
            if not bool(flat_action_mask[flat]):
                raise RuntimeError(("hierarchical actor sampled simulator-invalid action",flat,tpos,dest))

        ev=self.evaluate_actions(
            obs,pair_mask,
            torch.tensor(op,device=obs.device),
            torch.tensor(tpos,device=obs.device),
            torch.tensor(dest,device=obs.device),
        )
        return HierarchicalSample(
            flat_action=int(flat),
            operation=op,
            target_position=tpos if op==1 else -1,
            destination=dest if op==1 else -1,
            log_prob=float(ev["log_prob"].item()),
            p_proactive=float(ev["p_proactive"].item()),
            operation_entropy=float(ev["operation_entropy"].item()),
            target_entropy_norm=float(ev["target_entropy_norm"].item()),
            destination_entropy_norm=float(ev["destination_entropy_norm"].item()),
        )


class HierarchicalResourceCooperativeModel(nn.Module):
    """V5 CTDE model: existing Storage actor/critic + hierarchical shared YC actor."""

    def __init__(
        self,
        global_obs_dim:int,
        storage_obs_dim:int,
        hidden:int=128,
        candidate_hidden:int=32,
        proactive_bias:float=-2.197224577,
    ):
        super().__init__()
        expected_global=8+N_BLOCKS*8+(N_BLOCKS*STACKS_PER_BLOCK)*4
        expected_storage=4+N_BLOCKS*8+(N_BLOCKS*STACKS_PER_BLOCK)*4
        if global_obs_dim!=expected_global:
            raise ValueError(("global obs dim",global_obs_dim,expected_global))
        if storage_obs_dim!=expected_storage:
            raise ValueError(("storage obs dim",storage_obs_dim,expected_storage))
        self.storage_actor=SymmetricStorageActor(
            context_dim=4,n_blocks=N_BLOCKS,stacks_per_block=STACKS_PER_BLOCK,
            block_feat_dim=8,slot_feat_dim=4,hidden=hidden,
        )
        self.yc_actor=HierarchicalYCActor(
            hidden=hidden,candidate_hidden=candidate_hidden,proactive_bias=proactive_bias
        )
        self.critic=PermutationInvariantCritic(
            context_dim=8,n_blocks=N_BLOCKS,stacks_per_block=STACKS_PER_BLOCK,
            block_feat_dim=8,slot_feat_dim=4,hidden=hidden,
        )

    def storage_logits(self,obs:torch.Tensor)->torch.Tensor:
        return self.storage_actor(obs)

    def value(self,global_obs:torch.Tensor)->torch.Tensor:
        return self.critic(global_obs)


def heuristic_ranks_for_selected_pair(
    env: ResourceMARLYardEnv,
    block: int,
    target_position: int,
    destination: int,
) -> tuple[int,int]:
    """Return 1-based Target and Destination ranks under the current heuristics.

    Used only for V5 diagnostics; it does not constrain the learned policy.
    """
    sim=env.sim
    if sim is None:
        raise RuntimeError("environment is not initialized")
    cid=sim._cid_at_target_position(block,int(target_position))
    if cid is None:
        raise RuntimeError(("missing selected target",target_position))
    targets=list(sim.proactive_candidates(block))
    try:
        target_rank=targets.index(cid)+1
    except ValueError as exc:
        raise RuntimeError(("selected target is not proactive-eligible",cid)) from exc

    c=sim.containers[cid]
    if c.stack is None:
        raise RuntimeError(("selected target has no source stack",cid))
    source=int(c.stack)
    moving_id=sim.yard.top(block,source)
    if moving_id is None or moving_id==cid:
        raise RuntimeError(("selected target has no blocker",cid))
    moving=sim.containers[moving_id]
    scored=[]
    for dest in sim.yard.feasible_relocation_stacks(block,source):
        stack=sim.yard.stacks[block][dest]
        inversion=sum(1 for lower_id in stack if sim.containers[lower_id].eta < moving.eta)
        height=len(stack)/sim.yard.max_tier
        scored.append((float(inversion+0.35*height),int(dest)))
    scored.sort(key=lambda x:(x[0],x[1]))
    order=[d for _,d in scored]
    try:
        dest_rank=order.index(int(destination))+1
    except ValueError as exc:
        raise RuntimeError(("selected destination not feasible",destination)) from exc
    return int(target_rank),int(dest_rank)
