from __future__ import annotations

"""PPO trainer for the final V4 Storage + shared-policy YC CTDE model."""

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from v4_networks import SymmetricYardEncoder, SymmetricStorageActor, PermutationInvariantCritic, mlp
from yc_marl_env import (
    N_BLOCKS,
    STACKS_PER_BLOCK,
    ResourceMARLYardEnv,
    STORAGE_AGENT,
    YC_AGENT_PREFIX,
    YC_PROACTIVE_BASE,
    YC_TARGET_POSITIONS,
    YC_DEST_STACKS,
    YC_PAIR_COUNT,
    YC_ACTION_DIM,
)


def masked_distribution(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if logits.ndim == 1:
        if not bool(mask.any()):
            raise RuntimeError("Action mask contains no valid action")
    elif not bool(mask.any(dim=-1).all()):
        raise RuntimeError("At least one batch item has no valid action")
    return Categorical(logits=logits.masked_fill(~mask, torch.finfo(logits.dtype).min))


def nested_group_normalized_flat_yc_logits(
    operation_logits: torch.Tensor, target_scores: torch.Tensor, pair_scores: torch.Tensor,
    mask: torch.Tensor | None = None
) -> torch.Tensor:
    """LEGACY / UNUSED in the canonical policy.

    Flat 2502-action logits with Operation→Target→Destination normalization.

    The environment/action interface remains one flat categorical action.  The
    internal probability factorization is:
        P(pair t,d)=P(Proactive) P(t|Proactive) P(d|t,Proactive).
    This removes both the number-of-pairs bias and the number-of-destinations
    bias for each target while still returning a single flat action distribution.
    """
    squeeze = operation_logits.ndim == 1
    if squeeze:
        operation_logits=operation_logits.unsqueeze(0); target_scores=target_scores.unsqueeze(0); pair_scores=pair_scores.unsqueeze(0)
        if mask is not None: mask=mask.unsqueeze(0)
    if operation_logits.shape[-1] != 3: raise ValueError(operation_logits.shape)
    if target_scores.shape[-1] != YC_TARGET_POSITIONS: raise ValueError(target_scores.shape)
    if pair_scores.shape[-1] != YC_PAIR_COUNT: raise ValueError(pair_scores.shape)
    if mask is None:
        pair_mask=torch.ones_like(pair_scores,dtype=torch.bool)
    else:
        if mask.dtype!=torch.bool: mask=mask.bool()
        pair_mask=mask[...,YC_PROACTIVE_BASE:]
    b=pair_scores.shape[0]
    pm=pair_mask.reshape(b,YC_TARGET_POSITIONS,YC_DEST_STACKS)
    ps=pair_scores.reshape(b,YC_TARGET_POSITIONS,YC_DEST_STACKS)
    target_mask=pm.any(dim=-1)
    neg_inf=torch.finfo(pair_scores.dtype).min

    # Target conditional distribution among targets with at least one feasible destination.
    tmasked=target_scores.masked_fill(~target_mask,neg_inf)
    has_target=target_mask.any(dim=-1,keepdim=True)
    tz=torch.logsumexp(tmasked,dim=-1,keepdim=True)
    tz=torch.where(has_target,tz,torch.zeros_like(tz))
    tnorm=target_scores-tz

    # Destination conditional distribution separately within each target.
    dmasked=ps.masked_fill(~pm,neg_inf)
    has_dest=pm.any(dim=-1,keepdim=True)
    dz=torch.logsumexp(dmasked,dim=-1,keepdim=True)
    dz=torch.where(has_dest,dz,torch.zeros_like(dz))
    dnorm=ps-dz

    proactive_group=operation_logits[...,2:3]
    pair_logits=(proactive_group.unsqueeze(-1)+tnorm.unsqueeze(-1)+dnorm).reshape(b,YC_PAIR_COUNT)
    flat=torch.cat([operation_logits[...,:2],pair_logits],dim=-1)
    return flat.squeeze(0) if squeeze else flat


def group_normalized_flat_yc_logits(operation_logits: torch.Tensor, pair_scores: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Backward-compatible two-level helper used by legacy diagnostics only."""
    squeeze=operation_logits.ndim==1
    if squeeze:
        operation_logits=operation_logits.unsqueeze(0);pair_scores=pair_scores.unsqueeze(0)
        if mask is not None: mask=mask.unsqueeze(0)
    if mask is None: pair_mask=torch.ones_like(pair_scores,dtype=torch.bool)
    else: pair_mask=mask[...,YC_PROACTIVE_BASE:].bool()
    neg_inf=torch.finfo(pair_scores.dtype).min; masked=pair_scores.masked_fill(~pair_mask,neg_inf);has=pair_mask.any(dim=-1,keepdim=True)
    z=torch.logsumexp(masked,dim=-1,keepdim=True);z=torch.where(has,z,torch.zeros_like(z))
    flat=torch.cat([operation_logits[...,:2],operation_logits[...,2:3]+pair_scores-z],dim=-1)
    return flat.squeeze(0) if squeeze else flat


def structured_yc_entropy(dist: Categorical, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return operation entropy, normalized conditional pair entropy, and P(proactive).

    Pair entropy is normalized by log(number of feasible pairs), so its scale does
    not grow merely because one state exposes more relocation pairs than another.
    """
    probs = dist.probs
    squeeze = probs.ndim == 1
    if squeeze:
        probs = probs.unsqueeze(0)
        mask = mask.unsqueeze(0)
    if mask.dtype != torch.bool:
        mask = mask.bool()
    eps = torch.finfo(probs.dtype).eps
    p_m = probs[..., 0]
    p_i = probs[..., 1]
    p_p = probs[..., YC_PROACTIVE_BASE:].sum(dim=-1)
    group = torch.stack([p_m, p_i, p_p], dim=-1)
    op_entropy = -(group * torch.log(group.clamp_min(eps))).sum(dim=-1)

    pair_probs = probs[..., YC_PROACTIVE_BASE:]
    pair_mask = mask[..., YC_PROACTIVE_BASE:]
    p_p_safe = p_p.unsqueeze(-1).clamp_min(eps)
    conditional = torch.where(pair_mask, pair_probs / p_p_safe, torch.zeros_like(pair_probs))
    pair_entropy = -(conditional * torch.log(conditional.clamp_min(eps))).sum(dim=-1)
    n_feasible = pair_mask.sum(dim=-1).to(probs.dtype)
    denom = torch.log(n_feasible.clamp_min(2.0))
    pair_entropy_norm = torch.where(n_feasible > 1.0, pair_entropy / denom, torch.zeros_like(pair_entropy))
    if squeeze:
        return op_entropy.squeeze(0), pair_entropy_norm.squeeze(0), p_p.squeeze(0)
    return op_entropy, pair_entropy_norm, p_p


class RelocationPairYCActor(nn.Module):
    """Scores the 2500 flat Target×Destination proactive moves with shared weights.

    The action remains a single flat categorical choice.  Internally, each pair
    is encoded by a small shared network to keep the 2500-action head tractable.
    """

    CONTEXT_DIM = ResourceMARLYardEnv.YC_CONTEXT_DIM
    PAIR_FEAT_DIM = ResourceMARLYardEnv.YC_PAIR_FEAT_DIM
    PAIR_COUNT = YC_PAIR_COUNT

    def __init__(self, obs_dim: int, hidden: int = 128, pair_hidden: int = 16):
        super().__init__()
        expected = self.CONTEXT_DIM + self.PAIR_COUNT * self.PAIR_FEAT_DIM
        if obs_dim != expected:
            raise ValueError(f"Expected YC obs dim {expected}, got {obs_dim}")
        self.context_encoder = mlp(self.CONTEXT_DIM, hidden, hidden)
        self.context_pair_proj = nn.Linear(hidden, pair_hidden)
        self.pair_encoder = nn.Sequential(
            nn.Linear(self.PAIR_FEAT_DIM, pair_hidden), nn.Tanh(),
            nn.Linear(pair_hidden, pair_hidden), nn.Tanh(),
        )
        self.operation_head = mlp(self.CONTEXT_DIM, hidden, 3)
        self.pair_scorer = nn.Linear(pair_hidden, 1)

    def raw_components(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        squeeze = obs.ndim == 1
        if squeeze: obs = obs.unsqueeze(0)
        context = obs[:, : self.CONTEXT_DIM]
        pair_raw = obs[:, self.CONTEXT_DIM :].reshape(-1, self.PAIR_COUNT, self.PAIR_FEAT_DIM)
        ctx = self.context_pair_proj(self.context_encoder(context)).unsqueeze(1)
        pair = self.pair_encoder(pair_raw)
        pair_scores = self.pair_scorer(torch.tanh(pair + ctx)).squeeze(-1)
        operation_logits = self.operation_head(context)
        if squeeze: return operation_logits.squeeze(0), pair_scores.squeeze(0)
        return operation_logits, pair_scores

    def forward(self, obs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        operation_logits, pair_scores = self.raw_components(obs)
        logits = group_normalized_flat_yc_logits(operation_logits, pair_scores, mask)
        if logits.shape[-1] != YC_ACTION_DIM:
            raise RuntimeError(f"YC logits shape mismatch: {logits.shape}")
        return logits


class ActionConditionedYCCritic(nn.Module):
    """LEGACY PROTOTYPE. Disabled in the canonical model (use_action_q_critic=False).

    Centralized Q critic over the same 2502 flat YC actions.

    It uses the full global yard state plus a full-resource YC action-feature
    observation during training.  The pair scorer is shared across all
    Target×Destination actions so sampled returns generalize across similar moves.
    """
    CONTEXT_DIM = ResourceMARLYardEnv.YC_CONTEXT_DIM
    PAIR_FEAT_DIM = ResourceMARLYardEnv.YC_PAIR_FEAT_DIM
    PAIR_COUNT = YC_PAIR_COUNT

    def __init__(self, global_obs_dim: int, yc_obs_dim: int, hidden: int = 128, pair_hidden: int = 16):
        super().__init__()
        expected_global = 8 + N_BLOCKS * 8 + (N_BLOCKS * STACKS_PER_BLOCK) * 4
        expected_yc = self.CONTEXT_DIM + self.PAIR_COUNT * self.PAIR_FEAT_DIM
        if global_obs_dim != expected_global or yc_obs_dim != expected_yc:
            raise ValueError((global_obs_dim, yc_obs_dim, expected_global, expected_yc))
        self.global_encoder = SymmetricYardEncoder(8, N_BLOCKS, STACKS_PER_BLOCK, 8, 4, hidden)
        self.local_context_encoder = mlp(self.CONTEXT_DIM, hidden, hidden)
        self.base_q_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, YC_PROACTIVE_BASE))
        self.global_pair_proj = nn.Linear(hidden, pair_hidden)
        self.local_pair_proj = nn.Linear(hidden, pair_hidden)
        self.pair_encoder = nn.Sequential(
            nn.Linear(self.PAIR_FEAT_DIM, pair_hidden), nn.Tanh(),
            nn.Linear(pair_hidden, pair_hidden), nn.Tanh(),
        )
        self.pair_q = nn.Linear(pair_hidden, 1)

    def forward(self, global_obs: torch.Tensor, yc_obs: torch.Tensor) -> torch.Tensor:
        squeeze = global_obs.ndim == 1
        if squeeze:
            global_obs = global_obs.unsqueeze(0); yc_obs = yc_obs.unsqueeze(0)
        g, _, _ = self.global_encoder(global_obs)
        local_ctx = yc_obs[:, : self.CONTEXT_DIM]
        pairs = yc_obs[:, self.CONTEXT_DIM :].reshape(-1, self.PAIR_COUNT, self.PAIR_FEAT_DIM)
        lc = self.local_context_encoder(local_ctx)
        base_q = self.base_q_head(torch.cat([g, lc], dim=-1))
        ctx = (self.global_pair_proj(g) + self.local_pair_proj(lc)).unsqueeze(1)
        pe = self.pair_encoder(pairs)
        pair_q = self.pair_q(torch.tanh(pe + ctx)).squeeze(-1)
        q = torch.cat([base_q, pair_q], dim=-1)
        if q.shape[-1] != YC_ACTION_DIM:
            raise RuntimeError(q.shape)
        return q.squeeze(0) if squeeze else q


class ResourceCooperativeModel(nn.Module):
    """CTDE model: 100-stack storage actor, shared YC actor, global critic."""

    def __init__(self, global_obs_dim: int, storage_obs_dim: int, yc_obs_dim: int, hidden: int = 128):
        super().__init__()
        expected_global = 8 + N_BLOCKS * 8 + (N_BLOCKS * STACKS_PER_BLOCK) * 4
        expected_storage = 4 + N_BLOCKS * 8 + (N_BLOCKS * STACKS_PER_BLOCK) * 4
        if global_obs_dim != expected_global:
            raise ValueError(f"Expected global obs dim {expected_global}, got {global_obs_dim}")
        if storage_obs_dim != expected_storage:
            raise ValueError(f"Expected storage obs dim {expected_storage}, got {storage_obs_dim}")
        self.storage_actor = SymmetricStorageActor(
            context_dim=4,
            n_blocks=N_BLOCKS,
            stacks_per_block=STACKS_PER_BLOCK,
            block_feat_dim=8,
            slot_feat_dim=4,
            hidden=hidden,
        )
        self.yc_actor = RelocationPairYCActor(yc_obs_dim, hidden=hidden)
        self.yc_q_critic = ActionConditionedYCCritic(global_obs_dim, yc_obs_dim, hidden=hidden)
        self.critic = PermutationInvariantCritic(
            context_dim=8,
            n_blocks=N_BLOCKS,
            stacks_per_block=STACKS_PER_BLOCK,
            block_feat_dim=8,
            slot_feat_dim=4,
            hidden=hidden,
        )

    def storage_logits(self, obs: torch.Tensor) -> torch.Tensor:
        return self.storage_actor(obs)

    def yc_logits(self, obs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.yc_actor(obs, mask)

    def value(self, global_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(global_obs)

    def yc_q_values(self, global_obs: torch.Tensor, full_yc_obs: torch.Tensor) -> torch.Tensor:
        return self.yc_q_critic(global_obs, full_yc_obs)


def initialize_conservative_yc_actor(model: ResourceCooperativeModel, proactive_bias: float = -2.197224577, pair_init_std: float = 0.05) -> None:
    """Scratch YC initialization with ~10% proactive-group probability.

    Unlike the old -7.2 per-pair bias, this bias acts once on the proactive group
    and therefore does not depend on the number of feasible Target×Destination pairs.
    """
    operation_last=model.yc_actor.operation_head[-1];pair_last=model.yc_actor.pair_scorer
    if not all(isinstance(x,nn.Linear) for x in (operation_last,pair_last)): raise TypeError("Expected linear final heads in YC actor")
    with torch.no_grad():
        operation_last.weight.zero_();operation_last.bias.zero_();operation_last.bias[2]=float(proactive_bias)
        nn.init.normal_(pair_last.weight,mean=0.0,std=float(pair_init_std));pair_last.bias.zero_()


@dataclass
class ResourcePPOConfig:
    total_steps: int = 30000
    rollout_steps: int = 512
    min_storage_transitions: int = 64
    min_yc_transitions: int = 128
    max_rollout_multiplier: int = 4
    update_epochs: int = 2
    minibatch_size: int = 256
    gamma: float = 1.0
    gae_lambda: float = 1.0
    clip_coef: float = 0.20
    ent_coef: float = 0.01
    yc_op_ent_coef: float = 0.001
    yc_pair_ent_coef: float = 0.0001
    yc_proactive_init_bias: float = -2.197224577  # logit(0.10) when one base action competes with proactive
    yc_pair_init_std: float = 0.05
    vf_coef: float = 0.5
    learning_rate: float = 3e-4
    q_learning_rate: float = 3e-4
    q_update_epochs: int = 3
    q_adv_blend: float = 0.5
    use_action_q_critic: bool = False
    max_grad_norm: float = 0.5
    hidden: int = 128
    seed: int = 1
    arrival_rate_per_hour: float = 20.0  # calibrated single operating point
    include_resource_state: bool = True
    truck_wait_weight: float = 1.0
    storage_wait_weight: float = 1.0
    extra_move_weight: float = 0.10
    risk_shaping_weight: float = 0.0
    yc_queue_shaping_weight: float = 0.0
    enable_proactive: bool = True
    yc_move_time: float = 2.0
    episode_complete_rollout: bool = False


class ScenarioSampler:
    """Only scenario seeds vary; there are no Low/Medium/High load regimes."""
    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def next(self) -> int:
        return self.rng.randint(10_000, 9_999_999)


def set_global_seeds(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def compute_gae(rewards, values, next_values, dones, gamma, lam):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values[t] * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        advantages[t] = gae
    return advantages, advantages + values


def completed_episode_mc_diagnostics(rewards: np.ndarray, dones: np.ndarray, values: np.ndarray) -> dict:
    """MC diagnostics only for transitions whose terminal is present in this rollout.

    A cutoff rollout can end in the middle of an episode.  Those trailing transitions
    are excluded rather than pretending that the bootstrap value is a Monte-Carlo
    target.  Episode-complete rollouts should therefore report fraction ~= 1.0.
    """
    rewards=np.asarray(rewards,dtype=np.float64); dones=np.asarray(dones,dtype=np.float64); values=np.asarray(values,dtype=np.float64)
    mc=np.full(len(rewards),np.nan,dtype=np.float64); running=0.0; have_terminal=False
    for t in range(len(rewards)-1,-1,-1):
        if dones[t] > 0.5:
            running=float(rewards[t]); have_terminal=True; mc[t]=running
        elif have_terminal:
            running=float(rewards[t])+running; mc[t]=running
    valid=np.isfinite(mc)
    if not valid.any():
        return {'mc_completed_fraction':0.0,'mc_value_ev':float('nan'),'mc_value_rmse':float('nan'),'mc_n':0}
    y=mc[valid]; pred=values[valid]; var=float(np.var(y))
    ev=float('nan') if var < 1e-12 else float(1.0-np.var(y-pred)/var)
    rmse=float(np.sqrt(np.mean((y-pred)**2)))
    return {'mc_completed_fraction':float(valid.mean()),'mc_value_ev':ev,'mc_value_rmse':rmse,'mc_n':int(valid.sum())}


def load_balancing_weights(items: List[str] | None = None) -> np.ndarray:
    """Compatibility helper: single operating condition => uniform weights."""
    n = len(items) if items is not None else 0
    return np.ones(n, dtype=np.float32)


def rollout_ready(n: int, storage_n: int, yc_n: int, base: int, max_n: int, config) -> bool:
    if n >= max_n:
        return True
    return n >= base and storage_n >= config.min_storage_transitions and yc_n >= config.min_yc_transitions


def load_compatible_state_dict(model: nn.Module, state_dict: dict) -> tuple[list[str], list[str]]:
    """Load only keys whose names and shapes match the current architecture.

    This keeps Storage-BC checkpoints reusable when the YC policy head changes.
    """
    current=model.state_dict();compatible={};skipped=[]
    for k,v in state_dict.items():
        if k in current and current[k].shape==v.shape: compatible[k]=v
        else: skipped.append(k)
    result=model.load_state_dict(compatible,strict=False)
    return list(result.missing_keys),skipped


def _weighted_mean(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (x * w).sum() / w.sum().clamp_min(1e-8)


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def train_resource_marl(config: ResourcePPOConfig, out_dir: Path, init_checkpoint: Optional[Path] = None) -> Tuple[ResourceCooperativeModel, List[dict]]:
    set_global_seeds(config.seed)
    device = torch.device("cpu")
    sampler = ScenarioSampler(config.seed + 1000)
    env = ResourceMARLYardEnv(
        seed=sampler.next(), arrival_rate_per_hour=config.arrival_rate_per_hour,
        include_resource_state=config.include_resource_state,
        truck_wait_weight=config.truck_wait_weight, storage_wait_weight=config.storage_wait_weight,
        extra_move_weight=config.extra_move_weight, risk_shaping_weight=config.risk_shaping_weight,
        yc_queue_shaping_weight=config.yc_queue_shaping_weight,
        enable_proactive=config.enable_proactive, yc_move_time=config.yc_move_time,
    )
    env.reset()
    model = ResourceCooperativeModel(env.global_obs_dim, env.storage_obs_dim, env.yc_obs_dim, hidden=config.hidden).to(device)
    if init_checkpoint is not None:
        checkpoint = torch.load(Path(init_checkpoint), map_location=device, weights_only=False)
        state=checkpoint.get("state_dict",checkpoint)
        if bool(checkpoint.get("storage_only",False)):
            state={k:v for k,v in state.items() if not k.startswith("yc_actor.") and not k.startswith("yc_q_critic.")}
        load_compatible_state_dict(model,state)
        if bool(checkpoint.get("storage_only", False)):
            initialize_conservative_yc_actor(model, config.yc_proactive_init_bias, config.yc_pair_init_std)
    else:
        initialize_conservative_yc_actor(model, config.yc_proactive_init_bias, config.yc_pair_init_std)
    main_params = [p for n, p in model.named_parameters() if not n.startswith("yc_q_critic.")]
    optimizer = torch.optim.Adam(main_params, lr=config.learning_rate)
    q_optimizer = torch.optim.Adam(model.yc_q_critic.parameters(), lr=config.q_learning_rate)

    global_step = update_idx = 0
    episode_records: List[dict] = []
    update_records: List[dict] = []

    while global_step < config.total_steps:
        global_obs_buf=[]; actor_obs_buf=[]; masks_buf=[]; roles_buf=[]; yc_critic_obs_buf=[]
        actions_buf=[]; old_logp_buf=[]; rewards_buf=[]; team_buf=[]
        storage_local_buf=[]; yc_local_buf=[]; dones_buf=[]; values_buf=[]; next_values_buf=[]
        yc_op_entropy_buf=[]; yc_pair_entropy_buf=[]; yc_proactive_prob_buf=[]; yc_feasible_pairs_buf=[]; yc_selected_proactive_buf=[]
        remaining=max(config.total_steps-global_step,1); base=min(config.rollout_steps,remaining)
        max_n=max(base,config.rollout_steps*config.max_rollout_multiplier); storage_n=yc_n=0

        rollout_target_reached=False
        while True:
            if (not config.episode_complete_rollout) and rollout_ready(len(actions_buf),storage_n,yc_n,base,max_n,config):
                break
            active=env.active_agent(); role=0 if active==STORAGE_AGENT else 1
            if role==1 and not active.startswith(YC_AGENT_PREFIX):
                raise RuntimeError(f"Invalid active agent {active}")
            aobs=env.actor_observation(); gobs=env.critic_observation(); mask=env.action_mask()
            yc_cobs = env.yc_observation(env.active_block(), resource_state=True).copy() if role == 1 else None
            aobs_t=torch.as_tensor(aobs,dtype=torch.float32,device=device)
            gobs_t=torch.as_tensor(gobs,dtype=torch.float32,device=device)
            mask_t=torch.as_tensor(mask,dtype=torch.bool,device=device)
            with torch.no_grad():
                logits=model.storage_logits(aobs_t) if role==0 else model.yc_logits(aobs_t, mask_t)
                dist=masked_distribution(logits,mask_t); action_t=dist.sample(); logp_t=dist.log_prob(action_t)
                value_t=model.value(gobs_t)
                if role == 1:
                    op_h, pair_h, p_pro = structured_yc_entropy(dist, mask_t)
                    yc_op_entropy_buf.append(float(op_h.item()))
                    yc_pair_entropy_buf.append(float(pair_h.item()))
                    yc_proactive_prob_buf.append(float(p_pro.item()))
                    yc_feasible_pairs_buf.append(int(mask_t[YC_PROACTIVE_BASE:].sum().item()))
                    yc_selected_proactive_buf.append(float(int(action_t.item()) >= YC_PROACTIVE_BASE))
            _,reward,done,_,info=env.step(int(action_t.item()))
            with torch.no_grad():
                next_value=0.0 if done else float(model.value(torch.as_tensor(env.critic_observation(),dtype=torch.float32,device=device)).item())
            global_obs_buf.append(gobs.copy()); actor_obs_buf.append(aobs.copy()); masks_buf.append(mask.copy()); roles_buf.append(role); yc_critic_obs_buf.append(yc_cobs)
            actions_buf.append(int(action_t.item())); old_logp_buf.append(float(logp_t.item())); rewards_buf.append(float(reward))
            comp=info.get("reward_components",{}); team_buf.append(float(comp.get("team",reward)))
            storage_local_buf.append(float(comp.get("storage_local",0.0))); yc_local_buf.append(float(comp.get("yc_local",0.0)))
            dones_buf.append(float(done)); values_buf.append(float(value_t.item())); next_values_buf.append(next_value)
            storage_n += int(role==0); yc_n += int(role==1); global_step += 1
            if config.episode_complete_rollout and rollout_ready(len(actions_buf),storage_n,yc_n,base,max_n,config):
                rollout_target_reached=True
            if done:
                episode_records.append({"global_step":global_step,"seed":env.seed,"arrival_rate_per_hour":env.arrival_rate_per_hour,
                    "episode_return":info["episode_return"],**info["kpis"]})
                env.reset(seed=sampler.next())
                if config.episode_complete_rollout and rollout_target_reached:
                    break

        rewards=np.asarray(rewards_buf,np.float32); values=np.asarray(values_buf,np.float32); next_values=np.asarray(next_values_buf,np.float32); dones=np.asarray(dones_buf,np.float32)
        mc_diag=completed_episode_mc_diagnostics(rewards,dones,values)
        adv,ret=compute_gae(rewards,values,next_values,dones,config.gamma,config.gae_lambda); adv=(adv-adv.mean())/(adv.std()+1e-8)
        roles_t=torch.as_tensor(roles_buf,dtype=torch.long,device=device); actions_t=torch.as_tensor(actions_buf,dtype=torch.long,device=device)
        old_logp_t=torch.as_tensor(old_logp_buf,dtype=torch.float32,device=device); adv_t=torch.as_tensor(adv,dtype=torch.float32,device=device)
        ret_t=torch.as_tensor(ret,dtype=torch.float32,device=device); global_obs_t=torch.as_tensor(np.asarray(global_obs_buf),dtype=torch.float32,device=device)
        policy_adv_t = adv_t.clone()
        q_loss_value = 0.0; q_adv_std = 0.0
        yc_all = [i for i, role in enumerate(roles_buf) if role == 1]
        if config.use_action_q_critic and yc_all:
            yc_obs_all = torch.as_tensor(np.stack([yc_critic_obs_buf[i] for i in yc_all]), dtype=torch.float32, device=device)
            yc_idx_all = torch.as_tensor(yc_all, dtype=torch.long, device=device)
            # Fit Q(s,a) to Monte-Carlo/GAE return targets before the actor update.
            q_losses=[]
            for _ in range(config.q_update_epochs):
                order_q=np.random.permutation(len(yc_all))
                for stq in range(0,len(yc_all),config.minibatch_size):
                    loc=order_q[stq:stq+config.minibatch_size]
                    if len(loc)==0: continue
                    loc_t=torch.as_tensor(loc,dtype=torch.long,device=device)
                    rid=yc_idx_all[loc_t]
                    q_all=model.yc_q_values(global_obs_t[rid],yc_obs_all[loc_t])
                    q_sel=q_all.gather(1,actions_t[rid].unsqueeze(1)).squeeze(1)
                    q_loss=0.5*(q_sel-ret_t[rid]).pow(2).mean()
                    q_optimizer.zero_grad();q_loss.backward();nn.utils.clip_grad_norm_(model.yc_q_critic.parameters(),config.max_grad_norm);q_optimizer.step()
                    q_losses.append(float(q_loss.item()))
            with torch.no_grad():
                q_all=model.yc_q_values(global_obs_t[yc_idx_all],yc_obs_all)
                q_sel=q_all.gather(1,actions_t[yc_idx_all].unsqueeze(1)).squeeze(1)
                v_sel=model.value(global_obs_t[yc_idx_all])
                q_adv=q_sel-v_sel
                q_adv=(q_adv-q_adv.mean())/(q_adv.std()+1e-8)
                q_adv_std=float(q_adv.std().item())
                policy_adv_t[yc_idx_all]=(1.0-config.q_adv_blend)*adv_t[yc_idx_all]+config.q_adv_blend*q_adv
            q_loss_value=float(np.mean(q_losses)) if q_losses else 0.0
        n=len(actions_buf)
        for _ in range(config.update_epochs):
            order=np.random.permutation(n)
            for start in range(0,n,config.minibatch_size):
                ids=order[start:start+config.minibatch_size]; mb=torch.as_tensor(ids,dtype=torch.long,device=device)
                pg_terms=[]; ent_terms=[]
                for role in (0,1):
                    rid_np=[i for i in ids if roles_buf[i]==role]
                    if not rid_np: continue
                    rid=torch.as_tensor(rid_np,dtype=torch.long,device=device)
                    obs=torch.as_tensor(np.stack([actor_obs_buf[i] for i in rid_np]),dtype=torch.float32,device=device)
                    masks=torch.as_tensor(np.stack([masks_buf[i] for i in rid_np]),dtype=torch.bool,device=device)
                    logits=model.storage_logits(obs) if role==0 else model.yc_logits(obs, masks)
                    dist=masked_distribution(logits,masks)
                    new_logp=dist.log_prob(actions_t[rid]); ratio=(new_logp-old_logp_t[rid]).exp()
                    pg1=-policy_adv_t[rid]*ratio; pg2=-policy_adv_t[rid]*torch.clamp(ratio,1-config.clip_coef,1+config.clip_coef)
                    pg_terms.append(torch.maximum(pg1,pg2).mean())
                    if role == 0:
                        ent_terms.append(config.ent_coef*dist.entropy().mean())
                    else:
                        op_h, pair_h, _ = structured_yc_entropy(dist, masks)
                        ent_terms.append(config.yc_op_ent_coef*op_h.mean() + config.yc_pair_ent_coef*pair_h.mean())
                pg_loss=torch.stack(pg_terms).mean(); entropy_bonus=torch.stack(ent_terms).mean()
                value=model.value(global_obs_t[mb]); v_loss=0.5*(value-ret_t[mb]).pow(2).mean()
                loss=pg_loss+config.vf_coef*v_loss-entropy_bonus
                optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),config.max_grad_norm); optimizer.step()
        update_idx+=1
        update_records.append({"update":update_idx,"global_step":global_step,"storage_decisions":storage_n,"yc_decisions":yc_n,
            "mean_reward":float(np.mean(rewards)),"mean_team_reward":float(np.mean(team_buf)),
            "mean_storage_local":float(np.mean(storage_local_buf)),"mean_yc_local":float(np.mean(yc_local_buf)),
            "yc_operation_entropy":float(np.mean(yc_op_entropy_buf)) if yc_op_entropy_buf else 0.0,
            "yc_pair_entropy_norm":float(np.mean(yc_pair_entropy_buf)) if yc_pair_entropy_buf else 0.0,
            "yc_proactive_probability":float(np.mean(yc_proactive_prob_buf)) if yc_proactive_prob_buf else 0.0,
            "yc_feasible_pairs":float(np.mean(yc_feasible_pairs_buf)) if yc_feasible_pairs_buf else 0.0,
            "yc_sampled_proactive_rate":float(np.mean(yc_selected_proactive_buf)) if yc_selected_proactive_buf else 0.0,
            "rollout_mode":"episode_complete" if config.episode_complete_rollout else "cutoff",
            "rollout_decisions":int(len(actions_buf)),"rollout_ended_at_terminal":bool(dones_buf and dones_buf[-1]>0.5),
            **mc_diag,
            "yc_q_loss":q_loss_value,"yc_q_adv_std":q_adv_std})

    out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    checkpoint_payload={"state_dict":model.state_dict(),"config":asdict(config),"global_obs_dim":env.global_obs_dim,
        "storage_obs_dim":env.storage_obs_dim,"yc_obs_dim":env.yc_obs_dim,
        "init_checkpoint":str(init_checkpoint) if init_checkpoint is not None else None,
        "continuation_mode":"weights_only_warm_start" if init_checkpoint is not None else "scratch",
        "run_global_step":int(global_step),"update_count":int(update_idx),"current_env_seed":int(env.seed),
        "optimizer_state_dict":optimizer.state_dict(),"python_random_state":random.getstate(),
        "numpy_random_state":np.random.get_state(),"torch_rng_state":torch.get_rng_state(),
        "scenario_sampler_state":sampler.rng.getstate()}
    if config.use_action_q_critic:
        checkpoint_payload["q_optimizer_state_dict"]=q_optimizer.state_dict()
    torch.save(checkpoint_payload,out_dir/"resource_marl_final.pt")
    _write_csv(out_dir/"resource_marl_train_episodes.csv",episode_records); _write_csv(out_dir/"resource_marl_updates.csv",update_records)
    (out_dir/"resource_marl_config.json").write_text(json.dumps(asdict(config),indent=2),encoding="utf-8")
    return model,update_records


if __name__ == "__main__":
    train_resource_marl(ResourcePPOConfig(), Path("resource_marl_run"))
