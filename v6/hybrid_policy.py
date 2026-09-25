from __future__ import annotations

"""V6 candidate-aware hybrid policies.

V6 keeps the frozen simulator/action interface but removes the dead learned
2,500-pair head from the final hybrid policy.  The shared YC actor learns only
an operation decision:

    Default vs Proactive

where Proactive maps to the single deterministic rule-resolved Target×Destination
move.  Crucially, the actor sees the exact 9-d feature vector of that resolved
move before deciding whether to execute it.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn

from v4_networks import (
    SymmetricStorageActor,
    SymmetricYardEncoder,
    PermutationInvariantCritic,
    mlp,
)
from yc_marl_env import (
    N_BLOCKS,
    STACKS_PER_BLOCK,
    ResourceMARLYardEnv,
    YC_MANDATORY,
    YC_IDLE,
    YC_PROACTIVE_BASE,
)


YC_CONTEXT_DIM = ResourceMARLYardEnv.YC_CONTEXT_DIM
RESOLVED_PAIR_FEAT_DIM = ResourceMARLYardEnv.YC_PAIR_FEAT_DIM
OP_FLAG_DIM = 2
HYBRID_YC_OBS_DIM = YC_CONTEXT_DIM + RESOLVED_PAIR_FEAT_DIM + OP_FLAG_DIM

OP_DEFAULT = 0
OP_PROACTIVE = 1
OP_ACTION_DIM = 2


@dataclass(frozen=True)
class HybridYCDecision:
    """Actor input plus the exact mapping back to the simulator flat action."""

    obs: np.ndarray
    operation_mask: np.ndarray
    default_flat_action: Optional[int]
    proactive_flat_action: Optional[int]

    def flat_action(self, operation: int) -> int:
        operation = int(operation)
        if operation == OP_DEFAULT:
            if self.default_flat_action is None:
                raise RuntimeError("Default operation is masked")
            return int(self.default_flat_action)
        if operation == OP_PROACTIVE:
            if self.proactive_flat_action is None:
                raise RuntimeError("Proactive operation is masked")
            return int(self.proactive_flat_action)
        raise ValueError(operation)


def split_hybrid_yc_obs(
    obs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    squeeze = obs.ndim == 1
    if squeeze:
        obs = obs.unsqueeze(0)
    if obs.shape[-1] != HYBRID_YC_OBS_DIM:
        raise ValueError(
            f"Expected V6 YC obs dim {HYBRID_YC_OBS_DIM}, got {obs.shape[-1]}"
        )
    p = 0
    context = obs[:, p : p + YC_CONTEXT_DIM]
    p += YC_CONTEXT_DIM
    pair = obs[:, p : p + RESOLVED_PAIR_FEAT_DIM]
    p += RESOLVED_PAIR_FEAT_DIM
    flags = obs[:, p : p + OP_FLAG_DIM]
    if squeeze:
        return context.squeeze(0), pair.squeeze(0), flags.squeeze(0)
    return context, pair, flags


def build_hybrid_yc_operation_input(
    env: ResourceMARLYardEnv,
    block: int,
    resource_state: bool | None = None,
) -> HybridYCDecision:
    """Build the 29-d V6 YC operation input at the active YC decision.

    The environment must be in rule-resolved mode.  The exact proactive flat
    action exposed by the simulator is recovered from its mask and its existing
    9-d pair feature row is injected into the learned operation observation.
    """
    if env.sim is None:
        raise RuntimeError("Environment is not initialized")
    if not env.rule_resolve_proactive_pair:
        raise RuntimeError("V6 requires rule_resolve_proactive_pair=True")
    if env.active_block() != int(block) or not env.active_agent().startswith("yc_"):
        raise RuntimeError(
            ("V6 YC input requested for inactive block", block, env.active_agent())
        )

    resource = env.include_resource_state if resource_state is None else bool(resource_state)
    flat_mask = np.asarray(env.action_mask(), dtype=np.bool_)

    default_flat: Optional[int]
    if bool(flat_mask[YC_MANDATORY]):
        default_flat = YC_MANDATORY
    elif bool(flat_mask[YC_IDLE]):
        default_flat = YC_IDLE
    else:
        default_flat = None

    proactive = (
        np.flatnonzero(flat_mask[YC_PROACTIVE_BASE:]) + YC_PROACTIVE_BASE
    ).astype(np.int64)
    if len(proactive) > 1:
        raise RuntimeError(
            f"V6 rule-resolved support exposed {len(proactive)} proactive pairs"
        )
    proactive_flat = int(proactive[0]) if len(proactive) == 1 else None

    # Cross-check the action mask against the simulator resolver itself.
    resolved = env.sim.rule_resolved_proactive_action(int(block))
    if proactive_flat is None:
        if resolved is not None:
            raise RuntimeError(("resolver/mask proactive mismatch", resolved))
    elif int(resolved) != proactive_flat:
        raise RuntimeError(("resolver/mask proactive mismatch", resolved, proactive_flat))

    context = env._yc_context_observation(
        int(block), resource_state=resource
    ).astype(np.float32, copy=False)

    pair = np.zeros(RESOLVED_PAIR_FEAT_DIM, dtype=np.float32)
    if proactive_flat is not None:
        all_pair_features = env._proactive_pair_features(
            int(block), resource_state=resource
        )
        pair_idx = proactive_flat - YC_PROACTIVE_BASE
        pair = all_pair_features[pair_idx].astype(np.float32, copy=True)
        if pair.shape != (RESOLVED_PAIR_FEAT_DIM,):
            raise RuntimeError(pair.shape)
        # First pair feature is physical feasibility in the existing V4 schema.
        if not np.isclose(float(pair[0]), 1.0):
            raise RuntimeError(
                ("resolved proactive pair is not marked feasible", proactive_flat, pair)
            )

    flags = np.asarray(
        [
            1.0 if default_flat == YC_MANDATORY else 0.0,
            1.0 if proactive_flat is not None else 0.0,
        ],
        dtype=np.float32,
    )
    obs = np.concatenate([context, pair, flags]).astype(np.float32, copy=False)
    if obs.shape != (HYBRID_YC_OBS_DIM,):
        raise RuntimeError((obs.shape, HYBRID_YC_OBS_DIM))

    op_mask = np.asarray(
        [default_flat is not None, proactive_flat is not None],
        dtype=np.bool_,
    )
    if not bool(op_mask.any()):
        raise RuntimeError("V6 YC decision has no operation action")

    return HybridYCDecision(
        obs=obs,
        operation_mask=op_mask,
        default_flat_action=default_flat,
        proactive_flat_action=proactive_flat,
    )


class CandidateAwareYCOperationActor(nn.Module):
    """Shared V6 YC actor: Default vs exact rule-resolved Proactive move."""

    def __init__(
        self,
        hidden: int = 128,
        pair_hidden: int = 32,
        proactive_bias: float = -2.197224577,
    ):
        super().__init__()
        self.context_encoder = mlp(YC_CONTEXT_DIM, hidden, hidden)
        self.pair_encoder = nn.Sequential(
            nn.Linear(RESOLVED_PAIR_FEAT_DIM, pair_hidden),
            nn.Tanh(),
            nn.Linear(pair_hidden, pair_hidden),
            nn.Tanh(),
        )
        self.operation_head = nn.Sequential(
            nn.Linear(hidden + pair_hidden + OP_FLAG_DIM, hidden),
            nn.Tanh(),
            nn.Linear(hidden, OP_ACTION_DIM),
        )
        self.initialize_operation_head(proactive_bias)

    def initialize_operation_head(self, proactive_bias: float) -> None:
        # Exact common initialization: when both operations are feasible,
        # P(Proactive)=0.10 independent of resource visibility/candidate values.
        # Candidate sensitivity is learned after the first PPO head update.
        last = self.operation_head[-1]
        if not isinstance(last, nn.Linear):
            raise TypeError("operation head final layer must be Linear")
        with torch.no_grad():
            last.weight.zero_()
            last.bias.zero_()
            last.bias[OP_PROACTIVE] = float(proactive_bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        squeeze = obs.ndim == 1
        if squeeze:
            obs = obs.unsqueeze(0)
        context, pair, flags = split_hybrid_yc_obs(obs)
        if context.ndim == 1:
            context = context.unsqueeze(0)
            pair = pair.unsqueeze(0)
            flags = flags.unsqueeze(0)
        c = self.context_encoder(context)
        p = self.pair_encoder(pair)
        logits = self.operation_head(torch.cat([c, p, flags], dim=-1))
        return logits.squeeze(0) if squeeze else logits


class HybridResourceCooperativeModel(nn.Module):
    """V6 CTDE model: Storage actor + shared candidate-aware YC actor + V(s)."""

    def __init__(
        self,
        global_obs_dim: int,
        storage_obs_dim: int,
        hidden: int = 128,
        pair_hidden: int = 32,
        proactive_bias: float = -2.197224577,
    ):
        super().__init__()
        expected_global = 8 + N_BLOCKS * 8 + (N_BLOCKS * STACKS_PER_BLOCK) * 4
        expected_storage = 4 + N_BLOCKS * 8 + (N_BLOCKS * STACKS_PER_BLOCK) * 4
        if global_obs_dim != expected_global:
            raise ValueError((global_obs_dim, expected_global))
        if storage_obs_dim != expected_storage:
            raise ValueError((storage_obs_dim, expected_storage))

        self.storage_actor = SymmetricStorageActor(
            context_dim=4,
            n_blocks=N_BLOCKS,
            stacks_per_block=STACKS_PER_BLOCK,
            block_feat_dim=8,
            slot_feat_dim=4,
            hidden=hidden,
        )
        self.yc_actor = CandidateAwareYCOperationActor(
            hidden=hidden,
            pair_hidden=pair_hidden,
            proactive_bias=proactive_bias,
        )
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

    def yc_operation_logits(self, obs: torch.Tensor) -> torch.Tensor:
        return self.yc_actor(obs)

    def value(self, global_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(global_obs)


class HybridCentralizedSingleModel(nn.Module):
    """Fair centralized V6 baseline with the same hybrid operation semantics."""

    CONTEXT_DIM = 8
    BLOCK_FEAT_DIM = 8
    SLOT_FEAT_DIM = 4
    N_SLOTS = N_BLOCKS * STACKS_PER_BLOCK

    def __init__(
        self,
        global_obs_dim: int,
        hidden: int = 128,
        pair_hidden: int = 32,
        proactive_bias: float = -2.197224577,
    ):
        super().__init__()
        expected = (
            self.CONTEXT_DIM
            + N_BLOCKS * self.BLOCK_FEAT_DIM
            + self.N_SLOTS * self.SLOT_FEAT_DIM
        )
        if global_obs_dim != expected:
            raise ValueError((global_obs_dim, expected))
        self.global_obs_dim = global_obs_dim

        self.actor_encoder = SymmetricYardEncoder(
            self.CONTEXT_DIM,
            N_BLOCKS,
            STACKS_PER_BLOCK,
            self.BLOCK_FEAT_DIM,
            self.SLOT_FEAT_DIM,
            hidden,
        )
        self.storage_scorer = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.local_context_encoder = mlp(YC_CONTEXT_DIM, hidden, hidden)
        self.pair_encoder = nn.Sequential(
            nn.Linear(RESOLVED_PAIR_FEAT_DIM, pair_hidden),
            nn.Tanh(),
            nn.Linear(pair_hidden, pair_hidden),
            nn.Tanh(),
        )
        self.operation_head = nn.Sequential(
            nn.Linear(2 * hidden + pair_hidden + OP_FLAG_DIM, hidden),
            nn.Tanh(),
            nn.Linear(hidden, OP_ACTION_DIM),
        )
        last = self.operation_head[-1]
        assert isinstance(last, nn.Linear)
        with torch.no_grad():
            last.weight.zero_()
            last.bias.zero_()
            last.bias[OP_PROACTIVE] = float(proactive_bias)

        self.critic = PermutationInvariantCritic(
            self.CONTEXT_DIM,
            N_BLOCKS,
            STACKS_PER_BLOCK,
            self.BLOCK_FEAT_DIM,
            self.SLOT_FEAT_DIM,
            hidden,
        )

    def _global_encode(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        squeeze = obs.ndim == 1
        if squeeze:
            obs = obs.unsqueeze(0)
        g, b, s = self.actor_encoder(obs)
        return g, b, s, squeeze

    def storage_logits(self, global_obs: torch.Tensor) -> torch.Tensor:
        g, b, s, squeeze = self._global_encode(global_obs)
        bf = b.unsqueeze(2).expand(-1, -1, STACKS_PER_BLOCK, -1)
        gf = g[:, None, None, :].expand(
            -1, N_BLOCKS, STACKS_PER_BLOCK, -1
        )
        logits = self.storage_scorer(torch.cat([gf, bf, s], dim=-1))
        logits = logits.squeeze(-1).reshape(-1, self.N_SLOTS)
        return logits.squeeze(0) if squeeze else logits

    def yc_operation_logits(
        self,
        global_obs: torch.Tensor,
        yc_obs: torch.Tensor,
    ) -> torch.Tensor:
        squeeze = global_obs.ndim == 1
        if squeeze:
            global_obs = global_obs.unsqueeze(0)
            yc_obs = yc_obs.unsqueeze(0)
        g, _, _, _ = self._global_encode(global_obs)
        context, pair, flags = split_hybrid_yc_obs(yc_obs)
        if context.ndim == 1:
            context = context.unsqueeze(0)
            pair = pair.unsqueeze(0)
            flags = flags.unsqueeze(0)
        c = self.local_context_encoder(context)
        p = self.pair_encoder(pair)
        logits = self.operation_head(torch.cat([g, c, p, flags], dim=-1))
        return logits.squeeze(0) if squeeze else logits

    def value(self, global_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(global_obs)
