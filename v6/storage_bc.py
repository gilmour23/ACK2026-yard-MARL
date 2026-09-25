from __future__ import annotations

"""Information-matched Storage behavior cloning for V6.

The teacher deliberately uses only variables observable to both resource-state
and no-resource Storage actors.  Labels are generated once per scenario state
and paired resource/no-resource observations are recorded from that exact state.
"""

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Iterable

import numpy as np
import torch
from torch import nn

from validate_yc_marl import heuristic_action
from yc_marl_env import ResourceMARLYardEnv, STORAGE_AGENT
from v6.hybrid_policy import (
    HybridResourceCooperativeModel,
    HybridCentralizedSingleModel,
)


@dataclass
class CommonStorageDataset:
    resource_local_obs: list[np.ndarray]
    noresource_local_obs: list[np.ndarray]
    global_obs: list[np.ndarray]
    masks: list[np.ndarray]
    actions: list[int]
    scenario_seeds: list[int]


def _storage_obs_for_resource_state(
    env: ResourceMARLYardEnv,
    resource_state: bool,
) -> np.ndarray:
    full = env.global_observation(resource_state=bool(resource_state))
    local = np.concatenate(
        [full[[2, 3, 4, 5]], full[env.GLOBAL_CONTEXT_DIM :]]
    ).astype(np.float32, copy=False)
    if local.shape != (env.storage_obs_dim,):
        raise RuntimeError((local.shape, env.storage_obs_dim))
    return local


def common_storage_teacher_action(env: ResourceMARLYardEnv) -> int:
    """Storage teacher using only inversion, height and physical feasibility."""
    sim = env.sim
    if sim is None or sim.pending_decision is None:
        raise RuntimeError("Environment has no pending decision")
    if env.active_agent() != STORAGE_AGENT:
        raise RuntimeError("Common Storage teacher called outside Storage decision")
    d = sim.pending_decision
    if d.cid is None:
        raise RuntimeError("Storage decision missing container")
    c = sim.containers[d.cid]

    scored: list[tuple[float, int, int]] = []
    for b, s in d.candidates or []:
        stack = sim.yard.stacks[b][s]
        inversion = sum(
            1
            for lower_id in stack
            if sim.containers[lower_id].eta < c.eta
        )
        height = len(stack) / sim.yard.max_tier
        score = float(inversion + 0.35 * height)
        scored.append((score, int(b), int(s)))
    if not scored:
        raise RuntimeError("No feasible Storage candidate")
    scored.sort()
    _, b, s = scored[0]
    action = b * sim.yard.stacks_per_block + s
    mask = env.action_mask()
    if action >= len(mask) or not bool(mask[action]):
        raise RuntimeError(("Teacher selected infeasible stack", action))
    return int(action)


def collect_common_storage_dataset(
    scenario_seeds: Iterable[int],
    arrival_rate_per_hour: float = 20.0,
) -> CommonStorageDataset:
    ds = CommonStorageDataset([], [], [], [], [], [])
    for scenario in scenario_seeds:
        env = ResourceMARLYardEnv(
            seed=int(scenario),
            arrival_rate_per_hour=arrival_rate_per_hour,
            include_resource_state=True,
            rule_resolve_proactive_pair=True,
        )
        env.reset()
        done = False
        while not done:
            if env.active_agent() == STORAGE_AGENT:
                action = common_storage_teacher_action(env)
                ds.resource_local_obs.append(
                    _storage_obs_for_resource_state(env, True).copy()
                )
                ds.noresource_local_obs.append(
                    _storage_obs_for_resource_state(env, False).copy()
                )
                ds.global_obs.append(
                    env.global_observation(resource_state=True).copy()
                )
                ds.masks.append(env.action_mask().copy())
                ds.actions.append(int(action))
                ds.scenario_seeds.append(int(scenario))
            else:
                # Fixed operational teacher only drives the common BC trajectory;
                # Storage labels themselves never consume resource-only fields.
                action = heuristic_action(env)
            _, _, done, _, _ = env.step(action)
    return ds


def _model_for_kind(
    kind: str,
    env: ResourceMARLYardEnv,
    hidden: int,
    pair_hidden: int,
) -> nn.Module:
    if kind in {"marl_resource", "marl_noresource"}:
        return HybridResourceCooperativeModel(
            env.global_obs_dim,
            env.storage_obs_dim,
            hidden=hidden,
            pair_hidden=pair_hidden,
        )
    if kind == "single":
        return HybridCentralizedSingleModel(
            env.global_obs_dim,
            hidden=hidden,
            pair_hidden=pair_hidden,
        )
    raise ValueError(kind)


def _storage_parameters(model: nn.Module, kind: str):
    if kind in {"marl_resource", "marl_noresource"}:
        return list(model.storage_actor.parameters())
    if kind == "single":
        return list(model.actor_encoder.parameters()) + list(
            model.storage_scorer.parameters()
        )
    raise ValueError(kind)


def _storage_state_dict(model: nn.Module, kind: str) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    if kind in {"marl_resource", "marl_noresource"}:
        prefixes = ("storage_actor.",)
    elif kind == "single":
        prefixes = ("actor_encoder.", "storage_scorer.")
    else:
        raise ValueError(kind)
    return {
        k: v.detach().cpu()
        for k, v in state.items()
        if k.startswith(prefixes)
    }


def train_common_storage_bc(
    kind: str,
    out_path: Path,
    scenario_seeds: Iterable[int] = range(1000, 1008),
    epochs: int = 3,
    batch_size: int = 128,
    lr: float = 5e-4,
    rng_seed: int = 20260926,
    arrival_rate_per_hour: float = 20.0,
    hidden: int = 128,
    pair_hidden: int = 32,
) -> dict:
    scenario_seeds = [int(x) for x in scenario_seeds]
    if not scenario_seeds:
        raise ValueError("scenario_seeds is empty")

    torch.manual_seed(int(rng_seed))
    np.random.seed(int(rng_seed))
    ds = collect_common_storage_dataset(
        scenario_seeds,
        arrival_rate_per_hour=arrival_rate_per_hour,
    )

    env = ResourceMARLYardEnv(
        seed=1,
        arrival_rate_per_hour=arrival_rate_per_hour,
        include_resource_state=True,
        rule_resolve_proactive_pair=True,
    )
    env.reset()
    model = _model_for_kind(kind, env, hidden, pair_hidden)

    if kind == "marl_resource":
        obs_rows = ds.resource_local_obs
    elif kind == "marl_noresource":
        obs_rows = ds.noresource_local_obs
    else:
        obs_rows = ds.global_obs

    params = _storage_parameters(model, kind)
    opt = torch.optim.Adam(params, lr=float(lr))
    idx = np.arange(len(ds.actions))
    rng = np.random.default_rng(int(rng_seed))

    for _ in range(int(epochs)):
        rng.shuffle(idx)
        for start in range(0, len(idx), int(batch_size)):
            ids = idx[start : start + int(batch_size)]
            if len(ids) == 0:
                continue
            obs = torch.as_tensor(
                np.stack([obs_rows[i] for i in ids]),
                dtype=torch.float32,
            )
            masks = torch.as_tensor(
                np.stack([ds.masks[i] for i in ids]),
                dtype=torch.bool,
            )
            acts = torch.as_tensor(
                [ds.actions[i] for i in ids],
                dtype=torch.long,
            )
            logits = model.storage_logits(obs)
            logits = logits.masked_fill(
                ~masks, torch.finfo(logits.dtype).min
            )
            loss = nn.functional.cross_entropy(logits, acts)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

    correct = 0
    with torch.no_grad():
        for i in range(len(ds.actions)):
            obs = torch.as_tensor(obs_rows[i], dtype=torch.float32)
            mask = torch.as_tensor(ds.masks[i], dtype=torch.bool)
            logits = model.storage_logits(obs)
            action = int(
                logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
                .argmax()
                .item()
            )
            correct += int(action == ds.actions[i])

    metrics = {
        "kind": kind,
        "storage_accuracy_train": correct / max(len(ds.actions), 1),
        "storage_samples": len(ds.actions),
        "scenario_seeds": scenario_seeds,
        "teacher": "common_observable_inversion_plus_0.35_height",
        "teacher_uses_resource_only_fields": False,
        "rng_seed": int(rng_seed),
        "arrival_rate_per_hour": float(arrival_rate_per_hour),
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "storage_state_dict": _storage_state_dict(model, kind),
            "kind": kind,
            "hidden": int(hidden),
            "pair_hidden": int(pair_hidden),
            "storage_only": True,
            "information_matched_teacher": True,
            "metrics": metrics,
        },
        out_path,
    )
    out_path.with_suffix(".json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics


def load_storage_bc_strict(
    model: nn.Module,
    checkpoint: Path,
    expected_kind: str,
) -> dict:
    ck = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    if ck.get("kind") != expected_kind:
        raise RuntimeError((ck.get("kind"), expected_kind))
    if not bool(ck.get("information_matched_teacher", False)):
        raise RuntimeError("V6 requires an information-matched BC checkpoint")
    state = ck.get("storage_state_dict")
    if not isinstance(state, dict) or not state:
        raise RuntimeError("Missing storage_state_dict")
    current = model.state_dict()
    unexpected = [k for k in state if k not in current]
    shape_mismatch = [
        k
        for k, v in state.items()
        if k in current and current[k].shape != v.shape
    ]
    if unexpected or shape_mismatch:
        raise RuntimeError(
            {
                "unexpected": unexpected,
                "shape_mismatch": shape_mismatch,
            }
        )
    result = model.load_state_dict(state, strict=False)
    # Missing non-storage tensors are expected; every supplied tensor must load.
    if result.unexpected_keys:
        raise RuntimeError(result.unexpected_keys)
    return ck.get("metrics", {})
