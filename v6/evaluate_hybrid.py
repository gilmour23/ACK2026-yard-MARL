from __future__ import annotations

"""Stochastic evaluator for V6 corrected hybrid policies."""

from pathlib import Path
import csv
import json
from typing import Iterable

import numpy as np
import torch

from train_yc_marl import masked_distribution
from yc_marl_env import ResourceMARLYardEnv, STORAGE_AGENT
from v6.hybrid_policy import (
    HybridResourceCooperativeModel,
    HybridCentralizedSingleModel,
    build_hybrid_yc_operation_input,
)


def load_v6_model(checkpoint: Path):
    ck = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    kind = str(ck["kind"])
    cfg = dict(ck.get("config", {}))
    hidden = int(ck.get("hidden", cfg.get("hidden", 128)))
    pair_hidden = int(ck.get("pair_hidden", cfg.get("pair_hidden", 32)))
    include_resource = bool(cfg.get("include_resource_state", True))

    env = ResourceMARLYardEnv(
        seed=1,
        arrival_rate_per_hour=float(cfg.get("arrival_rate_per_hour", 20.0)),
        include_resource_state=include_resource,
        rule_resolve_proactive_pair=True,
    )
    env.reset()

    if kind in {"marl_resource", "marl_noresource"}:
        model = HybridResourceCooperativeModel(
            env.global_obs_dim,
            env.storage_obs_dim,
            hidden=hidden,
            pair_hidden=pair_hidden,
            proactive_bias=float(cfg.get("proactive_init_bias", -2.197224577)),
        )
    elif kind == "single":
        model = HybridCentralizedSingleModel(
            env.global_obs_dim,
            hidden=hidden,
            pair_hidden=pair_hidden,
            proactive_bias=float(cfg.get("proactive_init_bias", -2.197224577)),
        )
    else:
        raise ValueError(kind)

    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()
    return model, kind, cfg


def run_v6_episode(
    scenario_seed: int,
    model,
    kind: str,
    config: dict,
    policy_seed: int,
    stochastic: bool = True,
) -> dict:
    include_resource = bool(config.get("include_resource_state", True))
    env = ResourceMARLYardEnv(
        seed=int(scenario_seed),
        arrival_rate_per_hour=float(config.get("arrival_rate_per_hour", 20.0)),
        include_resource_state=include_resource,
        truck_wait_weight=float(config.get("truck_wait_weight", 1.0)),
        storage_wait_weight=float(config.get("storage_wait_weight", 1.0)),
        extra_move_weight=float(config.get("extra_move_weight", 0.10)),
        risk_shaping_weight=float(config.get("risk_shaping_weight", 0.0)),
        yc_queue_shaping_weight=float(config.get("yc_queue_shaping_weight", 0.0)),
        enable_proactive=bool(config.get("enable_proactive", True)),
        yc_move_time=float(config.get("yc_move_time", 2.0)),
        rule_resolve_proactive_pair=True,
    )
    env.reset()
    rng = torch.Generator(device="cpu").manual_seed(int(policy_seed))

    done = False
    steps = 0
    yc_support2 = 0
    yc_decisions = 0
    proactive_selected = 0

    while not done and steps < 100000:
        gobs = torch.as_tensor(
            env.critic_observation(), dtype=torch.float32
        )

        if env.active_agent() == STORAGE_AGENT:
            mask = torch.as_tensor(env.action_mask(), dtype=torch.bool)
            if kind == "single":
                obs = gobs
            else:
                obs = torch.as_tensor(
                    env.actor_observation(), dtype=torch.float32
                )
            with torch.inference_mode():
                dist = masked_distribution(model.storage_logits(obs), mask)
            if stochastic:
                action = int(
                    torch.multinomial(
                        dist.probs, 1, generator=rng
                    ).item()
                )
            else:
                action = int(dist.probs.argmax().item())
        else:
            block = env.active_block()
            if block is None:
                raise RuntimeError("YC decision missing block")
            resource = True if kind == "single" else include_resource
            h = build_hybrid_yc_operation_input(
                env, block, resource_state=resource
            )
            yobs = torch.as_tensor(h.obs, dtype=torch.float32)
            mask = torch.as_tensor(h.operation_mask, dtype=torch.bool)
            with torch.inference_mode():
                logits = (
                    model.yc_operation_logits(gobs, yobs)
                    if kind == "single"
                    else model.yc_operation_logits(yobs)
                )
                dist = masked_distribution(logits, mask)
            if stochastic:
                op = int(
                    torch.multinomial(
                        dist.probs, 1, generator=rng
                    ).item()
                )
            else:
                op = int(dist.probs.argmax().item())
            action = h.flat_action(op)
            yc_decisions += 1
            yc_support2 += int(int(h.operation_mask.sum()) >= 2)
            proactive_selected += int(op == 1)

        _, _, done, _, info = env.step(action)
        steps += 1

    if not done:
        raise RuntimeError(("V6 evaluation did not terminate", scenario_seed))
    k = info["kpis"]
    objective = float(
        k["mean_truck_completion_delay"]
        + k["mean_storage_completion_delay"]
        + 0.10 * k["extra_yc_minutes_per_retrieval"]
    )
    return {
        "scenario_seed": int(scenario_seed),
        "policy_seed": int(policy_seed),
        "kind": kind,
        "evaluation_mode": "stochastic" if stochastic else "greedy",
        "decisions": int(steps),
        "yc_decisions": int(yc_decisions),
        "yc_support_ge2_fraction": (
            float(yc_support2 / yc_decisions)
            if yc_decisions
            else 0.0
        ),
        "yc_sampled_proactive_rate": (
            float(proactive_selected / yc_decisions)
            if yc_decisions
            else 0.0
        ),
        "objective_proxy": objective,
        **k,
    }


def evaluate_v6(
    checkpoint: Path,
    out_csv: Path,
    scenarios: Iterable[int],
    repeats: int = 3,
    stochastic: bool = True,
) -> dict:
    model, kind, config = load_v6_model(checkpoint)
    rows = []
    scenarios = [int(s) for s in scenarios]
    for scenario in scenarios:
        nrep = int(repeats) if stochastic else 1
        for r in range(nrep):
            rows.append(
                run_v6_episode(
                    scenario,
                    model,
                    kind,
                    config,
                    policy_seed=scenario * 100 + r,
                    stochastic=stochastic,
                )
            )

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    metrics = [
        "objective_proxy",
        "mean_truck_completion_delay",
        "mean_storage_completion_delay",
        "rehandling_moves",
        "proactive_moves",
        "extra_yc_minutes_per_retrieval",
        "total_yc_moves",
        "mean_yc_utilization",
        "max_yc_queue",
        "yc_support_ge2_fraction",
        "yc_sampled_proactive_rate",
    ]
    summary = {
        m: float(np.mean([float(row[m]) for row in rows]))
        for m in metrics
    }
    summary.update(
        {
            "kind": kind,
            "n_scenarios": len(scenarios),
            "repeats": int(repeats if stochastic else 1),
            "evaluation_mode": "stochastic" if stochastic else "greedy",
            "rule_resolve_proactive_pair": True,
        }
    )
    out_csv.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
