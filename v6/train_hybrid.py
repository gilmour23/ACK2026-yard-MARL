from __future__ import annotations

"""V6 multi-episode PPO trainer for the corrected hybrid architecture.

Key differences from frozen V4:
- YC learns only Default vs candidate-aware Proactive operation.
- The exact deterministic resolved move features are fed to the YC actor.
- PPO updates batch multiple terminal-complete independently seeded episodes.
- Training budget is specified in complete episodes, not raw decision count.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
import random
from typing import List, Optional

import numpy as np
import torch
from torch import nn

from train_yc_marl import (
    ScenarioSampler,
    completed_episode_mc_diagnostics,
    compute_gae,
    masked_distribution,
    set_global_seeds,
)
from yc_marl_env import ResourceMARLYardEnv, STORAGE_AGENT
from v6.hybrid_policy import (
    HybridResourceCooperativeModel,
    HybridCentralizedSingleModel,
    build_hybrid_yc_operation_input,
    OP_PROACTIVE,
)
from v6.storage_bc import load_storage_bc_strict


@dataclass
class HybridPPOConfig:
    total_episodes: int = 24
    episodes_per_update: int = 4
    update_epochs: int = 2
    critic_extra_epochs: int = 18
    minibatch_size: int = 256
    gamma: float = 1.0
    gae_lambda: float = 1.0
    clip_coef: float = 0.20
    storage_ent_coef: float = 0.01
    yc_op_ent_coef: float = 0.001
    vf_coef: float = 0.5
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    hidden: int = 128
    pair_hidden: int = 32
    proactive_init_bias: float = -2.197224577
    seed: int = 71
    arrival_rate_per_hour: float = 20.0
    include_resource_state: bool = True
    truck_wait_weight: float = 1.0
    storage_wait_weight: float = 1.0
    extra_move_weight: float = 0.10
    risk_shaping_weight: float = 0.0
    yc_queue_shaping_weight: float = 0.0
    enable_proactive: bool = True
    yc_move_time: float = 2.0


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _make_model(
    kind: str,
    env: ResourceMARLYardEnv,
    config: HybridPPOConfig,
) -> nn.Module:
    if kind in {"marl_resource", "marl_noresource"}:
        return HybridResourceCooperativeModel(
            env.global_obs_dim,
            env.storage_obs_dim,
            hidden=config.hidden,
            pair_hidden=config.pair_hidden,
            proactive_bias=config.proactive_init_bias,
        )
    if kind == "single":
        return HybridCentralizedSingleModel(
            env.global_obs_dim,
            hidden=config.hidden,
            pair_hidden=config.pair_hidden,
            proactive_bias=config.proactive_init_bias,
        )
    raise ValueError(kind)


def _actor_param_prefixes(kind: str) -> tuple[str, ...]:
    if kind in {"marl_resource", "marl_noresource"}:
        return ("storage_actor.", "yc_actor.")
    if kind == "single":
        # Everything outside critic belongs to the centralized actor.
        return (
            "actor_encoder.",
            "storage_scorer.",
            "local_context_encoder.",
            "pair_encoder.",
            "operation_head.",
        )
    raise ValueError(kind)


def _finite(x) -> bool:
    if isinstance(x, (float, np.floating)):
        return bool(np.isfinite(float(x)))
    if isinstance(x, dict):
        return all(_finite(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return all(_finite(v) for v in x)
    return True


def train_hybrid_ppo(
    kind: str,
    config: HybridPPOConfig,
    out_dir: Path,
    storage_bc_checkpoint: Optional[Path] = None,
):
    if config.total_episodes <= 0:
        raise ValueError("total_episodes must be positive")
    if config.episodes_per_update <= 0:
        raise ValueError("episodes_per_update must be positive")
    if kind == "marl_resource" and not config.include_resource_state:
        raise ValueError("marl_resource requires include_resource_state=True")
    if kind == "marl_noresource" and config.include_resource_state:
        raise ValueError("marl_noresource requires include_resource_state=False")

    set_global_seeds(config.seed)
    device = torch.device("cpu")
    sampler = ScenarioSampler(config.seed + 1000)

    init_seed = sampler.next()
    init_env = ResourceMARLYardEnv(
        seed=init_seed,
        arrival_rate_per_hour=config.arrival_rate_per_hour,
        include_resource_state=config.include_resource_state,
        truck_wait_weight=config.truck_wait_weight,
        storage_wait_weight=config.storage_wait_weight,
        extra_move_weight=config.extra_move_weight,
        risk_shaping_weight=config.risk_shaping_weight,
        yc_queue_shaping_weight=config.yc_queue_shaping_weight,
        enable_proactive=config.enable_proactive,
        yc_move_time=config.yc_move_time,
        rule_resolve_proactive_pair=True,
    )
    init_env.reset()
    model = _make_model(kind, init_env, config).to(device)

    bc_metrics = {}
    if storage_bc_checkpoint is not None:
        bc_metrics = load_storage_bc_strict(
            model,
            Path(storage_bc_checkpoint),
            expected_kind=kind,
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    critic_extra_rng = np.random.default_rng(config.seed + 700_001)

    episodes_seen = 0
    global_step = 0
    update_idx = 0
    episode_records: list[dict] = []
    update_records: list[dict] = []

    while episodes_seen < config.total_episodes:
        batch_episodes = min(
            config.episodes_per_update,
            config.total_episodes - episodes_seen,
        )

        global_obs_buf = []
        storage_obs_buf = []
        yc_obs_buf = []
        masks_buf = []
        roles_buf = []
        actions_buf = []
        old_logp_buf = []
        rewards_buf = []
        dones_buf = []
        values_buf = []
        next_values_buf = []
        team_buf = []
        scenario_seed_buf: list[int] = []

        yc_entropy_all = []
        yc_entropy_support2 = []
        yc_support2 = []
        yc_p_pro_support2 = []
        yc_selected_proactive = []

        storage_n = 0
        yc_n = 0

        for _ in range(batch_episodes):
            scenario = sampler.next()
            scenario_seed_buf.append(int(scenario))
            env = ResourceMARLYardEnv(
                seed=scenario,
                arrival_rate_per_hour=config.arrival_rate_per_hour,
                include_resource_state=config.include_resource_state,
                truck_wait_weight=config.truck_wait_weight,
                storage_wait_weight=config.storage_wait_weight,
                extra_move_weight=config.extra_move_weight,
                risk_shaping_weight=config.risk_shaping_weight,
                yc_queue_shaping_weight=config.yc_queue_shaping_weight,
                enable_proactive=config.enable_proactive,
                yc_move_time=config.yc_move_time,
                rule_resolve_proactive_pair=True,
            )
            env.reset()
            done = False
            episode_decisions = 0

            while not done:
                active = env.active_agent()
                role = 0 if active == STORAGE_AGENT else 1
                gobs = env.critic_observation().copy()
                gobs_t = torch.as_tensor(
                    gobs, dtype=torch.float32, device=device
                )

                if role == 0:
                    mask = np.asarray(env.action_mask(), dtype=np.bool_).copy()
                    mask_t = torch.as_tensor(
                        mask, dtype=torch.bool, device=device
                    )
                    if kind == "single":
                        aobs = gobs.copy()
                    else:
                        aobs = env.actor_observation().copy()
                    aobs_t = torch.as_tensor(
                        aobs, dtype=torch.float32, device=device
                    )
                    with torch.no_grad():
                        logits = model.storage_logits(aobs_t)
                        dist = masked_distribution(logits, mask_t)
                        action_t = dist.sample()
                        logp_t = dist.log_prob(action_t)
                        value_t = model.value(gobs_t)
                    model_action = int(action_t.item())
                    env_action = model_action
                    storage_obs_buf.append(aobs)
                    yc_obs_buf.append(None)
                    storage_n += 1
                else:
                    block = env.active_block()
                    if block is None:
                        raise RuntimeError("YC decision missing block")
                    # Decentralized MARL sees its declared resource visibility.
                    # Centralized Single receives full resource information.
                    yc_resource = (
                        True if kind == "single"
                        else bool(config.include_resource_state)
                    )
                    h = build_hybrid_yc_operation_input(
                        env,
                        block,
                        resource_state=yc_resource,
                    )
                    mask = h.operation_mask.copy()
                    mask_t = torch.as_tensor(
                        mask, dtype=torch.bool, device=device
                    )
                    yobs = h.obs.copy()
                    yobs_t = torch.as_tensor(
                        yobs, dtype=torch.float32, device=device
                    )
                    with torch.no_grad():
                        if kind == "single":
                            logits = model.yc_operation_logits(
                                gobs_t, yobs_t
                            )
                        else:
                            logits = model.yc_operation_logits(yobs_t)
                        dist = masked_distribution(logits, mask_t)
                        action_t = dist.sample()
                        logp_t = dist.log_prob(action_t)
                        value_t = model.value(gobs_t)

                    model_action = int(action_t.item())
                    env_action = h.flat_action(model_action)
                    storage_obs_buf.append(None)
                    yc_obs_buf.append(yobs)
                    yc_n += 1

                    ent = float(dist.entropy().item())
                    support = int(mask.sum())
                    yc_entropy_all.append(ent)
                    yc_support2.append(float(support >= 2))
                    yc_selected_proactive.append(
                        float(model_action == OP_PROACTIVE)
                    )
                    if support >= 2:
                        yc_entropy_support2.append(ent)
                        yc_p_pro_support2.append(
                            float(dist.probs[OP_PROACTIVE].item())
                        )

                _, reward, done, _, info = env.step(env_action)
                with torch.no_grad():
                    next_value = (
                        0.0
                        if done
                        else float(
                            model.value(
                                torch.as_tensor(
                                    env.critic_observation(),
                                    dtype=torch.float32,
                                    device=device,
                                )
                            ).item()
                        )
                    )

                global_obs_buf.append(gobs)
                masks_buf.append(mask)
                roles_buf.append(role)
                actions_buf.append(model_action)
                old_logp_buf.append(float(logp_t.item()))
                rewards_buf.append(float(reward))
                team_buf.append(
                    float(info.get("reward_components", {}).get("team", reward))
                )
                dones_buf.append(float(done))
                values_buf.append(float(value_t.item()))
                next_values_buf.append(float(next_value))

                global_step += 1
                episode_decisions += 1

            if not done:
                raise RuntimeError("V6 episode did not terminate")
            if "kpis" not in info:
                raise RuntimeError("Terminal episode missing KPIs")
            episode_records.append(
                {
                    "training_seed": int(config.seed),
                    "scenario_seed": int(scenario),
                    "episode_index": int(episodes_seen + len(scenario_seed_buf)),
                    "decisions": int(episode_decisions),
                    "global_step": int(global_step),
                    "episode_return": float(info["episode_return"]),
                    **info["kpis"],
                }
            )

        if len(set(scenario_seed_buf)) != len(scenario_seed_buf):
            raise RuntimeError(
                ("duplicate scenario seed within PPO batch", scenario_seed_buf)
            )

        rewards = np.asarray(rewards_buf, dtype=np.float32)
        values = np.asarray(values_buf, dtype=np.float32)
        next_values = np.asarray(next_values_buf, dtype=np.float32)
        dones = np.asarray(dones_buf, dtype=np.float32)
        if int(dones.sum()) != int(batch_episodes):
            raise RuntimeError(
                ("terminal count mismatch", dones.sum(), batch_episodes)
            )

        mc_diag = completed_episode_mc_diagnostics(
            rewards, dones, values
        )
        if not np.isclose(mc_diag["mc_completed_fraction"], 1.0):
            raise RuntimeError(("non-terminal rollout content", mc_diag))

        adv, ret = compute_gae(
            rewards,
            values,
            next_values,
            dones,
            config.gamma,
            config.gae_lambda,
        )
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        roles_t = torch.as_tensor(
            roles_buf, dtype=torch.long, device=device
        )
        actions_t = torch.as_tensor(
            actions_buf, dtype=torch.long, device=device
        )
        old_logp_t = torch.as_tensor(
            old_logp_buf, dtype=torch.float32, device=device
        )
        adv_t = torch.as_tensor(
            adv, dtype=torch.float32, device=device
        )
        ret_t = torch.as_tensor(
            ret, dtype=torch.float32, device=device
        )
        global_obs_t = torch.as_tensor(
            np.asarray(global_obs_buf),
            dtype=torch.float32,
            device=device,
        )

        n = len(actions_buf)
        last_pg = last_v = last_ent = 0.0
        for _ in range(config.update_epochs):
            order = np.random.permutation(n)
            for start in range(0, n, config.minibatch_size):
                ids = order[start : start + config.minibatch_size]
                if len(ids) == 0:
                    continue
                mb = torch.as_tensor(ids, dtype=torch.long, device=device)
                pg_terms = []
                ent_terms = []

                storage_ids = [i for i in ids if roles_buf[i] == 0]
                if storage_ids:
                    rid = torch.as_tensor(
                        storage_ids, dtype=torch.long, device=device
                    )
                    obs = torch.as_tensor(
                        np.stack([storage_obs_buf[i] for i in storage_ids]),
                        dtype=torch.float32,
                        device=device,
                    )
                    masks = torch.as_tensor(
                        np.stack([masks_buf[i] for i in storage_ids]),
                        dtype=torch.bool,
                        device=device,
                    )
                    dist = masked_distribution(
                        model.storage_logits(obs), masks
                    )
                    new_logp = dist.log_prob(actions_t[rid])
                    ratio = (new_logp - old_logp_t[rid]).exp()
                    pg1 = -adv_t[rid] * ratio
                    pg2 = -adv_t[rid] * torch.clamp(
                        ratio,
                        1 - config.clip_coef,
                        1 + config.clip_coef,
                    )
                    pg_terms.append(torch.maximum(pg1, pg2).mean())
                    ent_terms.append(
                        config.storage_ent_coef * dist.entropy().mean()
                    )

                yc_ids = [i for i in ids if roles_buf[i] == 1]
                if yc_ids:
                    rid = torch.as_tensor(
                        yc_ids, dtype=torch.long, device=device
                    )
                    yobs = torch.as_tensor(
                        np.stack([yc_obs_buf[i] for i in yc_ids]),
                        dtype=torch.float32,
                        device=device,
                    )
                    masks = torch.as_tensor(
                        np.stack([masks_buf[i] for i in yc_ids]),
                        dtype=torch.bool,
                        device=device,
                    )
                    if kind == "single":
                        logits = model.yc_operation_logits(
                            global_obs_t[rid], yobs
                        )
                    else:
                        logits = model.yc_operation_logits(yobs)
                    dist = masked_distribution(logits, masks)
                    new_logp = dist.log_prob(actions_t[rid])
                    ratio = (new_logp - old_logp_t[rid]).exp()
                    pg1 = -adv_t[rid] * ratio
                    pg2 = -adv_t[rid] * torch.clamp(
                        ratio,
                        1 - config.clip_coef,
                        1 + config.clip_coef,
                    )
                    pg_terms.append(torch.maximum(pg1, pg2).mean())

                    support2 = masks.sum(dim=-1) >= 2
                    if bool(support2.any()):
                        yc_ent = dist.entropy()[support2].mean()
                        ent_terms.append(config.yc_op_ent_coef * yc_ent)

                if not pg_terms:
                    raise RuntimeError("PPO minibatch contains no actor samples")
                pg_loss = torch.stack(pg_terms).mean()
                entropy_bonus = (
                    torch.stack(ent_terms).mean()
                    if ent_terms
                    else torch.zeros((), dtype=torch.float32, device=device)
                )
                value = model.value(global_obs_t[mb])
                v_loss = 0.5 * (value - ret_t[mb]).pow(2).mean()
                loss = (
                    pg_loss
                    + config.vf_coef * v_loss
                    - entropy_bonus
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), config.max_grad_norm
                )
                optimizer.step()
                last_pg = float(pg_loss.item())
                last_v = float(v_loss.item())
                last_ent = float(entropy_bonus.item())

        actor_before = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if name.startswith(_actor_param_prefixes(kind))
        }
        critic_extra_steps = 0
        for _ in range(config.critic_extra_epochs):
            order = critic_extra_rng.permutation(n)
            for start in range(0, n, config.minibatch_size):
                ids = order[start : start + config.minibatch_size]
                if len(ids) == 0:
                    continue
                mb = torch.as_tensor(ids, dtype=torch.long, device=device)
                value = model.value(global_obs_t[mb])
                v_loss = 0.5 * (value - ret_t[mb]).pow(2).mean()
                optimizer.zero_grad(set_to_none=True)
                (config.vf_coef * v_loss).backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), config.max_grad_norm
                )
                optimizer.step()
                critic_extra_steps += 1

        critic_extra_actor_max_abs_delta = 0.0
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in actor_before:
                    critic_extra_actor_max_abs_delta = max(
                        critic_extra_actor_max_abs_delta,
                        float((p - actor_before[name]).abs().max().item()),
                    )
        if critic_extra_actor_max_abs_delta != 0.0:
            raise RuntimeError(
                "Actor changed during critic-only epochs: "
                f"{critic_extra_actor_max_abs_delta}"
            )

        with torch.no_grad():
            post_v = (
                model.value(global_obs_t)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
        target_v = np.asarray(ret, dtype=np.float64)
        target_var = float(np.var(target_v))
        post_ev = (
            float("nan")
            if target_var < 1e-12
            else float(
                1.0
                - np.var(target_v - post_v) / target_var
            )
        )
        post_rmse = float(
            np.sqrt(np.mean((target_v - post_v) ** 2))
        )

        episodes_seen += batch_episodes
        update_idx += 1
        record = {
            "update": int(update_idx),
            "episodes_seen": int(episodes_seen),
            "episodes_in_batch": int(batch_episodes),
            "scenario_seeds": json.dumps(scenario_seed_buf),
            "global_step": int(global_step),
            "rollout_decisions": int(n),
            "storage_decisions": int(storage_n),
            "yc_decisions": int(yc_n),
            "mean_reward": float(np.mean(rewards)),
            "mean_team_reward": float(np.mean(team_buf)),
            "mc_completed_fraction": float(mc_diag["mc_completed_fraction"]),
            "mc_value_ev": float(mc_diag["mc_value_ev"]),
            "mc_value_rmse": float(mc_diag["mc_value_rmse"]),
            "post_update_value_ev": float(post_ev),
            "post_update_value_rmse": float(post_rmse),
            "policy_loss_last": float(last_pg),
            "value_loss_last": float(last_v),
            "entropy_bonus_last": float(last_ent),
            "yc_operation_entropy_all": (
                float(np.mean(yc_entropy_all))
                if yc_entropy_all
                else 0.0
            ),
            "yc_operation_entropy_support_ge2": (
                float(np.mean(yc_entropy_support2))
                if yc_entropy_support2
                else 0.0
            ),
            "yc_support_ge2_fraction": (
                float(np.mean(yc_support2))
                if yc_support2
                else 0.0
            ),
            "yc_p_proactive_support_ge2": (
                float(np.mean(yc_p_pro_support2))
                if yc_p_pro_support2
                else 0.0
            ),
            "yc_sampled_proactive_rate": (
                float(np.mean(yc_selected_proactive))
                if yc_selected_proactive
                else 0.0
            ),
            "critic_extra_steps": int(critic_extra_steps),
            "critic_extra_actor_max_abs_delta": float(
                critic_extra_actor_max_abs_delta
            ),
        }
        if not _finite(record):
            # mc_value_ev may be undefined only for a degenerate target variance.
            bad = {
                k: v
                for k, v in record.items()
                if isinstance(v, float) and not np.isfinite(v)
            }
            if set(bad) != {"mc_value_ev"}:
                raise RuntimeError(("non-finite update record", bad))
        update_records.append(record)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "v6_hybrid_final.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "kind": kind,
            "config": asdict(config),
            "hidden": int(config.hidden),
            "pair_hidden": int(config.pair_hidden),
            "episodes_seen": int(episodes_seen),
            "global_step": int(global_step),
            "update_count": int(update_idx),
            "storage_bc_checkpoint": (
                str(storage_bc_checkpoint)
                if storage_bc_checkpoint is not None
                else None
            ),
            "storage_bc_metrics": bc_metrics,
            "optimizer_state_dict": optimizer.state_dict(),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "scenario_sampler_state": sampler.rng.getstate(),
        },
        checkpoint_path,
    )
    _write_csv(out_dir / "v6_train_episodes.csv", episode_records)
    _write_csv(out_dir / "v6_updates.csv", update_records)
    (out_dir / "v6_config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n",
        encoding="utf-8",
    )
    return model, update_records, episode_records
