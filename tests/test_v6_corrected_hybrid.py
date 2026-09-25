import numpy as np
import torch

from validate_yc_marl import heuristic_action
from yc_marl_env import (
    ResourceMARLYardEnv,
    STORAGE_AGENT,
    YC_MANDATORY,
    YC_IDLE,
    YC_PROACTIVE_BASE,
)
from v6.hybrid_policy import (
    HYBRID_YC_OBS_DIM,
    YC_CONTEXT_DIM,
    RESOLVED_PAIR_FEAT_DIM,
    CandidateAwareYCOperationActor,
    HybridResourceCooperativeModel,
    build_hybrid_yc_operation_input,
    OP_DEFAULT,
    OP_PROACTIVE,
)
from v6.storage_bc import (
    common_storage_teacher_action,
    _storage_obs_for_resource_state,
)
from v6.train_hybrid import HybridPPOConfig


def _find_rule_resolved_proactive_state():
    for seed in range(1, 80):
        env = ResourceMARLYardEnv(
            seed=seed,
            arrival_rate_per_hour=20.0,
            rule_resolve_proactive_pair=True,
        )
        env.reset()
        done = False
        for _ in range(5000):
            if env.active_agent() != STORAGE_AGENT:
                mask = env.action_mask()
                if int(mask[YC_PROACTIVE_BASE:].sum()) == 1:
                    return env
            action = heuristic_action(env)
            _, _, done, _, _ = env.step(action)
            if done:
                break
    raise AssertionError("No rule-resolved proactive state found")


def test_v6_candidate_aware_observation_matches_resolved_flat_pair():
    env = _find_rule_resolved_proactive_state()
    block = env.active_block()
    assert block is not None
    h = build_hybrid_yc_operation_input(env, block, resource_state=True)

    assert h.obs.shape == (HYBRID_YC_OBS_DIM,)
    assert HYBRID_YC_OBS_DIM == 29
    assert h.proactive_flat_action is not None
    assert h.operation_mask.tolist()[OP_PROACTIVE] is True

    pair_idx = int(h.proactive_flat_action) - YC_PROACTIVE_BASE
    expected = env._proactive_pair_features(
        block, resource_state=True
    )[pair_idx]
    got = h.obs[
        YC_CONTEXT_DIM : YC_CONTEXT_DIM + RESOLVED_PAIR_FEAT_DIM
    ]
    assert np.allclose(got, expected)
    assert np.isclose(got[0], 1.0)


def test_v6_operation_mapping_preserves_simulator_semantics():
    env = _find_rule_resolved_proactive_state()
    block = env.active_block()
    assert block is not None
    h = build_hybrid_yc_operation_input(env, block)

    assert h.flat_action(OP_PROACTIVE) == h.proactive_flat_action
    mask = env.action_mask()
    assert bool(mask[h.flat_action(OP_PROACTIVE)])

    if bool(mask[YC_MANDATORY]):
        assert h.default_flat_action == YC_MANDATORY
    else:
        assert bool(mask[YC_IDLE])
        assert h.default_flat_action == YC_IDLE
    assert h.flat_action(OP_DEFAULT) == h.default_flat_action


def test_v6_operation_initialization_is_exactly_common():
    actor = CandidateAwareYCOperationActor(hidden=32, pair_hidden=8)
    obs = torch.randn(HYBRID_YC_OBS_DIM, dtype=torch.float32)
    with torch.no_grad():
        logits = actor(obs)
        probs = torch.softmax(logits, dim=-1)
    assert np.isclose(float(probs[OP_PROACTIVE]), 0.10, atol=1e-6)


def test_v6_has_no_learned_2500_pair_scoring_head():
    env = ResourceMARLYardEnv(
        seed=1,
        arrival_rate_per_hour=20.0,
        rule_resolve_proactive_pair=True,
    )
    env.reset()
    model = HybridResourceCooperativeModel(
        env.global_obs_dim,
        env.storage_obs_dim,
        hidden=32,
        pair_hidden=8,
    )
    names = [name for name, _ in model.named_parameters()]
    assert not any("pair_scorer" in name for name in names)
    for module in model.yc_actor.modules():
        if isinstance(module, torch.nn.Linear):
            assert module.out_features != 2500


def test_v6_pair_features_can_change_operation_logits():
    actor = CandidateAwareYCOperationActor(
        hidden=32,
        pair_hidden=8,
    )
    # Make one explicit pair-feature -> pair-latent -> operation-logit path.
    with torch.no_grad():
        actor.pair_encoder[0].weight.zero_()
        actor.pair_encoder[0].bias.zero_()
        actor.pair_encoder[0].weight[0, 3] = 1.0
        actor.pair_encoder[2].weight.zero_()
        actor.pair_encoder[2].bias.zero_()
        actor.pair_encoder[2].weight[0, 0] = 1.0
        actor.operation_head[0].weight.zero_()
        actor.operation_head[0].bias.zero_()
        actor.operation_head[0].weight[0, 32] = 1.0
        actor.operation_head[2].weight.zero_()
        actor.operation_head[2].bias.zero_()
        actor.operation_head[2].weight[OP_PROACTIVE, 0] = 1.0

    x0 = torch.zeros(HYBRID_YC_OBS_DIM, dtype=torch.float32)
    x1 = x0.clone()
    x1[YC_CONTEXT_DIM + 3] = 1.0
    with torch.no_grad():
        y0 = actor(x0)
        y1 = actor(x1)
    assert not torch.allclose(y0, y1, atol=1e-10, rtol=0.0)


def test_v6_common_storage_teacher_is_resource_visibility_invariant():
    a = ResourceMARLYardEnv(
        seed=20260926,
        arrival_rate_per_hour=20.0,
        include_resource_state=True,
        rule_resolve_proactive_pair=True,
    )
    b = ResourceMARLYardEnv(
        seed=20260926,
        arrival_rate_per_hour=20.0,
        include_resource_state=False,
        rule_resolve_proactive_pair=True,
    )
    a.reset()
    b.reset()

    compared = 0
    for _ in range(10000):
        assert a.active_agent() == b.active_agent()
        assert np.array_equal(a.action_mask(), b.action_mask())

        if a.active_agent() == STORAGE_AGENT:
            aa = common_storage_teacher_action(a)
            bb = common_storage_teacher_action(b)
            assert aa == bb
            # Paired observations differ only through declared resource visibility,
            # while the common teacher label stays fixed.
            ra = _storage_obs_for_resource_state(a, True)
            na = _storage_obs_for_resource_state(a, False)
            assert ra.shape == na.shape == (a.storage_obs_dim,)
            action = aa
            compared += 1
        else:
            action = heuristic_action(a)
            assert action == heuristic_action(b)

        _, _, da, _, _ = a.step(action)
        _, _, db, _, _ = b.step(action)
        assert da == db
        if compared >= 20 or da:
            break

    assert compared >= 1


def test_v6_training_budget_is_episode_based_and_multi_episode():
    cfg = HybridPPOConfig(total_episodes=24, episodes_per_update=4)
    assert cfg.total_episodes == 24
    assert cfg.episodes_per_update == 4
    assert cfg.total_episodes // cfg.episodes_per_update == 6
    assert cfg.gamma == 1.0
    assert cfg.gae_lambda == 1.0
    assert cfg.extra_move_weight == 0.10
