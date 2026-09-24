import math

import numpy as np
import torch

from validate_yc_marl import heuristic_action
from yc_marl_env import ResourceMARLYardEnv, STORAGE_AGENT, YC_PROACTIVE_BASE
from v5.hierarchical_yc import (
    HIER_YC_OBS_DIM,
    HierarchicalYCActor,
    build_hierarchical_yc_input,
)


def find_candidate_state(min_targets=3):
    for scenario in range(1101,1111):
        env=ResourceMARLYardEnv(
            seed=scenario,
            arrival_rate_per_hour=20.0,
            include_resource_state=True,
            rule_resolve_proactive_pair=False,
        )
        env.reset()
        done=False
        for _ in range(100000):
            if env.active_agent()!=STORAGE_AGENT:
                block=env.active_block()
                obs,pair_mask=build_hierarchical_yc_input(env,block)
                if int(pair_mask.any(axis=1).sum())>=min_targets:
                    return env,block,obs,pair_mask
            action=heuristic_action(env)
            _,_,done,_,_=env.step(action)
            if done:
                break
    raise AssertionError("no V5 candidate state found")


def test_v5_compact_observation_matches_flat_physical_support():
    env,block,obs,pair_mask=find_candidate_state()
    assert obs.shape==(HIER_YC_OBS_DIM,)
    assert HIER_YC_OBS_DIM < env.yc_obs_dim
    flat=np.asarray(env.action_mask(),dtype=np.bool_)[YC_PROACTIVE_BASE:]
    assert np.array_equal(pair_mask.reshape(-1),flat)
    assert pair_mask.shape==(100,25)
    assert int(pair_mask.any(axis=1).sum())>=3


def test_v5_initial_policy_is_uniform_within_target_and_destination_branches():
    env,block,obs,pair_mask=find_candidate_state()
    actor=HierarchicalYCActor(proactive_bias=math.log(0.1/0.9))
    obs_t=torch.as_tensor(obs,dtype=torch.float32)
    pm_t=torch.as_tensor(pair_mask,dtype=torch.bool)
    op_logits,t_logits,d_logits,t_mask=actor.components(obs_t,pm_t)

    valid_t=t_logits[t_mask]
    assert torch.allclose(valid_t,torch.zeros_like(valid_t),atol=0,rtol=0)
    first_t=int(torch.nonzero(t_mask,as_tuple=False)[0].item())
    valid_d=d_logits[first_t][pm_t[first_t]]
    assert torch.allclose(valid_d,torch.zeros_like(valid_d),atol=0,rtol=0)

    first_d=int(torch.nonzero(pm_t[first_t],as_tuple=False)[0].item())
    ev=actor.evaluate_actions(
        obs_t,pm_t,
        torch.tensor(0),
        torch.tensor(-1),
        torch.tensor(-1),
    )
    assert abs(float(ev["p_proactive"].item())-0.1)<1e-6

    ev_pro=actor.evaluate_actions(
        obs_t,pm_t,
        torch.tensor(1),
        torch.tensor(first_t),
        torch.tensor(first_d),
    )
    expected=(
        math.log(0.1)
        -math.log(int(t_mask.sum().item()))
        -math.log(int(pm_t[first_t].sum().item()))
    )
    assert abs(float(ev_pro["log_prob"].item())-expected)<1e-5


def test_v5_sampled_action_is_always_accepted_by_simulator_mask():
    env,block,obs,pair_mask=find_candidate_state()
    actor=HierarchicalYCActor()
    obs_t=torch.as_tensor(obs,dtype=torch.float32)
    pm_t=torch.as_tensor(pair_mask,dtype=torch.bool)
    flat_mask=torch.as_tensor(env.action_mask(),dtype=torch.bool)
    for _ in range(200):
        sample=actor.sample(obs_t,pm_t,flat_mask)
        assert bool(flat_mask[sample.flat_action])
        if sample.operation==1:
            assert sample.target_position>=0
            assert sample.destination>=0
            assert bool(pm_t[sample.target_position,sample.destination])
        else:
            assert sample.target_position==-1
            assert sample.destination==-1


def test_v5_proactive_logprob_backpropagates_to_both_conditional_branches():
    env,block,obs,pair_mask=find_candidate_state()
    actor=HierarchicalYCActor()
    obs_t=torch.as_tensor(obs,dtype=torch.float32)
    pm_t=torch.as_tensor(pair_mask,dtype=torch.bool)
    t=int(np.flatnonzero(pair_mask.any(axis=1))[0])
    d=int(np.flatnonzero(pair_mask[t])[0])

    ev=actor.evaluate_actions(
        obs_t,pm_t,
        torch.tensor(1),
        torch.tensor(t),
        torch.tensor(d),
    )
    loss=-ev["log_prob"]
    actor.zero_grad(set_to_none=True)
    loss.backward()

    assert actor.target_scorer.weight.grad is not None
    assert float(actor.target_scorer.weight.grad.abs().sum().item())>0
    assert actor.destination_bias.weight.grad is not None
    assert float(actor.destination_bias.weight.grad.abs().sum().item())>0
    assert actor.operation_head[-1].weight.grad is not None
    assert float(actor.operation_head[-1].weight.grad.abs().sum().item())>0
