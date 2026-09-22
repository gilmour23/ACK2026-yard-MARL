import numpy as np
import torch

from yc_marl_env import (
    N_BLOCKS, STACKS_PER_BLOCK, MAX_TIER, TOTAL_SLOTS, INITIAL_CONTAINERS,
    OPERATING_HORIZON_MIN, YC_MOVE_TIME_MIN, YC_ACTION_DIM, YC_PAIR_COUNT,
    YC_PROACTIVE_BASE, YC_MANDATORY, ResourceMARLYardEnv,
    build_resource_marl_scenario, encode_proactive_action, decode_proactive_action,
    _sample_future_pickup_delay_min, RELOCATION_SLACK_SLOTS_PER_BLOCK,
)
from train_yc_marl import (
    ResourceCooperativeModel, ResourcePPOConfig, initialize_conservative_yc_actor,
    masked_distribution, structured_yc_entropy,
)
from train_yc_single import CentralizedSingleModel
from validate_yc_marl import heuristic_action


def test_v4_fixed_dimensions_and_actions():
    env=ResourceMARLYardEnv(seed=1,arrival_rate_per_hour=20.0)
    assert (N_BLOCKS,STACKS_PER_BLOCK,MAX_TIER,TOTAL_SLOTS)==(4,25,4,400)
    assert env.storage_action_dim==100
    assert env.storage_obs_dim==436
    assert env.global_obs_dim==440
    assert YC_PAIR_COUNT==2500 and YC_ACTION_DIM==2502
    a=encode_proactive_action(99,24)
    assert decode_proactive_action(a)==(99,24)


def test_initial_state_exactly_268_and_67_per_block():
    sim=build_resource_marl_scenario(7,20.0)
    assert sum(len(st) for b in sim.yard.stacks for st in b)==INITIAL_CONTAINERS
    assert [sum(len(st) for st in sim.yard.stacks[b]) for b in range(4)]==[67]*4
    assert all(len(st)<=4 for b in sim.yard.stacks for st in b)




def test_future_pickup_proxy_is_multiday_right_skewed_and_bounded():
    import random
    rng=random.Random(20260921)
    hours=np.asarray([_sample_future_pickup_delay_min(rng)/60.0 for _ in range(10000)])
    assert hours.min()>=24.0 and hours.max()<=168.0
    assert 68.0 <= float(np.median(hours)) <= 76.0
    assert float(hours.mean()) > float(np.median(hours))

def test_new_inbound_never_retrievable_same_episode():
    sim=build_resource_marl_scenario(11,20.0)
    new=[c for c in sim.containers.values() if c.cohort=='new_inbound']
    assert new and all(not c.retrievable_this_episode for c in new)
    assert all(c.truck_actual>OPERATING_HORIZON_MIN for c in new)
    assert all(24.0 <= (c.truck_actual-c.yard_arrival)/60.0 <= 168.0 for c in new)


def test_same_seed_same_exogenous_stream():
    a=build_resource_marl_scenario(19,20.0);b=build_resource_marl_scenario(19,20.0)
    sig=lambda s:[(round(e.time,8),e.kind,e.payload.get('cid'),round(float(e.payload.get('eta',0.0)),8)) for e in sorted(s.events)]
    assert sig(a)==sig(b)
    assert [[list(st) for st in block] for block in a.yard.stacks]==[[list(st) for st in block] for block in b.yard.stacks]




def test_three_relocation_slack_slots_preserved_per_block_for_storage_admission():
    sim=build_resource_marl_scenario(29,20.0)
    assert sim.yard.relocation_buffer_slots_per_block==RELOCATION_SLACK_SLOTS_PER_BLOCK==3
    # Fill block 0 through the normal storage-admission capacity (97/100).
    # Physical capacity is still 100; three slots remain unavailable to new storage.
    next_id=0
    while sum(len(st) for st in sim.yard.stacks[0]) < 97:
        cand=[(b,s) for b,s in sim.yard.feasible_storage_slots() if b==0]
        assert cand
        b,st=cand[0]
        cid=f"SLACK_{next_id}"; next_id+=1
        from yard_simulator import Container
        c=Container(cid=cid,yard_arrival=0.0,truck_actual=9999.0,eta=9999.0,prev_eta=9999.0,cohort='test',block=b,stack=st,stored=True)
        sim.containers[cid]=c;sim.yard.push(b,st,cid)
    assert sum(len(st) for st in sim.yard.stacks[0])==97
    assert not any(b==0 for b,_ in sim.yard.feasible_storage_slots())
    assert 100-sum(len(st) for st in sim.yard.stacks[0])==3

def test_capacity_slack_is_not_proactive_hard_mask():
    sim=build_resource_marl_scenario(23,20.0)
    # Find a blocked retrieval target and force its observed ETA close enough to
    # create negative slack while keeping it physically proactive-eligible.
    target=None
    for c in sim.containers.values():
        if c.retrievable_this_episode and c.stored and sim.yard.blockers_above(c.cid,sim.containers)>0:
            target=c;break
    assert target is not None
    target.eta=sim.now+1.0;target.prev_eta=target.eta
    block=target.block
    # Add mandatory work to make slack negative.
    from yard_simulator import Task
    for _ in range(3): sim.ycs[block].queue.append(Task('STORAGE',target.cid,sim.move_time,created_at=sim.now))
    assert sim.target_capacity_slack(target.cid)<sim.move_time
    actions=sim.proactive_pair_actions(block)
    assert actions  # still feasible because slack is observation-only




def test_group_normalized_proactive_mass_is_invariant_to_pair_count():
    env=ResourceMARLYardEnv(seed=5,arrival_rate_per_hour=20.0);env.reset()
    model=ResourceCooperativeModel(env.global_obs_dim,env.storage_obs_dim,env.yc_obs_dim,hidden=32)
    cfg=ResourcePPOConfig()
    assert np.isclose(cfg.yc_proactive_init_bias,-2.197224577)
    assert np.isclose(cfg.yc_op_ent_coef,0.001)
    assert np.isclose(cfg.yc_pair_ent_coef,0.0001)
    assert np.isclose(cfg.yc_pair_init_std,0.05)
    initialize_conservative_yc_actor(model,cfg.yc_proactive_init_bias,cfg.yc_pair_init_std)
    obs=torch.zeros(env.yc_obs_dim,dtype=torch.float32)
    masses=[]
    for n_pairs in (10,150,500):
        mask=torch.zeros(YC_ACTION_DIM,dtype=torch.bool)
        mask[YC_MANDATORY]=True
        mask[YC_PROACTIVE_BASE:YC_PROACTIVE_BASE+n_pairs]=True
        logits=model.yc_logits(obs,mask)
        dist=masked_distribution(logits,mask)
        masses.append(float(dist.probs[YC_PROACTIVE_BASE:].sum().item()))
        pair_cond=dist.probs[YC_PROACTIVE_BASE:YC_PROACTIVE_BASE+n_pairs]/dist.probs[YC_PROACTIVE_BASE:].sum()
        assert torch.isclose(pair_cond.sum(),torch.tensor(1.0),atol=1e-6)
    assert max(masses)-min(masses)<1e-6
    assert all(0.095<=m<=0.105 for m in masses)


def test_mandatory_logprob_has_zero_gradient_on_pair_ranking():
    env=ResourceMARLYardEnv(seed=6,arrival_rate_per_hour=20.0);env.reset()
    model=ResourceCooperativeModel(env.global_obs_dim,env.storage_obs_dim,env.yc_obs_dim,hidden=16)
    initialize_conservative_yc_actor(model,-2.197224577,0.05)
    # Move to a YC decision.
    for _ in range(1000):
        if env.active_agent()!='storage': break
        env.step(int(np.flatnonzero(env.action_mask())[0]))
    assert env.active_agent()!='storage'
    obs=torch.tensor(env.actor_observation(),dtype=torch.float32)
    mask=torch.tensor(env.action_mask(),dtype=torch.bool)
    if not bool(mask[YC_MANDATORY]) or int(mask[YC_PROACTIVE_BASE:].sum())==0:
        # Synthetic mask is enough to test the mathematical decomposition.
        mask=torch.zeros(YC_ACTION_DIM,dtype=torch.bool);mask[YC_MANDATORY]=True;mask[YC_PROACTIVE_BASE:YC_PROACTIVE_BASE+20]=True
    logits=model.yc_logits(obs,mask);dist=masked_distribution(logits,mask)
    loss=-dist.log_prob(torch.tensor(YC_MANDATORY))
    model.zero_grad();loss.backward()
    grad=model.yc_actor.pair_scorer.weight.grad
    assert grad is not None
    assert float(grad.abs().max())<1e-7


def test_structured_entropy_is_pair_count_normalized():
    env=ResourceMARLYardEnv(seed=8,arrival_rate_per_hour=20.0);env.reset()
    model=ResourceCooperativeModel(env.global_obs_dim,env.storage_obs_dim,env.yc_obs_dim,hidden=16)
    initialize_conservative_yc_actor(model,-2.197224577,0.05)
    obs=torch.zeros(env.yc_obs_dim,dtype=torch.float32)
    vals=[]
    for n_pairs in (10,100,500):
        mask=torch.zeros(YC_ACTION_DIM,dtype=torch.bool);mask[YC_MANDATORY]=True;mask[YC_PROACTIVE_BASE:YC_PROACTIVE_BASE+n_pairs]=True
        dist=masked_distribution(model.yc_logits(obs,mask),mask)
        op_h,pair_h,p=structured_yc_entropy(dist,mask)
        vals.append(float(pair_h.item()))
        assert 0.095<=float(p.item())<=0.105
    assert all(0.0 <= v <= 1.0 + 1e-6 for v in vals)
    assert max(vals)-min(vals)<0.03


def test_model_heads_match_100_and_2502_actions():
    env=ResourceMARLYardEnv(seed=3,arrival_rate_per_hour=20.0);env.reset()
    marl=ResourceCooperativeModel(env.global_obs_dim,env.storage_obs_dim,env.yc_obs_dim,hidden=32)
    single=CentralizedSingleModel(env.global_obs_dim,env.yc_obs_dim,hidden=32)
    for _ in range(500):
        if env.active_agent()=='storage':
            assert marl.storage_logits(torch.tensor(env.actor_observation())).shape==(100,)
            assert single.storage_logits(torch.tensor(env.centralized_actor_observation())).shape==(100,)
        else:
            mask=torch.tensor(env.action_mask(),dtype=torch.bool)
            assert marl.yc_logits(torch.tensor(env.actor_observation()),mask).shape==(2502,)
            assert single.yc_logits(torch.tensor(env.centralized_actor_observation()),mask).shape==(2502,)
            break
        env.step(int(np.flatnonzero(env.action_mask())[0]))


def test_heuristic_episode_drains_all_mandatory_work():
    env=ResourceMARLYardEnv(seed=31,arrival_rate_per_hour=20.0);_,info=env.reset();done=False
    for _ in range(100000):
        _,_,done,_,info=env.step(heuristic_action(env))
        if done:break
    assert done
    k=info['kpis']
    assert k['unretrieved_containers']==0.0
    assert k['unstored_inbound_containers']==0.0
    assert k['episode_end_time']>=OPERATING_HORIZON_MIN
    assert all(not yc.busy and len(yc.queue)==0 for yc in env.sim.ycs)


def test_reward_normalization_uses_nt_ns_and_extra_minutes():
    env=ResourceMARLYardEnv(seed=37,arrival_rate_per_hour=20.0,extra_move_weight=0.1)
    _,info=env.reset();before=env.sim.cost_snapshot();action=int(np.flatnonzero(info['action_mask'])[0])
    _,r,_,_,info=env.step(action);after=env.sim.cost_snapshot();nt=max(env.sim.n_retrieval_requests,1);ns=max(env.sim.n_storage_requests,1)
    expected=-(after[0]-before[0])/nt-(after[1]-before[1])/ns-0.1*(after[2]-before[2])*YC_MOVE_TIME_MIN/nt
    assert np.isclose(info['reward_components']['team'],expected)


def test_single_baseline_uses_groupnorm_and_mask_signature():
    env=ResourceMARLYardEnv(seed=41,arrival_rate_per_hour=20.0);env.reset()
    single=CentralizedSingleModel(env.global_obs_dim,env.yc_obs_dim,hidden=32)
    from train_yc_single import initialize_conservative_yc_head
    initialize_conservative_yc_head(single,-2.197224577,0.05)
    # Reach a YC decision and verify the proactive group mass does not depend on pair count.
    obs=torch.zeros(env.global_obs_dim+env.yc_obs_dim,dtype=torch.float32)
    masses=[]
    for n_pairs in (10,150,500):
        mask=torch.zeros(YC_ACTION_DIM,dtype=torch.bool);mask[YC_MANDATORY]=True;mask[YC_PROACTIVE_BASE:YC_PROACTIVE_BASE+n_pairs]=True
        dist=masked_distribution(single.yc_logits(obs,mask),mask)
        masses.append(float(dist.probs[YC_PROACTIVE_BASE:].sum().detach()))
    assert max(masses)-min(masses)<1e-6
    assert all(0.095<=m<=0.105 for m in masses)


def test_completed_mc_diagnostics_distinguishes_cutoff_tail():
    from train_yc_marl import completed_episode_mc_diagnostics
    rewards=np.asarray([-1.0,-2.0,-3.0,-4.0],dtype=np.float32)
    dones=np.asarray([0.0,1.0,0.0,0.0],dtype=np.float32)
    values=np.zeros(4,dtype=np.float32)
    d=completed_episode_mc_diagnostics(rewards,dones,values)
    assert d['mc_n']==2
    assert np.isclose(d['mc_completed_fraction'],0.5)
