import numpy as np

from yc_marl_env import (
    ResourceDecision,
    build_resource_marl_scenario,
    decode_proactive_action,
    YC_PROACTIVE_BASE,
)


def test_rule_resolved_proactive_support_is_single_and_matches_existing_heuristics():
    found=False
    for seed in range(1, 101):
        sim=build_resource_marl_scenario(
            seed,
            20.0,
            rule_resolve_proactive_pair=True,
        )
        for block in range(sim.yard.n_blocks):
            all_pairs=sim.proactive_pair_actions(block)
            if not all_pairs:
                continue
            chosen=sim.rule_resolved_proactive_action(block)
            assert chosen is not None
            assert chosen in all_pairs

            target_pos,dest=decode_proactive_action(chosen)
            target=sim.proactive_candidates(block)[0]
            assert target_pos == sim._target_position(target)

            c=sim.containers[target]
            assert c.stack is not None
            moving=sim.yard.top(block,c.stack)
            assert moving is not None and moving != target
            expected_dest=sim.policy.choose_relocation_destination(
                sim,moving,block,c.stack
            )
            assert dest == expected_dest

            sim.pending_decision=ResourceDecision("yc",block=block)
            mask=sim.yc_action_mask(block)
            proactive=np.flatnonzero(mask[YC_PROACTIVE_BASE:]) + YC_PROACTIVE_BASE
            assert proactive.tolist() == [chosen]
            found=True
            break
        if found:
            break
    assert found, "No proactive state found in deterministic seed search"
