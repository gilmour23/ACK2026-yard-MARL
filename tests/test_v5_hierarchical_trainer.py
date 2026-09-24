from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from v5.train_hierarchical_marl import HierarchicalPPOConfig


def test_v5_trainer_frozen_core_settings():
    cfg=HierarchicalPPOConfig()
    assert cfg.total_steps==6000
    assert cfg.update_epochs==2
    assert cfg.critic_extra_epochs==18
    assert cfg.gamma==1.0
    assert cfg.gae_lambda==1.0
    assert cfg.clip_coef==0.20
    assert cfg.learning_rate==3e-4
    assert cfg.extra_move_weight==0.10
    assert cfg.episode_complete_rollout is True
    assert cfg.include_resource_state is True
    assert cfg.yc_proactive_init_bias < 0
