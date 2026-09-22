from __future__ import annotations

"""Minimal A/B pilot requested by the independent Astra audit.

A: current cutoff rollout.
B: once the rollout threshold is reached, continue to the current episode terminal
   before updating.  Actor, critic, reward and PPO hyperparameters are otherwise
   unchanged.

This script intentionally does NOT add ETA bonuses, target heuristics, top-K action
pruning, YC behavior cloning, or an action-conditioned critic.
"""

from pathlib import Path
import argparse, csv, json
import numpy as np

from train_yc_marl import ResourcePPOConfig, train_resource_marl
from evaluate_yc_policies import evaluate


def _mean_finite(rows, key):
    vals=[float(r[key]) for r in rows if key in r and np.isfinite(float(r[key]))]
    return float(np.mean(vals)) if vals else float('nan')


def run(checkpoint: Path, out: Path, steps: int = 2000, train_seeds=(21,22,23), eval_seeds=range(401,411), repeats: int = 3):
    checkpoint=Path(checkpoint); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    results={}
    for episode_complete in (False, True):
        label='B_episode_complete' if episode_complete else 'A_cutoff'
        results[label]={}
        for seed in train_seeds:
            run_dir=out/label/f'seed{seed}'
            cfg=ResourcePPOConfig(
                total_steps=steps,
                rollout_steps=512,
                min_storage_transitions=64,
                min_yc_transitions=128,
                max_rollout_multiplier=4,
                update_epochs=2,
                minibatch_size=256,
                gamma=1.0,
                gae_lambda=1.0,
                seed=int(seed),
                arrival_rate_per_hour=20.0,
                episode_complete_rollout=episode_complete,
            )
            _, updates=train_resource_marl(cfg,run_dir,checkpoint)
            eval_csv=run_dir/'eval_stochastic.csv'
            eval_summary=evaluate(
                eval_csv,'marl',checkpoint=run_dir/'resource_marl_final.pt',
                seeds=eval_seeds,repeats=repeats,arrival_rate_per_hour=20.0,stochastic=True,
            )
            train_summary={
                'actual_decisions':int(updates[-1]['global_step']),
                'n_updates':len(updates),
                'mc_value_ev_mean':_mean_finite(updates,'mc_value_ev'),
                'mc_value_rmse_mean':_mean_finite(updates,'mc_value_rmse'),
                'mc_completed_fraction_mean':_mean_finite(updates,'mc_completed_fraction'),
                'yc_pair_entropy_norm_mean':_mean_finite(updates,'yc_pair_entropy_norm'),
                'yc_proactive_probability_mean':_mean_finite(updates,'yc_proactive_probability'),
                'yc_sampled_proactive_rate_mean':_mean_finite(updates,'yc_sampled_proactive_rate'),
            }
            results[label][str(seed)]={'training':train_summary,'evaluation':eval_summary}
    (out/'pilot_summary.json').write_text(json.dumps(results,indent=2,allow_nan=True),encoding='utf-8')
    return results


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--out',default='credit_assignment_pilot')
    ap.add_argument('--steps',type=int,default=2000)
    ap.add_argument('--train-seeds',type=int,nargs='+',default=[21,22,23])
    ap.add_argument('--eval-start',type=int,default=401)
    ap.add_argument('--eval-count',type=int,default=10)
    ap.add_argument('--repeats',type=int,default=3)
    a=ap.parse_args()
    run(Path(a.checkpoint),Path(a.out),a.steps,tuple(a.train_seeds),range(a.eval_start,a.eval_start+a.eval_count),a.repeats)
