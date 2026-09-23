# Actor pair-signal diagnostic — 2026-09-23

Status: **pair gradient path is live, but conditional Target×Destination credit is weak.**

## Provenance

- Workflow run: `35810176232`
- Branch commit: `18060b306d23bb9ca5f022c6c19671cb001dd314`
- Arms: value2 control / value20 treatment
- Seeds: 21, 22, 23
- Read-only instrumentation; no reward, actor, critic, environment, or PPO hyperparameter change.

## What was measured

At each episode-complete rollout:
- proactive-sample advantage distribution;
- operation-head, pair-path, and pair-scorer gradient norms;
- conditional pair KL/TV movement before vs after PPO actor updates;
- selected pair-score change vs advantage / return;
- advantage correlations with target/destination pair features.

## Rollout-2 aggregate

Rollout 2 is the informative comparison because value20 has already received the extra critic-only passes from rollout 1.

| Metric | value2 | value20 |
| --- | ---: | ---: |
| critic EV before update | 0.00187 | **0.81114** |
| pair TV(uniform), pre | 0.01473 | 0.01473 |
| pair distribution movement TV/update | 0.00205 | **0.00162** |
| pair distribution KL/update | 0.000016 | **0.000010** |
| operation-head grad L2 | 0.06285 | 0.05920 |
| pair-path grad L2 | 0.00556 | 0.00564 |
| pair-scorer grad L2 | 0.00466 | 0.00474 |
| corr(pair-score delta, raw advantage) | 0.191 | 0.098 |
| corr(pair-score delta, return) | 0.191 | 0.152 |

The pair path receives non-zero gradients, so this is not a disconnected-head bug. However, pair gradients are roughly an order of magnitude smaller than the operation-head gradient and produce very little movement in the conditional pair distribution.

## Feature-level signal under value20

Mean rollout-2 correlation between proactive raw advantage and selected-pair features:

- target ETA lead: -0.055
- ETA advance: +0.031
- blockers: -0.084
- capacity slack: -0.056
- moving blocker ETA: +0.199
- destination height: +0.031
- destination inversion: +0.093
- destination nearest ETA: +0.070

These correlations are weak and unstable across seeds, especially for destination-specific features. Critic recovery therefore does not reveal a strong, consistent Target×Destination learning signal.

## Interpretation

The coarse value critic was a real failure mechanism, but it was not the only reason the flat 2500-pair policy stayed near-uniform. Once value20 repairs coarse EV, the actor still sees weak action-specific separation among feasible proactive pairs.

This motivates testing a simpler action formulation rather than forcing the same flat pair head with more epochs, entropy pressure, or reward tuning.
