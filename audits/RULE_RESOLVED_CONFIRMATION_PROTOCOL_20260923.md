# Rule-resolved proactive bounded confirmation protocol — 2026-09-23

Status before execution: **2k structural pilot PASS; Full 30k remains No-Go.**

## Purpose

The 2k pilot on training seeds 21–23 and evaluation scenarios 851–860 showed a
large benefit from simplifying YC action support to one transparent proactive
pair while leaving the RL operation decision intact.

This confirmation tests whether that result persists with:
- a longer bounded training budget;
- completely new training seeds;
- a new diagnostic evaluation bank.

It is not a final test and does not use scenarios 901+.

## Arms

Control:
- current flat 2500 Target×Destination proactive action support;
- episode-complete PPO;
- value20 critic training;
- Q critic OFF.

Treatment:
- identical settings;
- `rule_resolve_proactive_pair=True`;
- target priority = ETA lead -> blockers -> ETA advance;
- destination = existing information-aware inversion + height heuristic.

## Fixed configuration

- canonical init checkpoint SHA-256:
  `2bbd3a2e795a215d3fcbd654d58fad154d4298b55bb50ce92f54ad50b82ddb59`
- training seeds: **31, 32, 33**
- decision threshold: **6,000**, then finish current episode
- rollout_steps=512
- actor+critic epochs=2
- critic-only extra epochs=18
- minibatch_size=256
- learning_rate=3e-4
- max_grad_norm=0.5
- gamma=lambda=1
- reward/environment dynamics/Storage actor/value critic unchanged
- stochastic evaluation

## Confirmation evaluation bank

Scenarios **871–880**, 3 repeats/scenario.

Banks already consumed:
- 701–710: value-budget diagnostic
- 801–810: Q-credit diagnostic
- 851–860: 2k structural pilot

Scenarios **901+ remain untouched and reserved for final testing**.

## Pre-registered confirmation gate

Let
`J = truck delay + storage delay + 0.1 * extra YC minutes/retrieval`.

The simplified formulation is confirmed only if:

1. integrity checks pass for all six runs;
2. treatment seed-mean J is at least **5% lower** than flat control;
3. at least **2/3 training seeds** improve;
4. paired bootstrap over scenario-level treatment-control differences
   (10,000 resamples, RNG seed 20260923) has 95% CI upper bound < 0.

Record truck delay, storage delay, rehandling, proactive count, YC overhead,
utilization, queue, P(Proactive), and value EV as secondary diagnostics.

## Decision rule

- PASS: the rule-resolved formulation becomes the candidate canonical YC action
  formulation. This still does **not** automatically authorize Full 30k; first
  perform a code/lineage review and define the final baseline matrix.
- FAIL: keep Full 30k blocked and investigate whether the 2k result depended on
  short-horizon training or the specific training/evaluation bank.

No reward tuning, Q-credit tuning, entropy tuning, or further pair-head tuning
is permitted inside this confirmation.
