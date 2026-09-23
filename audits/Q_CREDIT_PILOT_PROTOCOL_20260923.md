# Q-credit pilot protocol — 2026-09-23

Status before execution: **Full 30k No-Go**.

## Rationale

The value-budget pilot showed that episode-complete value20 training fixes coarse critic underfitting (held-out EV about 0.88–0.93), but the conditional Target×Destination policy remains almost uniform and seed-mean J does not materially improve. The read-only actor pair-signal diagnostic then showed:

- pair-scorer gradients are non-zero, so the pair path is not disconnected;
- conditional-pair distribution movement per PPO update is very small;
- after critic recovery, selected-pair score changes have only weak/unstable correlation with realized return;
- destination-level return correlations are weak compared with the operation/state-level signal.

The next intervention therefore targets **action-specific pair credit**, not critic capacity, reward, entropy, or actor epoch count.

## Single change

Keep the value20 episode-complete configuration fixed and toggle the already-existing action-conditioned YC Q-critic path.

### Control
- `episode_complete_rollout=True`
- `update_epochs=2`
- `critic_extra_epochs=18`
- `use_action_q_critic=False`

### Treatment
Same as control, plus:
- `use_action_q_critic=True`
- `q_update_epochs=3`
- `q_adv_blend=0.5`

No other training or environment parameter changes.

## Fixed settings

- canonical init checkpoint SHA-256: `2bbd3a2e795a215d3fcbd654d58fad154d4298b55bb50ce92f54ad50b82ddb59`
- training seeds: 21, 22, 23
- decision threshold: 2,000, then finish the current episode
- rollout_steps: 512
- minibatch_size: 256
- learning_rate: 3e-4
- max_grad_norm: 0.5
- gamma=lambda=1
- reward/environment/actor/critic architecture unchanged
- stochastic evaluation

## Diagnostic evaluation bank

Use scenarios **801–810**, 3 repeats per scenario. This bank was not used for the earlier 701–710 value-budget decision. It is still a diagnostic/validation bank, **not** the final unseen test set.

Reserve 901+ for later final testing.

## Pre-registered gates

### Safety / integrity
All seeds must:
- terminate normally with episode-complete coverage;
- produce finite policy/value/Q losses and finite evaluation metrics;
- preserve canonical checkpoint lineage and fixed settings.

### Pair-credit mechanism gate
Treatment must show meaningful non-uniform conditional pair behavior:
- evaluation conditional-pair TV(uniform) >= 0.05 in at least 2/3 seeds;
- and treatment TV must exceed the paired control TV for those seeds.

### Policy gate
Using J = truck delay + storage delay + 0.1 * extra YC minutes/retrieval:
- seed-mean treatment J must be at least 2% lower than control;
- at least 2/3 seeds must improve;
- scenario-level paired bootstrap, 10,000 resamples with RNG seed 20260923, must have the 95% CI upper bound for treatment-control below 0.

Passing only the pair gate is not enough. Passing only J is not enough.

## Decision rule

- If safety fails: reject the pilot result and fix instrumentation/runtime only.
- If Q-credit mechanism and policy gates pass: Q-credit becomes the candidate actor-credit mechanism for a longer bounded pilot, **not** automatic 30k Go.
- If the Q-credit mechanism gate fails: do not increase actor epochs or lower pair entropy as the next response; revisit action/credit formulation.
- If mechanism passes but policy fails: inspect whether the learned pair preference targets the wrong operational features before any reward change.

**Full 30k remains No-Go regardless of this 2k pilot until a separate Go review.**
