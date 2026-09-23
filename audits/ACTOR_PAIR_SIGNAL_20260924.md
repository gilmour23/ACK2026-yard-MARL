# Actor pair-signal diagnostic result — 2026-09-24

Status: **pair gradient path is alive, but pair-credit signal is weak/unstable; Full 30k remains No-Go.**

## Evidence source

GitHub Actions run `35810176232` on `experiment/critic-diagnostic`, commit `18060b306d23bb9ca5f022c6c19671cb001dd314`.

Matrix:
- control: episode-complete value2
- treatment: episode-complete value20
- seeds 21, 22, 23
- same canonical checkpoint SHA-256 `2bbd3a2e795a215d3fcbd654d58fad154d4298b55bb50ce92f54ad50b82ddb59`

Instrumentation did not consume random numbers and did not change actor/reward/environment.

## 1. Pair gradient path is not broken

Across the six control/treatment rollouts, the YC pair path receives non-zero gradients.

Representative rollout-2 values:

| Seed | Arm | Critic EV before update | Operation-head grad | Pair-path grad | Pair/Operation |
|---|---|---:|---:|---:|---:|
| 21 | value2 | 0.0022 | 0.0562 | 0.00492 | 0.0876 |
| 21 | value20 | 0.8653 | 0.0381 | 0.00502 | 0.1318 |
| 22 | value2 | 0.0019 | 0.0641 | 0.00566 | 0.0882 |
| 22 | value20 | 0.8093 | 0.0733 | 0.00620 | 0.0846 |
| 23 | value2 | 0.0015 | 0.0682 | 0.00609 | 0.0893 |
| 23 | value20 | 0.7588 | 0.0663 | 0.00570 | 0.0861 |

Therefore the near-uniform pair policy is not explained by a detached pair scorer or zero-gradient implementation bug.

## 2. Better critic changes coarse proactive advantage, but not pair learning

In rollout 2, the normalized mean advantage of sampled proactive actions changes from negative under value2 to slightly positive under value20:

| Seed | value2 | value20 |
|---|---:|---:|
| 21 | -0.2249 | +0.0561 |
| 22 | -0.0795 | +0.0381 |
| 23 | -0.0956 | +0.0330 |

This is consistent with the previous value-budget result: critic undertraining was real and materially affected operation-level credit.

However, conditional pair movement remains tiny. Rollout-2 conditional pre→post TV is only:

- seed21 value20: 0.00352
- seed22 value20: 0.00050
- seed23 value20: 0.00085

Post-update conditional TV from uniform remains only 0.0083, 0.0177, 0.0164 respectively, and normalized pair entropy remains approximately 1.

## 3. Pair update direction is not stably aligned with realized return

For value20 rollout 2, correlation between selected-pair score change and realized return is:

- seed21: +0.148
- seed22: +0.095
- seed23: +0.213

These are weak. Under value2 the sign is not stable across seeds.

The actor therefore receives a pair gradient, but the update direction is not strongly or consistently associated with which sampled pair eventually produced the better terminal-complete return.

## 4. Interpretation

Current evidence separates two failures:

1. **Coarse value failure:** largely repaired by value20.
2. **Conditional Target×Destination credit:** still unresolved.

The remaining pair problem is not a simple critic-EV problem and is not a zero-gradient implementation bug. The plausible remaining mechanisms are:

- within-state returns may differ very little among feasible proactive pairs under the current objective;
- pair-specific return differences may exist, but a state-value GAE advantage is too noisy to identify them from sparse on-policy pair samples;
- the 2,500-pair formulation may be over-granular relative to the operational signal, especially for destination choice.

Do **not** respond by merely increasing critic epochs, actor epochs, or entropy.

## 5. Next diagnostic

Before enabling the legacy action-conditioned Q critic or changing the action formulation, run a same-state counterfactual sensitivity test:

> clone the same YC state, force different feasible Target×Destination pairs, then compare terminal return under a fixed continuation policy.

If same-state return spread is small, simplify/factor the pair action. If spread is material but actor score does not align, action-specific credit estimation becomes the justified next intervention.

## Full 30k

**No-Go.**
