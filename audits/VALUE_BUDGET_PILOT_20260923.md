# Value-budget pilot result — 2026-09-23

Status: **critic mechanism PASS; policy/pair gates FAIL; Full 30k remains No-Go.**

## Exact execution provenance

The pilot was reproduced from the exact GitHub branch code by GitHub Actions.

- Repository: `gilmour23/ACK2026-yard-MARL`
- Branch: `experiment/critic-diagnostic`
- Execution commit: `2d65d5e67ad02bea8dd480d0607fc7cd9f6acf0b`
- Workflow run: `35777547765`
- Workflow result: **success**
- Artifact: `value-budget-pilot-20260923`
- Artifact ID: `10717004897`
- Artifact size: 27,278,618 bytes
- Artifact ZIP SHA-256 reported by GitHub Actions: `1b2d3ad5a2265c3c18cc115f795af6c26635eebd511a0373c1e0f939cc763679`
- Canonical checkpoint SHA-256: `2bbd3a2e795a215d3fcbd654d58fad154d4298b55bb50ce92f54ad50b82ddb59`
- `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`

The exact GitHub Actions artifact reproduces the same numerical results as the independent local run.

## Protocol

- Training seeds: 21, 22, 23
- Validation scenarios: 701–710
- Evaluation: stochastic, 3 repeats/scenario
- Control: episode-complete PPO, 2 joint actor+critic epochs, 0 extra critic epochs
- Treatment: same control plus 18 critic-only epochs per rollout (20 value exposures total)
- rollout_steps=512, minibatch_size=256, lr=3e-4, max_grad_norm=0.5
- No reward, actor architecture, critic architecture, environment, gamma/lambda, or entropy change.

## Held-out critic result

| Seed | EV control | EV treatment | RMSE control | RMSE treatment |
| --- | ---: | ---: | ---: | ---: |
| 21 | 0.0238 | 0.9228 | 3.3696 | 1.3783 |
| 22 | 0.0210 | 0.8789 | 3.5494 | 1.5381 |
| 23 | 0.0247 | 0.9265 | 3.4734 | 1.1690 |

The pre-registered critic mechanism gate passes in all three seeds.

## Policy objective

| Seed | J control | J treatment | treatment change |
| --- | ---: | ---: | ---: |
| 21 | 10.3116 | 10.4737 | +1.57% |
| 22 | 11.2434 | 10.9303 | -2.79% |
| 23 | 10.6730 | 10.7797 | +1.00% |

Seed-mean:
- control J = 10.7427
- treatment J = 10.7279
- relative change = -0.138%

Only 1/3 training seeds improves. Scenario-level paired bootstrap (10,000 resamples, RNG seed 20260922) gives treatment-control mean difference -0.0148 with 95% CI [-0.1226, +0.0956]. The policy gate therefore fails.

## Pair policy

Treatment conditional-pair TV(uniform):
- seed21: 0.0073
- seed22: 0.0138
- seed23: 0.0140

Treatment normalized pair entropy remains about 0.9999. The pre-registered TV>=0.05 gate fails in all three seeds.

The exact workflow artifact also reports expected Target ETA lead, destination height, and destination inversion extremely close to the uniform-feasible-pair baseline. For treatment, expected ETA minus uniform is only +0.0166, +0.0590, and -0.0675 minutes for seeds 21–23.

## Interpretation

Increasing critic update exposure fixes the coarse value-function underfitting mechanism, but it does **not** make the Target×Destination actor learn a useful non-uniform pair policy within the 2k pilot. Critic undertraining was real, but it was not the sole cause of pair-credit failure.

Do not increase critic epochs further as the next main intervention.

## Next minimal diagnostic

Keep value20 and the existing actor/critic/reward/environment fixed. Instrument only the actor pair-learning signal during PPO:

- proactive-sample advantage distribution;
- advantage versus chosen target/destination features;
- pair-scorer gradient norm versus operation-head gradient norm;
- pair-logit standard-deviation change per update;
- conditional-pair KL / TV movement per update;
- correlation between pair-score update direction and realized terminal-complete MC return.

Do this before changing actor epochs, entropy, architecture, reward, heuristic top-K, ETA bonus, YC BC, or Q critic.

## Full 30k

**No-Go.** The critic gate passes, but both the policy-objective gate and pair-learning gate fail.

## Artifacts

Large checkpoints and exact workflow artifacts are stored in Google Drive under:

`05_ASTRA_AUDITS/20260923_value_budget_pilot/`.
