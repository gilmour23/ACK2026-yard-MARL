# Value-budget pilot result — 2026-09-23

Status: **critic mechanism PASS; policy/pair gates FAIL; Full 30k remains No-Go.**

## Protocol

- Canonical checkpoint SHA-256: `2bbd3a2e795a215d3fcbd654d58fad154d4298b55bb50ce92f54ad50b82ddb59`
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

A separate shared 500-state diagnostic also found expected Target ETA lead, destination height, and destination inversion to be extremely close to the uniform-feasible-pair baseline.

## Interpretation

Increasing critic update exposure fixes the coarse value-function underfitting mechanism, but it does **not** make the Target×Destination actor learn a useful non-uniform pair policy within the 2k pilot. Critic undertraining was real, but it was not the sole cause of pair-credit failure.

Do not increase critic epochs further as the next main intervention. The next diagnostic should keep value20 and the existing actor/critic/reward/environment fixed, and instrument the actor pair-learning signal: proactive-sample advantage distribution, pair-scorer vs operation-head gradient norms, conditional-pair KL/probability movement, pair-logit spread, and relation to realized terminal-complete return.

## Provenance caveat

The local execution copy used for this pilot had byte-identical `yc_marl_env.py`, `v4_networks.py`, and `evaluate_yc_policies.py` relative to the branch blobs, and mirrored the branch's value-budget modification in `train_yc_marl.py`. The private GitHub connector did not provide a direct repository-to-runtime materialization path, so the trainer file used locally was reconstructed rather than byte-for-byte materialized. The exact branch code passed CI. Before promotion to `main`, Astra should independently spot-check or rerun the pilot from this branch.

Large checkpoints and result artifacts are stored in Google Drive under:
`05_ASTRA_AUDITS/20260923_value_budget_pilot/`.
