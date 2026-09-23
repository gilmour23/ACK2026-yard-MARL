# ACK2026 final experiment protocol — 2026-09-24

Status before execution: **pre-registered; execute only after FINAL_MATRIX_FAIRNESS_SMOKE integrity PASS.**

## 1. Research comparison

The final experiment fixes the YC formulation to the validated hybrid design:

- Storage Allocation is learned.
- YC Operation is learned.
- If Proactive is selected, Target×Destination is resolved by the same transparent information-aware rule for every learned policy.
- The rule resolver is not described as learned or globally optimal.

Final policies:

1. **Information-aware heuristic** — non-learning operational baseline.
2. **Centralized Single PPO** — one centralized controller with global state; same rule-resolved proactive support.
3. **MARL w/o Resource State** — same cooperative MARL and same rule resolver, but resource-state features removed from decentralized actor observations.
4. **Proposed Cooperative MARL** — decentralized Storage/YC actors with resource-state information, CTDE, and rule-resolved proactive moves.

The historical flat 2,500-pair PPO and Q-credit variants are diagnostic evidence, not final primary baselines.

## 2. Fair initialization

All learned policies start from **Storage-only behavior cloning**, not from the historical 12k policy checkpoint.

- BC data scenario seeds: 1000–1007
- BC epochs: 3
- BC batch size: 128
- BC learning rate: 5e-4
- BC RNG seed: 20260924
- YC policy is not behavior cloned.
- Each architecture receives its matching Storage-only BC checkpoint.
- MARL w/o Resource State is BC-trained with resource-state features disabled.

The historical 12k checkpoint is retained only as audit lineage and is not used to initialize the final comparison.

## 3. Common RL training

Training seeds: **51, 52, 53**.

For every learned arm:

- decision threshold = 30,000, then finish the current episode;
- episode-complete rollout;
- rollout_steps = 512;
- actor+critic PPO epochs = 2;
- critic-only extra epochs = 18 (value20);
- minibatch = 256;
- learning rate = 3e-4;
- clip = 0.20;
- max_grad_norm = 0.5;
- gamma = 1.0;
- GAE lambda = 1.0;
- Q critic OFF;
- rule-resolved proactive pair ON;
- reward unchanged;
- environment unchanged;
- no YC behavior cloning;
- no positive proactive shaping.

## 4. Final untouched evaluation bank

Final scenarios: **901–930**.

- Learned policies: 3 stochastic repeats per scenario and training seed.
- Heuristic: deterministic once per scenario.
- Policy seed for learned evaluation: deterministic function of scenario, repeat, and training seed.
- Scenarios 901–930 must not be used for model selection, hyperparameter changes, or rerunning a failed design.
- After these results are opened, changes create a new study/version rather than replacing this final test.

Previously consumed diagnostic banks remain non-final:
601–610, 701–710, 801–810, 851–860, 871–880, 881–885.

## 5. Outcomes

Training objective proxy:

`J = mean truck completion delay + mean storage completion delay + 0.1 × extra YC minutes/retrieval`.

Report, without replacing them by reward alone:

- mean truck completion delay;
- mean storage completion delay;
- reactive rehandling moves;
- proactive moves;
- extra YC minutes/retrieval;
- total YC moves;
- mean YC utilization;
- maximum YC queue;
- J.

## 6. Statistical analysis

For each learned policy, first average stochastic repeats within each (training seed, scenario). Then report:

- seed-level means;
- overall mean and standard deviation across training seeds;
- scenario-level means averaged across the three training seeds.

For Proposed MARL versus each baseline, compute scenario-paired differences on scenarios 901–930. Use 10,000 paired bootstrap resamples with RNG seed 20260924 and report the 95% CI.

No post-hoc success threshold is introduced for the paper. The final result is reported as observed, including if a baseline performs better.

## 7. Integrity requirements

Before accepting a learned run:

- every training rollout ends at terminal under episode-complete collection;
- all losses/KPIs finite;
- critic-only extra epochs have zero actor parameter change;
- treatment action support is rule-resolved as declared;
- checkpoints load under the common evaluator;
- exact config, git commit, seed, and artifact hashes are recorded.

## 8. Safety / branch rule

Execution remains on `experiment/critic-diagnostic` until the final matrix is complete and audited. Do not merge experimental code to `main` before the result and lineage review.
