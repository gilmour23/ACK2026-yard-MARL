# ACK2026 final-result independent review — 2026-09-24

Status: **final artifacts are internally consistent; result interpretation requires narrowing. No additional training was performed.**

## 1. Scope and provenance

Final experiment:
- repository: `gilmour23/ACK2026-yard-MARL`
- execution commit: `734c234024ff039fea7a68d114c0993035f39e18`
- GitHub Actions run: `35951014861`
- final scenarios: 901–930
- learned policies: 3 training seeds (51, 52, 53) × 3 stochastic repeats/scenario
- heuristic: deterministic once/scenario
- final learned evaluation episodes: 810
- heuristic evaluation episodes: 30

The workflow completed successfully: all 9 training jobs, 9 learned evaluation jobs, heuristic evaluation, and aggregate job passed.

## 2. Raw-result integrity: PASS

Independent re-aggregation from the 9 learned evaluation CSVs plus the heuristic CSV reproduced every reported seed mean, overall mean/SD, and paired-bootstrap value in `final_summary.json` to floating-point precision (maximum absolute difference approximately `4.44e-16`).

For every learned arm/seed:
- exactly 90 rows were present;
- scenarios were exactly 901–930;
- each scenario had policy seeds `scenario*100 + {0,1,2}` exactly once;
- no duplicate scenario/policy-seed rows were present;
- stochastic evaluation was used;
- rule-resolved proactive support was enabled;
- the resource-state flag matched the declared arm;
- all evaluation checkpoint SHA-256 values matched the corresponding training manifests;
- evaluation/training git commits and config hashes matched.

All 9 training manifests used the same final config hash and execution commit. All rollout records were terminal-complete and critic-only epochs changed no actor parameters.

## 3. Reproduced final objective J

| Method | Mean J | SD across training seeds |
|---|---:|---:|
| Information-aware heuristic | 7.8393 | — |
| Proposed MARL | 8.4359 | 0.2578 |
| Centralized Single PPO | 8.5390 | 0.6802 |
| MARL w/o Resource State | 9.8845 | 0.2744 |

The pre-registered scenario-paired analysis averaged the three trained models within each scenario and then bootstrapped scenarios. Under that fixed-model estimand:
- Proposed − Single PPO = **−0.1030**, 95% CI **[−0.1648, −0.0417]**.
- Proposed − MARL w/o Resource State = **−1.4486**, 95% CI **[−1.5615, −1.3423]**.
- Proposed − Heuristic = **+0.5966**, 95% CI **[+0.4170, +0.7750]**.

These numbers are correctly implemented for the pre-registered estimand.

## 4. Training-seed robustness: Proposed vs Single PPO is not established

The seed-level J values are:

| Seed | Proposed | Single PPO | Proposed − Single |
|---|---:|---:|---:|
| 51 | 8.3685 | 8.0881 | +0.2804 |
| 52 | 8.2185 | 9.3213 | −1.1028 |
| 53 | 8.7208 | 8.2075 | +0.5133 |

Only seed 52 favors Proposed. Seeds 51 and 53 favor Single PPO.

The mean seed-level difference is still −0.1030, but its SD across the three training seeds is **0.8736**, much larger than the mean effect. A diagnostic two-stage bootstrap that resamples both training seeds and scenarios gives an approximate 95% interval of **[−1.07, +0.51]**. A t interval over the three seed means is also very wide (**[−2.27, +2.07]**).

Therefore the paper should not state that Proposed MARL robustly or statistically outperforms Single PPO with respect to training randomness. The defensible wording is:

> For the three trained models and final scenario bank, Proposed MARL had a 1.2% lower mean J than Single PPO under the pre-registered scenario-paired aggregation, but the direction was not consistent across training seeds.

The seed effect is dominant in the Proposed–Single difference matrix: descriptively, about 82% of the observed squared variation is associated with the three training-seed means. This is diagnostic only because n=3.

## 5. Resource-state ablation is substantially more stable

Seed-level Proposed − MARL w/o Resource State differences are:
- seed 51: −1.2557
- seed 52: −1.9526
- seed 53: −1.1376

All 3/3 seeds favor Proposed. A diagnostic two-stage bootstrap gives approximately **[−1.91, −1.09]**, and the three-seed t interval is approximately **[−2.54, −0.35]**.

This supports the strongest result of the final experiment:

> Providing the declared resource-state features to the decentralized actors materially improved performance relative to the otherwise matched MARL ablation.

However, the ablation removes resource-state features from **both Storage and YC decentralized actor observations**. The experiment does not isolate which agent is responsible for the gain. Do not attribute the full improvement specifically to YC scheduling.

## 6. Heuristic comparison

Proposed − heuristic J by training seed:
- seed 51: +0.5292
- seed 52: +0.3792
- seed 53: +0.8815

Thus all three Proposed seeds have higher J than the deterministic heuristic on the same final scenario bank.

The heuristic uses:
- its information-aware storage rule;
- mandatory-first YC operation;
- the same ETA/blocker target ordering and relocation-destination heuristic used by the rule resolver when proactive work is selected.

It also performs far more proactive moves (472.1 vs 121.0 on average) and uses substantially more YC capacity.

It is **not justified** to state that the heuristic wins *because* the extra-YC weight is 0.1. The final experiment does not identify that causal mechanism. A post-hoc fixed-trajectory arithmetic check places the heuristic/Proposed J crossover near an extra-move weight of 0.26, but this does not represent retrained policies and must not be presented as a reward-sensitivity experiment.

## 7. Important hybrid-policy architecture limitation

The final hybrid support exposes at most one proactive Target×Destination pair per YC decision.

The current YC actor computes:
- `operation_logits = operation_head(context)`
- conditional pair scores from the 9-dimensional pair features
- group-normalized flat logits.

With exactly one feasible proactive pair,

`pair_score - logsumexp(pair_score) = 0`.

Therefore:
- the probability mass assigned to Proactive is determined by the operation head and does not depend on the resolved pair feature vector;
- the pair scorer has no influence on action probability in rule-resolved states;
- the selected Target/Destination's detailed ETA, destination height/inversion, etc. are not directly fed to the operation head;
- final Proposed training recorded normalized pair entropy = 0 throughout, as expected for one-pair support.

The operation head still receives an 18-dimensional context with aggregated urgency, blocker, queue, occupancy, and capacity information. The limitation is narrower:

> The learned YC policy decides whether to allocate capacity to proactive work from aggregated block context; it does not evaluate the detailed attributes of the specific rule-resolved move.

This does not invalidate the final result, but it must constrain the paper's model description. The model is a **hybrid operation-level MARL + deterministic move resolver**, not a policy that learns Target/Destination quality.

A future model revision could inject the resolved move's feature vector into the operation head or use a dedicated operation actor. Such a change would require a new study/version and retraining; it is not applied to the frozen final result.

## 8. Storage-only BC reproducibility caveat

Although the final protocol used one fixed BC RNG seed (20260924), the parallel jobs independently rebuilt BC checkpoints. State-dict comparison found small but real numerical differences in the trained storage actor:

- Proposed: seed 51 and 53 BC weights were identical; seed 52 differed, maximum absolute tensor difference about **7.52e-5**.
- MARL w/o Resource State: seed 51 and 52 were identical; seed 53 differed by up to **8.47e-4**.
- Single PPO: seed 52 and 53 were identical; seed 51 differed by up to **1.18e-3**.

Only storage-network tensors differed; all arms reported identical BC training accuracy within architecture. The exact cause is not proven, but independent CPU numerical execution across parallel runners is consistent with the observed small differences.

This is a reproducibility caveat rather than evidence that the final comparison is invalid. However, training-seed variability also contains this small BC numerical variation.

For any new study/version, build each architecture-matched BC checkpoint **once**, hash it, and fan the same artifact out to all training-seed jobs.

## 9. Training scenario parity

Within each training seed, Proposed and Single PPO consumed the exact same 26 scenario-seed sequence. The no-resource arm consumed the same common prefix and, for seeds 51 and 53, one additional terminal-complete episode because of different episode lengths.

Thus the Proposed–Single seed comparison is not explained by different exogenous training scenario sequences.

## 10. Post-run aggregation code issues and fixes

The original final aggregation code had two validation gaps:
1. duplicate learned arm/seed manifests could silently overwrite one another;
2. duplicated stochastic policy-seed rows could pass if row counts remained unchanged.

It also did not cross-check evaluation checkpoint hashes against training manifests.

The actual final artifacts were independently checked and contain none of these faults.

Post-run, audit-only integrity fixes were committed on `experiment/critic-diagnostic`:
- reject duplicate/missing learned arm/seed records;
- require the exact 90 scenario/policy-seed pairs for every learned evaluation;
- reject duplicated heuristic scenarios;
- cross-check evaluation checkpoint SHA, config hash, and git commit against training manifests;
- add regression tests for duplicate detection.

These changes do **not** rerun training, do not reopen scenarios 901–930, and do not alter the reported final result.

Latest post-audit test run: **22 passed**.

## 11. Paper-safe conclusion

The final evidence supports:
1. a clear resource-state ablation benefit in the cooperative MARL formulation;
2. a lower aggregate J for Proposed than Single PPO in the fixed final matrix, but with strong training-seed sensitivity and no robust seed-general superiority claim;
3. a lower J for the information-aware heuristic than all three Proposed training seeds;
4. an operational trade-off in which the heuristic uses much more proactive YC capacity;
5. a hybrid architecture in which RL learns storage allocation and YC operation-level scheduling while Target×Destination is rule-resolved.

No additional training is required to report these results if the claims are narrowed accordingly. Additional seeds would only be needed to make a stronger algorithm-level claim against Single PPO.
