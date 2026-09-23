# Rule-resolved proactive pilot result — 2026-09-23

Status: **integrity PASS; short-pilot policy gate PASS; Full 30k still No-Go pending bounded confirmation.**

## Provenance

- Workflow run: `35838558211`
- Execution commit: `9dbbca39964b6f326a340fd29e43d71c98ec3399`
- Branch: `experiment/critic-diagnostic`
- Canonical init checkpoint SHA-256: `2bbd3a2e795a215d3fcbd654d58fad154d4298b55bb50ce92f54ad50b82ddb59`
- Training seeds: 21, 22, 23
- Evaluation: scenarios 851–860, 3 stochastic repeats/scenario
- 2,000-decision threshold followed by terminal completion
- Latest test workflow for the execution commit: success

The protocol was committed before implementation/execution in
`audits/RULE_RESOLVED_PROACTIVE_PILOT_PROTOCOL_20260923.md`.

## Arms

Control:
- current flat 2500 Target×Destination proactive action support;
- episode-complete PPO;
- actor+critic epochs=2;
- critic-only extra epochs=18 (value20);
- Q critic OFF.

Treatment:
- identical settings;
- at most one proactive pair exposed per YC decision;
- target resolved by existing ETA lead -> blockers -> ETA advance priority;
- destination resolved by existing information-aware inversion + height heuristic;
- actor therefore learns the operation-level Mandatory/Idle versus Proactive trade-off.

No reward, move time, queue rule, Storage actor, value critic, learning rate, entropy coefficient, gamma/lambda, or canonical checkpoint change.

## Primary objective

| Seed | J control | J rule-resolved | Change |
| --- | ---: | ---: | ---: |
| 21 | 10.3159 | 8.5982 | **-16.65%** |
| 22 | 10.6815 | 8.3006 | **-22.29%** |
| 23 | 10.6013 | 8.7029 | **-17.91%** |

Seed mean:
- control J = **10.53288**
- treatment J = **8.53389**
- relative change = **-18.98%**

All 3/3 training seeds improve.

Scenario-level paired treatment-control mean differences were negative in all scenarios 851–860:

| Scenario | ΔJ treatment-control |
| --- | ---: |
| 851 | -1.9483 |
| 852 | -2.2820 |
| 853 | -1.8206 |
| 854 | -1.8085 |
| 855 | -1.6662 |
| 856 | -2.3945 |
| 857 | -1.9628 |
| 858 | -1.8404 |
| 859 | -1.6644 |
| 860 | -2.6022 |

Mean paired difference = **-1.99899**.

Paired bootstrap over the 10 scenario means, 10,000 resamples with RNG seed 20260923:
- 95% CI = **[-2.19776, -1.82155]**

The pre-registered short-pilot policy gate passes.

## Operational decomposition

Seed-mean metrics:

| Metric | Flat control | Rule-resolved | Relative change |
| --- | ---: | ---: | ---: |
| Truck completion delay | 6.3821 | 4.9267 | **-22.80%** |
| Storage completion delay | 3.8178 | 3.3412 | **-12.49%** |
| Reactive rehandling moves | 129.88 | 76.68 | **-40.96%** |
| Proactive moves | 136.13 | 135.86 | -0.20% |
| Extra YC min/retrieval | 3.3298 | 2.6600 | **-20.12%** |
| Mean YC utilization | 0.6010 | 0.5456 | **-9.22%** |
| Max YC queue | 4.78 | 4.40 | -7.91% |
| P(Proactive) | 0.1388 | 0.1351 | -2.66% |
| Held-out value EV | 0.9379 | 0.9397 | +0.2% relative |

Treatment evaluation confirmed proactive support max = **1** for every seed, with zero support violations.

## Interpretation

The gain is **not** explained by simply suppressing proactive work:

- P(Proactive) changes only slightly: 0.1388 -> 0.1351.
- proactive move count is essentially unchanged: 136.13 -> 135.86.

Instead, the same approximate amount of proactive YC capacity is used on much better-chosen moves. Reactive rehandling falls by about 41%, and both truck and storage delays fall simultaneously.

This resolves the previous diagnostic sequence coherently:

1. value20 repaired the coarse critic;
2. flat 2500-pair PPO still failed to learn meaningful pair preferences;
3. action-conditioned Q credit increased proactive quantity but did not fix pair quality and worsened J;
4. replacing learned pair selection with a transparent current-information resolver retains RL/MARL for the resource-allocation decision and sharply improves operations.

The result supports **simplifying the YC policy formulation**, rather than adding more pair-learning machinery.

## Research implication

For the ACK study, the cleaner formulation is now:

- Storage Allocation Agent: learns inbound stack placement;
- shared YC Operation Agents: learn whether scarce block YC capacity should serve mandatory work or an ETA-driven proactive move;
- proactive target/destination: resolved transparently from current ETA/blocker/inversion/height information.

This is a hybrid MARL + heuristic formulation. The heuristic resolver must not be described as learned or globally optimal.

## Next step

Do not merge to `main` yet and do not start Full 30k.

Run one longer bounded confirmation with the simplified formulation against the flat control on a fresh diagnostic bank. Preserve 901+ as final unseen scenarios. Only after the confirmation should the 30k guard be reconsidered.

**Full 30k: still No-Go.**
