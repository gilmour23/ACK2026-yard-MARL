# CURRENT MODEL — ACK2026 항만 야드 MARL

Updated: 2026-09-22
Status: **Astra independent audit fixes applied; 30k full training remains No-Go**

## 1. Canonical model

현재 기준모델은 **Group-normalized Flat Target×Destination PPO**이다.

- Environment action space: Flat discrete
  - Mandatory
  - Idle
  - Proactive(Target × Destination)
- Maximum proactive pairs: 100 target positions × 25 destination stacks = 2,500
- Maximum YC actions: 2,502
- Canonical checkpoint: `checkpoints/groupnorm_12k_resource_marl_final.pt`
- Canonical training code: `code/train_yc_marl.py`
- Canonical evaluation code: `code/evaluate_yc_policies.py`
- Canonical comparison code: `code/train_yc_single.py`
- Credit-assignment pilot runner: `code/run_credit_assignment_pilot.py`

## 2. Astra independent audit findings now treated as ground truth for the next audit

The 2026-09-21 independent audit found:

1. **Training GroupNorm math/log-prob/PPO ratio was not found to contain a core mathematical error.**
2. **The provided evaluation helper had a GroupNorm mask bug.** The YC mask was applied only after pair normalization, producing a different policy distribution.
3. **`greedy proactive = 0%` is not a valid standalone failure metric** for the current grouped flat policy because proactive mass is spread across many pair actions.
4. The earlier interpretation that **Destination was already well learned was too strong**. The top-ranked pair looked reasonable, but the full conditional pair distribution was nearly uniform (`normalized entropy ≈ 0.99988`).
5. The current critic does not represent detailed Target ETA information well and its value prediction was almost constant on the audited trajectories (`MC explained variance ≈ 0.000074`).
6. Reward integration/drain accounting and GroupNorm training log-prob checks passed the audit.

Original audit materials are stored under `05_ASTRA_AUDITS/20260921_independent_audit/`.

## 3. Fixes already applied to canonical code

### 3.1 Evaluation GroupNorm mask fix

`evaluate_yc_policies.py`

- Before: `model.yc_logits(obs)` then external mask
- Now: `model.yc_logits(obs, mask)` so the feasible mask participates in GroupNorm pair normalization.

### 3.2 Evaluation protocol

- Learned policies are evaluated with **stochastic sampling by default**, matching the PPO policy distribution.
- Flat greedy remains available only as a separate diagnostic.
- Do not use flat-greedy proactive rate as a primary learning-success metric.

### 3.3 Centralized Single PPO fairness fixes

`train_yc_single.py`

- uses the same Group-normalized Flat Target×Destination YC policy parameterization as MARL
- uses the same proactive-group initialization and pair-score initialization family
- uses structured operation + normalized pair entropy
- passes the YC mask into GroupNorm normalization
- minibatch permutation bug fixed: one permutation per epoch

This was necessary so future MARL-vs-Single comparisons are not confounded by different action-policy parameterizations.

### 3.4 Episode-complete rollout option

`train_yc_marl.py`

Added `ResourcePPOConfig.episode_complete_rollout`:

- `False`: existing cutoff rollout
- `True`: once the rollout threshold is reached, continue collecting until the current episode reaches terminal/drain, then update

No actor architecture, reward, environment dynamics, gamma/lambda, or PPO objective was changed by this option.

### 3.5 Critic diagnostics

Each update can now log:

- `rollout_mode`
- `rollout_decisions`
- `rollout_ended_at_terminal`
- `mc_completed_fraction`
- `mc_value_ev`
- `mc_value_rmse`

### 3.6 Checkpoint lineage

New checkpoints include additional run/update/optimizer/RNG lineage metadata. The historical 12k checkpoint is still a weights-only warm-start source because the original full optimizer/RNG lineage was not available.

## 4. Verification after fixes

- Unit/integration tests: **15 / 15 passed**
- Episode-complete smoke test:
  - cutoff A: 256 decisions, non-terminal, MC completed fraction 0
  - episode-complete B: 1,188 decisions, terminal, MC completed fraction 1
  - B critic EV remained ~0.000073, RMSE ~3.93

This smoke test validates implementation behavior only. It does **not** establish that episode-complete rollout improves policy performance.

## 5. Partial credit-assignment pilot status

A longer local A/B pilot was attempted after the fixes, but the available execution window was insufficient to finish the planned 2k × 3-seed comparison.

Completed partial result:

### Cutoff A, training seed 21, 500 decisions

- rollout decisions: 500
- Storage decisions: 64
- YC decisions: 436
- mean P(Proactive): ~0.1467
- sampled proactive rate: ~0.1628
- normalized conditional pair entropy: ~0.99991
- operation entropy: ~0.4153
- rollout did not end at terminal
- MC completed fraction: 0
- MC EV/RMSE: not computable from completed tails

Do **not** use this partial result to choose A or B.

## 6. Current interpretation

The current canonical policy should be described as:

> The grouped operation probability is non-collapsed, but the conditional Target×Destination policy remains almost uniform. The strongest independently verified weakness is the critic/value representation and the handling of delayed effects across rollout cutoffs, not a proven defect in the actor probability formula or reward integration.

Do not describe the model as "Destination learned, Target failed" without qualification.

## 7. Not canonical

These remain experimental/failed branches and must not be treated as the current model:

- Target/Destination representation split
- Autoregressive Operation → Target → Destination
- Action-conditioned Q critic prototype

## 8. Fixed environment settings

- Import containers only
- 4 blocks × 25 stacks/block × 4 tiers = 400 physical slots
- Initial containers = 268 (67%)
- 1 fixed YC/block = 4 YCs
- No inter-block YC redeployment
- Operating horizon = 480 min
- YC move time = 2 min/container move
- Storage and Retrieval homogeneous Poisson
- λS = λR = 20/h
- Dynamic truck ETA
- New inbound containers are storage-only in current episode
- Low/Medium/High workload conditions are not used

## 9. Reward

`r_t = -ΔW_truck/N_T - ΔW_storage/N_S - 0.1·ΔT_YC_extra/N_T`

No positive proactive reward, no YC BC, no heuristic top-K.

## 10. Next audit / experiment request

Before any 30k training, independently verify the **post-audit-fix canonical code** and then run or supervise the planned A/B credit-assignment pilot:

- A: cutoff rollout
- B: episode-complete rollout
- same canonical 12k weights
- same hyperparameters
- training seeds 21, 22, 23
- first checkpoint: 2k additional decisions per arm, then 4k/6k only if justified

Primary diagnostics:

1. on-policy MC critic explained variance and RMSE
2. conditional pair entropy
3. stochastic objective J
4. Truck delay / Storage delay / extra YC work
5. sampled proactive rate
6. Target-randomization evaluation intervention, if useful, without modifying training

Do not reintroduce hand-crafted ETA target scores, heuristic top-K, YC BC, proactive positive reward, or workload-level experiments.
