# V6 corrected hybrid implementation status — 2026-09-26

Status: **implementation complete enough for preflight; not yet authorized for long training.**

Branch: `experiment/critic-diagnostic`

## Implemented

### Architecture
- New package: `v6/`
- Explicit two-action YC operation policy:
  - Default -> Mandatory when compulsory work exists, otherwise Idle;
  - Proactive -> exact single rule-resolved Target×Destination move.
- Removed learned 2,500-pair scoring from the V6 policy.
- V6 YC operation observation = 29 floats:
  - 18 context;
  - exact resolved pair's existing 9 features;
  - 2 operation flags.
- Resource MARL and centralized Single baseline both use the same hybrid action semantics.
- Exact initial `P(Proactive)=0.10` when both operations are feasible.

### BC fairness
- Added common observable Storage teacher:
  `inversion + 0.35 × normalized height`.
- Teacher labels do not use queue/busy/inbound resource-only fields.
- Resource/no-resource paired observations are collected from the exact same state and share the same teacher action.
- BC checkpoints contain Storage-related tensors only.
- Intended workflow: build each architecture-matched BC artifact once, hash it, fan the same bytes to every training seed.

### Training
- New trainer uses complete **episode count** as the primary budget.
- `episodes_per_update` batches multiple independent terminal-complete scenarios before PPO updates.
- Training scenario seeds are unique within a training run.
- Existing reward, gamma/lambda, PPO clip, LR, critic schedule and environment dynamics remain unchanged.
- Entropy reporting distinguishes:
  - all YC decisions;
  - decisions with operation support >= 2;
  - support>=2 fraction.
- Critic-only actor-parameter immutability check retained.

### Evaluation
- Added strict V6 checkpoint loader and stochastic evaluator.
- Simulator action semantics remain frozen through the legacy flat action IDs.

### Tests / preflight
Added unit tests for:
- 29-d observation;
- resolved-pair feature equality;
- operation->flat-action mapping;
- absence of learned 2,500-pair head;
- exact common 0.10 operation initialization;
- architectural candidate-feature sensitivity;
- common Storage teacher invariance to resource visibility;
- episode-based multi-episode training config.

Added bounded preflight:
`scripts/run_v6_corrected_hybrid_preflight.py`

Preflight performs, for all three learned kinds:
- tiny information-matched BC;
- 2 complete training episodes in one PPO update;
- critic-only actor immutability check;
- one stochastic diagnostic evaluation on scenario 1401.

## Files

- `audits/V6_CORRECTED_HYBRID_ARCHITECTURE_SPEC_20260926.md`
- `v6/hybrid_policy.py`
- `v6/storage_bc.py`
- `v6/train_hybrid.py`
- `v6/evaluate_hybrid.py`
- `tests/test_v6_corrected_hybrid.py`
- `scripts/run_v6_corrected_hybrid_preflight.py`

## Not yet done

- No V6 PPO training result exists yet.
- No new BC artifact has yet been accepted for a study.
- No V6 preflight result has yet been accepted.
- No final training episode budget has been frozen.
- No V6 final evaluation bank has been allocated.
- Nothing is merged to `main`.

The GitHub-hosted Actions account is currently blocked by billing/spending limits, and the self-hosted runner has not yet taken the queued recovery job. Therefore code execution validation must wait for the local/self-hosted runner or another executable environment.

## Go / No-Go

- Code design -> **ready for preflight**
- Long training -> **NO-GO until preflight PASS**
- Reuse final scenarios 901–930 -> **NO-GO**
- Merge to main -> **NO-GO**
