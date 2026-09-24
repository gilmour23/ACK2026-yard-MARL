# V5 hierarchical YC implementation status — 2026-09-24

Status: **implementation + unit/integration preflight PASS; bounded 6k scientific pilot NOT yet run.**

## Branch / isolation

- Repository: `gilmour23/ACK2026-yard-MARL`
- Branch: `experiment/critic-diagnostic`
- Frozen V4/final source files were not modified for V5.
- V5 lives under the isolated `v5/` package.
- No push triggered legacy value-budget/rule/Q pilots because V5 files are outside `src/**`.

## Implemented architecture

YC policy is factorized as:

1. Default vs Proactive
2. Target conditional on Proactive
3. Destination conditional on selected Target

The simulator still receives the existing encoded flat YC action.

For proactive actions:

```
log pi(a|s)
= log pi_op(Proactive|s)
+ log pi_target(Target|s)
+ log pi_destination(Destination|s,Target)
```

PPO clipping uses this joint log probability.

## Observation

Legacy dense YC actor observation:
- 18 + 2,500×9 = 22,518 floats.

V5 learned YC observation:
- 18 context features
- 100×8 Target-position features
- 25×10 Destination-stack features
- total **1,068 floats**

The physical 100×25 pair-feasibility mask is supplied separately for masking only.

Target features include:
- ETA lead
- ETA advance
- blockers
- capacity slack
- moving blocker ETA
- source-stack height
- target tier
- number of feasible destinations

Destination features include:
- occupied/reserved height
- reservation load
- per-tier ETA lead
- per-tier occupancy flags

## Candidate-aware factorization

- Operation gate sees context plus masked Target/Destination summaries and learned pair-score statistics.
- Target score sees its own features **and** mean/max learned quality over that Target's feasible Destination set.
- Destination score combines context-conditioned Target and Destination embeddings.
- All eligible Targets and all physically feasible Destinations are retained; no heuristic Top-K pruning is used.

## Initialization

- P(Proactive) initial prior remains 0.10.
- Target conditional policy initializes exactly uniform.
- Destination conditional policy initializes exactly uniform.
- No YC heuristic imitation is used.
- Storage/critic can load only a Storage-only MARL BC checkpoint.

## Trainer

`v5/train_hierarchical_marl.py` preserves the first-pilot core settings:
- episode-complete rollout
- gamma = 1
- lambda = 1
- PPO clip = 0.20
- LR = 3e-4
- actor+critic epochs = 2
- critic-only extra epochs = 18 (value20)
- reward weights 1 / 1 / 0.1
- resource state ON
- Q-credit absent

Diagnostics recorded per update:
- operation entropy
- normalized Target entropy
- normalized Destination entropy
- P(Proactive)
- feasible Target count
- feasible pair count
- sampled proactive rate
- selected Target heuristic rank
- selected Destination heuristic rank
- critic diagnostics and actor-delta integrity

## Evaluation

`v5/evaluate_hierarchical.py` supports:
- strict V5 checkpoint loading
- stochastic evaluation with explicit Torch generator / policy seed
- hierarchical greedy mode
- ordinary operational KPIs
- V5 branch diagnostics

## Verification

Latest V5 implementation preflight:
- workflow: `v5-hierarchical-preflight`
- run: `35986227089`
- result: **success**

V5 unit tests:
- **5 passed**
- compact observation exactly matches simulator proactive action support
- initial Target/Destination policies are uniform
- initial P(Proactive) = 0.10
- sampled hierarchical actions are always simulator-valid
- proactive joint log-prob propagates gradients into operation, Target, and Destination branches

End-to-end tiny trainer smoke:
- result: **success**
- one terminal-complete episode
- actual decisions: **1,183**
- mean feasible Targets at YC decisions: **8.289**
- mean feasible Target×Destination pairs: **143.588**
- critic-only actor max parameter delta: **0**

Generic repository tests for the same implementation lineage also passed.

## Not yet done

No 6k V5 learning result exists yet.

Before a new 30k study, the next scientific step is a pre-registered bounded pilot using:
- shared one-time Storage BC artifact
- current rule-resolved hybrid control vs V5 hierarchical treatment
- seeds 61, 62, 63
- 6k decisions then terminal completion
- fresh diagnostic/evaluation bank
- branch-learning diagnostics as pre-specified

Do not merge V5 to `main` before bounded pilot result and lineage review.
