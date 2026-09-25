# V6 corrected hybrid cooperative MARL architecture spec — 2026-09-26

Status: **design frozen before retraining.** This document defines the minimum correction of the final V4 hybrid architecture. V4/V5 code and final artifacts remain frozen for audit lineage.

## 1. Research scope

V6 intentionally studies **Storage Allocation + YC operation scheduling coordination**.

It does **not** claim end-to-end learning of proactive Target or Destination selection.

When proactive work is available:
- Target is resolved by the existing transparent ETA/blocker/ETA-advance priority rule.
- Destination is resolved by the existing information-aware relocation heuristic.
- RL decides whether to execute that exact resolved proactive move now or take the default compulsory/idle operation.

The method is therefore:

> Candidate-aware hybrid cooperative MARL with deterministic proactive move resolution.

## 2. Agents

- Storage Agent: one yard-level actor selecting one of 100 stacks.
- YC Agents: one decision identity per block (`yc_0 ... yc_3`) with a **shared YC operation policy**.
- Centralized critic: shared global state-value function for CTDE.

The simulator remains asynchronous/event-driven. Only the active agent acts at each decision epoch.

## 3. YC operation action space

Remove the dead learned 2,500-pair scoring head from the V6 policy.

The learned YC action is a two-action categorical:

- `Default`
  - maps to `Mandatory` when compulsory work exists;
  - otherwise maps to `Idle`.
- `Proactive`
  - maps to the **single rule-resolved Target×Destination action**.

The simulator still receives the existing flat action ID, preserving environment transition semantics.

## 4. Candidate-aware YC observation

The YC operation actor receives:

1. existing 18-dimensional block/context features;
2. the existing 9-dimensional feature vector for the **exact rule-resolved proactive pair**;
3. two explicit flags:
   - `default_is_mandatory`;
   - `proactive_available`.

Total YC operation input: **29 floats**.

When proactive is unavailable, the 9 pair features are zeros and the proactive action is masked.

This directly fixes the V4 information bottleneck: the learned operation decision can now depend on the exact move that the deterministic resolver proposes.

## 5. Reward and environment

Unchanged from the frozen final study:

`r_team = -Δ truck delay - Δ storage delay - 0.1 × Δ extra YC minutes/retrieval`

- truck weight = 1;
- storage weight = 1;
- extra YC weight = 0.1;
- risk shaping = 0;
- YC queue shaping = 0;
- arrival rate = 20/h;
- 4 blocks × 25 stacks × 4 tiers;
- one fixed YC per block;
- dynamic Truck ETA;
- proactive horizon and simulator mechanics unchanged.

No reward retuning is authorized by this architecture correction.

## 6. Storage behavior-cloning fairness

The V4 final resource-state ablation used an information-aware BC teacher that could observe YC queue/inbound-pressure variables hidden from the no-resource student.

V6 removes this confound.

The common Storage BC teacher may use **only information observable to both resource and no-resource actors**:
- ETA inversion count;
- stack height;
- physical feasibility.

Fixed common teacher score:

`score = inversion + 0.35 × normalized stack height`

with deterministic block/stack tie-breaking.

For a fixed BC scenario, all architectures use the same teacher actions.

Resource and no-resource observations may differ as declared, but the teacher label-generation rule must not consume resource-only variables.

Each architecture-matched BC checkpoint is built **once**, hashed, then the exact same bytes are fanned out to all training seeds.

## 7. Training batch unit

V4 used about 26 complete scenario realizations per 30k-decision training seed, with effectively one complete episode per PPO update.

V6 training is controlled primarily by **complete independent episodes**, not raw decision count.

A PPO actor update collects multiple complete independently seeded episodes in one batch.

Required config fields:
- `total_episodes`;
- `episodes_per_update`.

Every update must record:
- exact scenario seeds in the batch;
- episode count;
- Storage decisions;
- YC decisions;
- total decisions.

No episode may be truncated for an actor update.

## 8. PPO

Retain unless a later protocol explicitly changes them:
- PPO clip = 0.20;
- actor+critic epochs = 2;
- critic-only extra epochs = 18;
- minibatch = 256;
- learning rate = 3e-4;
- gamma = 1.0;
- GAE lambda = 1.0;
- max grad norm = 0.5;
- action-Q critic OFF.

The common team advantage remains valid for the active asynchronous actor.

## 9. Centralized Single PPO fairness

If a new Single PPO comparison is run, it must use the same corrected hybrid action semantics:
- same deterministic Target/Destination resolver;
- same exact resolved-pair feature vector available to its YC operation head;
- same common information-matched Storage BC labels;
- same episode batching and optimization budget.

## 10. Validation gates before any long run

V6 must pass:

1. exact rule support: at most one proactive flat action;
2. operation mapping: Default→Mandatory/Idle and Proactive→resolved pair;
3. exact 29-dimensional YC operation observation;
4. resolved pair features in the actor input exactly match the simulator's selected flat action;
5. perturbing resolved pair features can affect YC operation logits;
6. no learned 2,500-pair scoring parameters in V6;
7. common Storage BC teacher action is invariant to resource-state visibility;
8. multi-episode PPO batch contains the declared number of terminal-complete scenario seeds;
9. critic-only epochs change actor parameters by exactly zero;
10. checkpoint strict-load and stochastic evaluation smoke pass.

## 11. Branch / study rule

- Work remains on `experiment/critic-diagnostic`.
- Do not merge V6 to `main` before the corrected study is completed and audited.
- Do not reuse final scenarios 901–930 for V6 model selection or final claims.
- V6 requires a fresh evaluation bank if promoted to a paper result.
- No long training is authorized merely by this specification; implementation/preflight comes first.
