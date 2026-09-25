# V4 Cooperative MARL architecture independent audit — 2026-09-26

Status: **core MARL formulation is structurally valid, but the frozen final V4 is a hybrid operation-level MARL rather than an end-to-end learned yard-control MARL. Several methodological issues should be fixed before a new retraining study.**

Scope:
- branch: `experiment/critic-diagnostic`
- final execution commit reviewed: `734c234024ff039fea7a68d114c0993035f39e18`
- current branch code reviewed through `aa0edb8` lineage
- final training artifacts: Proposed MARL seeds 51/52/53 and no-resource BC artifacts
- no new policy training was performed in this audit

## 1. Bottom-line verdict

### What is valid

The V4 model is a legitimate cooperative CTDE formulation:

- one Storage actor chooses among 100 yard stacks;
- four block-specific YC decision makers are represented as `yc_0 ... yc_3`;
- the four YCs share one policy network, which is valid parameter sharing for homogeneous cranes;
- only the currently active asynchronous agent acts at an event-driven decision epoch;
- a centralized critic receives a global engineered yard observation;
- active actors are updated with the common team return;
- the final reward telescopes to the declared terminal objective
  `J = mean truck delay + mean storage delay + 0.1 * extra YC minutes/retrieval`
  because local shaping is zero in the final protocol.

Therefore the code is not merely “single-agent PPO mislabeled as MARL”.

### What is not valid to claim

The frozen final V4 does **not** learn the full proactive pre-marshalling decision.

With `rule_resolve_proactive_pair=True`:
- RL learns Storage Allocation;
- RL learns YC operation-level choice (Mandatory/Idle versus Proactive);
- proactive Target is resolved by the ETA/blocker/ETA-advance rule;
- proactive Destination is resolved by the information-aware relocation heuristic.

The final architecture should therefore be described as:

> **Hybrid cooperative MARL with learned Storage Allocation and learned YC operation scheduling, plus deterministic proactive move resolution.**

It should not be described as learning Target and Destination quality end to end.

## 2. Agent / state / action / reward audit

### 2.1 Agents — PASS

The asynchronous Markov-game representation is coherent.

- Storage is one yard-level planning actor.
- Each physical block has one YC decision identity.
- YC parameters are shared across blocks.
- The active agent is selected by event progression rather than by a simultaneous joint-action clock.

Simultaneous actions are not required for a valid MARL formulation. This is an asynchronous cooperative control problem.

### 2.2 Observations — PASS with a material limitation

Storage receives yard-wide block/stack information plus incoming-container information.

YC receives:
- an 18-dimensional operation context;
- Target×Destination feature tensors.

The centralized critic receives a global engineered observation and an active-block indicator.

However, under the final rule-resolved action support, the detailed Target×Destination feature tensor does not affect the learned operation probability. The operation head consumes only the 18-dimensional context.

Thus the learned YC operation cannot directly condition its Proactive/Default decision on:
- the selected destination inversion;
- selected destination height;
- the exact selected blocker ETA;
- the resolved move's detailed pair feature vector.

The context contains useful aggregate urgency/queue/blocker features, so this is not a blind policy, but it is an information bottleneck.

### 2.3 Actions — STRUCTURALLY VALID, SCOPE NARROWED

Storage:
- 100 stack actions with physical feasibility masking.

YC:
- canonical flat interface has 2502 action IDs;
- final rule-resolved support exposes at most one proactive pair;
- therefore an actual final YC decision is usually a two-way operation choice:
  - Mandatory vs Proactive, or
  - Idle vs Proactive.

The pair scorer is mathematically inactive in these final states because there is only one feasible proactive pair:

`pair_score - logsumexp(pair_score) = 0`.

The final network therefore contains a 2500-pair scoring component that is dead with respect to policy probability in the frozen final experiment.

This does not invalidate the policy, but the architecture should be simplified in any new study.

### 2.4 Team reward — PASS

The final protocol has:
- truck weight = 1;
- storage weight = 1;
- extra YC weight = 0.1;
- storage risk shaping = 0;
- YC queue shaping = 0.

The environment accumulates waiting-time areas and extra relocation moves and applies the delta cost between decision epochs.

With terminal-complete episodes and all mandatory jobs drained, the undiscounted cumulative team reward is aligned with the negative terminal objective J.

Using `gamma=1` for this finite undiscounted operational objective is therefore defensible.

## 3. PPO / critic audit

### 3.1 Shared team advantage — PASS

The active Storage or YC policy is updated using the common team advantage.

For cooperative CTDE, the same scalar team advantage can validly update different active agents. The prior hypothesis that “the same advantage is inherently wrong for hierarchical/heterogeneous branches” is too strong.

Branch-specific action-value critics may reduce variance in a future design, but they are not required for the current MARL to be mathematically valid.

### 3.2 One centralized V(s) critic — PASS

A single centralized state-value critic is acceptable because the observation encodes the current decision type and active YC block.

The final artifacts do not support the claim that the critic remained broken.

For Proposed MARL seeds 51/52/53, after excluding the first rollout, the **pre-update** completed-episode MC explained variance on subsequent newly collected episodes was approximately:

- seed51: mean **0.947**
- seed52: mean **0.957**
- seed53: mean **0.959**

The critic-only extra epochs also changed actor parameters by exactly zero in the final integrity audit.

A role-conditioned critic can still be studied, but it is not the first structural repair priority.

### 3.3 Role-wise actor loss averaging — NOT A CORE MARL BUG

The PPO code computes a mean policy loss separately for Storage and YC transitions and then averages the role losses.

The final training data contain approximately:
- 4.1k Storage decisions;
- 26.0k YC decisions;
- YC/Storage decision ratio about **6.3:1**.

At first glance this appears to reweight per-transition gradients strongly.

However, the V4 MARL Storage actor and YC actor have disjoint parameters. Therefore role-wise averaging mainly rescales two separate actor gradients rather than forcing a trade-off through shared actor weights.

This choice should be documented, but it is not a sufficient reason by itself to reject or rebuild V4.

For the centralized Single PPO baseline, actor parameters are more shared, so this weighting deserves separate fairness sensitivity analysis if a stronger MARL-vs-Single claim is pursued.

### 3.4 Global advantage normalization — ACCEPTABLE, not proven optimal

Advantages are normalized across the complete rollout before role-specific actor updates.

This is not mathematically invalid. Per-role normalization could change finite-sample variance, but no current evidence establishes that global normalization caused the observed seed sensitivity.

Do not change it post hoc without a new pre-registered study.

## 4. High-priority methodological issues

### HIGH-1. Final architecture does not match the original end-to-end decision scope

If the research question is:

> “Can MARL jointly learn Storage Allocation, proactive Target selection, Destination selection, and shared-YC scheduling?”

then V4 is incomplete.

If the research question is narrowed to:

> “Can cooperative MARL coordinate Storage Allocation and YC operation scheduling while a transparent rule resolves individual proactive moves?”

then V4 is coherent.

This scope choice must be explicit before retraining.

### HIGH-2. Only about 26 independent training episodes were used per seed

The 30k-decision budget sounds large, but the episode-complete final rollout produced only **26 terminal-complete training scenarios per seed**.

Observed totals:

| Seed | Training episodes | Storage decisions | YC decisions | Total decisions |
|---:|---:|---:|---:|---:|
| 51 | 26 | 4,104 | 26,040 | 30,144 |
| 52 | 26 | 4,188 | 26,328 | 30,516 |
| 53 | 26 | 4,124 | 26,279 | 30,403 |

Each PPO update is effectively one complete stochastic scenario.

This is the strongest current explanation for why the final Proposed-vs-Single result is highly training-seed sensitive: the policy sees many correlated decisions but relatively few independent scenario realizations.

This is a training-design limitation, not a proof of a MARL formulation error.

For a new study, the training budget should be specified and audited in **complete episodes / independent scenario seeds as well as decisions**, and rollout batches should contain multiple independent episodes before each actor update.

### HIGH-3. Resource-state ablation is confounded by behavior-cloning observability

This affects the strongest final result.

Storage BC is generated by the information-aware heuristic. That heuristic uses:
- actual YC queue length;
- inbound-pressure information.

For `marl_noresource`, those resource-state features are removed from the actor observation, but the BC teacher still uses them to select the target action.

Therefore the no-resource student is trained to imitate an expert whose relevant information is partly hidden from the student.

Observed BC train accuracy:
- Proposed resource-state MARL: **0.5871**
- No-resource MARL: **0.5209**

Thus the two final arms do not begin from equally observable imitation tasks.

The final 14.7% resource-state advantage is still an observed pipeline result, but it does **not** cleanly isolate the causal effect of adding resource state during PPO alone.

A new ablation should remove this confound.

Preferred options, in order:
1. use a common information-matched Storage BC teacher based only on features visible to both arms, with extra resource inputs initialized neutrally in the Proposed arm;
2. or train the ablation study from scratch with enough episodes;
3. at minimum, separately report pre-RL BC performance and the incremental RL gain.

## 5. Medium-priority limitations

### MEDIUM-1. YC operation actor cannot see the exact rule-resolved move

The deterministic resolver selects a specific pair, but the operation head uses only aggregate context.

A minimal hybrid revision should expose the resolved move feature vector directly to the operation actor.

This is much smaller and cleaner than returning to a 2500-way flat policy.

### MEDIUM-2. New inbound containers are not retrieved in the same episode

New inbound containers receive multi-day future pickup proxies but their own future retrieval is outside the 480-minute operating episode.

Therefore Storage Allocation is optimized primarily for:
- current-window storage delay;
- interference with same-episode initial-stock retrievals;
- current resource use.

Longer-term consequences of storing new inbound containers are truncated.

This is acceptable for a one-window import-yard study if stated explicitly, but it limits claims about long-horizon yard planning.

### MEDIUM-3. BC checkpoints were independently rebuilt in parallel final jobs

The existing audit already found small numerical differences across supposedly identical BC builds.

For any new study:
- build one architecture-matched BC artifact once;
- hash it;
- fan out exactly the same bytes to all training seeds.

The later V5 pilot workflow already adopted this practice.

## 6. What should NOT be changed based on current evidence

Do not rebuild the following merely because final performance was mixed:

- CTDE itself;
- YC parameter sharing;
- asynchronous event-driven decision epochs;
- common team reward;
- `gamma=1` finite-horizon objective;
- one centralized V(s) critic;
- action masking.

All are defensible in the current formulation.

Do not introduce a branch-Q critic simply because V5 failed. The current evidence does not prove shared state-value advantage is the root cause.

## 7. Recommended next architecture

Two scientifically distinct paths exist.

### Path A — paper-focused hybrid MARL

Use this if the ACK paper contribution is Storage–YC resource coordination.

Keep:
- event-driven environment;
- Storage actor;
- shared YC actor;
- centralized critic;
- common reward;
- deterministic Target/Destination resolver.

Change before a new training study:
1. replace the dead 2500-pair YC head with an explicit operation policy;
2. feed the exact resolved proactive move features into that operation policy;
3. fix Storage BC fairness across resource/no-resource arms;
4. batch multiple independent complete episodes per PPO update;
5. define the training budget by both episode count and decision count;
6. build one BC artifact per architecture and fan it to all seeds.

This is the lowest-risk reconstruction.

### Path B — end-to-end proactive MARL

Use this only if Target and Destination learning is itself the research contribution.

Then the V5 factorized hierarchy is the right action-space family, but the 6k pilot already showed that the current implementation should not simply be extended to 30k.

Before retraining:
- finish the offline source-policy diagnostic;
- correct entropy accounting to support>=2 states;
- separate pooled versus source-policy-specific critic fits;
- preserve raw counterfactual data and held-out predictions;
- only then choose a revised credit/feature design.

## 8. Current Go / No-Go decision

### For reporting the frozen V4 result
**GO, with narrowed claims.**

It is a valid hybrid cooperative MARL experiment.

### For claiming end-to-end learned yard pre-marshalling
**NO-GO.**

Target/Destination are deterministic in the final policy.

### For simply rerunning the same V4 code for more seeds
**NO-GO as the first action.**

The BC-ablation confound and operation-information bottleneck should be resolved first if a new training study is opened.

### For a new corrected hybrid training study
**GO after protocol freeze.**

The minimal corrected hybrid architecture is the recommended next step if time permits.

## 9. Final assessment

The existing research does **not** need to be discarded.

The simulator, cooperative reward, asynchronous CTDE structure, shared YC policy, feasibility masking, and final critic training are technically usable.

The two issues that most materially weaken the present scientific interpretation are:

1. **the final policy learns only operation-level pre-marshalling control, not Target/Destination selection;**
2. **the strongest resource-state ablation is partially confounded by an information-advantaged BC teacher.**

The main training-quality limitation is that each 30k run contains only about **26 independent stochastic episodes**, despite tens of thousands of decisions.

Those are the points that should drive the next retraining decision.
