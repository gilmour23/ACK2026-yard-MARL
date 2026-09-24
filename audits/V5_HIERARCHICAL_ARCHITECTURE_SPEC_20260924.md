# V5 hierarchical YC architecture specification — 2026-09-24

Status: implementation specification. No training is authorized by this file.

## Motivation

The completed no-training diagnostics established that:
- nominal flat Target×Destination support (2,500 proactive actions) is unnecessarily large;
- physically feasible proactive support averaged about 130 pairs in the audited probe states;
- current Target and Destination heuristics both leave material counterfactual regret;
- current Destination heuristic Top-K pruning is unsafe (counterfactual-best Destination median heuristic rank 8; p90 rank 14).

Therefore V5 must retain all physically feasible candidates while factorizing the learned decision.

## YC policy factorization

At each YC decision:

1. **Operation**
   - action 0: Default
   - action 1: Proactive
   - Default maps to Mandatory if compulsory work exists; otherwise Idle.

2. **Target | Proactive**
   - categorical over all currently eligible proactive Targets.

3. **Destination | Target, Proactive**
   - categorical over all physically feasible relocation Destinations for the selected Target.

The simulator still receives the existing encoded flat proactive action. The factorization is internal to the actor/trainer.

For proactive actions:

```
log pi(a|s)
= log pi_op(Proactive|s)
+ log pi_target(target|s)
+ log pi_dest(destination|s,target)
```

Default uses only `log pi_op(Default|s)`.

PPO clipping applies to this **joint log probability**.

## Observation redesign

Do not feed the legacy 18 + 2,500×9 observation to the V5 YC actor.

Use:
- current 18-dimensional YC block context;
- fixed 100-position Target feature table with a dynamic eligibility mask;
- fixed 25-stack Destination feature table;
- physical Target×Destination feasibility mask used only for masking, not as learned dense pair features.

Target features include current-information quantities such as ETA lead/advance, blockers, capacity slack, moving-blocker ETA, source-stack height, and tier.

Destination features encode stack occupancy/capacity and per-tier ETA information. Target-conditioned Destination scoring is learned by combining Target, Destination, and context embeddings.

## Candidate-aware operation gate

The operation gate must depend on the current candidate set, not context alone.

Its input combines:
- context embedding;
- masked Target embedding summary;
- masked Target×Destination interaction summary.

Thus P(Proactive) can respond to the quality of the currently available proactive moves.

## Initialization

- Storage actor/critic may warm-start from a **single shared Storage-only BC checkpoint** per architecture.
- YC operation proactive prior remains 0.10 for the first pilot.
- Target and Destination conditional scorers initialize to uniform preference (zero final scoring weights/bias).
- No YC imitation from the existing heuristic.

## Frozen first-pilot settings

Do not simultaneously change:
- reward weights;
- gamma/lambda;
- PPO clip;
- learning rate;
- episode-complete rollout;
- value20 critic schedule;
- resource-state formulation;
- environment dynamics.

Q-credit remains disabled.

## Required diagnostics

Record separately:
- operation entropy and P(Proactive);
- Target normalized entropy;
- Destination normalized entropy;
- number of eligible Targets and feasible pairs;
- selected Target heuristic rank;
- selected Destination heuristic rank;
- sampled proactive rate;
- ordinary operational KPIs and J.

The first bounded pilot must verify that Target/Destination policies move away from uniform in a seed-consistent manner before any new 30k study is authorized.
