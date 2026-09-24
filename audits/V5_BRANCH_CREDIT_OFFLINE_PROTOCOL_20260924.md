# V5 branch-credit offline sufficiency protocol — 2026-09-24

Status: **pre-registered diagnostic; no policy retraining.**

## Purpose

The V5 hierarchical 6k pilot passed implementation integrity but failed both:
- conditional-policy mechanism learning; and
- objective performance.

The next question is narrower:

> Do the current V5 state/candidate features contain enough action-specific signal for separate Target and Destination value models to rank alternatives out of sample?

This diagnostic fits **offline supervised critics only**. It does not update any policy, environment parameter, reward weight, PPO hyperparameter, or final/pilot checkpoint.

## Source policies

Frozen V5 treatment checkpoints from pilot run `35989214777`:
- training seed 61;
- training seed 62;
- training seed 63.

## Fresh diagnostic bank

Scenarios: **1221–1230**.

These scenarios are diagnostic-only and become consumed after this study.

For each source checkpoint × scenario:
- reproduce one stochastic V5 trajectory using policy seed `scenario*100`;
- collect the first YC decision with at least 3 eligible proactive Targets;
- deep-copy that exact state.

Maximum intended probe states: 30.

## Counterfactual labels

At each probe state enumerate **all physically feasible Target×Destination proactive pairs**.

For each pair:
1. force that proactive pair once;
2. after the forced action, continue with the same frozen V5 policy in **hierarchical greedy** mode until terminal;
3. record terminal
   `J = truck delay + storage delay + 0.1 * extra YC minutes/retrieval`.

Define Destination action value target:

`Q_D label = -J`.

For each Target, define Target action value as the frozen V5 Destination policy-weighted expectation over that Target's feasible Destinations at the probe state:

`Q_T(s,t) = sum_d pi_D(d|s,t) * (-J(s,t,d))`.

This diagnostic therefore evaluates representational/ranking sufficiency under a frozen deterministic continuation after the first forced move. It is not an unbiased estimator of the stochastic-policy Q function.

## Offline critic inputs

### Destination critic
Input:
- 18-dimensional YC context;
- selected Target's 8-dimensional V5 Target features;
- candidate Destination's 10-dimensional V5 Destination features.

No Target/Destination IDs, scenario IDs, training seed IDs, heuristic ranks, or counterfactual J values may be used as input.

### Target critic
Input:
- 18-dimensional YC context;
- Target's 8-dimensional V5 Target features;
- Destination-feature mean under the frozen Destination policy for that Target;
- Destination-feature max over feasible Destinations for that Target.

Again, no identifiers, heuristic ranks, or labels may be used as input.

## Cross-validation

Use 5 grouped folds by scenario:
- fold1 test: 1221–1222
- fold2: 1223–1224
- fold3: 1225–1226
- fold4: 1227–1228
- fold5: 1229–1230

All three source-model states for a held-out scenario remain in the test fold.

For each fold and branch:
- standardize inputs using training-fold statistics only;
- standardize labels using training-fold statistics only;
- fit a two-hidden-layer MLP offline with Adam/MSE;
- train 3 deterministic critic initializations and average test predictions.

The critic is diagnostic only and is not inserted into PPO in this study.

## Primary sufficiency metrics

Computed only on held-out predictions.

For Target and Destination separately:
- within-state pairwise ordering accuracy;
- within-state Spearman rank correlation;
- critic-selected Top-1 regret:
  `best actual J - selected actual J` expressed as nonnegative J regret;
- uniform-random expected regret;
- current heuristic Top-1 regret;
- fold-specific ordering accuracy.

Because labels are stored as `-J`, larger critic prediction is better.

## Pre-registered branch sufficiency gate

A branch passes only if all are true:

1. pooled held-out pairwise ordering accuracy >= **0.60**;
2. critic-selected mean Top-1 regret <= **75%** of uniform-random mean regret;
3. at least **4 of 5** folds have pairwise ordering accuracy > **0.55**.

Heuristic regret is reported as a reference but is not a required threshold because previous diagnostics already showed that the current heuristic is not an oracle.

## Decision rule

- both Target and Destination pass -> branch-specific critic representation is sufficiently promising for a bounded hierarchical credit pilot;
- only one branch passes -> do not train a full branch-Q policy; use the passed critic only as a diagnostic and redesign features/credit for the failed branch;
- neither passes -> current V5 features are insufficient for reliable branch-specific value ranking; improve observation/label formulation before any new RL pilot.

No 30k or policy retraining is authorized by this protocol.
