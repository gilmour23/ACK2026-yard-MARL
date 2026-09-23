# Pair counterfactual diagnostic protocol — 2026-09-24

Status: **pre-registered read-only diagnostic; Full 30k remains No-Go.**

## Question

The value20 pilot fixed coarse critic underfitting, but the conditional Target×Destination policy remained nearly uniform. The actor-signal audit then showed non-zero pair-path gradients but only very small conditional-pair distribution movement.

The next question is therefore:

> **At the same YC state, do different feasible proactive Target×Destination choices produce enough terminal-return separation to support learnable pair credit?**

This is a mechanism diagnostic, not a policy-performance experiment.

## Frozen items

- Canonical checkpoint SHA-256: `2bbd3a2e795a215d3fcbd654d58fad154d4298b55bb50ce92f54ad50b82ddb59`
- Actor/critic/reward/environment: unchanged
- No optimizer step and no training
- Primary policy continuation: frozen canonical stochastic actor
- Main branch: unchanged
- Diagnostic code only on `experiment/critic-diagnostic`

## Diagnostic bank

Scenarios: 801, 802, 803, 804. These are diagnostic scenarios and must not later be called final unseen test scenarios.

From each scenario, collect 3 YC decision states with at least two feasible proactive pairs, separated by at least 80 decisions.

At each state:
- deep-copy the exact environment state;
- include actor-score highest and lowest feasible pair;
- add 10 deterministic random feasible pairs;
- force each candidate pair once;
- continue to terminal under the frozen stochastic actor;
- use 2 common-random-number policy repeats per candidate.

The same continuation policy seed is reused across candidates within the same state/repeat. No random number is consumed by the instrumentation before the forced action.

## Primary outputs

1. Within-state future-return range and standard deviation across candidate pairs.
2. Between-state standard deviation of candidate-mean return.
3. Ratio of within-state candidate SD to between-state state-mean SD.
4. Within-state-centered correlation between actor pair score and realized future return.
5. Within-state-centered correlations between pair features and realized future return.
6. Regret of the actor-score-top candidate relative to the best tested candidate.

Because only a subset of feasible pairs is evaluated, the measured return range is a sampled diagnostic and not the exhaustive full-action range.

## Interpretation gate

- If within-state action variation is very small relative to between-state variation and actor-score/feature correlations are weak, the current 2500-pair action formulation has little identifiable pair signal under the present objective. The next design step should simplify/factor the pair decision rather than merely increase PPO epochs.
- If within-state action variation is material but actor-score alignment remains weak, pair credit estimation is the stronger failure mode; an action-conditioned advantage/Q formulation becomes the next justified pilot.
- No result from this diagnostic by itself authorizes Full 30k.

