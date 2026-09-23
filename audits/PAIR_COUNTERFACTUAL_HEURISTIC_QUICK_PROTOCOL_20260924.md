# Pair counterfactual heuristic-continuation quick protocol — 2026-09-24

Registered before the actor-continuation counterfactual jobs returned results.

Purpose: obtain a fast, low-noise check of whether the forced proactive pair itself changes downstream terminal cost. The forced first action is evaluated from the same cloned YC state, but all later decisions follow the deterministic information-aware heuristic rather than the neural actor.

- Scenarios: 801, 802
- 2 probe states/scenario
- gap >= 80 decisions
- actor-score top/bottom + 4 deterministic random feasible pairs
- 1 continuation per candidate
- canonical 12k checkpoint used only to collect states and score candidate pairs
- downstream continuation: deterministic heuristic
- no training, no optimizer step, no main-branch change
- Full 30k remains No-Go

This quick stage is a mechanism check and does not replace the larger stochastic-actor continuation diagnostic.
