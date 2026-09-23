# Pair counterfactual quick-stage protocol — 2026-09-24

This is a runtime-reduced stage-1 diagnostic registered before the full counterfactual run produced any result.

- Scenarios: 801, 802
- 2 probe states/scenario
- minimum gap: 80 decisions
- actor-score top/bottom pair + 4 deterministic random feasible pairs
- 1 common-random-number continuation per candidate
- frozen canonical checkpoint and stochastic continuation policy
- no training and no source-model changes

Purpose: obtain an early mechanism check. It does not replace the larger pre-registered diagnostic if that run completes. Full 30k remains No-Go.
