# FINAL RUN AUTHORIZED — 2026-09-24

User authorization received to execute the pre-registered ACK2026 final experiment matrix.

Frozen protocol:
- learned arms: proposed_marl, marl_noresource, single_rule
- training seeds: 51, 52, 53
- 30,000-decision threshold + terminal completion
- episode-complete rollout
- value20 critic schedule
- rule-resolved proactive pair support
- final evaluation bank: scenarios 901-930
- learned evaluation: 3 stochastic repeats/scenario
- heuristic: deterministic once/scenario
- paired bootstrap: 10,000 resamples, RNG seed 20260924

This file exists only to provide an auditable one-shot push trigger for
.github/workflows/final_experiment_matrix.yml.

After this final bank is opened, the results are to be reported as observed.
No post-hoc hyperparameter tuning or replacement final test is permitted under
the same study version.
