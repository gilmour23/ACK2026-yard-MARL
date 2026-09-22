# Value-budget pilot protocol — 2026-09-23

Status: **pre-registered diagnostic; do not merge to main before results are audited.**

Basis: `ACK2026_ASTRA_Critic_Diagnostic_20260922.md`.

## Question

Does increasing only the critic's update exposure in episode-complete PPO repair the online value underfitting strongly enough to improve held-out value fit, without changing the actor update budget or any environment/reward/model architecture?

## Arms

Both arms start from the canonical 12k checkpoint:

- `groupnorm_12k_resource_marl_final.pt`
- SHA-256: `2bbd3a2e795a215d3fcbd654d58fad154d4298b55bb50ce92f54ad50b82ddb59`

Shared:
- episode-complete rollout = true
- rollout threshold = 512
- actor+critic joint PPO epochs = 2
- minibatch = 256
- lr = 3e-4
- max grad norm = 0.5
- gamma = lambda = 1
- environment/reward/actor/critic architecture unchanged
- training seeds = 21, 22, 23
- threshold = 2,000 decisions, then finish current episode
- validation scenarios = 701–710
- stochastic repeats = 3 per scenario

Control:
- `critic_extra_epochs = 0`
- total value exposures = canonical 2 joint epochs

Treatment:
- `critic_extra_epochs = 18`
- canonical 2 joint epochs + 18 value-only epochs
- same Adam optimizer
- actor grads must remain None during extra steps
- dedicated NumPy RNG for extra critic permutations
- actor max absolute delta during extra steps must be exactly 0

## Primary mechanism gate

All runs must have finite losses, terminal-complete rollouts and complete minibatch coverage.

Treatment succeeds as a critic mechanism only if:
1. held-out MC RMSE is lower than control in all 3 training seeds;
2. at least 2/3 treatment seeds have held-out critic EV >= 0.50;
3. the remaining treatment seed has held-out critic EV >= 0.20.

These are project engineering gates, not literature thresholds.

## Policy diagnostics

Mechanism success is not policy success. Record:
- objective J
- Truck completion delay
- Storage completion delay
- reactive rehandling
- proactive moves
- extra YC min/retrieval
- P(Proactive)
- normalized conditional pair entropy
- TV distance of conditional feasible-pair distribution from uniform
- expected Target ETA lead vs uniform-feasible baseline
- destination height/inversion expected values vs uniform-feasible baseline

Do not claim the critic fix solved pair credit unless policy-level evidence supports it.

## Full-30k rule

Full 30k remains No-Go regardless of offline critic fit.

No 30k run is allowed from this protocol.

## Code

- implementation: `src/train_yc_marl.py`, config `critic_extra_epochs`
- runner: `scripts/run_value_budget_pilot.py`
- branch: `experiment/critic-diagnostic`
- main must remain unchanged until audited results justify a merge.
