# V5 hierarchical 6k bounded pilot result — 2026-09-24

Status: **integrity PASS; mechanism FAIL; performance FAIL; do not extend to longer training yet.**

## Provenance

- Repository: `gilmour23/ACK2026-yard-MARL`
- Branch: `experiment/critic-diagnostic`
- Workflow run: `35989214777`
- Execution commit: `762f4488d402d83e6b1250f93ea3731b20ee1c06`
- Protocol: `audits/V5_HIERARCHICAL_6K_PILOT_PROTOCOL_20260924.md`
- Config: `configs/v5_hierarchical_pilot_20260924.json`
- Training seeds: 61, 62, 63
- Pilot evaluation bank: 1201–1210, 3 stochastic repeats/scenario
- Shared Storage-only BC SHA-256:
  `59dfd073753a82420603362f2eae9fa53130609fa0493b2026f132946a20846c`

No additional post-result training was performed in this audit.

## Integrity: PASS

All workflow jobs completed successfully:
- shared BC;
- 3 control-rule train/eval jobs;
- 3 V5 hierarchical train/eval jobs;
- aggregate.

Actual terminal-complete decisions:
- control seed61: 7,048
- control seed62: 7,128
- control seed63: 7,056
- V5 seed61: 6,926
- V5 seed62: 7,092
- V5 seed63: 6,934

All six runs:
- 6 PPO updates;
- all rollouts terminal-complete;
- critic-only actor max abs parameter delta = 0;
- exact same shared BC SHA;
- exact same six training-scenario sequence within each seed across arms;
- exact 1201–1210 × 3 stochastic evaluation matrix.

Thus the result is interpretable as an architecture-pilot result rather than an implementation failure.

## Primary objective

| Seed | Control rule J | V5 hierarchical J | V5 − Control |
|---:|---:|---:|---:|
| 61 | 9.3680 | 9.8598 | +0.4918 |
| 62 | 8.6860 | 9.5519 | +0.8659 |
| 63 | 8.8837 | 10.5755 | +1.6918 |

V5 was worse in **all 3/3 training seeds**.

Across the ten pilot scenarios, after averaging repeats and training seeds:
- mean V5 − Control J = **+1.0165**
- relative increase vs Control mean = **+11.32%**
- scenario bootstrap 95% CI = **[+0.7142, +1.3794]**
- all **10/10** scenario mean differences were positive.

Therefore the pre-registered performance gate fails clearly.

## Operational decomposition

Mean across the three training seeds and all pilot evaluation rows:

| KPI | Control rule | V5 | V5 − Control | Relative |
|---|---:|---:|---:|---:|
| Truck completion delay | 5.2918 | 6.2195 | +0.9278 | +17.53% |
| Storage completion delay | 3.4182 | 3.4640 | +0.0458 | +1.34% |
| Reactive rehandling moves | 98.03 | 132.02 | +33.99 | +34.67% |
| Proactive moves | 115.06 | 115.47 | +0.41 | +0.36% |
| Extra YC min/retrieval | 2.6924 | 3.1218 | +0.4294 | +15.95% |
| Total YC moves | 534.39 | 568.79 | +34.40 | +6.44% |
| Mean YC utilization | 0.5518 | 0.5888 | +0.0370 | +6.70% |
| Max YC queue | 4.433 | 4.678 | +0.244 | +5.51% |

The key pattern is:

> V5 did **not** perform materially more proactive moves than Control, but its proactive move choices produced substantially more later rehandling, more YC work, and higher truck delay.

This localizes the performance failure primarily to **which Target/Destination pair is selected**, not simply to the frequency of proactive operation.

## Mechanism gate: FAIL

Pre-registered normalized entropy drops (early two updates minus late two updates):

| Seed | Target entropy drop | Destination entropy drop | Mechanism gate |
|---:|---:|---:|---|
| 61 | -0.0095 | -0.0049 | FAIL |
| 62 | -0.0012 | -0.0059 | FAIL |
| 63 | +0.0204 | +0.0048 | PASS |

Only 1/3 seeds passed; the protocol required at least 2/3.

Late-window normalized conditional entropies remained high:
- Target: approximately 0.92–0.94
- Destination: approximately 0.95–0.98

Evaluation-bank means were similarly high:
- seed61: Target 0.938, Destination 0.970
- seed62: Target 0.931, Destination 0.969
- seed63: Target 0.924, Destination 0.961

The learned branches therefore remained close to broad/uniform conditional distributions.

## Heuristic-rank diagnostic

During V5 evaluation:

- mean feasible Targets per YC decision: approximately 7.8
- mean feasible Target×Destination pairs: approximately 129–136
- selected Target heuristic rank: approximately 4.48–4.55
- selected Destination heuristic rank: approximately 8.62–9.46

For roughly 7.8 eligible Targets, an approximately uniform selection has an expected rank near 4.4.
For roughly 16–17 feasible Destinations per selected Target, an approximately uniform selection has an expected rank near 8.5–9.

Thus the selected-rank statistics are consistent with the entropy result: the V5 Target and Destination branches remain near-uniform rather than learning a stable action-quality ordering.

This does **not** mean the heuristic ranks are globally optimal; prior counterfactual work showed they are not. The rank statistics are used here only as a convenient reference for detecting non-uniform learned selection.

## Parameter movement corroboration

The final V5 checkpoints show only small movement in the conditional scoring heads:

- `pair_scale`: seed61 -0.0066, seed62 +0.00065, seed63 -0.0059
- Target scorer weight norm: about 0.010–0.017
- Destination bias weight norm: about 0.021–0.032
- operation-head proactive bias remains very close to the initial `logit(0.10) = -2.1972`.

This is consistent with the mechanism result: the hierarchy is differentiable and executable, but the shared trajectory-level PPO advantage did not produce enough branch-specific discrimination over 6k.

## Interpretation

The factorized action representation solved the **cardinality/representation problem**:
- no 2,500-way flat categorical decision;
- exact feasibility masking;
- gradients reach Operation, Target, and Destination branches;
- compact YC observation;
- stable training.

It did **not** solve the **branch-specific credit-assignment problem**.

The same scalar PPO/GAE advantage is currently multiplied by:
- operation log-probability;
- selected Target log-probability;
- selected Destination log-probability.

When two proactive pairs have different long-run consequences, the current state-value advantage provides only a noisy sampled-action signal for both conditional branches. The observed near-uniform conditional entropies and worse rehandling are consistent with this limitation.

## Next step

Do **not**:
- extend this V5 to 30k;
- increase actor epochs;
- raise P(Proactive);
- tune entropy coefficients;
- change reward weights;
- reintroduce the previous flat Q-credit design.

The next diagnostic should isolate branch-specific credit.

The most direct candidate is a **hierarchical action-value critic with counterfactual branch baselines**:
- operation advantage for Default vs Proactive;
- Target advantage conditional on Proactive;
- Destination advantage conditional on selected Target.

For example:

```
A_D(s,T,D) = Q_D(s,T,D) - sum_d pi_D(d|s,T) Q_D(s,T,d)

A_T(s,T) = Q_T(s,T) - sum_t pi_T(t|s) Q_T(s,t)
```

with the operation branch continuing to use the existing state-value/GAE signal or its own operation-level Q baseline.

This differs materially from the failed historical flat Q-credit pilot because the credit would be applied **separately to the Target and Destination conditional branches**, rather than blending one flat-action Q advantage into the whole YC actor.

A no-training/offline sufficiency diagnostic should precede implementation: fit candidate-conditioned Target/Destination critics on completed V5 trajectories and verify held-out action-ranking signal before another RL pilot.
