# Rule-resolved proactive 6k bounded confirmation — 2026-09-24

Status: **confirmation PASS; rule-resolved hybrid formulation is the candidate canonical YC formulation. Full final matrix is not started until baseline fairness checks complete.**

## Provenance

- Repository: `gilmour23/ACK2026-yard-MARL`
- Branch: `experiment/critic-diagnostic`
- Confirmation workflow run: `35839804721`
- Training seeds: 31, 32, 33
- Evaluation bank: scenarios 871–880, 3 stochastic repeats/scenario
- Decision threshold: 6,000, then finish the current episode
- Canonical init checkpoint SHA-256:
  `2bbd3a2e795a215d3fcbd654d58fad154d4298b55bb50ce92f54ad50b82ddb59`
- Protocol: `audits/RULE_RESOLVED_CONFIRMATION_PROTOCOL_20260923.md`
- Scenarios 901+ remain unused by this confirmation.

All six workflow jobs completed successfully. Treatment support exposed at most one proactive action at each YC decision and recorded zero support violations.

## Primary objective

`J = truck completion delay + storage completion delay + 0.1 * extra YC minutes/retrieval`.

| Seed | Flat control J | Rule-resolved J | Relative change |
|---|---:|---:|---:|
| 31 | 11.1353 | 9.5420 | -14.31% |
| 32 | 12.4070 | 9.0036 | -27.43% |
| 33 | 11.3127 | 9.2437 | -18.29% |

Seed mean:
- flat control = **11.61834**
- rule-resolved = **9.26309**
- relative change = **-20.27%**

All 3/3 training seeds improve.

Scenario-level paired treatment-control differences for scenarios 871–880 are:

| Scenario | ΔJ |
|---|---:|
| 871 | -2.6178 |
| 872 | -3.1086 |
| 873 | -1.7578 |
| 874 | -2.1079 |
| 875 | -1.9184 |
| 876 | -3.1983 |
| 877 | -2.3644 |
| 878 | -2.2554 |
| 879 | -2.1088 |
| 880 | -2.1151 |

Mean paired difference = **-2.35525**.

Paired bootstrap over the 10 scenario means, 10,000 resamples with RNG seed 20260923:
- 95% CI = **[-2.65823, -2.09009]**.

The pre-registered bounded-confirmation gate therefore passes.

## Operational decomposition

Seed-mean metrics:

| Metric | Flat control | Rule-resolved | Relative change |
|---|---:|---:|---:|
| Truck completion delay | 7.1029 | 5.3577 | **-24.57%** |
| Storage completion delay | 4.1519 | 3.6123 | **-13.00%** |
| Reactive rehandling | 135.80 | 77.63 | **-42.83%** |
| Proactive moves | 156.51 | 157.77 | +0.80% |
| Extra YC min/retrieval | 3.6359 | 2.9305 | **-19.40%** |
| Mean YC utilization | 0.6380 | 0.5785 | **-9.33%** |
| Max YC queue | 5.456 | 4.967 | **-8.96%** |

The improvement is not caused by suppressing proactive work: proactive-move volume is essentially unchanged. The simplified resolver spends approximately the same proactive capacity on substantially better moves, reducing reactive rehandling and both truck and storage delay.

## Code / lineage review

The treatment is a support simplification, not a reward or transition change.

- `ResourcePPOConfig.rule_resolve_proactive_pair` is passed directly into `ResourceMARLYardEnv`.
- `ResourceMarlSimulator.rule_resolved_proactive_action()` selects the first existing proactive candidate under the current ETA-lead → blocker-count → ETA-advance priority.
- The moved object is the current top blocker.
- Destination is selected with the existing `InformationAwareHeuristic.choose_relocation_destination()` using inversion/height information.
- `yc_action_mask()` exposes this one encoded proactive pair, while Mandatory remains available when compulsory work exists.
- The environment still executes the ordinary encoded proactive action. Task duration, reward equation, queue dynamics, and physical move transition are unchanged.
- Training and evaluation both instantiate the environment with the same treatment flag.
- The 6k confirmation used episode-complete rollouts, value20 critic training, Q critic OFF, and stochastic evaluation as pre-registered.

## Research interpretation

The evidence now supports a **hybrid cooperative MARL formulation**:

- Storage Allocation Agent learns inbound stack placement.
- Shared-policy YC Operation Agents learn whether scarce YC capacity should process mandatory work or an ETA-driven proactive move.
- Proactive target/destination selection is a transparent information-aware resolver, not a learned policy and not claimed to be globally optimal.

This is more consistent with the study's resource-contention question than asking PPO to learn a 2,500-way Target×Destination choice whose conditional policy remained nearly uniform.

## Next gate

Before final 30k experiments, the Centralized Single-PPO baseline must be brought to the same episode-complete/value20/rule-resolved action support. The MARL resource-state ablation must use the same rule resolver as the proposed model. Only after a short baseline-fairness smoke test passes should the final multi-seed matrix use scenarios 901+ for untouched evaluation.
