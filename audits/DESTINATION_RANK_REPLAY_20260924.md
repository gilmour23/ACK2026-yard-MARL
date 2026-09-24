# Destination heuristic-rank replay result — 2026-09-24

Status: **completed; current destination heuristic is not suitable for aggressive Top-K pruning. No training or new counterfactual continuation was performed.**

## Provenance

- Repository: `gilmour23/ACK2026-yard-MARL`
- Branch: `experiment/critic-diagnostic`
- Workflow run: `35983513679`
- Replay commit: `589c66f9dbbd3fd7772765ed5b41fe7345dd770b`
- Source counterfactual run: `35975570052`
- Source final Proposed checkpoints: training seeds 51, 52, 53 from final run `35951014861`
- Diagnostic scenarios: 1101–1110
- Counterfactual J values were reused from the completed exhaustive destination diagnostic.
- This replay only regenerated the exact probe state and reconstructed the current destination-heuristic ordering.

## Result

Across 148 tested Target cases:

- mean feasible Destinations per Target: **15.74**
- counterfactual-best Destination heuristic rank:
  - median: **8**
  - 90th percentile: **14**

Coverage and residual regret when retaining the current heuristic's Top-K Destinations:

| K | Best-Destination coverage | Mean residual regret | P90 residual regret |
|---:|---:|---:|---:|
| 1 | 8.8% | 0.3222 J | 0.6139 J |
| 2 | 16.9% | 0.2036 J | 0.4688 J |
| 3 | 20.9% | 0.1686 J | 0.4688 J |
| 5 | 43.9% | 0.0594 J | 0.1851 J |

Seed-specific patterns were similar:

- seed 51: Top-5 coverage 46%, mean Top-5 regret 0.0550 J
- seed 52: Top-5 coverage 40%, mean Top-5 regret 0.0719 J
- seed 53: Top-5 coverage 45.8%, mean Top-5 regret 0.0510 J

Therefore the current `inversion + 0.35*height` destination score is better than random ranking only weakly and does not concentrate the counterfactual-best Destination near the top strongly enough to justify Top-2/Top-3/Top-5 pruning.

## Dynamic feasible action size

Using the same 30 probe states and all eligible proactive Targets:

- mean eligible Targets per state: **8.57**
- mean total physically feasible Target×Destination pairs: **130.37**
- median: **134**
- 90th percentile: **169.3**
- maximum: **190**

Thus the physically meaningful candidate set is already much smaller than the nominal 2,500 flat actions, but is still too large for an arbitrary heuristic Top-K cut if the current heuristic score is used.

## Architecture implication

The previous candidate plan should be revised.

Do **not**:
- keep only Top-2/Top-3/Top-5 Destinations using the current destination heuristic;
- return to the nominal flat 2,500-action policy.

The next architecture should first test a **factorized hierarchical proactive policy** over the complete feasible sets:

1. Operation: Default vs Proactive.
2. If Proactive: select one eligible Target (typically 3–14, mean 8.57).
3. Conditional on Target: select one physically feasible Destination (typically about 16).
4. Encode the selected Target×Destination back into the simulator's existing flat action representation.

This avoids discarding potentially good Destinations while reducing each learned categorical decision from roughly 130 simultaneous feasible pairs to two small conditional decisions.

The PPO joint log-probability should be:

```
log pi(a|s)
= log pi_operation(Proactive|s)
+ log pi_target(target|s,Proactive)
+ log pi_destination(destination|s,target,Proactive)
```

for proactive actions; Default uses only the operation term.

The current centralized critic/value20 schedule, episode-complete rollout, reward weights, and resource-state formulation should remain unchanged in the first architecture pilot.

## Important caution

Factorization reduces the action-branch cardinality but does **not by itself prove that action-specific credit will be sufficient**. The earlier flat-pair diagnostic showed that pair-specific GAE signal was weak. Therefore the first bounded V5 pilot should instrument:

- Target entropy and Destination entropy separately;
- selected Target/Destination score changes;
- rank distribution of selected Targets/Destinations;
- proactive probability;
- realized return correlation by branch;
- J and operational KPIs by seed.

Do not add Q-credit, reward shaping, or different PPO hyperparameters in the same first pilot. If the hierarchical actor remains nearly uniform despite the smaller conditional action spaces, then the next intervention should target branch-specific credit estimation rather than further action-space pruning.
