# Target–Destination counterfactual decomposition protocol — 2026-09-24

Status: **pre-registered diagnostic; no training.**

## Purpose

The frozen final study showed that:
- direct PPO over 2,500 Target×Destination proactive actions did not learn stable pair preferences;
- rule-resolved operation-level control substantially improved the short and 6k pilots;
- the final hybrid policy therefore learned YC operation-level allocation while Target×Destination was resolved by a transparent heuristic.

Before defining a new retrained model, this diagnostic separates the operational value of:
1. **Target choice**, and
2. **Destination choice**.

The result will determine whether the next architecture should be:
- candidate-aware binary operation control with the current top-1 resolver;
- Top-K learned Target selection with heuristic Destination; or
- a small Top-K Target×Destination candidate policy.

No optimizer step, reward change, environment change, or retraining is permitted in this diagnostic.

## Frozen source models and states

Source policies:
- final Proposed MARL checkpoints from GitHub Actions run `35951014861`;
- training seeds 51, 52, 53.

Fresh diagnostic scenarios:
- **1101–1110**;
- these scenarios are not part of any previous diagnostic bank or the final 901–930 test bank.

For each source model × scenario:
- run the final rule-resolved policy stochastically with a fixed collection RNG seed;
- collect the first YC state with at least 3 eligible proactive targets;
- if none exists before terminal, record the scenario as unprobed rather than changing the threshold.

Target/destination counterfactual branches use a deep copy of the exact same state.

## Counterfactual continuation

For the forced first proactive action only:
- temporarily expose the full physically feasible proactive action set;
- force the selected Target×Destination pair.

After that forced move:
- all later decisions use the deterministic information-aware heuristic;
- the environment dynamics, move time, arrivals, ETA updates, queue discipline, and objective remain unchanged.

Because continuation is deterministic, one continuation is used per branch.

## Target decomposition

For **every eligible proactive target** in a probe state:
- choose its Destination with the current information-aware relocation heuristic;
- force that Target + heuristic Destination;
- record terminal objective J and operational KPIs.

Per state compute:
- global best target under heuristic Destination;
- current heuristic top-1 target regret;
- best-within-top-3 regret;
- best-within-top-5 regret;
- whether the globally best target is contained in top-1/top-3/top-5 heuristic ranking;
- target objective range.

This directly tests how much value is lost by fixing Target to the current top-1 priority rule.

## Destination decomposition

For the heuristic-ranked top **5 Targets** (or all Targets if fewer):
- enumerate **all physically feasible relocation Destinations**;
- force each Target×Destination pair;
- record terminal objective J and operational KPIs.

Per Target compute:
- best Destination;
- heuristic Destination;
- heuristic-Destination regret;
- Destination objective range;
- whether the heuristic Destination is counterfactually best.

Per state aggregate the mean/median Destination regret and range over the tested Targets.

## Interpretation gates

These are architecture-selection diagnostics, not hypothesis-test thresholds.

### Candidate-aware binary top-1 is supported when:
- heuristic top-1 Target regret is consistently small;
- top-1 coverage of the counterfactual best Target is high;
- heuristic Destination regret is also small.

### Top-K Target + heuristic Destination is supported when:
- top-1 Target regret is material;
- top-3/top-5 largely recover the global best Target;
- heuristic Destination regret is small relative to Target spread.

### Top-K Pair is supported when:
- Destination regret/range is also material for top-ranked Targets;
- therefore fixing Destination to the current heuristic would impose a meaningful performance ceiling.

No architecture will be selected from one model seed alone. Results must be summarized across all 30 source-model/scenario combinations that yield valid probe states.

## Provenance requirements

Record:
- source checkpoint SHA-256;
- source training seed;
- scenario seed;
- policy collection seed;
- probe decision index and block;
- number of eligible Targets;
- number of feasible Destinations evaluated;
- all forced Target/Destination IDs and heuristic ranks;
- terminal J and component KPIs.

This diagnostic must remain on `experiment/critic-diagnostic` and must not modify `main`.
