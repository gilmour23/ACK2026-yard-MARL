# V5 hierarchical 6k bounded pilot protocol — 2026-09-24

Status: **pre-registered; execute only on `experiment/critic-diagnostic`.**

## Purpose

Test whether the new factorized YC policy

`Default/Proactive -> Target -> Destination`

can learn useful conditional Target/Destination preferences without the flat 2,500-action burden, while holding the objective, environment, Storage policy family, critic schedule, PPO settings, and resource-state formulation fixed.

This is a bounded architecture pilot, not a final experiment.

## Arms

### Control: `control_rule`
Current validated hybrid formulation:
- learned Storage Allocation actor;
- shared YC operation actor;
- at most one proactive Target×Destination pair exposed by the current rule resolver;
- rule-resolved proactive pair support ON.

### Treatment: `treatment_v5`
New V5 formulation:
- same Storage actor family;
- shared YC hierarchical actor;
- operation: Default vs Proactive;
- Target: categorical over all eligible proactive Targets;
- Destination: categorical over all physically feasible relocation Destinations conditional on the selected Target;
- no heuristic Top-K pruning;
- rule-resolved proactive pair support OFF.

The simulator receives the same encoded flat proactive action after the V5 hierarchy samples Target and Destination.

## Shared initialization

Generate **one** Storage-only MARL BC checkpoint and reuse the exact artifact for all six learned jobs.

BC:
- scenario seeds 1000–1007;
- 3 epochs;
- batch size 128;
- LR 5e-4;
- RNG seed 20260924;
- resource state ON;
- no YC BC.

Record and verify one BC SHA-256 across every arm/seed.

V5 YC initialization:
- initial P(Proactive) = 0.10;
- Target conditional logits uniform;
- Destination conditional logits uniform.

## Training

Training seeds: **61, 62, 63**.

Common settings:
- 6,000 decision threshold then finish current episode;
- episode-complete rollout;
- rollout size 512;
- min Storage transitions 64;
- min YC transitions 128;
- max rollout multiplier 4;
- PPO actor+critic epochs 2;
- critic-only extra epochs 18;
- minibatch 256;
- LR 3e-4;
- PPO clip 0.20;
- max grad norm 0.5;
- gamma = 1;
- lambda = 1;
- resource state ON;
- Q-credit OFF;
- proactive enabled;
- arrival rate 20/h;
- truck/storage/extra-YC objective weights 1 / 1 / 0.1;
- no risk or YC-queue shaping.

The two arms use the same `ScenarioSampler(seed+1000)`. Because episode lengths can differ, compare the common training-scenario prefix and record any terminal-completion tail difference.

## Pilot evaluation bank

Fresh pilot-only scenarios: **1201–1210**.

For each trained arm/seed:
- 3 stochastic repeats per scenario;
- common policy RNG seed rule: `scenario*100 + repeat_index_0_based`.

The 1201–1210 bank becomes consumed for architecture selection and must not be reused as a future final test bank.

## Outcomes

Primary pilot outcome:
- `J = truck completion delay + storage completion delay + 0.1 * extra YC minutes/retrieval`.

Operational:
- truck delay;
- storage delay;
- reactive rehandling;
- proactive moves;
- total YC moves;
- extra YC minutes/retrieval;
- mean YC utilization;
- max YC queue.

V5 mechanism:
- P(Proactive);
- operation entropy;
- normalized Target entropy;
- normalized Destination entropy;
- feasible Target count;
- feasible pair count;
- selected Target heuristic rank;
- selected Destination heuristic rank.

## Integrity gates

All six learned jobs must satisfy:
- actual decisions >= 6,000;
- every rollout ends at a terminal;
- all tracked losses/KPIs finite;
- critic-only actor max absolute delta = 0;
- shared BC SHA identical across all six jobs;
- checkpoints strict-load through the matching evaluator;
- exact 1201–1210 × 3 evaluation matrix.

Failure of an integrity gate is an implementation/infrastructure failure, not a performance result.

## Pre-registered architecture viability gates

These gates only decide whether V5 merits a longer confirmation; they do not establish final superiority.

### Mechanism gate
For at least **2 of 3 V5 training seeds**, at least one conditional branch must depart measurably from its initially uniform policy:
- late-window mean normalized Target entropy is at least **0.01 lower** than early-window mean, **or**
- late-window mean normalized Destination entropy is at least **0.01 lower** than early-window mean.

Early window = first two PPO updates with YC observations.
Late window = last two PPO updates.

### Performance gate
Both must hold:
1. V5 has lower seed-mean J than control for at least **2 of 3** training seeds.
2. The scenario-paired V5-control mean J difference over 1201–1210 is **< 0** after averaging repeats and training seeds.

A 10,000-resample scenario bootstrap (RNG 20260924) is reported descriptively but is not an additional pass/fail threshold for this bounded pilot.

## Decision rule

- Integrity fail -> fix implementation only; do not interpret performance.
- Integrity pass + mechanism fail -> do not extend; inspect branch-specific credit.
- Integrity pass + mechanism pass + performance fail -> do not extend directly; inspect whether learned candidate preferences are operationally misaligned.
- All gates pass -> eligible for a longer bounded confirmation. **No automatic 30k run is authorized.**

No reward, PPO, critic, or heuristic changes may be made using 1201–1210 results and then re-tested on the same pilot bank as if unseen.
