# Final-matrix baseline fairness smoke protocol — 2026-09-24

Status: **pre-registered integrity smoke test; Full 30k remains blocked.**

## Purpose

The 6k rule-resolved MARL confirmation passed. Before a final 30k comparison matrix, all learned baselines must use the same operational formulation and critic-training corrections.

This smoke test checks implementation parity only. It is not used to select the winning policy and does not consume scenarios 901+.

## Arms

1. `proposed_marl`
   - local-observation cooperative MARL
   - resource state ON
   - rule-resolved proactive pair
2. `marl_noresource`
   - same MARL and rule resolver
   - resource state OFF
3. `single_rule`
   - centralized Single-PPO baseline
   - same rule-resolved proactive support

## Common training settings

- Storage-only BC warm start generated separately for the matching architecture.
- BC collection: 4 scenario seeds, 2 epochs for smoke only.
- YC is not behavior cloned.
- 2,000-decision threshold, then complete the current episode.
- episode-complete rollout
- actor+critic PPO epochs = 2
- extra state-value critic epochs = 18 (value20)
- minibatch = 256
- lr = 3e-4
- gamma = lambda = 1
- Q critic OFF
- rule-resolved proactive pair ON
- reward and environment unchanged
- smoke training seeds = 41, 42

## Evaluation

- scenarios 881–885
- 2 stochastic repeats/scenario
- same scenario/repeat policy seeds across learned policies
- scenarios 901+ remain untouched for final evaluation

## Integrity gates

Every run must:
- finish all rollouts at terminal;
- have finite training/evaluation metrics;
- report zero actor change during critic-only extra epochs;
- use the declared rule-resolved support;
- produce a checkpoint that loads under the common evaluator.

This smoke test has **no relative-performance gate**. Its only purpose is to confirm apples-to-apples baseline plumbing before the final matrix.
