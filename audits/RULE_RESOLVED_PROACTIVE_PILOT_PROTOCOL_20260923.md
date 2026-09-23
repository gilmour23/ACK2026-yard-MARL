# Rule-resolved proactive pilot protocol — 2026-09-23

Status before execution: **Full 30k No-Go.**

## Motivation

Two completed diagnostics now constrain the next step:

1. value20 fixes the coarse value critic but does not make the 2500 Target×Destination conditional policy non-uniform;
2. the pair-scorer receives gradients, but pair-distribution movement and pair-specific return correlations remain weak;
3. adding the existing action-conditioned Q critic increases proactive usage strongly while leaving Target×Destination preference nearly uniform and worsens J by about 17%.

The current research question is resource contention: whether scarce YC capacity should be spent on mandatory work or on an ETA-driven proactive move. Learning the relocation destination is not required for that contribution, and the earlier research formulation already allowed a rule-based destination/resolver.

## Single structural change

Keep the current flat-action environment interface, but in the treatment expose **only one deterministic proactive pair** at each YC decision:

1. choose the proactive target using the simulator's existing transparent priority:
   - smaller ETA lead;
   - then more blockers;
   - then larger ETA advance;
   - deterministic container-ID tie break;
2. move the current top blocker of that target's stack;
3. choose its destination using the existing `InformationAwareHeuristic.choose_relocation_destination()` score:
   - fewer ETA inversions;
   - then lower stack height.

The actor therefore learns only the operation-level trade-off:
- Mandatory versus Proactive when mandatory work exists;
- Idle versus Proactive when only speculative work exists.

The underlying action passed to the simulator is still a valid encoded Target×Destination action. No reward or simulator transition rule is changed.

## Arms

### Control — flat-pair value20
- current 2500 proactive Target×Destination actions
- `episode_complete_rollout=True`
- actor+critic epochs=2
- critic-only extra epochs=18
- Q critic OFF

### Treatment — rule-resolved value20
Exactly the same settings, except:
- `rule_resolve_proactive_pair=True`
- at most one proactive pair is exposed in the YC action mask

## Fixed settings

- canonical init checkpoint SHA-256: `2bbd3a2e795a215d3fcbd654d58fad154d4298b55bb50ce92f54ad50b82ddb59`
- training seeds: 21, 22, 23
- 2,000 decision threshold, then finish the current episode
- rollout_steps=512
- minibatch_size=256
- learning_rate=3e-4
- max_grad_norm=0.5
- gamma=lambda=1
- reward/environment dynamics/Storage actor/value critic unchanged
- no Q critic
- stochastic evaluation

## Diagnostic evaluation bank

Use scenarios **851–860**, 3 repeats/scenario.

- 701–710 were used by the value-budget pilot.
- 801–810 were used by the Q-credit pilot.
- 851–860 are new for this structural pilot.
- 901+ remain reserved for later final testing and must not be used here.

## Pre-registered integrity checks

All seeds must:
- end every training rollout at a terminal state;
- have finite losses and evaluation metrics;
- have no invalid actions;
- in treatment, expose no more than one proactive action at any YC decision;
- use the same canonical checkpoint and fixed hyperparameters except the stated action-support change.

## Pre-registered policy gate

Let

`J = truck completion delay + storage completion delay + 0.1 * extra YC minutes / retrieval`.

Treatment passes the short-pilot policy gate only if:
- seed-mean J is at least **2% lower** than control;
- at least **2/3 training seeds** improve;
- scenario-level paired bootstrap over scenarios 851–860, 10,000 resamples, RNG seed 20260923, has treatment-control 95% CI upper bound < 0.

Record truck delay, storage delay, rehandling, proactive moves, YC overhead, P(Proactive), and value EV as secondary diagnostics. Do not create a gate that rewards a particular proactive count.

## Interpretation rule

A pass would show that the **simpler hybrid formulation** is a better candidate for this study than learning 2500 relocation pairs. It would not prove the heuristic pair resolver is globally optimal.

A fail would mean pair-action complexity is not the only reason for poor performance; inspect the economic value of proactive work itself before changing reward weights or adding more learning machinery.

Regardless of outcome, **this 2k pilot does not authorize Full 30k**. A separate bounded confirmation and Go review are required.
