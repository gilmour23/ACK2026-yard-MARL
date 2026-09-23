# Q-credit pilot result — 2026-09-23

Status: **safety PASS; pair-credit mechanism FAIL; policy gate FAIL; Full 30k remains No-Go.**

## Provenance

- Repository: `gilmour23/ACK2026-yard-MARL`
- Branch: `experiment/critic-diagnostic`
- Workflow run: `35836388136`
- Execution commit: `1293f52e023435f5e33aaa28fd95cc1842ce544f`
- Canonical init checkpoint SHA-256: `2bbd3a2e795a215d3fcbd654d58fad154d4298b55bb50ce92f54ad50b82ddb59`
- Training seeds: 21, 22, 23
- Evaluation bank: scenarios 801–810, 3 stochastic repeats/scenario
- Decision threshold: 2,000 then current episode completed
- All six GitHub Actions jobs completed successfully.

The protocol was committed before the runner/workflow in `audits/Q_CREDIT_PILOT_PROTOCOL_20260923.md`.

## Intervention

Control:
- episode-complete PPO
- actor+critic epochs=2
- critic-only extra epochs=18 (value20)
- action-conditioned YC Q critic OFF

Treatment:
- identical control
- `use_action_q_critic=True`
- `q_update_epochs=3`
- `q_adv_blend=0.5`

No reward, environment, actor architecture, value-critic architecture, LR, entropy coefficient, gamma/lambda, or checkpoint change.

## Primary result

| Seed | J control | J Q-credit | Change | Pair TV control | Pair TV Q-credit |
| --- | ---: | ---: | ---: | ---: | ---: |
| 21 | 10.7903 | 12.8480 | +19.07% | 0.00735 | 0.01175 |
| 22 | 11.5603 | 13.1465 | +13.72% | 0.01407 | 0.01483 |
| 23 | 11.3461 | 13.4522 | +18.56% | 0.01393 | 0.01556 |

Seed mean:
- control J = **11.23224**
- Q-credit J = **13.14889**
- relative change = **+17.06% (worse)**

Scenario-level paired treatment-control differences were positive for every scenario 801–810. Mean difference was **+1.91665**. Paired bootstrap with 10,000 resamples and RNG seed 20260923 gave 95% CI **[+1.55560, +2.27949]**. The policy gate fails decisively.

## Operational decomposition

Seed-mean Q-credit relative to control:

- truck completion delay: **+15.59%**
- storage completion delay: **+18.83%**
- reactive rehandling moves: **-12.31%**
- proactive moves: **+65.37%**
- extra YC minutes/retrieval: **+26.88%**
- P(Proactive): **+72.44%**
- conditional pair TV(uniform): 0.01178 -> 0.01404, still near-uniform
- held-out value EV: 0.91872 -> 0.82644

The Q treatment does reduce reactive rehandling, but it does so by spending much more YC capacity on proactive moves. That capacity diversion increases both truck and storage delay and worsens the overall objective.

## Interpretation

The existing action-conditioned Q path does **not** solve the Target×Destination credit problem.

The read-only actor diagnostic had already shown that the pair-scorer path is connected and receives non-zero gradients, but the conditional pair distribution moves only minimally. The Q pilot now shows that the current Q-advantage blend mostly changes the **operation-level probability of doing proactive work**, while conditional Target×Destination preference remains close to uniform.

This is consistent with the implementation: the Q advantage is blended into the advantage of the **entire flat YC action**, rather than decomposing operation credit from conditional Target/Destination credit.

Therefore:
- do not tune `q_adv_blend` as the next step;
- do not increase actor epochs or reduce pair entropy merely to force non-uniformity;
- do not alter reward weights to rescue this arm;
- do not run Full 30k.

## Next decision

The current 2500 Target×Destination proactive action is now the main formulation burden. The next bounded test should reduce the learned YC decision to the actual research question—whether scarce YC capacity should be spent on mandatory work or proactive work—while resolving the proactive move with the transparent ETA/blocker + relocation-destination heuristic already present in the simulator.

This is a structural simplification pilot, not a claim that the rule resolver is optimal.

**Full 30k: No-Go.**
