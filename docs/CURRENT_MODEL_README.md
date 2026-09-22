# CURRENT MODEL — ACK2026 항만 야드 MARL

Updated: 2026-09-22  
Status: **post-audit-fix canonical code; Astra 2k A/B re-audit pilot completed; Full 30k remains No-Go**

## 1. Source of truth

현재 코드의 source of truth는 GitHub `main` branch이다.

- Core MARL trainer: `src/train_yc_marl.py`
- Environment: `src/yc_marl_env.py`
- Simulator: `src/yard_simulator.py`
- Centralized PPO baseline: `src/train_yc_single.py`
- Evaluation: `src/evaluate_yc_policies.py`
- Credit-assignment pilot: `scripts/run_credit_assignment_pilot.py`
- Tests: `tests/test_v4_final.py`
- Frozen canonical settings: `configs/v4_canonical.yaml`

대형 checkpoint와 실험 산출물은 GitHub에 저장하지 않고 Google Drive master repository에 보관한다.

Canonical checkpoint:
- file: `groupnorm_12k_resource_marl_final.pt`
- SHA-256: `2bbd3a2e795a215d3fcbd654d58fad154d4298b55bb50ce92f54ad50b82ddb59`

## 2. Canonical model

현재 기준모델은 **Group-normalized Flat Target×Destination PPO**이다.

Environment action space:
- Mandatory
- Idle
- Proactive(Target × Destination)

Maximum proactive pairs = 100 target positions × 25 destination stacks = 2,500.  
Maximum YC actions = 2,502.

현재 actor는 **2-level GroupNorm**을 사용한다:

[
P(a)=P(Operation)P(Target,Destinationmid Proactive)
]

`nested_group_normalized_flat_yc_logits()`는 legacy/unused helper이며 canonical actor가 호출하지 않는다.  
`ActionConditionedYCCritic`도 legacy prototype compatibility를 위해 코드에 남아 있지만 canonical 설정에서는 `use_action_q_critic=False`다.

## 3. Independent audit findings already incorporated

2026-09-21 Astra independent audit에서 확인된 주요 사실:

1. training GroupNorm math, joint log-prob, PPO ratio에서 핵심 수학 오류는 발견되지 않았다.
2. 기존 evaluation helper는 YC mask를 GroupNorm 내부 normalization에 전달하지 않는 오류가 있었다.
3. flat greedy proactive=0%는 grouped flat policy 구조상 standalone failure metric으로 쓰면 안 된다.
4. top-ranked pair가 합리적으로 보여도 full conditional pair distribution은 거의 uniform이었다.
5. critic value는 audited trajectories에서 거의 상수였고 MC explained variance가 약 0에 가까웠다.
6. reward integration, drain accounting, request normalization은 감사 범위에서 일치했다.

원본 감사 문서: `audits/ASTRA_INDEPENDENT_AUDIT_20260921.md`.

## 4. Canonical fixes

### 4.1 Evaluation mask

YC evaluation은 `model.yc_logits(obs, mask)`를 사용하여 feasible mask가 GroupNorm pair normalization에 들어간다.

### 4.2 Evaluation protocol

- PPO learned policy의 **primary evaluation은 stochastic sampling**이다.
- `evaluate()`의 API default도 stochastic이다.
- flat greedy는 별도 deployment/diagnostic 지표로만 사용한다.

### 4.3 MARL vs Single PPO fairness

현재 두 trainer의 canonical optimization 기본값은 다음으로 통일했다.

- rollout_steps = 512
- update_epochs = 2
- minibatch_size = 256
- gamma = 1.0
- gae_lambda = 1.0
- learning_rate = 3e-4
- YC operation entropy coefficient = 0.001
- normalized conditional pair entropy coefficient = 0.0001
- proactive initial bias = -2.197224577
- pair scorer init std = 0.05

Single PPO는 같은 GroupNorm action parameterization, feasible mask, structured entropy, minibatch-permutation 방식을 사용한다.

### 4.4 Episode-complete rollout option

`ResourcePPOConfig.episode_complete_rollout`:

- `False`: cutoff rollout
- `True`: threshold 도달 후 현재 episode terminal/drain까지 수집한 뒤 update

Actor architecture, environment, reward, gamma/lambda는 이 switch로 바뀌지 않는다.

### 4.5 Checkpoint continuation semantics

새 checkpoint는 optimizer/RNG/step metadata를 **저장**하지만 현재 `init_checkpoint` load path는 이를 exact resume에 사용하지 않는다.

따라서 현재 continuation semantics는 명시적으로:

> **weights-only warm start, not exact resume**

이다.

Historical 12k checkpoint 역시 exact optimizer/RNG lineage가 완전하게 복원되지 않는다.

## 5. Astra 2k A/B re-audit pilot — interim result

아래 수치는 사용자가 전달한 Astra 재감사 중간 실행 로그를 기록한 것이며, **최종 감사 보고서가 아직 제공되지 않았으므로 final evidence로 확정하지 않는다.**

Training seeds:
- 21
- 22
- 23

Validation scenarios:
- 601–610
- stochastic evaluation
- 총 210 evaluation episodes

실행 범위:
- A cutoff: seed별 약 2,059–2,127 decisions
- B episode-complete: 약 2,137–2,301 decisions
- B는 threshold 이후 현재 episode terminal까지 수집
- B의 해당 rollout samples는 terminal return에 포함

중간 결과:
- B는 3 seeds 모두에서 critic EV와 RMSE를 개선
- 그러나 critic EV는 약 0.017–0.021에 머묾
- objective J는 2 seeds에서 악화, 1 seed에서 소폭 개선
- conditional pair distribution은 여전히 거의 uniform
- 따라서 episode-complete는 critic training에 영향을 주지만 Target/pair credit failure를 해결했다고 볼 수 없음

추가 진단:
- critic observation에 이미 존재하는 시간 변수만 사용한 scenario-split 단순 회귀가 약 EV 0.958을 보였다는 중간 결과가 보고됨
- 따라서 현재 critic failure를 세부 ETA feature 부족만으로 설명하기 어렵고 **critic optimization/training failure를 별도로 분리할 필요**가 있음

## 6. Current interpretation

현재 가장 방어적인 해석:

> GroupNorm operation mass 자체는 붕괴하지 않았지만 conditional Target×Destination policy는 거의 균등하다. Episode-complete rollout은 value fit을 일부 개선했으나 objective와 pair learning을 일관되게 개선하지 않았다. 현재 다음 우선 가설은 actor 재설계가 아니라 critic optimization/training failure이며, fine-grained critic representation limitation과 rollout-boundary effect는 별도 요인으로 분리해 검증해야 한다.

## 7. Fixed environment

- Import containers only
- 4 blocks × 25 stacks/block × 4 tiers = 400 physical slots
- Initial containers = 268
- 1 fixed YC/block
- No inter-block YC redeployment
- Operating horizon = 480 min
- YC move time = 2 min
- Storage and Retrieval homogeneous Poisson
- λS = λR = 20/h
- Dynamic truck ETA
- New inbound is storage-only in current episode
- no Low/Medium/High workload regimes

## 8. Reward

[
r_t=-Delta W_{truck}/N_T-Delta W_{storage}/N_S-0.1Delta T_{YC,extra}/N_T
]

No positive proactive reward, no YC behavior cloning, no heuristic top-K.

## 9. Training safety

Full 30k multi-seed training remains **No-Go**.

`scripts/run_v4_experiments.py` blocks non-quick 30k execution unless `--allow-full-30k` is explicitly supplied after an audit Go decision.

## 10. Next step

Do **not** extend the episode-complete experiment to 4k/6k merely because EV improved.

Next work should keep the actor fixed and isolate critic training behavior:
- value target scale/distribution
- critic loss trajectory
- gradient norms and joint clipping interaction
- critic learning rate / optimizer behavior
- minibatch coverage
- value prediction vs MC-return calibration
- simple-observation baselines

Do not reintroduce hand-crafted ETA target scores, heuristic top-K, YC BC, proactive positive reward, or workload-level experiments before this diagnostic is resolved.
