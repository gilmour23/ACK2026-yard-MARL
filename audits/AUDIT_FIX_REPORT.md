# ACK2026 항만 야드 MARL — Astra 독립감사 반영 수정

작성일: 2026-09-21
기준 체크포인트: `groupnorm_12k_resource_marl_final.pt`

## 현재 판정

- 30k multi-seed full training: **No-Go 유지**
- 현재 MARL actor/reward/environment 구조: **변경하지 않음**
- 다음 검증: cutoff rollout(A) vs episode-complete rollout(B) 2k→4k→6k pilot

## 반영한 수정

### 1. GroupNorm 평가 마스크 오류 수정

`evaluate_yc_policies.py`

- 기존: YC 평가에서 `model.yc_logits(obs)` 호출 후 외부 mask 적용
- 수정: `model.yc_logits(obs, mask)`로 GroupNorm 내부 normalization에 동일 mask 전달
- MARL/Single 모두 동일 API 사용
- 평가 결과에 `evaluation_mode = stochastic | flat_greedy` 기록

### 2. PPO 기본 평가 프로토콜 정리

`run_v4_experiments.py`

- learned policy 기본 성능평가는 stochastic sampling으로 저장
- flat argmax는 별도 `*_flat_greedy` diagnostic으로 분리
- heuristic은 deterministic 유지

### 3. Centralized Single PPO 비교 공정성 수정

`train_yc_single.py`

- MARL과 동일한 Group-normalized Flat Target×Destination action parameterization 적용
- proactive group 초기 bias = -2.197224577
- pair scorer small-random init std = 0.05
- operation entropy + normalized conditional pair entropy 사용
- YC mask를 GroupNorm 내부에 전달
- epoch마다 permutation을 한 번만 생성하도록 minibatch bug 수정
- storage-only BC checkpoint의 YC head shape 차이를 안전하게 무시

### 4. Episode-complete rollout 실험 옵션 추가

`train_yc_marl.py`

`ResourcePPOConfig.episode_complete_rollout` 추가. 기본값은 `False`로 canonical actor behavior를 바꾸지 않는다.

- `False`: 기존 cutoff 방식
- `True`: rollout threshold 도달 후 현재 episode terminal/drain까지 수집한 뒤 update

Actor, critic, reward, gamma/lambda, entropy, PPO objective는 변경하지 않는다.

### 5. Critic/return 진단 로깅 추가

각 PPO update에 다음을 기록한다.

- `rollout_mode`
- `rollout_decisions`
- `rollout_ended_at_terminal`
- `mc_completed_fraction`
- `mc_value_ev`
- `mc_value_rmse`

cutoff 끝의 미완료 episode tail은 MC diagnostic에서 제외한다.

### 6. Checkpoint lineage 강화

새 체크포인트에 다음을 저장한다.

- `run_global_step`
- `update_count`
- `continuation_mode`
- `init_checkpoint`
- optimizer state
- Python / NumPy / Torch RNG state
- scenario sampler RNG state
- current environment seed

과거 12k checkpoint에서 시작하는 pilot은 `weights_only_warm_start`임을 명시한다.

### 7. A/B pilot runner 추가

`run_credit_assignment_pilot.py`

동일 12k 가중치에서:

- A: cutoff rollout
- B: episode-complete rollout

을 동일 hyperparameter와 training seeds로 비교하도록 구현했다.

## 검증

### Unit/integration tests

- **15 / 15 passed**
- 기존 Astra audit에서 실패했던 Single `yc_logits(obs, mask)` signature test 해결
- Single GroupNorm pair-count invariance test 추가
- completed-MC tail exclusion test 추가

### Smoke test

동일 canonical 12k checkpoint에서 짧은 smoke:

- A cutoff: rollout 256 decisions, terminal 아님, `mc_completed_fraction=0.0`
- B episode-complete: rollout 1,188 decisions, terminal에서 종료, `mc_completed_fraction=1.0`
- B의 초기 critic EV는 약 `7.3e-05`, RMSE 약 `3.93`으로 여전히 좋지 않음

이는 episode-complete 옵션이 의도대로 **bootstrap cutoff를 제거했다는 구현 검증**이지, 성능 개선을 입증한 결과는 아니다.

## 의도적으로 하지 않은 수정

- hand-crafted ETA target score 없음
- heuristic top-K 없음
- YC behavior cloning 없음
- proactive positive reward 없음
- actor architecture 변경 없음
- critic architecture 변경 없음
- action-conditioned Q critic 활성화 없음
- Low/Medium/High workload 재도입 없음

## 다음 단계

`run_credit_assignment_pilot.py`로 2k pilot을 3개 training seed에 대해 수행하고, 다음을 비교한다.

1. MC critic explained variance / RMSE
2. conditional pair entropy
3. stochastic objective J
4. Truck delay / Storage delay / extra YC work
5. sampled proactive rate

2k에서 방향성이 확인되면 4k, 6k로 진행한다. 그 전에는 30k full training을 실행하지 않는다.
