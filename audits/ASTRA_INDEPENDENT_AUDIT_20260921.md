# ACK2026 항만 야드 MARL 독립 감사

작성일: 2026-09-21  
대상: Group-normalized Flat Target×Destination PPO, 제공된 12k 기준 체크포인트  
판정: **현재 30k multi-seed 본학습은 No-Go. 평가 경로 정리와 짧은 credit-assignment 실험을 먼저 수행한다.**

`ASTRA_WORK_README.md`와 `CURRENT_MODEL_README.md`를 먼저 읽고, 최신 `ASTRA_WORK_PROMPT.md`의 순서에 따라 감사했다. 기존 설명을 코드의 정답으로 간주하지 않았다. 원본 모델·환경·체크포인트를 수정하거나 학습시키지 않았으며, 별도의 감사 스크립트에서 수치 검사와 평가만 수행했다.

핵심 판단은 다음과 같다.

- 학습 중 GroupNorm 확률, log-prob, PPO ratio와 보상 적분에서 핵심 수학 오류는 발견하지 못했다.
- **제공된 평가 함수는 YC 마스크를 GroupNorm 내부에 전달하지 않는다.** 동일 상태에서 proactive 확률이 14.01%에서 0.58%로 바뀐다.
- `greedy proactive = 0%`에는 flat argmax의 확률 분산 효과가 있다. 이것만으로 Target-credit 실패를 판정하면 안 된다.
- **Destination이 충분히 학습됐다는 기존 해석은 과하다.** 가장 높은 확률을 가진 후보는 낮은 스택을 선호하지만, 조건부 행동분포는 거의 균등하다.
- 현재 critic은 거의 상수이며, 세부 Target ETA를 표현하지 못한다. rollout 경계 밖의 지연 효과를 이 critic에 맡기는 구조가 우선 점검할 가설이다. 이것을 유일한 원인으로 확정하지는 않는다.

## 1. 실제 코드 기준 모델 재구성

### 1.1 기준 코드와 자료의 경계

기준 환경은 handoff 안의 `baseline_full_project/ACK2026_yard_marl_V4_CONTINUE_20260921.zip`에서 확보했다. 그 위에 `groupnorm_code/train_yc_marl.py`와 `groupnorm_code/test_v4_final.py`를 덮어쓴 별도 감사용 사본을 구성했다. 이 사본을 이하 `canonical_code/`라고 부른다.

별도로 제공된 `ACK2026_yard_marl_V4_GROUPNORM_20260921.zip`의 두 코드 파일은 handoff의 파일과 바이트 단위로 동일했다. 체크포인트는 `strict=True`로 모든 키와 shape를 정상 로드했다.

| 항목 | 확인 결과 |
|---|---|
| 기준 체크포인트 | `groupnorm_12k/resource_marl_final.pt` |
| 체크포인트 SHA-256 | `2bbd3a2e795a215d3fcbd654d58fad154d4298b55bb50ce92f54ad50b82ddb59` |
| 기준 학습 코드 SHA-256 | `5ae663aa2b9d872a13c8fb1fbf3439a1ef2a64b1a6849291b8f3b1ba9e60ff5e` |
| 기준 환경 코드 SHA-256 | `b12d561bebc1736a464624cd1fa8b260969d05e0cf36b38be0833751235b3e04` |
| 실행 환경 | Python 3.12.14, PyTorch 2.14.0+cpu, NumPy 2.3.5, SciPy 1.17.0, CPU 1 thread |
| 학습 실행 | 없음 |
| 주요 평가 실행 | 21 episodes: 공통 heuristic 3 + 정책·마스크 비교 12 + Target 규칙 반사실 비교 6 |

`train_yc_marl.py`에는 `nested_group_normalized_flat_yc_logits()`와 Q critic 클래스도 남아 있다. 그러나 기준 actor의 `forward()`는 179행에서 **2단계 `group_normalized_flat_yc_logits()`를 호출**하고, 체크포인트 설정은 `use_action_q_critic=False`다. 따라서 현재 모델을 autoregressive 또는 action-conditioned Q 모델로 재구성하면 틀린다.

### 1.2 환경과 의사결정

| 구성 | 실제 코드의 동작 |
|---|---|
| 범위 | 수입 컨테이너만, 4 blocks × 25 stacks × 4 tiers |
| 초기 재고 | 268개, block당 67개 |
| 물리 용량 | 400 slots |
| 신규 storage 입고 허용 | block당 `used + reserved < 97`인 경우만 허용; relocation 여유 3 slots 보존 |
| 크레인 | block당 고정 YC 1대, inter-block 이동 없음 |
| 시간 | 분; 물리 이동 1회 2분 |
| 외생 요청 | storage와 retrieval 각각 20/h Poisson, 480분 전까지 생성 |
| 반출 대상 | 초기 재고 중 사전 선정된 대상만 해당 episode에 반출 |
| 신규 입고 | 이번 episode에는 반출하지 않음; 미래 pickup proxy는 median 72h lognormal, 24–168h clipping |
| proactive | 아직 truck이 오지 않은, 적재되어 있고 blocker가 있는 대상; 관측 ETA lead가 `(0, 180]`분일 때 노출 |
| mandatory | retrieval와 storage 중 대기시간에 따른 결정적 규칙; 세부 작업과 reactive destination은 휴리스틱 |
| idle | mandatory가 없고 proactive 후보가 있을 때만 노출 |
| 종료 | 480분 이후 proactive·외생 요청 종료, 남은 mandatory 작업을 drain한 뒤 종료 |

실제 wrapper는 `ResourceMARLYardEnv`라는 **custom Gym-like 클래스**다. 이번 코드에는 Gymnasium `Env` 상속이나 표준 `spaces` 객체가 없다. 의사결정은 storage 또는 어느 한 YC가 활성화되는 비동기 순서이며, 여러 물리 작업은 이벤트 큐에서 동시에 진행될 수 있다.

Storage는 100개 stack 중 하나를 고른다. YC는 다음 2,502개 flat ID를 사용한다.

\[
a=0:\ Mandatory,\quad a=1:\ Idle,\quad a=2+25t+d:\ Proactive(t,d)
\]

여기서 `t = source_stack × 4 + tier`, `d = destination_stack`이다. proactive는 Target 자체를 이동하는 것이 아니라 **그 Target 위의 현재 최상단 blocker 1개**를 옮긴다. 두 Target이 같은 source stack에 묻혀 있으면 서로 다른 ID가 같은 물리 이동을 뜻할 수 있다.

### 1.3 Actor와 critic

| 구성 | 입력과 구조 |
|---|---|
| Storage actor | 436차원; block/stack permutation을 고려하는 shared encoder와 stack scorer |
| YC actor | 22,518차원 = context 18 + pair 2,500 × feature 9 |
| YC operation head | context 18만 보고 Mandatory / Idle / Proactive logits 생성 |
| YC pair scorer | 모든 pair에 같은 9→16→16 Tanh encoder 적용, context embedding과 결합해 scalar score 생성 |
| 공유 | 4개 YC가 같은 actor weights 사용; 활성 block의 local context와 pair feature는 각기 다름 |
| Value critic | global observation 440차원, permutation-invariant encoder → scalar value |
| Q critic | 모델 객체와 가중치는 존재하지만 현재 학습·추론에 사용하지 않음 |

pair feature 순서는 다음과 같다. `yc_marl_env.py::_proactive_pair_features()`, 1262–1313행에서 확인했다.

| 인덱스 | Feature | 정규화 |
|---:|---|---|
| 0 | feasible | 0/1 |
| 1 | Target ETA lead | `/180`, proactive 후보에서는 0–1 |
| 2 | ETA advance | `max(-eta_delta, 0)/60`, 최대 2 |
| 3 | blocker count | `/4` |
| 4 | Target capacity slack | `/180`, -2–2 clipping |
| 5 | moving blocker ETA lead | `/1440`, -1–2 clipping |
| 6 | destination height | `/4` |
| 7 | destination inversion count | `/4` |
| 8 | destination nearest ETA lead | `/1440`, 최대 2 |

`ResourceCooperativeModel.value()`는 YC의 pair feature를 받지 않는다. global critic의 stack feature는 height, coarse inversion, queue work, feasible 네 가지다. YC 시점 inversion 기준은 `now + 720`분이라, 같은 stack의 서로 다른 수십 분 단위 Target ETA를 구분하지 못할 수 있다.

### 1.4 GroupNorm과 실제 학습 설정

유효 proactive pair 집합을 \(F(s)\), operation logit을 \(u\), pair score를 \(z\)라고 하면:

\[
\ell_0=u_M,\qquad \ell_1=u_I,\qquad
\ell_{t,d}=u_P+z_{t,d}-\log\sum_{(i,j)\in F(s)}e^{z_{i,j}}.
\]

이후 같은 mask로 Categorical을 구성한다. 유효한 base operation은 현재 환경에서 Mandatory 또는 Idle 하나이므로:

\[
\pi(t,d\mid s)=P(P\mid s)\,P(t,d\mid P,s).
\]

sampling과 PPO update가 모두 이 동일한 joint log-prob를 사용한다. 기존 코드가 target log-prob 또는 destination log-prob를 별도로 더하는 구조는 아니다.

아래는 클래스 기본값이 아니라 **실제 체크포인트의 config**다.

| 설정 | 실제 값 |
|---|---:|
| 저장된 마지막 continuation의 `total_steps` | 6,144 |
| 이전 초기화 경로 | `groupnorm_randominit_6k_cont/resource_marl_final.pt` |
| seed | 15 |
| rollout 기본 크기 | 512 decisions |
| 최소 storage / YC samples | 64 / 128 |
| 최대 rollout 배수 | 4 |
| PPO epochs / minibatch | 2 / 256 |
| gamma / lambda | 1.0 / 1.0 |
| learning rate | 0.0003 |
| clip coefficient / gradient norm cap | 0.2 / 0.5 |
| Storage entropy coefficient | 0.01 |
| YC operation / normalized pair entropy coefficient | 0.001 / 0.0001 |
| value coefficient | 0.5; 코드의 `v_loss` 안에 추가 0.5가 있음 |
| proactive 초기 bias | -2.197224577, 한 base와 경쟁 시 약 10% |

`12k`는 이전 6k와 continuation을 합친 이름이다. 이번 패키지에는 이전 체크포인트와 모든 학습 로그가 없어 **정확한 누적 12k 계보를 독립적으로 복원할 수는 없다.** 현재 가중치와 저장된 마지막 실행 설정은 확인했다.

### 1.5 보상

\[
r_t=-\frac{\Delta W_T}{N_T}-\frac{\Delta W_S}{N_S}
-0.1\frac{2(\Delta n_{reactive}+\Delta n_{proactive})}{N_T}.
\]

`W_T`와 `W_S`는 단순 queue 대기만이 아니라 **요청/도착부터 물리 작업 완료까지**의 미완료 개수 적분이다. 기본 mandatory 반출·입고 이동은 delay에 포함되지만 extra move 항에는 들어가지 않는다. local shaping 두 개는 0이다.

drain이 완료되면 다음 등식이 성립한다.

\[
-\sum_t r_t = J
=\overline{D_T}+\overline{D_S}+0.1\frac{T_{YC,extra}}{N_T}.
\]

## 2. 구현 오류·불일치와 재현 결과

### 2.1 요청한 12개 구현 항목의 판정

| 항목 | 판정과 근거 |
|---|---|
| 1. flat index ↔ Target/Destination | 정상. 2,500개 ID 왕복 검사 및 실제 stack/tier/이동 컨테이너 확인 |
| 2. mask가 물리 불가능만 제거하는가 | **아님.** ETA 180분 범위, overdue ETA 제외, mandatory 때 idle 금지, storage 97-slot admission 규칙도 포함 |
| 3. feature indexing | 검사 범위 정상. 한 실제 상태의 133개 feasible pair를 물리 상태와 독립 계산해 최대 오차 5.38e-8 |
| 4. GroupNorm log-prob | mask를 내부까지 넘기면 정상. 명시적 분해식과 최대 확률 오차 1.11e-16, log-prob 오차 1.78e-15, float64 검사 |
| 5. PPO old/new ratio | 학습 경로 정상. 실제 상태 64개에서 single rollout 계산과 batched update 재계산의 최대 ratio 오차 1.43e-6, float32 수준 |
| 6. entropy와 action count | operation mass에서 pair 개수 bias 제거 확인. 조건부 entropy는 `log(K)`로 정규화; 실제 분포가 매우 균등한 것은 별도 문제 |
| 7. masked normalization | feasible pair 0/1/10/100/500/2500 검사에서 NaN 없음, masked probability 0 |
| 8. reservation/release | 선택 action의 대상 blocker와 destination 일치. 12개 정책 비교 episode의 매 step에서 예약·용량·컨테이너 보존 검사 통과 |
| 9. reward interval와 drain | 정상. 12개 episode의 `episode_return + J` 최대 절대오차 2.13e-14; wait-area도 개별 완료 지연 합과 일치 |
| 10. N_T/N_S 공정성 | 같은 scenario seed의 외생 요청으로 고정, 정책 중 변하지 않음. 정책별 동일 cohort와 요청 수 확인 |
| 11. critic input | **표현 불충분 확인.** 세부 ETA가 다른 상태가 같은 critic observation으로 들어감 |
| 12. GAE·episode·drain | recurrence와 terminal reset은 정상. gamma=lambda=1이므로 시간 할인 소실은 아님. rollout 절단과 약한 bootstrap critic의 결합은 원인 가설 |

### 2.2 P1 — 평가 함수의 GroupNorm mask 누락

**위치:** `evaluate_yc_policies.py::run_episode()`, 30행. 33행에서 Categorical mask를 적용하더라도, 이미 잘못 계산된 pair normalization은 복구되지 않는다.

```python
# 제공된 MARL 평가 경로
logits = model.yc_logits(obs)
dist = masked_distribution(logits, mask)

# 학습과 동일한 확률을 평가하려면 필요한 호출
logits = model.yc_logits(obs, mask)
dist = masked_distribution(logits, mask)
```

mask가 생략되면 GroupNorm은 2,500개 전체 pair를 normalization 분모에 넣는다. 이후 불가능한 pair를 지우면서 proactive group의 확률 질량도 함께 줄인다. 단순한 masking 스타일 차이가 아니라 **다른 정책분포를 평가하는 오류**다.

동일한 heuristic trajectory의 2,094개 상태에서:

| 호출 방식 | 평균 P(Proactive) |
|---|---:|
| 학습과 동일하게 내부 mask 전달 | 0.140098 = **14.01%** |
| 제공 평가 코드처럼 내부 mask 생략 | 0.005823 = **0.58%** |

이 오류는 현재 학습 수식 자체의 오류가 아니다. `train_yc_marl.py`의 sampling 436행과 PPO 재계산 506행은 모두 mask를 전달한다. 과거 평가 전부가 잘못됐다고 단정할 수도 없다. 제공된 **평가 함수가 잘못된 것**은 확정이며, 다른 외부 진단 스크립트 사용 여부는 별도로 확인해야 한다.

### 2.3 P1 — greedy 0%의 해석 문제

**위치:** `evaluate_yc_policies.py::run_episode()`, 34행의 flat `argmax()`.

유효 base action이 하나일 때, 가장 높은 conditional pair 확률을 \(q_{max}\)라 하면 proactive가 flat greedy로 선택되려면:

\[
P(P)q_{max}>1-P(P)
\quad\Longleftrightarrow\quad
P(P)>\frac{1}{1+q_{max}}.
\]

현재 상태 집합에서 평균 feasible pair는 96.70개이고, 위 **상태별 필요 group probability의 평균은 98.36%**다. 현재 실제 group probability 범위는 11.27–17.14%다.

따라서 14%의 group mass가 여러 pair에 퍼져 있는 정책에서 greedy 0%가 나오는 것은 자연스럽다. flat argmax 자체가 수학적으로 틀린 것은 아니지만, stochastic PPO의 탐색·사용 행동을 설명하는 지표로 그대로 대체할 수는 없다.

또한 지금은 operation 수준 argmax로 바꿔도 P(Proactive)<50%라 모두 base action을 고른다. **group-greedy로 변경하면 해결된다는 제안도 현재 증거와 맞지 않는다.**

### 2.4 P1 — critic의 세부 ETA aliasing과 거의 상수인 예측

**위치:** `yc_marl_env.py::global_observation()`, 1156–1210행; 특히 1193–1206행. `critic_observation()`, 1332–1333행. `v4_networks.py::PermutationInvariantCritic`, 26–32행.

실제 seed 301, 시각 0.821분, `I3_049`의 관측 ETA lead를 다음처럼 변경했다.

| 항목 | 변경 전 | 변경 후 |
|---|---:|---:|
| Target ETA lead | 31.403분 | 61.403분 |
| actor observation 최대 변화 | — | 0.166667 |
| critic observation 최대 변화 | — | **0.0** |
| critic 예측 | -8.1785936 | **-8.1785936** |

이는 actor가 구분하는 중요한 관측을 critic이 구조적으로 구분하지 못하는 사례다. 다만 value critic이 action 자체를 입력받지 않는 것은 정상적인 PPO 설계이며, **Q critic이 없다는 사실을 버그로 보지는 않는다.**

정상 mask를 사용한 stochastic rollout 3개, 총 3,500 transitions에서도 다음을 확인했다.

| 진단 | 결과 |
|---|---:|
| value 예측 범위 | -8.180284 ~ -8.177218 |
| 실제 episode-end MC return 범위 | -11.087954 ~ -0.011560 |
| MC return explained variance | **0.000074** |
| MC return RMSE | **4.032600** |
| chunk 내 advantage와 decision 순서의 평균 상관 | 0.989708 |

남은 시간이 줄면 남은 누적 비용도 줄어야 하는데, 현재 value는 이를 거의 반영하지 않는다. 이것은 표현 제한과 별개로 **현재 critic의 학습 상태 자체가 좋지 않다는 증거**다. 정확한 학습 실패 원인은 추가 개입 없이 하나로 확정할 수 없다.

### 2.5 기존 Destination 학습 해석의 수정

기존의 `selected Target`, `destination height`는 실제 deployed action이 아니라, **proactive를 택한다고 가정했을 때 최고 확률인 후보**의 특성이었다. 전체 greedy rollout에서는 proactive를 한 번도 실행하지 않는다.

| 동일 2,094개 상태에서 측정 | Target ETA lead | Destination height | Destination inversion |
|---|---:|---:|---:|
| 최고 확률 proactive pair | 0.622787 | 0.284981 | 0.164518 |
| 학습된 conditional pair 분포의 기대값 | 0.608582 | 0.587116 | 0.302073 |
| feasible pair 균등분포의 기대값 | 0.607340 | 0.588395 | 0.306236 |

정규화된 conditional pair entropy는 **0.999880**이고 pair score 표준편차는 0.030522다. Destination에 약한 순위 선호는 형성됐지만, 샘플링되는 행동분포의 실질적 집중은 거의 없다.

따라서 현재 문제는 **Target만 못 배우고 Destination은 완성된 상태**라고 보기 어렵다. Target urgency가 특히 약하고, 동시에 pair 정책 전체가 아직 매우 약하게 분화된 상태라고 보는 편이 정확하다.

### 2.6 비교모델과 패키지의 불일치

제공된 테스트 함수를 직접 실행한 결과는 **13개 중 12개 통과, 1개 실패**다. pytest fixture가 없는 함수를 직접 호출했으며, 오류를 무시하거나 테스트를 고치지 않았다.

- 실패 위치: `test_v4_final.py::test_model_heads_match_100_and_2502_actions()`, 176행.
- 실제 오류: `CentralizedSingleModel.yc_logits() takes 2 positional arguments but 3 were given`.
- 원인: 테스트는 single model에 mask 인자를 넘기지만, baseline의 `train_yc_single.py::yc_logits()`, 53행은 이를 받지 않는다.

더 중요한 비교 confound도 있다.

| 위치 | 문제 | 의미 |
|---|---|---|
| `train_yc_single.py:53–67, 72–85, 138` | single은 legacy flat logits, -7.2 per-pair 초기화, 전체 Categorical entropy 사용 | MARL만 GroupNorm과 구조화 entropy를 사용하므로 향후 차이를 MARL/CTDE 효과만으로 귀속할 수 없음 |
| `train_yc_single.py:128–130` | minibatch slice마다 새 permutation을 생성 | epoch당 모든 sample을 한 번씩 방문하지 않음. n=1024, batch=128 예시에서 659개만 방문, 365개 누락/중복 |
| `train_yc_marl.py:395–414, 531–536` | checkpoint에는 optimizer/RNG/누적 step 상태가 없음 | `init_checkpoint` continuation은 정확한 중단 복원이 아니라 가중치 warm-start |

MARL trainer의 permutation은 496행에서 epoch마다 한 번 생성하므로 위 minibatch 오류가 없다. single의 문제를 현재 MARL 학습의 원인으로 섞어서 해석하지 않았다.

### 2.7 환경·관측의 추가 한계

**마스크 설명과 실제 규칙.** `yc_marl_env.py::_proactive_eligible()`, 375–393행은 관측 ETA 181분 또는 음수 lead인 대상의 물리적으로 가능한 이동도 배제한다. `yard_simulator.py::feasible_storage_slots()`, 56–63행은 3개 slack slot을 신규 storage에서 제외한다. 현재 환경을 임의로 바꿀 필요는 없지만, “물리 불가능만 mask”라는 문장은 수정해야 한다. capacity slack 점수 자체는 proactive hard mask에 쓰이지 않았다.

**대기시간 관측의 reset.** reactive blocker 이동 완료 때 `yc_marl_env.py:270–275`에서 새 `RETRIEVAL_STEP`의 `created_at`을 현재 시각으로 지정한다. `_yc_context_observation():1230, 1247`은 이를 retrieval age로 사용한다. seed 301의 3번째 decision에서 truck은 이미 2분 기다렸지만 이 feature는 0이었다. 이 feature를 truck 누적 대기시간으로 설명하면 틀린다. critic에도 해당 truck의 실제 누적 대기가 세밀하게 표현되지 않는다.

**이벤트 시각과 hidden actual.** `_eta_update_times():822–830`은 `actual-180, …, actual-30`에 업데이트를 배치한다. 관측 ETA 값은 noisy지만 업데이트 발생 시점은 true arrival에 결부되어 있다. 실제 arrival 값을 바꿨을 때 현재 actor observation/mask가 그대로인 것은 확인했으나, 이벤트 timing을 통한 간접 정보까지 차단된 것은 아니다. 특히 역사 정보를 쓰는 정책에는 더 중요한 가정이다. 현재 모델의 성능 향상을 위해 임의로 바꾸기보다 synthetic ETA 생성 규칙으로 명시해야 한다.

**Target ID와 물리 행동의 중복.** 동일 source stack에서 서로 다른 buried target을 골라도 같은 top blocker와 destination을 고르면 물리 이동이 같다. 실제 두 action을 독립 실행해 yard, reservations, container 상태, 완료 후 YC observation이 동일한 사례를 확인했다. 공통 상태 집합에서 이런 중복 ID의 비중은 상태별 평균 2.79%다. 전체 실패를 설명할 정도로 크다고 단정할 수 없으나, Target rank를 해석할 때 같은 이동을 만드는 Target들을 구분해야 한다. 이동 중 pending-target 표시는 다를 수 있어 전체 event history까지 언제나 같다고 주장하지 않는다.

### 2.8 기존 진단의 재현과 재현되지 않은 부분

공통 trajectory는 정보 인식 heuristic, scenario seeds 301–303이다. proactive 후보가 있는 YC 상태는 2,094개, Target이 2개 이상 있는 상태는 1,895개로 원자료와 일치했다.

| 진단 | 기존 값 | 독립 재현 |
|---|---:|---:|
| P(Proactive), 2,094 states | 약 0.1401 | **0.140098** |
| greedy proactive | 0% | **0%** |
| 최고 proactive 후보 Target ETA lead | 약 0.623 | **0.622787** |
| nearest feasible Target ETA lead | 약 0.336 | **0.336252** |
| 최고 proactive 후보 destination height | 약 0.285 | **0.284981** |
| 최고 proactive 후보 inversion | 약 0.165 | **0.164518** |
| ETA-swap 평균 expected-direction effect | -0.001782910 | **-0.001782910** |
| ETA-swap top target 변경률 | 9.0237467% | **9.0237467%** |
| 1,895-state subset의 최고 후보 ETA | 0.633211 | **0.633211** |

`eta_swap.csv`의 1,895개 행을 직접 대조했다. seed/near/far ID 불일치는 0개, effect 최대 절대차는 1.64e-7이었다. README의 약 0.623과 summary의 약 0.633은 **상태 집합 차이**로 설명된다.

별도로 명시적 percentile을 계산했다. Target이 2개 이상인 상태에서, 선택 후보보다 ETA가 작은 Target 수를 `N_target-1`로 나눈 값의 평균은 **0.503346**이었다. 0은 가장 임박, 1은 가장 먼 Target이다. 이는 사실상 중간 순위다.

다음은 정확한 숫자 재현으로 포장하지 않았다.

- 원래 fixed-destination 진단은 1,692 states, Spearman -0.08648인데, 어떤 destination·필터를 썼는지 스크립트가 없다. 이번에는 “최고 pair의 destination에 고정”한 명시적 절차로 1,895 states, -0.03582를 얻었다. **동일 실험이 아니므로 원래 값이 틀렸다고 판정하지 않는다.**
- feature mean ablation의 원래 500-state 선택·평균 치환 절차도 제공되지 않았다. 이번에는 첫 500개 multi-target state에서 feasible pair의 상태 내 평균으로 치환했다. Target ETA의 TV는 0.001439, moving blocker ETA의 TV는 0.011786으로, 상대적 중요도 방향은 일치하지만 숫자는 다르다.
- raw-score gradient 진단은 명시적으로 균등 간격 64 states를 사용했다. scale-adjusted 민감도는 Target ETA 0.004325, ETA advance 0.015376, moving blocker ETA 0.028539였다. 원본과 동일 표본이라는 주장은 하지 않는다.
- ETA lead만 바꾸는 feature-level swap은 slack·context와의 일관성을 완전히 유지하는 물리 반사실이 아니다. feature 민감도 검사로 해석해야 하며, 전체 ETA 의존성의 인과 추정으로 확대하면 안 된다.

### 2.9 실제 rollout과 Target 반사실 확인

아래는 seeds 301–303, 각 방법·seed당 1회 평가 평균이다. stochastic 평가의 policy seed는 `scenario_seed×100`이다. 학습을 추가로 돌린 결과가 아니며 통계적 우월성 주장이 아니다.

| 평가 방식 | Truck delay | Storage delay | Proactive moves | J |
|---|---:|---:|---:|---:|
| 12k 모델, 올바른 mask, stochastic | 6.1523 | 3.5640 | 131.00 | **10.0465** |
| 12k 모델, 내부 mask 누락, stochastic | 6.2398 | 3.4704 | 12.67 | **9.9378** |
| 12k 모델, 올바른 mask, flat greedy | 5.1923 | 3.0146 | 0.00 | **8.3775** |
| heuristic proactive ON | 3.8987 | 3.3754 | 440.33 | **7.8846** |
| heuristic proactive OFF | 5.1286 | 2.8318 | 0.00 | **8.1296** |

확률 오류를 바로잡았다고 성능이 바로 좋아지는 것은 아니다. 수정 후 stochastic policy가 거의 균등한 pair를 더 많이 실행하기 때문이다. 또한 stochastic과 greedy 비교는 storage의 sampling 방식도 다르므로, 그 차이 전부를 YC proactive 효과로 해석하면 안 된다.

추가로 operation은 mandatory-first, destination은 같은 기존 휴리스틱으로 유지하고, Target 선택만 바꾸었다.

| Target 진단 규칙 | 평균 J | 평균 Truck delay | 평균 Rehandling |
|---|---:|---:|---:|
| nearest ETA | **7.8846** | **3.8987** | **32.33** |
| most blockers | 8.2518 | 4.1157 | 40.33 |
| farthest ETA | 9.0328 | 4.9245 | 85.67 |

이 3개 시나리오에서는 ETA urgency가 의미 있다는 기존 방향을 독립적으로 지지한다. 다만 seed 303에서는 most-blockers의 J가 nearest보다 낮았다. **nearest를 모든 상태의 정답 또는 학습 label로 사용하라는 근거는 아니다.** 위 규칙은 감사용 비교이며 actor에 주입하지 않았다.

## 3. Why Target credit fails — 원인 순위

현재 증거는 단일 원인을 확정하는 실험이 아니라, 검증된 사실을 바탕으로 다음 개입의 우선순위를 정하는 근거다. 학습 자체의 문제와 평가상 문제를 구분한다.

### 3.1 학습 실패 가설의 우선순위

| 순위 | 가설 | 증거와 판단 |
|---:|---|---|
| 1 | critic/value-function mismatch | 세부 ETA를 바꿔도 critic 입력이 같고, 실제 rollout value도 거의 상수다. 남은 비용을 설명하지 못하는 critic이 현재 가장 강하게 확인된 약점 |
| 2 | 긴 지연 효과와 rollout 경계의 결합 | 실제 config는 512 기준 rollout. 감사용 재구성에서 proactive 393건 중 49.36%는 Target truck이 chunk 이후 도착. 그 효과를 잘못된 value로 대체할 가능성 |
| 3 | 유효 pair 학습 표본과 학습 신호 부족 | conditional entropy 0.99988, Target percentile 0.503. pair의 task-reward gradient는 proactive가 실제 선택된 transition에서만 발생; base action log-prob의 pair gradient는 0으로 확인 |
| 4 | actor representation 및 Target/Destination 결합 | indexing은 정상이고 ETA gradient도 0이 아니다. 작은 shared pair encoder가 어렵게 학습할 가능성은 남지만, 새 representation이 해결책이라는 직접 근거는 없음 |
| 5 | event timing / observation aliasing | critic의 coarse stack 요약, task-age reset, 진행 중 작업의 세부 잔여시간 누락, 일부 동일 물리 action의 Target ID 중복이 존재. 이들 각각의 성능 기여는 아직 분리되지 않음 |
| 6 | ETA feature scaling | Target lead 자체는 0–1 범위, 표준편차도 충분하여 단순 포화·인덱스 오류설은 약함. 미래 pickup/moving blocker의 clipping은 있지만 Target urgency 실패의 주원인으로 확정 불가 |
| 7 | shared YC policy가 block을 구별하지 못함 | 활성 block의 local context와 pair 정보가 다르게 들어간다. 동일 구조 block에서 ID를 주지 않는 것은 그 자체로 버그가 아니며, 현재 증거로 우선순위 낮음 |
| 8 | PPO probability/log-prob 구현 오류 | **현재 학습 경로에서는 지지되지 않음.** 확률 분해·ratio·mask 검사를 통과. 별도로 평가 함수의 확률 오류는 확정 |
| 9 | 보상 적분·normalization·drain 누락 | **검사 범위에서 지지되지 않음.** reward-KPI 항등식, 완료 작업, 요청 수가 일치 |

exploration을 단순히 “Proactive를 전혀 안 해본다”라고 설명하는 것도 맞지 않는다. 올바른 stochastic 실행에서 3개 episode 동안 **393회의 proactive**가 실행되었다. 초기 약 10%보다 최종 held-out group probability 약 14%가 높으므로, 기준 GroupNorm 모델의 operation 확률이 붕괴했다고 말할 근거도 없다. 실패 branch의 확률 붕괴를 현재 모델에 옮겨 적으면 안 된다.

### 3.2 gamma와 lambda에 대한 정확한 해석

`compute_gae()`, `train_yc_marl.py:337–345`는 표준 recurrence를 구현한다. 현재 gamma=lambda=1이므로 episode 끝까지 수집한 구간에서는 return이 실제 남은 누적 reward로 telescoping된다. **할인율 때문에 먼 미래 보상이 사라졌다는 진단은 틀리다.**

rollout이 중간에서 끝나면:

\[
\hat G_t=\sum_{k=t}^{T_{cut}-1}r_k+V(s_{T_{cut}})
\]

가 된다. 문제 가설은 discount가 아니라 cutoff 이후의 action-dependent 결과를 잘 설명하지 못하는 bootstrap value다. [GAE 원논문](https://arxiv.org/pdf/1506.02438)의 bias/variance 구분과 일치하는 해석이다.

감사용 3개 episode에서 512 기준 chunk를 재구성했을 때 cut return과 full MC return의 평균 절대차는 4.094620이었다. 다만 이 진단은 **episode마다 chunk counter를 다시 시작한 고정 정책 검사**이며, 실제 trainer의 episode를 가로지르는 전체 학습 로그를 복원한 결과는 아니다.

또한 terminal이 없는 한 chunk에서는 gamma=lambda=1일 때 이 차이가 모든 transition에 같은 상수로 들어갈 수 있다. 코드의 batch advantage centering이 그 상수 부분을 상쇄할 수 있으므로, **4.09라는 차이만으로 정책 gradient가 그만큼 편향됐다고 주장하지 않는다.** 확인된 것은 critic의 낮은 설명력, 경계 밖 지연 효과의 비중, 시간 순서에 강하게 묶인 advantage다.

### 3.3 gradient와 entropy에서 과잉 해석하지 않을 점

한 진단 minibatch에서 gradient norm은 Storage actor 0.3361, YC actor 0.0756, critic 15.0370이었다. joint gradient clipping 배수는 약 0.03324였다. `train_yc_marl.py:517–519`의 공통 clipping이 actor/critic 학습을 결합하는 점은 기록해야 한다. 그러나 Adam은 gradient rescaling에 부분적으로 불변하므로 **actor의 실제 학습률이 정확히 30분의 1이 됐다는 결론은 내릴 수 없다.** 단 한 batch 측정으로 이것을 주원인으로 확정하지 않는다.

현재 entropy는 `H(operation) + H(pair|proactive)/log(K)` 형태의 별도 regularizer다. 두 번째 항이 `P(Proactive)`로 가중된 표준 joint entropy와 같은 것은 아니다. 그렇다고 잘못된 확률분포를 만드는 오류는 아니며, operation의 pair-count bias가 다시 생겼다는 증거도 발견하지 못했다. 거의 균등한 조건부 분포의 원인이 entropy coefficient라고 단정하려면 별도 ablation이 필요하다.

PPO ratio와 clipping 판정은 [PPO 원논문](https://arxiv.org/pdf/1707.06347)의 joint action probability ratio를 기준으로 했다. 조건부 분해라도 최종 sampled flat action의 정확한 log-prob를 쓰면 현재 방식은 성립한다.

## 4. 가장 작은 다음 실험 — 1개만 제안

### 4.1 실험 전에 필요한 정확성 정리

아래는 성능을 높이기 위한 실험 변수가 아니라, 학습과 평가를 같은 의미로 만드는 선행 조건이다. 이번 감사에서는 원본에 적용하지 않았다.

1. `evaluate_yc_policies.py:30`의 MARL 호출에 mask를 전달한다. single 호출에 무조건 같은 인자를 추가하면 현재 signature 오류가 나므로 모델별 API를 먼저 맞춘다.
2. stochastic flat sampling을 학습분포 그대로 평가하는 기본 지표로 두고, flat greedy는 별도 실행 방식으로 같이 보고한다. deterministic 운영이 필수라면 그 조건을 따로 고정한다.
3. 테스트의 single API 불일치와 single PPO의 permutation 오류를 정리한다. 최종 MARL 대 single 비교 전에 normalization·entropy·초기화 조건을 맞추거나 별도 비교 요인으로 명시한다.
4. checkpoint에 실제 cumulative decisions, update count, scenario seed manifest, 설정, optimizer/RNG 상태 또는 의도적인 warm-start 여부를 남긴다. 진단에 사용한 상태 선택 스크립트도 보존한다.

### 4.2 실험: episode-complete rollout 대 현재 cutoff

**변경할 것은 rollout 종료 기준 하나다.** 새로운 Target head, Q critic, ETA score, top-K, YC BC, proactive bonus를 넣지 않는다.

| 항목 | 설계 |
|---|---|
| 가설 | 지연된 물리 효과를 약한 cutoff value로 대체하는 것이 pair 학습을 방해한다면, episode 완료까지 수집할 때 학습 신호가 개선된다 |
| 대조군 A | 평가 mask만 정리한 현재 모델. 현행 512 기준과 최소 role count를 만족하면 update |
| 처리군 B | 같은 조건을 만족해도 현재 episode의 mandatory drain과 terminal까지 수집한 뒤 update |
| 출발점 | 동일한 canonical 12k 가중치; 두 군 모두 같은 방식으로 optimizer 초기화 또는 동일 상태 복원 |
| 유지 항목 | actor/critic 구조, reward, gamma=1, lambda=1, LR, epochs=2, minibatch=256, entropy, 단일 운영조건 |
| pilot 예산 | 군별 2k→4k→6k 추가 environment decisions 시점에서 점검; B는 마지막 episode 완료에 따른 실제 초과량 명시 |
| 반복 | 3개 사전 등록 training RNG seeds; A/B에 같은 scenario-seed 생성 규칙 사용 |
| 평가 | 301–303은 재현용 bank로만 유지. 모델 선택에는 별도 미사용 scenario 10개와 stochastic policy seed 3개를 사전 고정 |
| 기록 | 실제 decisions/episodes/proactive samples, optimizer updates·minibatches, MC EV/RMSE, operation mass, conditional entropy, J, component KPI |

**코드 위치:** `train_yc_marl.py::train_resource_marl()`, 417–461행; `rollout_ready()`, 354–358행. 현재 455–458행은 terminal 즉시 reset하기 때문에, “현재 episode가 끝났는지”를 별도로 유지하여 불필요하게 다음 episode를 시작하지 않도록 수집 종료 순서를 검토해야 한다. `compute_gae()`의 수식을 바꿀 필요는 없다.

**예상 관찰 변화:** B에서는 수집 구간 밖으로 나가는 동일 episode의 Target truck arrival이 없어지고, terminal 기준 return이 실제 누적 비용과 맞아야 한다. 원인 가설이 유효하면 critic의 시간별 비용 설명력, 실제 sampled pair의 구별력, J가 대조군보다 개선되어야 한다. 단지 top pair의 ETA가 작아지거나 proactive 횟수가 증가하는 것만으로는 성공이 아니다.

**실패 판정:** 6k에서 critic EV가 계속 0 부근이고 pair 분포 및 Target 무작위화 평가가 대조군과 구별되지 않으면 이 변경을 채택하지 않는다. 지연 경계를 제거했는데도 학습 신호가 개선되지 않으면, rollout 길이만으로는 해결되지 않는다는 결과다. 그때 critic 표현·최적화 문제를 별도 과제로 좁힌다.

episode-complete 수집은 update 시점과 batch 길이도 바꾼다. 따라서 이것은 “rollout 범위의 변경 실험”이지 critic 구조의 효과만을 고립시키는 실험은 아니다. 실제 optimizer update 수와 처리한 sample 수를 함께 보고해 비교를 숨기지 않는다.

## 5. Full 30k 학습 Go / No-Go 기준

아래 수치는 문헌의 보편적 기준이 아니라, **이번 프로젝트에서 학습 상태를 확인하기 위한 제안된 사전 운영 기준**이다. 결과를 본 뒤 유리하게 바꾸지 않는다.

| Gate | Go 조건 | 현재 상태 |
|---|---|---|
| 확률·테스트 무결성 | 현재 intended configuration 테스트 전부 통과; sampling/evaluation/update의 동일 상태 log-prob 차이 ≤1e-5(float32); invalid action 및 NaN 0 | 학습 계산은 통과, 평가와 single API는 실패 |
| 물리·보상 무결성 | validation 모든 episode 완료, 잔여 예약 0, 누락된 mandatory 0, `abs(return+J)≤1e-8` | 이번 검사 범위 통과 |
| return 처리 | B의 모든 학습 transition이 terminal까지의 episode return에 포함; 예산 끝에서 미완료 tail을 조용히 폐기하지 않음 | 현행 cutoff는 미충족 |
| critic 학습 | pilot 평가의 on-policy MC EV ≥0.20, RMSE가 동등 예산 대조군보다 ≥20% 감소; 3 training seeds 중 최소 2개에서 충족 | 현재 EV 0.000074, 미충족 |
| 실제 Target 사용 가치 | 아래 Target 무작위화 대조보다 J가 평균 ≥1% 낮고, 3 training seeds 중 최소 2개에서 개선 방향 일치 | 현재 미검증; 거의 균등한 분포로 근거 부족 |
| 전체 목표 | 같은 평가 protocol에서 B의 평균 J가 A보다 ≥3% 감소, 3 training seeds 중 최소 2개에서 개선; Truck/Storage/extra work를 함께 제시 | pilot 미실행 |
| 재현성·평가 분리 | 누적 step과 계보 확인, canonical 코드 고정, 진단/validation/final-test scenario 분리 | 원래 12k 전체 계보·일부 진단 스크립트 부족 |

Target 무작위화 대조는 operation probability와 학습된 `P(destination|target)`를 그대로 두고, `P(target|proactive)`만 feasible Target 간 균등하게 만드는 **평가 전용 intervention**이다. 이는 학습에 휴리스틱 Target score를 넣는 것이 아니며, Target 선택이 실제 목표 J에 기여하는지를 검사한다. 동일 물리 이동을 만드는 Target alias도 함께 집계하여 label 차이만을 성과로 세지 않는다.

`greedy proactive > 0%`, 더 낮은 Target ETA, 더 많은 proactive 횟수는 독립 Go 조건으로 쓰지 않는다. stochastic 정책으로 충분한 성능과 Target 사용 가치가 확인되면 flat greedy 0%만으로 탈락시키지 않는다. 반대로 실제 운영 방식을 deterministic flat greedy로 고정한다면 그 방식의 J와 행동을 별도로 통과해야 한다.

MARL이 heuristic을 이겨야 한다는 조건으로 reward를 조정하지 않는다. 위 gate는 **학습 가능성과 변경의 유효성**을 확인하는 기준이며, 30k 본실험에서 heuristic 또는 single PPO보다 우월하다는 결론은 별도 미사용 시나리오와 다중 학습 seed의 결과로 판단한다. 현재 301–305는 여러 pilot 선택에 재사용되어 왔으므로 최종 미사용 test set으로 취급하지 않는다.

**현재 최종 판정: No-Go.** 우선 평가 분포의 불일치를 바로잡고, actor 구조를 유지한 episode-complete pilot 1개로 다음 분기를 결정하는 것이 가장 작고 근거 있는 순서다.

## 재현 자료

함께 제공하는 `ACK2026_ASTRA_Audit_Evidence_20260921.zip`에는 다음을 포함한다.

- 수정하지 않은 감사 대상 `canonical_code/`와 기준 checkpoint.
- 감사 스크립트 `scripts/`: 고정 상태 재현, 확률·GAE 검사, 정책 rollout, 보조 관측 검사, 제공 테스트 실행.
- `evidence/`: JSON 요약, 상태별 CSV, rollout별 KPI, on-policy transitions, 오류 기록, 실행 버전과 원본 해시.
- 재현 절차 `REPRODUCE.md`.

큰 `state_bank.npz`는 ZIP에 넣지 않았다. `collect_diagnostics.py`로 같은 2,094개 상태를 다시 생성할 수 있다. 원본 소스의 문제는 보고서에 기록했으며, 감사 자료 안에서도 해당 코드를 임의로 수정하지 않았다.
