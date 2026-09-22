# ACK 2026 항만 야드 MARL V4 통합 수정 명세 — 단일 운영조건 확정본

## 0. 문서 목적

본 문서는 다음 내용을 하나의 최신 기준으로 통합한 수정 명세다.

1. `ACK2026_항만야드_MARL_Environment_V4_수정명세.md`
2. `ACK2026_항만야드_MARL_보상함수_재설계안.md`
3. 이후 확정된 YC Action 설계
4. 이후 확정된 **단일 운영조건 사용 방침**

이번 버전에서는 다음 사항을 확정한다.

- Low / Medium / High 부하조건 **완전 삭제**
- Training 시 workload range sampling **삭제**
- Evaluation 시 12 / 20 / 28 events/hour 비교 **삭제**
- Storage와 Retrieval은 하나의 고정 arrival rate \(\lambda^*=20/h\) 사용
- YC Action은 **Flat Target × Destination**
- Extra YC Work 정규화 분모는 **\(N_T\)** 사용

---

# 1. 최종 연구 문제

본 연구는 다음 문제를 다룬다.

> **4개 Block, 400개 physical slot, Block별 1대의 고정 YC로 구성된 수입 컨테이너 야드에서 신규 수입 컨테이너 적재와 기존 수입 컨테이너 반출 요청이 확률적으로 동시에 발생할 때, Storage Allocation과 ETA 기반 Proactive Relocation을 협조적으로 학습하여 Truck delay, Storage delay 및 불필요한 YC workload를 줄이는 문제**

두 종류의 RL 의사결정이 존재한다.

```text
[신규 수입 컨테이너 도착]
        ↓
Storage Allocation Agent
        ↓
100개 Stack 중 적재 위치 선택


[YC decision epoch]
        ↓
YC Operation Agent
        ↓
Mandatory
또는
Proactive(Target, Destination)
또는
Idle
```

---

# 2. Yard Environment

## 2.1 Layout

```text
Blocks = 4
Stacks per Block = 25
Max Tier = 4
```

총 physical capacity:

\[
4 \times 25 \times 4 = 400
\]

Block별 capacity:

\[
25 \times 4 = 100
\]

총 Stack 수:

\[
4 \times 25 = 100
\]

## 2.2 Yard Crane

```text
YC per Block = 1
Total YC = 4
```

YC는 다른 Block으로 이동하지 않는다.

이번 연구에서는 다음은 모델링하지 않는다.

- Block 간 YC redeployment
- stack 간 실제 gantry travel distance 차이
- trolley distance 차이
- bay/row별 이동시간 차이

모든 physical container move는 동일한 service time을 사용한다.

---

# 3. 시간 구조

모든 시간단위는 **minute**으로 통일한다.

```text
Operating horizon = 480 min
YC move time = 2 min / physical container move
```

다음 작업은 모두 1회당 2분으로 둔다.

- Storage
- Retrieval target handling
- Mandatory rehandling
- Proactive relocation

예:

```text
Target 위 blocker 2개
→ Mandatory rehandling 2회 = 4분
→ Target retrieval = 2분
→ 총 6분
```

---

# 4. 초기 야드 상태

## 4.1 Initial stock

```text
Initial containers = 268
Initial occupancy = 67%
```

Block별:

```text
Block 0 = 67
Block 1 = 67
Block 2 = 67
Block 3 = 67
```

각 Block에는 초기 33개의 빈 physical slot이 존재한다. 이 중 운영 중 신규 Storage admission은 Block당 최소 3개의 relocation slack slot을 남기도록 제한한다. 따라서 physical capacity는 100 slots/block을 유지하지만 신규 Storage가 점유할 수 있는 운영상 상한은 97 slots/block이다. 이 3-slot slack은 buried target의 source stack에 최대 2개의 빈 tier가 남을 수 있다는 점을 고려해, source 이외에 최소 1개의 relocation destination을 보장하기 위한 안전제약이다.

## 4.2 Initial layout

초기 컨테이너는 retrieval 순서에 맞게 정렬하지 않는다.

생성 절차:

```text
1. 각 Block에 67개 배치
2. height < 4인 Stack 중 하나 random sampling
3. 해당 Stack top에 배치
4. 67개가 될 때까지 반복
```

필수 조건:

- floating container 금지
- stack height > 4 금지
- Block별 정확히 67개
- 동일 seed → 동일 initial layout
- 동일 scenario seed → 모든 정책에 동일 layout 적용

초기 위치와 실제 Truck arrival 순서는 독립적으로 생성한다.

---

# 5. Container Cohort

모든 컨테이너는 Import container이지만 episode 내 역할을 두 집단으로 분리한다.

## 5.1 Initial Import Stock

Episode 시작 전에 이미 야드에 존재하는 268개 컨테이너.

이 집단만 현재 480분 episode의 Truck Retrieval 대상이 될 수 있다.

## 5.2 New Import Containers

Episode 중 선박 양하 후 yard에 유입되는 신규 컨테이너.

처리 흐름:

```text
Yard arrival
→ Storage Allocation
→ YC Storage
```

같은 episode에서는 Retrieval 대상이 되지 않는다.

권장 필드:

```python
container.cohort = "initial" | "new_inbound"
container.retrievable_this_episode = True | False
```

---

# 6. Arrival Process

Storage request와 Truck Retrieval request 모두 homogeneous Poisson process로 생성한다.

\[
N_S(t)\sim Poisson(\lambda_S t)
\]

\[
N_R(t)\sim Poisson(\lambda_R t)
\]

Inter-arrival time:

\[
\Delta t\sim Exp(\lambda)
\]

시간단위는 minute이므로 hour rate는 내부적으로:

\[
\lambda_{min}=\frac{\lambda_{hour}}{60}
\]

으로 변환한다.

---

# 7. 단일 운영조건

기존의 모든 Low / Medium / High 부하조건은 삭제한다.

다음 구조를 사용하지 않는다.

```text
Low / Medium / High workload
12 / 20 / 28 events/hour evaluation
λ ~ Uniform(12, 28) training
balanced load cycle
load-specific result table
load sensitivity analysis
```

최종 환경은 하나의 고정 arrival rate만 사용한다.

\[
\boxed{\lambda_S=\lambda_R=20/h}
\]

즉 모든 training/evaluation episode에서 평균 arrival intensity는 동일하다.

Episode마다 달라지는 것은 다음 stochastic realization뿐이다.

- Initial yard layout
- Poisson arrival timestamps
- Retrieval target mapping
- ETA noise
- New inbound future pickup proxy
- 기타 exogenous random variables

---

# 8. 고정 Arrival Rate 확정

Environment-only calibration을 통해 단일 operating point를 다음과 같이 확정한다.

\[
\boxed{\lambda^*=20\text{ events/hour}}
\]

따라서 모든 training 및 evaluation episode에서:

\[
\boxed{\lambda_S=\lambda_R=20/h}
\]

를 사용한다.

Calibration은 본 실험의 부하수준 비교가 아니라 하나의 운영조건을 정하기 위한 사전 환경설정 절차다. 12, 16, 20, 24 events/hour를 heuristic-only 환경에서 짧게 확인한 결과, 20/h에서 평균 YC utilization 약 0.84, 평균 max queue 약 3.67, 평균 mandatory drain 약 0.69분으로 자원경합이 존재하면서도 포화가 지속되지 않았다. 이후 본 실험에서는 arrival rate를 변경하지 않는다.

---

# 9. Episode Initialization 시 Event Stream 생성

Reward normalization을 위해 Storage와 Retrieval event 수를 episode 시작 시 알 수 있어야 한다.

따라서 두 stream 모두 initialization 단계에서 `[0,480)`의 전체 Poisson timestamp를 생성해 event queue에 등록한다.

```text
Storage timestamps:
{t1S, t2S, ..., tNS}

Retrieval timestamps:
{t1R, t2R, ..., tNT}
```

이로부터:

```text
N_S = 해당 episode의 Storage request 수
N_T = 해당 episode의 Retrieval request 수
```

를 episode 시작 시 확정한다.

---

# 10. Retrieval 대상 생성

절차:

```text
1. λ* 기준 [0,480) Retrieval timestamps 생성
2. Initial 268 containers 중 without replacement로 대상 선택
3. timestamp와 container를 1:1 mapping
4. container.truck_actual 저장
5. 해당 hidden truth를 기반으로 ETA 생성
```

Retrieval 대상은 Initial Import Stock에서만 선택한다.

## Retrieval event 수가 eligible initial container 수를 초과하는 경우

eligible initial container가 소진되면 추가 Retrieval event는 등록하지 않는다.

New inbound를 같은 episode Retrieval 대상으로 전환하지 않는다.

---

# 11. New Inbound Future Pickup Proxy

Storage Allocation Agent가 신규 컨테이너의 미래 retrieval priority를 고려할 수 있도록 future pickup proxy를 생성한다.

신규 컨테이너의 yard arrival이 \(t_i^S\)일 때:

\[
A_i=t_i^S+D_i
\]

Future pickup delay는 수입 컨테이너의 수일 단위 체류시간을 모사하기 위해 양의 왜도를 갖는 synthetic lognormal distribution으로 생성한다.

\[
D_i/60\sim Lognormal(\ln 72,\;0.5^2)\;\text{hour}
\]

그리고:

\[
24\le D_i/60\le168
\]

로 clipping한다. 즉 중앙값은 약 72시간(3일)이며 허용범위는 1~7일이다.

따라서 모든 신규 inbound의 future pickup은 현재 480분 episode 밖에 위치하며 같은 episode에 Retrieval event는 발생하지 않는다.

이 분포는 특정 터미널의 dwell-time 데이터를 직접 fitting한 값이 아니라, Storage Agent에 현실적인 시간척도의 미래 retrieval priority를 제공하기 위한 synthetic proxy다.

---

# 12. Dynamic Truck ETA

## 12.1 Hidden actual arrival

Retrieval 대상 initial container \(i\)에는 실제 service request time \(A_i\)가 존재한다.

```text
truck_actual = A_i
```

Actor에게 직접 공개하지 않는다.

## 12.2 Initial coarse ETA

\[
\hat A_{i,0}=A_i+\epsilon_{i,0}
\]

\[
\epsilon_{i,0}\sim N(0,60^2)
\]

그리고:

\[
\hat A_{i,0}\ge0
\]

으로 clipping한다.

## 12.3 Fine ETA update

Actual arrival까지 180분 이내이면 30분 간격으로 ETA를 갱신한다.

\[
0<A_i-t\le180
\]

\[
\hat A_{i,t}=A_i+\epsilon_{i,t}
\]

\[
\sigma_t=
\min(30,\max(5,0.2(A_i-t)))
\]

\[
\epsilon_{i,t}\sim N(0,\sigma_t^2)
\]

그리고:

\[
\hat A_{i,t}\ge t
\]

로 clipping한다.

Container state:

```text
truck_actual
eta_current
eta_previous
eta_delta
```

\[
eta\_delta=eta\_current-eta\_previous
\]

---

# 13. Storage Agent Action

야드 Stack 수는 100개다.

\[
\boxed{|A_{storage}|=100}
\]

예:

```text
0~24   = Block 0
25~49  = Block 1
50~74  = Block 2
75~99  = Block 3
```

Full Stack은 physical infeasibility로 mask한다.

Tier는 별도 선택하지 않고 선택된 Stack의 top에 적재한다.

---

# 14. Storage Observation

```text
Context = 4
Block features = 4 × 8
Stack features = 100 × 4
```

따라서:

\[
4+32+400=\boxed{436}
\]

수정 대상:

- Storage Actor input shape
- Stack embedding 반복 수
- Stack scorer 반복 대상
- Storage action mask
- Storage heuristic score array
- BC warm-start dataset/action
- Centralized critic global-state 구성
- Rollout buffer shape
- Evaluation stack-index mapping

---

# 15. YC Agent Action — Flat Target × Destination 확정

Hierarchical Action은 사용하지 않는다.

최종 YC Action은 하나의 Flat discrete action space로 구성한다.

\[
\boxed{
YC\ Action=
\{
Mandatory,\;
Idle,\;
Proactive(Target_i,Destination_j)
\}
}
\]

Block 하나의 physical target position:

\[
25\ stacks\times4\ tiers=100
\]

Destination stack:

\[
25
\]

따라서 최대 proactive pair:

\[
100\times25=2500
\]

기본적인 최대 action 수:

\[
\boxed{2502}
\]

즉:

```text
Action 0 = Mandatory
Action 1 = Idle

Action 2...
= Target physical position × Destination stack pair
```

실제 feasible action은 action mask에 의해 크게 줄어든다.

---

# 16. Flat Action Mapping

권장 mapping:

```python
ACTION_MANDATORY = 0
ACTION_IDLE = 1

pair_index = target_position_index * NUM_STACKS_PER_BLOCK + destination_stack
action_index = 2 + pair_index
```

여기서:

```text
target_position_index ∈ [0, 99]
destination_stack ∈ [0, 24]
```

역변환:

```python
pair_index = action_index - 2
target_position_index = pair_index // 25
destination_stack = pair_index % 25
```

---

# 17. Mandatory 내부 Scheduling

RL이 `Mandatory`를 선택하면 Retrieval과 Storage 중 oldest waiting class를 환경이 선택한다.

Retrieval waiting:

\[
t-TruckActual
\]

Storage waiting:

\[
t-YardArrival
\]

개별 작업:

```text
Retrieval → actual truck arrival이 빠른 순
Storage → FIFO
```

따라서 RL이 직접 최적화하는 영역은:

> Mandatory 작업 자체의 상세 sequencing이 아니라  
> Mandatory capacity와 Proactive relocation 간 자원배분 및 proactive move의 Target/Destination 선택

이다.

---

# 18. Proactive Target Candidate

기존 hard condition은 폐기한다.

```text
ETA lead <= 10
eta_delta <= -2
ETA lead <= 3
```

최종 candidate 조건:

```text
1. storage 완료
2. retrieval 완료 전
3. actual truck arrival 전
4. retrieval request 상태 아님
5. premarshal 수행 중 아님
6. blocker >= 1
7. current ETA 존재
8. 0 < current ETA lead <= 180 min
```

즉:

\[
0<\hat A_i-t\le180
\]

이면 target candidate가 될 수 있다.

`eta_delta`는 hard filter가 아니라 target feature로 사용한다.

---

# 19. Proactive 실행 단위

```text
Proactive(Target, Destination)
→ 선택 Target 위 top blocker 1개 이동
→ 2분 소요
→ 새 YC decision epoch
```

한 action에서 blocker 전체를 제거하지 않는다.

---

# 20. Proactive Destination

Destination도 RL이 직접 선택한다.

Feasible 조건:

- 같은 Block 내부
- source stack과 destination stack이 다름
- destination stack에 빈 physical slot 존재
- 실행 시점에 물리적으로 배치 가능
- reservation conflict 없음

기존 heuristic score에서 사용하던 요소는 observation feature로 이동한다.

예:

- destination stack height
- residual capacity
- ETA inversion risk
- destination stack 내 future pickup ETA 구조
- moved blocker의 future pickup priority
- source/destination relationship

---

# 21. YC Observation

Flat Target×Destination action을 사용하므로 각 pair를 평가할 정보가 필요하다.

## 21.1 YC summary

예:

- current time
- yard occupancy
- current block occupancy
- retrieval queue
- storage queue
- YC busy
- oldest retrieval waiting
- oldest storage waiting
- future inbound pressure
- mandatory workload
- capacity slack summary

## 21.2 Target feature

각 target physical position에 대해:

- candidate 여부
- current ETA lead
- eta_delta
- blocker count
- target stack height
- target-specific urgency
- target-specific capacity slack
- future retrieval priority

## 21.3 Destination feature

각 destination stack에 대해:

- stack height
- residual capacity
- ETA inversion risk
- earliest/nearest future pickup ETA
- moved blocker 배치 시 inversion change
- physical feasibility

## 21.4 Pair representation

Flat action을 사용하더라도 2500 pair를 단순 raw-vector로 모두 펼치는 방식만 가능한 것은 아니다.

권장:

```text
Target embedding
+
Destination embedding
+
Pair interaction scorer
→ 각 Target×Destination action score
```

즉 Action은 Flat이지만 Actor 내부에서는 shared pair scorer를 사용하여 parameter 수와 일반화 문제를 줄인다.

---

# 22. Capacity Slack

정의:

\[
CapacitySlack=
UrgentETAlead-MandatoryYCWorkloadTime
\]

모든 값은 minute 단위다.

Capacity Slack은 observation으로만 사용한다.

다음 hard mask는 사용하지 않는다.

```text
if CapacitySlack < YCMoveTime:
    proactive = infeasible
```

---

# 23. Action Mask 원칙

Action mask는 물리적 infeasibility만 표현한다.

## Mask 가능

- target candidate 아님
- blocker 없음
- destination full
- source == destination
- YC busy
- container already moving
- reservation conflict
- physical placement 불가능

## Mask 금지

- mandatory queue가 많음
- CapacitySlack이 작음
- Truck waiting이 큼
- Storage waiting이 큼
- 특정 action이 운영적으로 불리할 것으로 예상됨

이러한 trade-off는 RL이 reward를 통해 학습한다.

---

# 24. Proactive Destination Reservation

Proactive action을 시작하면:

```text
1. Destination slot reserve
2. YC 2분 move
3. Completion 직전 reservation release
4. blocker 실제 배치
```

Storage decision이 같은 마지막 slot을 동시에 선택하지 못하도록 한다.

---

# 25. Mandatory Reactive Rehandling

Truck arrival 이후 target 위 blocker가 존재하면 mandatory rehandling으로 처리한다.

Blocker 수가 \(b_i\)이면:

\[
ServiceTime_i=(b_i+1)\times2
\]

Reactive mandatory rehandling destination은 이번 연구에서 RL에 넘기지 않는다.

구분:

```text
Proactive relocation
→ Target + Destination RL

Reactive mandatory rehandling
→ deterministic / heuristic handling
```

---

# 26. Storage Capacity 부족

## Relocation slack

각 Block은 신규 Storage admission에서 3개의 physical slot을 relocation slack으로 보존한다. 이는 별도 가상공간이 아니라 100개 physical slot 중 신규 적재가 사용하지 않는 최소 여유공간이다.

## 일부 Stack full

Full Stack만 Storage action에서 mask.

## 전체 Yard full

신규 container를 삭제하거나 reject하지 않는다.

```text
Input waiting queue에 유지
→ Retrieval로 capacity 발생
→ Storage decision 재개
```

따라서 full-yard 상태는 Storage waiting 증가로 반영한다.

---

# 27. Episode 종료

## Operating phase

\[
0\le t<480
\]

동안:

- Storage arrival
- Truck arrival
- ETA update
- Mandatory
- Proactive

수행.

## Drain phase

\[
t\ge480
\]

이후:

### 중단

- 신규 Storage arrival
- 신규 Truck arrival
- 신규 ETA update
- 신규 Proactive action

### 계속

- 기존 Storage queue
- 기존 Retrieval queue
- 진행 중 Mandatory work

종료:

```text
storage_queue == 0
retrieval_queue == 0
all_yc_idle == True
```

Drain 동안 waiting은 KPI와 reward에 계속 포함한다.

---

# 28. Event Priority

동일 timestamp에서는 deterministic priority를 사용한다.

```text
1. Physical completion
2. Actual demand realization
3. Information update
4. Agent decision
```

구체적으로:

```text
1.
STORAGE_COMPLETE
RETRIEVAL_COMPLETE
REHANDLING_COMPLETE
PROACTIVE_COMPLETE

2.
TRUCK_ACTUAL_ARRIVAL
NEW_IMPORT_ARRIVAL

3.
TRUCK_ETA_UPDATE

4.
STORAGE_DECISION
YC_DECISION
```

---

# 29. Reward 최종 기본형

정규화 reward:

\[
\boxed{
r_t=
-w_T\frac{\Delta W_t^{truck}}{N_T}
-w_S\frac{\Delta W_t^{storage}}{N_S}
-w_Y\frac{\Delta T_t^{YC,extra}}{N_T}
}
\]

Extra YC Work 정규화 분모는:

\[
\boxed{N_C=N_T}
\]

로 확정한다.

기본 가중치:

\[
w_T=1,\quad w_S=1,\quad w_Y=0.1
\]

따라서 baseline:

\[
\boxed{
r_t=
-\frac{\Delta W_t^{truck}}{N_T}
-\frac{\Delta W_t^{storage}}{N_S}
-0.1\frac{\Delta T_t^{YC,extra}}{N_T}
}
\]

---

# 30. Reward 각 항의 의미

## Truck

\[
\frac{\sum_t\Delta W_t^{truck}}{N_T}
=
MeanTruckCompletionDelay
\]

## Storage

\[
\frac{\sum_t\Delta W_t^{storage}}{N_S}
=
MeanStorageCompletionDelay
\]

## Extra YC Work

\[
\frac{T_{YC,extra}}{N_T}
\]

은:

> **Retrieval 1건당 평균 추가 YC workload time**

으로 해석한다.

---

# 31. Extra YC Work

YC move time이 2분이므로:

\[
\Delta T_t^{YC,extra}
=
2
(
\Delta N_t^{rehandling}
+
\Delta N_t^{proactive}
)
\]

Proactive에 직접 positive reward를 주지 않는다.

Proactive 1회가 미래 rehandling 1회를 제거하면 YC workload 측면에서는 상쇄되고, Truck delay 감소 여부가 실제 이득이 된다.

---

# 32. Reward Sensitivity

부하조건 sensitivity는 수행하지 않는다.

Reward 자체에 대한 sensitivity만 별도로 수행할 수 있다.

## YC penalty

\[
w_Y\in\{0,0.05,0.1,0.25\}
\]

## Truck priority

선택된 \(w_Y\)에서:

\[
w_S=1
\]

고정,

\[
w_T\in\{1,1.5,2\}
\]

비교.

이 실험은 동일한 고정 \(\lambda_S=\lambda_R=20/h\) 환경에서 수행한다.

---

# 33. Local Reward Shaping

현재:

```text
risk_shaping = 0
yc_queue_shaping = 0
```

유지.

Environment 변경, reward normalization, local shaping을 동시에 변경하지 않는다.

---

# 34. Storage BC

기존 BC checkpoint는 재사용하지 않는다.

이유:

- Storage actions 40 → 100
- Storage obs 196 → 436
- Yard 160 → 400 slots
- Initial occupancy 변화
- workload generator 변화
- ETA 구조 변화

따라서 새로운 Environment에서 Storage-only BC dataset을 다시 생성한다.

YC에는 BC를 적용하지 않는다.

---

# 35. Centralized PPO

Centralized PPO도 동일한 Flat Target×Destination Action interface를 사용한다.

즉:

```text
Mandatory
Idle
Target×Destination pair
```

Action 표현은 MARL과 동일하다.

차이는 observation 범위다.

```text
MARL Actor
→ local observation

Centralized PPO
→ global state
```

---

# 36. CTDE

MARL은 CTDE 구조를 유지한다.

Training:

```text
Actor → local observation
Critic → global observation
```

Execution:

```text
Actor only
```

---

# 37. 비교 정책

부하별 비교가 아니라 하나의 operating point에서 정책 자체를 비교한다.

예:

```text
Information-aware Heuristic
Centralized PPO
MARL w/o Resource State
MARL w/o Proactive
Proposed MARL
```

모든 정책은 동일 held-out scenario seeds를 사용한다.

---

# 38. Evaluation KPI

필수 KPI:

## Truck

\[
RetrievalComplete-TruckActual
\]

## Storage

\[
StorageComplete-YardArrival
\]

## Rehandling Moves

Mandatory retrieval 과정에서 blocker를 옮긴 횟수.

## Proactive Moves

Truck actual arrival 이전 선제 relocation 횟수.

## Total YC Workload

- physical moves
- busy time
- utilization

## Queue

- mean/max YC queue
- retrieval waiting
- storage waiting

## Policy behavior

- proactive target 선택 빈도
- destination 선택 분포
- ETA lead별 proactive 행동
- proactive 후 실제 future rehandling 감소 여부

---

# 39. 동일 Seed 공정성

동일 scenario seed에서 모든 비교정책이 공유해야 하는 것:

- Initial layout
- Storage arrival timestamps
- Retrieval arrival timestamps
- Retrieval target mapping
- truck actual values
- ETA noise realization
- New inbound future pickup proxy
- 기타 exogenous randomness

정책에 따라 달라져야 하는 것은 endogenous yard state와 decision 결과뿐이다.

---

# 40. Environment Sanity Check

RL 학습 전에 검증한다.

## Capacity

- 400 slots
- start containers = 268
- 67 per block
- no floating
- tier ≤ 4

## Arrival

고정 20/h에서 여러 seed를 반복하여 실제 평균 event 수가:

\[
E[N]\approx20\times8=160
\]

과 일치하는지 확인.

## Occupancy

\[
\lambda_S=\lambda_R
\]

에서 평균 occupancy가 구조적으로 한 방향으로만 증가/감소하지 않는지 확인.

## YC contention

고정 20/h에서:

- YC가 대부분 idle하지 않음
- queue가 무한히 발산하지 않음
- Storage/Retrieval competition이 발생
- proactive opportunity가 충분히 존재

## Proactive

기록:

- ETA update 수
- candidate target 수
- blocked target 수
- feasible Target×Destination pair 수
- proactive move 수

## ETA

\[
|\hat A-A|
\]

가 actual arrival 접근 시 감소하는지 확인.

## Same-episode retrieval

New inbound 중:

```text
truck_actual <= 480
```

인 container가 0개인지 assert.

## Drain

480분 이후:

- 신규 exogenous event 없음
- 신규 proactive 없음
- mandatory queue 모두 처리
- all YC idle

## Action execution

선택한 Target×Destination pair와 실제 blocker 이동 결과가 정확히 일치하는지 검증.

---

# 41. 연구질문

부하 변화 관련 RQ는 삭제한다.

## RQ1

> Information-aware heuristic과 학습 정책은 동일한 stochastic yard operating condition에서 Truck delay, Storage delay 및 YC workload 측면에서 어떤 차이를 보이는가?

## RQ2

> Local observation 기반 CTDE MARL은 global state를 사용하는 Centralized PPO와 비교해 어떤 운영성과 차이를 보이는가?

## RQ3

> Dynamic ETA 및 YC resource information을 활용한 cooperative policy는 proactive relocation의 Target과 Destination을 어떻게 선택하며, 이러한 의사결정이 운영성과에 어떤 영향을 미치는가?

## RQ4

> Resource-state information 및 proactive relocation 기능의 제거는 운영성과에 어떤 영향을 미치는가?

---

# 42. 삭제할 기존 설정 및 표현

다음은 모두 폐기한다.

```text
Low / Medium / High

Low = 12/h
Medium = 20/h
High = 28/h

λ ~ Uniform(12,28)

balanced load sampling

load sensitivity

부하가 증가할수록 MARL 효과가 증가하는가?

80 / 150 / 200 inbound container

arrival window = 0~120

initial_containers = 60

4×10×4 = 160 slots

Storage actions = 40
Storage obs = 196

ETA update = 60/30/15 or 30/15/5

premarshal_horizon = 10
eta_advance_threshold = 2
emergency_eta_threshold = 3

CapacitySlack hard action mask

Proactive target = rule-based
Proactive destination = rule-based

Hierarchical YC Action
```

---

# 43. 미확정사항 없음

현재 Environment / Action / Reward의 핵심 구조는 모두 확정했다.

- Arrival rate: 20/h 고정
- Storage / Retrieval rate: 동일
- YC Action: Flat Target × Destination
- Extra YC Work normalization: \(N_T\)
- CapacitySlack: observation only
- Low / Medium / High 부하조건: 사용하지 않음

---

# 44. 최종 코드 수정 순서

## Stage 1 — Environment

1. Yard 4×25×4
2. Initial stock 268
3. minute / 480 / 2min
4. Storage action 100 / obs 436
5. fixed-20/h Poisson Storage stream
6. fixed-20/h Poisson Retrieval stream
7. cohort 분리
8. same-episode new inbound retrieval 차단
9. dynamic ETA
10. proactive candidate horizon 180 min
11. CapacitySlack hard mask 제거
12. drain phase
13. deterministic event priority
14. same-seed replay

## Stage 2 — YC Action

15. Flat 2500 Target×Destination pairs
16. Target features
17. Destination features
18. shared pair scorer
19. physical mask
20. destination reservation
21. Centralized PPO 동일 action interface

## Stage 3 — Reward

22. \(N_T,N_S\) initialization 시 확정
23. \(N_C=N_T\)
24. normalized reward 구현
25. raw + normalized metric logging

## Stage 4 — Calibration 완료

26. Environment-only calibration 완료
27. \(\lambda^*=20/h\) 확정
28. 이후 workload 조건 변경 금지

## Stage 5 — Training

29. Storage BC 재생성
30. YC scratch initialization
31. 짧은 PPO pilot
32. reward sensitivity
33. multi-seed full training
34. held-out evaluation
35. statistical analysis

---

# 45. 최종 모델 한 문장

> **4개 Block, 100개 Stack, 400개 physical slot과 Block별 1대의 고정 YC로 구성된 단일 수입 컨테이너 야드 운영조건에서, 동일한 고정 평균률 20/h의 Poisson Storage 및 Retrieval request와 실제 도착에 가까워질수록 정확도가 개선되는 dynamic Truck ETA를 생성하고, Storage Agent가 신규 컨테이너의 적재 Stack을 선택하며 YC Agent가 Flat Target×Destination action을 통해 Mandatory 처리와 ETA 기반 최적 Proactive Relocation을 협조적으로 학습하는 CTDE MARL 모델이다.**
