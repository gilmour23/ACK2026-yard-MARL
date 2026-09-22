# ACK 2026 항만 야드 MARL 연구계획 V4 FINAL

## 1. 연구 주제

**동적 외부트럭 ETA와 제한된 Yard Crane 자원을 고려한 수입 컨테이너 야드의 Storage Allocation 및 Proactive Relocation 협조학습**

본 연구는 수입 컨테이너 야드에서 신규 컨테이너의 적재 위치와 향후 외부트럭 반출을 대비한 선제 재배치를 동시에 다룬다. Storage Allocation Agent는 신규 컨테이너의 적재 Stack을 선택하고, Block별 YC Operation Agent는 필수 작업과 ETA 기반 proactive relocation 사이에서 YC capacity를 배분하면서 proactive target과 destination을 직접 선택한다.

---

## 2. 연구 범위

### 포함

- Import container only
- 신규 수입 컨테이너 Storage
- 기존 장치 컨테이너 Truck Retrieval
- Dynamic Truck ETA
- Mandatory rehandling
- Proactive relocation
- Block별 고정 YC 자원경합
- Storage Allocation + YC Operation CTDE MARL

### 제외

- Export / transshipment container
- Quay crane scheduling
- Internal yard truck routing
- Block 간 YC redeployment
- Gantry/trolley 실제 이동거리 기반 travel time
- Truck appointment 자체의 재스케줄링
- Reactive rehandling destination의 RL 최적화

---

## 3. Environment

### Yard

- Blocks: 4
- Stacks per Block: 25
- Max Tier: 4
- Physical slots: 400
- Initial containers: 268
- Initial occupancy: 67%
- Initial containers per Block: 67

초기 배치는 retrieval 순서와 독립적으로 random feasible layout으로 생성한다.

운영 중 신규 Storage admission은 각 Block에 최소 3개의 relocation slack physical slot을 남기도록 제한한다. Physical capacity는 100 slots/block을 유지하지만 신규 Storage가 사용할 수 있는 상한은 97 slots/block이다. 이는 buried target의 blocker를 same-block 내 다른 stack으로 이동시킬 수 있는 최소 여유공간을 보장하기 위한 feasibility 제약이다.

### Yard Crane

- 1 YC per Block
- Total YC: 4
- Block 간 이동 없음
- 모든 physical container move: 2 min

### Episode

- Operating window: 480 min
- 480분 이후 신규 Storage / Retrieval / ETA update / Proactive action 중단
- 이미 발생한 Storage 및 Retrieval mandatory queue는 모두 처리한 뒤 종료

---

## 4. 단일 운영조건

본 연구에서는 Low / Medium / High 부하조건을 사용하지 않는다.

Storage request와 Retrieval request는 독립적인 homogeneous Poisson process로 생성하고 평균률은 동일하게 고정한다.

\[
\boxed{\lambda_S=\lambda_R=20\text{ events/hour}}
\]

20/h는 environment-only calibration을 통해 선정하였다. 12, 16, 20, 24/h 후보를 heuristic 환경에서 확인한 결과, 20/h에서 평균 YC utilization 약 0.84, max queue 약 3.67, mandatory drain 약 0.69분으로 자원경합이 존재하면서도 지속 포화가 발생하지 않았다.

본 학습과 평가에서는 arrival rate를 변경하지 않고 scenario seed만 변화시킨다.

---

## 5. Container Cohort

### Initial Import Stock

Episode 시작 시 이미 야드에 존재하는 268개 컨테이너다. 현재 episode의 Retrieval 대상은 이 집단에서만 선택한다.

### New Import Containers

Episode 중 선박 양하 후 Storage request로 발생한다.

```text
Yard arrival
→ Storage Allocation
→ YC Storage
```

동일 episode에서는 Retrieval 대상으로 전환하지 않는다. 신규 컨테이너에는 Storage priority 계산을 위한 future pickup proxy만 부여한다. 이 proxy는 median 72시간, log-SD 0.5의 lognormal 분포를 사용하고 24~168시간으로 clipping한다. 이는 특정 터미널 dwell-time을 직접 추정한 값이 아니라 수일 단위 체류시간을 표현하는 synthetic priority 분포다.

---

## 6. Truck ETA

Retrieval 대상 컨테이너의 실제 service request time은 simulator hidden truth로 유지한다.

초기 ETA:

\[
\hat A_{i,0}=A_i+\epsilon_{i,0},\qquad \epsilon_{i,0}\sim N(0,60^2)
\]

실제 도착까지 180분 이내에서는 30분 간격으로 ETA를 갱신한다.

\[
\sigma_t=\min(30,\max(5,0.2(A_i-t)))
\]

실제 도착에 가까워질수록 ETA uncertainty가 감소한다.

---

## 7. Agent 및 Action

### 7.1 Storage Allocation Agent

신규 컨테이너가 도착하면 100개 Stack 중 하나를 선택한다.

\[
|A_{storage}|=100
\]

Full Stack만 physical infeasibility로 mask한다.

### 7.2 YC Operation Agent

Block별 YC Agent는 하나의 shared policy를 사용한다.

최종 flat action은:

\[
\boxed{
A_{YC}=\{Mandatory, Idle, Proactive(Target_i,Destination_j)\}
}
\]

Block 내 target physical position은 최대 100개, destination Stack은 25개다.

\[
100\times25=2500
\]

따라서 최대 YC action dimension은:

\[
\boxed{2502}
\]

#### Mandatory

RL이 Mandatory를 선택하면 Retrieval과 Storage 중 oldest-wait class를 deterministic rule로 선택한다.

- Retrieval 내부: actual truck arrival이 빠른 작업 우선
- Storage 내부: FIFO

#### Proactive

RL이 직접 다음을 선택한다.

1. 향후 반출 대상 Target container
2. Target 위 top blocker의 Destination Stack

한 action에서 blocker 1개만 이동하고 2분 후 새로운 decision epoch를 만든다.

#### Reactive Rehandling

Truck arrival 이후 target 위 blocker를 치우는 mandatory rehandling은 연구범위 확장을 막기 위해 destination heuristic을 유지한다.

---

## 8. Proactive Candidate

Target candidate 조건:

- storage 완료
- retrieval 완료 전
- actual truck arrival 전
- retrieval request 전
- premarshal 진행 중이 아님
- blocker ≥ 1
- current ETA 존재
- current ETA lead ≤ 180 min

기존 ETA advance hard threshold는 사용하지 않는다.

ETA lead, ETA delta, blocker count 및 CapacitySlack은 action을 제한하는 rule이 아니라 policy observation으로 사용한다.

---

## 9. Action Mask

물리적으로 불가능한 action만 mask한다.

### Mask

- target candidate 아님
- blocker 없음
- destination full
- source = destination
- reservation conflict
- YC busy

### Mask하지 않음

- mandatory queue가 많음
- CapacitySlack이 작음
- Truck waiting이 큼
- Storage waiting이 큼

CapacitySlack은 observation only다.

---

## 10. Observation

### Storage

\[
4 + 4\times8 + 100\times4 = 436
\]

### YC

- 18-d local/resource context
- 2500 Target×Destination pair 각각 9개 feature

Pair feature에는 ETA lead, ETA delta, blocker, capacity slack, moving blocker pickup priority, destination height, inversion risk, feasibility 등이 포함된다.

Actor는 2500개 action을 각각 독립 parameter로 학습하지 않고 shared pair scorer를 사용한다.

---

## 11. Learning Architecture

### Proposed MARL

CTDE 구조를 사용한다.

```text
Training:
Actor → local observation
Critic → global yard state

Execution:
Actor only
```

Storage Actor와 shared YC Actor는 서로 다른 action head를 사용한다.

### Storage warm-start

Storage Agent만 information-aware heuristic을 이용한 BC warm-start를 사용한다.

YC Agent에는 BC를 적용하지 않고 PPO scratch learning을 사용한다.

### Centralized PPO

동일한 100-action Storage 및 2502-action YC interface를 사용하되 global information을 추가로 제공하는 stronger-information baseline으로 사용한다.

---

## 12. Reward

기본 reward:

\[
\boxed{
r_t=
-\frac{\Delta W_t^{truck}}{N_T}
-\frac{\Delta W_t^{storage}}{N_S}
-0.1\frac{\Delta T_t^{YC,extra}}{N_T}
}
\]

- \(N_T\): episode Retrieval request 수
- \(N_S\): episode Storage request 수
- \(\Delta T^{YC,extra}\): Rehandling + Proactive의 추가 YC 작업시간

2분/move이므로:

\[
\Delta T^{YC,extra}=2(\Delta N^{rehandling}+\Delta N^{proactive})
\]

각 항은 각각 평균 Truck completion delay, 평균 Storage completion delay, Retrieval 1건당 Extra YC workload와 직접 연결된다.

Proactive action 자체에 positive reward를 부여하지 않는다.

---

## 13. 비교 정책

- Information-aware Heuristic
- Centralized PPO
- MARL w/o Resource State
- MARL w/o Proactive
- Proposed MARL

모든 정책은 동일 held-out scenario seed의 동일 exogenous event stream을 사용한다.

---

## 14. 주요 KPI

- Mean Truck Completion Delay
- Mean Storage Completion Delay
- Rehandling Moves
- Proactive Moves
- Extra YC Minutes per Retrieval
- Total YC Moves
- Mean YC Utilization
- Max YC Queue
- Proactive target / destination 선택 특성

최종 평가는 greedy policy를 사용한다.

---

## 15. 연구질문

### RQ1

동일한 stochastic yard operating condition에서 information-aware heuristic과 학습 정책은 Truck delay, Storage delay 및 YC workload 측면에서 어떤 차이를 보이는가?

### RQ2

Local observation 기반 CTDE MARL은 global information을 사용하는 Centralized PPO와 비교해 어떤 운영성과 차이를 보이는가?

### RQ3

Dynamic ETA와 YC resource information을 활용한 policy는 proactive target과 destination을 어떻게 선택하며, 이러한 의사결정은 운영성과에 어떤 영향을 미치는가?

### RQ4

Resource-state information 또는 proactive relocation 기능을 제거했을 때 운영성과가 어떻게 변화하는가?

---

## 16. 실험 절차

1. Environment invariant test
2. λ=20/h 고정
3. Storage-only BC 재학습
4. Short PPO smoke test
5. Reward sensitivity
6. Training seeds 1/2/3 full PPO
7. Held-out scenario seeds에서 greedy evaluation
8. 정책별 paired statistical comparison
9. ACK 2–3페이지 범위에 맞춰 핵심 KPI와 ablation 결과 중심 보고
