# Astra re-audit interim note — 2026-09-22

이 문서는 최종 감사 보고서가 아니라 사용자가 전달한 Astra 재감사 **중간 실행 로그**를 보존하기 위한 메모다.

확인된 중간 상태:

- 지정 문서 및 `00_CURRENT_MODEL/code/` 재검증 진행
- GroupNorm 평가 mask 수정 확인
- Single PPO GroupNorm/셔플 수정 확인
- 제공 테스트 15개 통과
- exact resume 미구현: weights는 이어지나 step/update/optimizer 상태가 실제 재호출 시 복원되지 않음
- seed 21/22/23 cutoff(A) vs episode-complete(B) 2k pilot 완료
- A: 약 2,059–2,127 decisions
- B: 약 2,137–2,301 decisions, episode terminal까지 수집
- validation scenarios 601–610, 총 210 stochastic evaluation episodes 완료
- B는 3 seeds 모두 critic EV/RMSE 개선
- 그러나 EV는 약 0.017–0.021로 낮음
- objective J는 2 seeds 악화, 1 seed 소폭 개선
- pair distribution은 여전히 거의 uniform
- 시간 변수만 이용한 scenario-split 단순 회귀는 EV 약 0.958을 보여, critic 실패를 ETA 관측 부족만으로 설명하기 어려움
- 잠정 판정: Full 30k No-Go, 4k/6k episode-complete 연장 보류
- 다음 우선 실험: actor를 고정하고 critic optimization/training failure를 분리 진단

최종 Astra 보고서가 확보되면 이 문서의 수치를 그 보고서/재현 자료와 대조해야 한다.
