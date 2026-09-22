# ACK2026 Yard MARL

ACK 2026 스마트해운물류 트랙용 수입 컨테이너 야드 운영 최적화 연구 코드 저장소입니다.

## Canonical source

`main` branch가 현재 canonical source of truth입니다. 대형 checkpoint와 experiment artifact는 Google Drive에서 별도 관리합니다.

현재 상태:
- Group-normalized Flat Target×Destination PPO
- 4 blocks × 25 stacks × 4 tiers = 400 slots
- initial import stock = 268
- fixed YC 1/block
- storage/retrieval Poisson rate = 20/h
- GroupNorm evaluation-mask fix 반영
- MARL/Single PPO action parameterization과 canonical PPO optimization defaults 정렬
- episode-complete A/B option 구현
- Astra 2k A/B re-audit pilot 완료(중간 결과 기록)
- Full 30k multi-seed training = **No-Go**

## Layout

```text
src/        simulator, environment, model, training, evaluation
scripts/    experiment/pilot runners
tests/      unit/integration tests
configs/    canonical settings
docs/       current model/methodology
audits/     Astra audit records
artifacts/  external artifact manifest only
```

## Setup

```bash
python -m venv .venv
pip install -r requirements.txt
PYTHONPATH=src pytest -q tests
```

Windows PowerShell:

```powershell
$env:PYTHONPATH="src"
pytest -q tests
```

## Canonical optimization defaults

MARL과 centralized PPO의 기본 optimization settings:

- rollout_steps = 512
- update_epochs = 2
- minibatch_size = 256
- gamma = 1.0
- gae_lambda = 1.0
- learning_rate = 3e-4

## Evaluation

Stochastic policy sampling이 learned PPO의 primary evaluation protocol이다. Flat greedy는 별도 diagnostic으로만 사용한다.

## Checkpoint policy

`.pt`, 대형 CSV/JSON, run outputs는 GitHub에 commit하지 않는다.

Canonical checkpoint:
- `groupnorm_12k_resource_marl_final.pt`
- SHA-256: `2bbd3a2e795a215d3fcbd654d58fad154d4298b55bb50ce92f54ad50b82ddb59`

자세한 위치는 `artifacts/ARTIFACT_MANIFEST.md` 참고.

현재 `init_checkpoint`는 **weights-only warm start**이며 exact optimizer/RNG resume이 아니다.

## Development rule

- `main`: canonical code only
- experimental changes: `experiment/<name>`
- verified changes only merge to `main`
- no post-hoc reward/gate tuning
- no Full 30k until explicit audit Go

현재 다음 실험 branch는 `experiment/critic-diagnostic`이며, actor를 고정한 critic optimization 진단용으로 사용한다.

## Current diagnosis

Astra 재감사 중간 결과에서는 episode-complete rollout이 critic EV/RMSE를 개선했지만 EV는 약 0.017–0.021로 낮았고, objective J는 seed 간 일관되게 개선되지 않았으며 pair distribution은 여전히 거의 uniform이었다.

따라서 다음 우선순위는 actor 재설계가 아니라 **critic optimization/training failure의 분리 진단**이다. 이 수치들은 최종 Astra 보고서가 아니라 중간 실행 로그에서 온 것이므로 final evidence로 확정하지 않는다.
