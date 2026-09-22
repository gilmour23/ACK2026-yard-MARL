# ACK2026 Yard MARL

ACK 2026 스마트해운물류 트랙용 수입 컨테이너 야드 운영 최적화 연구 코드 저장소입니다.

## Canonical code

`main` branch가 현재 검증 대상인 canonical source of truth입니다. 대형 체크포인트와 실험 산출물은 GitHub에 커밋하지 않고 Google Drive에 보관합니다.

현재 canonical 계열:
- Group-normalized Flat Target×Destination PPO
- 4 blocks × 25 stacks × 4 tiers = 400 slots
- initial import stock 268 containers
- fixed YC 1/block
- storage/retrieval Poisson rate 20/h
- GroupNorm evaluation mask audit fix 반영
- Single PPO GroupNorm/entropy/minibatch 비교 정렬 반영
- `episode_complete_rollout` A/B 실험 옵션 반영
- Full 30k multi-seed training: **No-Go**

## Repository layout

```text
src/        core simulator, environment, models, training, evaluation
scripts/    reproducible experiment/pilot runners
tests/      unit/integration tests
configs/    frozen canonical/pilot configuration notes
docs/       current model and methodology documents
audits/     Astra audit and audit-fix records
artifacts/  external checkpoint/result manifest only
```

## Setup

```bash
python -m venv .venv
pip install -r requirements.txt
```

코드는 flat-module import를 사용하므로 repository root에서 다음처럼 실행합니다.

```bash
PYTHONPATH=src pytest -q tests
PYTHONPATH=src python scripts/run_credit_assignment_pilot.py --help
```

Windows PowerShell:

```powershell
$env:PYTHONPATH="src"
pytest -q tests
```

## Artifact policy

`.pt`, 대형 CSV/JSON, experiment output은 GitHub에 올리지 않습니다. 기준 체크포인트의 파일명과 SHA-256은 `artifacts/ARTIFACT_MANIFEST.md`에 기록하고 실제 파일은 Google Drive master repository에서 관리합니다.

## Development rule

- `main`: 감사 완료/현재 canonical code만 유지
- 실험 변경: `experiment/<name>` branch
- 성공한 변경만 PR로 `main`에 merge
- 결과를 보고 reward나 gate를 사후 조정하지 않음

## Current research status

Astra 재감사 중간 결과에서는 episode-complete rollout이 critic EV/RMSE를 개선했지만 pair distribution은 거의 균등했고 objective J 개선은 일관되지 않았습니다. 이 결과는 아직 최종 감사 보고서가 아니라 중간 실행 로그로 취급합니다. 다음 우선순위는 actor를 재설계하는 것이 아니라 critic optimization/training failure를 분리 진단하는 것입니다.
