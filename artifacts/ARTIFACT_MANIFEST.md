# Artifact Manifest

GitHub에는 대형 바이너리 checkpoint를 저장하지 않는다. 아래 항목은 Google Drive master repository와 연결되는 외부 artifact다.

## Canonical checkpoint

- File: `groupnorm_12k_resource_marl_final.pt`
- SHA-256: `2bbd3a2e795a215d3fcbd654d58fad154d4298b55bb50ce92f54ad50b82ddb59`
- Drive location: `ACK2026_항만야드_MARL_MASTER_20260921/00_CURRENT_MODEL/checkpoints/`
- Role: 현재 canonical GroupNorm 12k 기준 가중치
- Important: legacy checkpoint는 exact resume state가 아니라 weight warm-start 기준으로 취급한다.

## Audit / experiment artifacts

대형 CSV/JSON/ZIP 및 Astra evidence는 Drive의 다음 위치에서 관리한다.

- `03_RESULTS/`
- `05_ASTRA_AUDITS/`
- `90_ARCHIVE/`

모든 새 실험은 가능한 경우 다음 메타데이터를 함께 기록한다.

- Git commit SHA
- checkpoint SHA-256
- config snapshot
- training seed
- scenario seed bank
- actual environment decisions
- optimizer update count
- evaluation protocol
