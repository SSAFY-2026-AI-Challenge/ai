# 교실 경제 신용평가 AI

초등 교실 경제활동 시뮬레이션에서 **신용점수**를 매기는 딥러닝 서비스입니다.
로컬은 학급 체험 데이터로 동작하고, 배포 서버는 GCP Secret Manager 의 DB 시크릿으로 MySQL 에 붙습니다.

## 쓰는 값

모델은 아래 세 가지를 입력으로 받습니다.

| 파라미터 | JSON 키 | 출처 |
| --- | --- | --- |
| 지출 추이 | `spending_trend` | 거래 현금흐름·소비. 0=검소, 1=과소비 |
| 숙제 완료율 | `homework_rate` | `member.cs_homework` |
| 출결률 | `attendance_rate` | `member.cs_attendance` |

점수는 **300~1000**이고, 등급은 최우수(A+) · 우수(A) · 양호(B) · 보통(C) · 노력(D)입니다.
MLP가 출석×숙제 시너지, 과소비 감점 같은 비선형 관계를 학습합니다.

## 로컬 실행

Python 3.12, [`uv`](https://docs.astral.sh/uv/) 기준입니다.

```bash
cp .env.example .env
uv sync --group dev
uv run credit-ai train
uv run credit-ai serve --host 0.0.0.0 --port 48765
```

브라우저에서 `http://127.0.0.1:48765` 로 학급 명단과 시뮬레이터를 엽니다.

직접 채점:

```bash
uv run credit-ai score --spending 0.25 --homework 0.95 --attendance 0.98
```

테스트:

```bash
uv run pytest
```

## GCP 배포 (self-hosted runner)

`main` 푸시 또는 Actions 수동 실행 시 GCE **self-hosted runner**가 앱을 `$HOME/credit-ai`에 설치합니다. 모델 파일이 없으면 학습한 뒤 **48765** 포트로 올립니다.

| 워크플로 | 실행 | 하는 일 |
| --- | --- | --- |
| **Deploy to GCP** | `main` push, 또는 Actions에서 수동 | Secret Manager 확인, 코드 동기화, 모델 없으면 학습, 서비스 기동 |
| **Verify Secret Manager** | Actions에서 수동 | `DB_*` 시크릿을 읽을 수 있는지만 확인 (값은 로그에 안 남김) |
| **Train model** | Actions에서 수동 | 모델을 다시 학습하고 서비스를 재시작 |

배포 VM 서비스 계정에는 `roles/secretmanager.secretAccessor` 가 있어야 하고, 인스턴스 API 범위는 Cloud API 전체 허용(`cloud-platform`)이어야 합니다.
GitHub Actions 잡이 끝나도 앱이 살아 있으려면 러너 계정에 passwordless sudo 또는 `loginctl enable-linger` 가 있어야 합니다.

GitHub **Actions** 탭에서 워크플로를 고른 뒤 Run workflow 하면 됩니다.
GCE 방화벽은 **TCP 48765**만 외부에 여세요. **3306은 열지 마세요.**

VM에서 직접:

```bash
bash deploy/gcp.sh verify-secrets
bash deploy/gcp.sh deploy
bash deploy/gcp.sh train
```

## API

- `GET /health`
- `GET /v1/status`
- `POST /v1/score` — 피처 JSON으로 즉시 채점
- `POST /v1/score/member/{id}` — 학생 한 명 채점
- `POST /v1/score/class` — 학급 전체
- `GET /v1/class` — 순위 목록
- `GET /docs` — OpenAPI

## 모델

`src/credit_ai/model/network.py` 의 `CreditMLP` (64-32-16, GELU, LayerNorm).
학습 데이터는 실제 라벨이 아직 없으므로 `training/synthetic.py` 의 교사 함수로 합성합니다.
실데이터 라벨이 쌓이면 같은 학습 진입점만 바꿔 재학습하면 됩니다.
산출물은 `models/credit_mlp.pt`, `models/scaler.json` 입니다.
