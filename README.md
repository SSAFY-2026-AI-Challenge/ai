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

## 배포 시 DB 연결

GCE에서 `.env` 에 `APP_MODE=db` 만 두고 올리면, 앱이 Secret Manager 에서 아래 시크릿을 읽습니다.

| 시크릿 이름 | 용도 |
| --- | --- |
| `DB_HOST` | MySQL 호스트 |
| `DB_PORT` | 포트 |
| `DB_USER` | 사용자 |
| `DB_PASSWORD` | 비밀번호 |
| `DB_NAME` | 데이터베이스 |

VM 서비스 계정에 **Secret Manager 보안 비밀 접근자** (`roles/secretmanager.secretAccessor`) 가 있어야 합니다.
`GCP_PROJECT_ID` 를 비우면 GCE/ADC 가 프로젝트 ID를 찾습니다. 로컬에서만 DB를 붙일 때는 같은 키를 `.env` 에 넣거나 `SECRETS_SOURCE=env` 로 두면 됩니다.

연결되면 `member` / `account` / `transaction` / `payroll` 에서 피처를 만들고,
결과 요약·추천은 `ai_report`, `ai_recommendation` 에 저장합니다.
`ai_report.member_id` 가 고유하므로 같은 학생은 리포트가 갱신됩니다.

확인:

```bash
curl http://127.0.0.1:48765/v1/status
```

`secrets_source` 가 `gcp` 이고 `using_database` 가 `true` 이면 Secret Manager 로 붙은 상태입니다.

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
