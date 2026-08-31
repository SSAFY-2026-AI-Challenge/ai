# 교실 경제 신용평가 AI

초등 교실 경제활동 시뮬레이션에서 **신용점수**를 매기는 딥러닝 서비스입니다.
DB 접속 정보는 배포 서버의 env로 넣고, 지금은 연결 없이 학급 체험 데이터로 동작합니다.

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

API 문서는 `http://127.0.0.1:48765/docs` 에서 확인할 수 있습니다.

직접 채점:

```bash
uv run credit-ai score --spending 0.25 --homework 0.95 --attendance 0.98
```

테스트:

```bash
uv run pytest
```

## 배포 시 DB 연결

`.env` 에 MySQL 정보를 채우고 `APP_MODE=db` 로 올립니다.

```
APP_MODE=db
DB_HOST=...
DB_PORT=3306
DB_USER=...
DB_PASSWORD=...
DB_NAME=...
```

연결되면 `member` / `account` / `transaction` / `payroll` 에서 피처를 만들고,
결과 요약·추천은 `ai_report`, `ai_recommendation` 에 저장합니다.
`ai_report.member_id` 가 고유하므로 같은 학생은 리포트가 갱신됩니다.

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
