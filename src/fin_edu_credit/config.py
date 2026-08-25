"""환경 설정 로딩.

DB 접속 정보와 모델 하이퍼파라미터를 한 곳에서 관리한다.
값은 `.env` 파일 또는 환경 변수에서 읽는다(`cp .env.example .env`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports"
SQL_DIR = PROJECT_ROOT / "sql"


def load_dotenv(path: Path | None = None) -> None:
    """`.env` 파일을 os.environ 에 반영한다(이미 설정된 값은 덮어쓰지 않음)."""
    path = path or PROJECT_ROOT / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class DBConfig:
    host: str
    port: int
    user: str
    password: str
    database: str
    charset: str = "utf8mb4"

    @classmethod
    def from_env(cls) -> "DBConfig":
        load_dotenv()
        return cls(
            host=os.environ.get("FIN_EDU_DB_HOST", "127.0.0.1"),
            port=int(os.environ.get("FIN_EDU_DB_PORT", "3306")),
            user=os.environ.get("FIN_EDU_DB_USER", "root"),
            password=os.environ.get("FIN_EDU_DB_PASSWORD", ""),
            database=os.environ.get("FIN_EDU_DB_NAME", "fin_edu"),
        )


@dataclass(frozen=True)
class SimulationConfig:
    """더미 모의 경제 생성 파라미터."""

    period_start: date = date(2026, 1, 5)   # 월요일
    period_end: date = date(2026, 8, 24)    # 월요일, 약 33주
    grades: tuple[int, ...] = (3, 4, 5, 6)
    classes_per_grade: int = 3
    students_per_class: int = 25
    random_seed: int = 20260825

    @property
    def total_students(self) -> int:
        return len(self.grades) * self.classes_per_grade * self.students_per_class


@dataclass(frozen=True)
class ModelConfig:
    """신용평가 모델 설정."""

    # 관측 윈도우: 기준일 직전 N주의 행동을 피처로 사용
    observation_weeks: int = 8
    # 성과 윈도우: 기준일 직후 N주 안에 연체가 발생하면 '부실(bad)'
    performance_weeks: int = 4
    # 학습용 패널 데이터 스냅샷 간격(일)
    snapshot_stride_days: int = 14

    # PD -> 점수 변환 (표준 스코어카드 스케일링)
    # 학급 전체의 평균 연체율이 20~25% 수준이므로, 그 지점(odds 3:1)을 650점에
    # 놓아야 학생들이 1~10등급에 고르게 분포한다.
    base_score: int = 650          # 기준 점수
    base_odds: float = 3.0         # 기준 점수에서의 정상:연체 odds
    pdo: float = 40.0              # odds 가 2배 좋아질 때 늘어나는 점수
    score_floor: int = 350
    score_cap: int = 1000

    test_size: float = 0.25
    cv_folds: int = 5
    random_state: int = 20260825
    regularization_c: float = 0.3  # 로지스틱 회귀 L2 정규화 강도(작을수록 강하게 규제)


# 등급 구간: (등급, 점수 하한, 등급명). 1등급이 가장 우수하다.
GRADE_TABLE: tuple[tuple[int, int, str], ...] = (
    (1, 860, "최우수"),
    (2, 815, "우수"),
    (3, 780, "양호"),
    (4, 745, "보통 상"),
    (5, 700, "보통"),
    (6, 660, "보통 하"),
    (7, 620, "주의"),
    (8, 570, "관찰 필요"),
    (9, 520, "경고"),
    (10, 0, "위험"),
)
