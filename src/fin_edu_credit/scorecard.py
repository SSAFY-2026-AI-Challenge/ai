"""부실확률(PD) -> 신용점수 -> 신용등급 변환과 교실용 규칙기반 점수.

모델이 내놓는 값은 '앞으로 4주 안에 연체할 확률(PD)'이다.
그대로 보여 주면 초등학생에게 와닿지 않으므로, 실제 신용평가사가 쓰는
스코어카드 스케일링(PDO 방식)으로 점수를 만든다.

    score = offset + factor * ln(odds),  odds = (1 - PD) / PD
    factor = PDO / ln(2)
    offset = base_score - factor * ln(base_odds)

즉 '정상:연체 odds 가 2배 좋아질 때마다 PDO(40)점씩 올라간다'는 뜻이다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import GRADE_TABLE, ModelConfig


def scaling_params(cfg: ModelConfig | None = None) -> tuple[float, float]:
    cfg = cfg or ModelConfig()
    factor = cfg.pdo / np.log(2)
    offset = cfg.base_score - factor * np.log(cfg.base_odds)
    return float(factor), float(offset)


def pd_to_score(probabilities, cfg: ModelConfig | None = None) -> np.ndarray:
    """PD 배열을 신용점수로 변환한다."""
    cfg = cfg or ModelConfig()
    factor, offset = scaling_params(cfg)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    odds = (1 - p) / p
    score = offset + factor * np.log(odds)
    return np.clip(np.round(score), cfg.score_floor, cfg.score_cap).astype(int)


def score_to_grade(score) -> np.ndarray:
    """점수를 1(최우수)~10(위험) 등급으로 변환한다."""
    scores = np.atleast_1d(np.asarray(score, dtype=float))
    result = np.full(scores.shape, GRADE_TABLE[-1][0], dtype=int)
    for grade, cutoff, _ in GRADE_TABLE:
        result = np.where(scores >= cutoff, np.minimum(result, grade), result)
    return result


def grade_label(grade: int) -> str:
    for g, _, label in GRADE_TABLE:
        if g == int(grade):
            return label
    return GRADE_TABLE[-1][2]


# --- 교실용 규칙기반 점수 ----------------------------------------------------
# 모델 점수는 '왜 이 점수인지'를 과바로 설명하기 어렵다.
# 선생님이 칠판에 적어 설명할 수 있도록 가중합 방식의 투명한 점수도 함께 낸다.
RULE_WEIGHTS: dict[str, float] = {
    "attendance": 0.15,
    "homework": 0.15,
    "book": 0.07,
    "repayment": 0.25,
    "savings": 0.13,
    "spending": 0.10,
    "liquidity": 0.08,
    "debt": 0.07,
}

RULE_COMPONENT_LABELS: dict[str, str] = {
    "attendance": "출석",
    "homework": "숙제",
    "book": "도서 반납",
    "repayment": "약속한 납부 지키기",
    "savings": "저축",
    "spending": "소비 관리",
    "liquidity": "여유 자금",
    "debt": "빚 관리",
}

RULE_SCORE_FLOOR = 350
RULE_SCORE_RANGE = 650


def _unit(series: pd.Series, low: float, high: float) -> pd.Series:
    return ((series - low) / (high - low)).clip(0, 1)


def rule_components(features: pd.DataFrame) -> pd.DataFrame:
    """규칙기반 점수의 항목별 달성도(0~1)."""
    f = features
    late = f["late_events_8w"].clip(0, 4)
    comp = pd.DataFrame(index=f.index)
    comp["attendance"] = _unit(f["cs_attendance"], 0.75, 1.0)
    comp["homework"] = _unit(f["cs_homework"], 0.60, 1.0)
    comp["book"] = 1 - (f["cs_overdue"].clip(0, 6) / 6)
    comp["repayment"] = (
        0.45 * f["repay_ontime_ratio"] + 0.25 * f["tax_paid_ratio"] + 0.30 * (1 - late / 4)
    ).clip(0, 1)
    comp["savings"] = (0.7 * _unit(f["savings_rate"], 0.0, 0.20) + 0.3 * f["has_savings"]).clip(0, 1)
    comp["spending"] = (
        0.7 * (1 - _unit(f["consumption_income_ratio"], 0.6, 1.2))
        + 0.3 * (1 - _unit(f["discretionary_ratio"], 0.20, 0.60))
    ).clip(0, 1)
    comp["liquidity"] = _unit(f["liquidity_weeks"], 0.0, 3.0)
    comp["debt"] = 1 - _unit(f["debt_to_income"], 0.0, 2.0)
    return comp


def rule_based_score(features: pd.DataFrame) -> pd.Series:
    """가중합 기반의 설명 가능한 학급 신용점수(350~1000)."""
    comp = rule_components(features)
    weighted = sum(comp[name] * weight for name, weight in RULE_WEIGHTS.items())
    return (RULE_SCORE_FLOOR + RULE_SCORE_RANGE * weighted).round().astype(int)


def grade_distribution(scores: pd.Series) -> pd.DataFrame:
    grades = score_to_grade(scores.to_numpy())
    out = (
        pd.Series(grades, name="grade")
        .value_counts()
        .rename_axis("grade")
        .sort_index()
        .reset_index(name="count")
    )
    out["label"] = out["grade"].map(grade_label)
    out["ratio"] = (out["count"] / out["count"].sum()).round(4)
    return out
