"""사유코드 생성.

스코어카드는 '피처별 배점의 합'이라서, 왜 이 점수가 나왔는지 정확히 분해할 수 있다.

    점수 = Σ (피처별 배점)
    기여도 = (내 배점) - (학급 중앙값 배점)

기여도가 크게 음수인 항목이 점수를 깎은 이유이고, 크게 양수인 항목이 강점이다.
근사치가 아니라 정확한 분해이므로, 학생에게 그대로 보여 줄 수 있다.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .config import ModelConfig
from .features import FEATURE_BY_NAME


def reason_codes(
    model,
    features: pd.DataFrame,
    cfg: ModelConfig | None = None,
    top_negative: int = 3,
    top_positive: int = 2,
) -> dict[int, dict[str, list[dict[str, Any]]]]:
    """학생별 감점/가점 사유를 만든다. 키는 features 의 인덱스(member_id)."""
    cfg = cfg or ModelConfig()
    points = model.points(features, cfg)
    reference = pd.Series(model.reference_points)
    contribution = points - reference

    out: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for member_id, row in contribution.iterrows():
        ordered = row.sort_values()
        negatives = [
            _entry(model, features, member_id, name, value, is_negative=True)
            for name, value in ordered.head(top_negative).items()
            if value < -0.5
        ]
        positives = [
            _entry(model, features, member_id, name, value, is_negative=False)
            for name, value in ordered.tail(top_positive)[::-1].items()
            if value > 0.5
        ]
        out[member_id] = {"minus": negatives, "plus": positives}
    return out


def _entry(model, features: pd.DataFrame, member_id, name: str, delta: float, is_negative: bool):
    spec = FEATURE_BY_NAME[name]
    value = float(features.loc[member_id, name])
    entry = {
        "feature": name,
        "label": spec.label,
        "value": round(value, 4),
        "points_delta": round(float(delta), 1),
    }
    if is_negative:
        entry["advice"] = spec.advice
    return entry


def summarize(name: str, score: int, grade: int, grade_label_: str, reasons: dict) -> str:
    """리포트용 한 문단 요약."""
    minus = reasons.get("minus", [])
    plus = reasons.get("plus", [])
    parts = [f"{name} 학생의 학급 신용점수는 {score}점, {grade}등급({grade_label_})입니다."]
    if plus:
        strengths = ", ".join(p["label"] for p in plus)
        parts.append(f"잘하고 있는 점은 {strengths} 입니다.")
    if minus:
        weakest = minus[0]
        others = ", ".join(m["label"] for m in minus[1:])
        tail = f" 그 밖에 {others} 도 살펴보면 좋아요." if others else ""
        parts.append(
            f"점수를 가장 많이 깎은 항목은 {weakest['label']}({weakest['points_delta']:+.0f}점)입니다. "
            f"{weakest['advice']}{tail}"
        )
    else:
        parts.append("눈에 띄게 부족한 항목이 없어요. 지금 습관을 그대로 이어 가세요.")
    return " ".join(parts)
