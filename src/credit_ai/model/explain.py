from __future__ import annotations

from dataclasses import dataclass, replace

from credit_ai.features.schema import FEATURE_LABELS_KO, FEATURE_ORDER, CreditFeatures
from credit_ai.model.scorer import CreditScorer

IDEAL_FEATURES = CreditFeatures(
    spending_trend=0.18,
    homework_rate=1.0,
    attendance_rate=1.0,
)


@dataclass(slots=True)
class FeatureContribution:
    feature: str
    label: str
    current_value: float | int
    ideal_value: float | int
    score_lift: float
    direction: str


def explain_score(scorer: CreditScorer, features: CreditFeatures) -> list[FeatureContribution]:
    current = scorer.predict(features)
    rows: list[FeatureContribution] = []
    for name in FEATURE_ORDER:
        counter = replace(features, **{name: getattr(IDEAL_FEATURES, name)})
        lifted = scorer.predict(counter)
        delta = float(lifted - current)
        if delta > 1.5:
            direction = "improve"
        elif delta < -1.5:
            direction = "already_good"
        else:
            direction = "neutral"
        rows.append(
            FeatureContribution(
                feature=name,
                label=FEATURE_LABELS_KO[name],
                current_value=features.as_dict()[name],
                ideal_value=getattr(IDEAL_FEATURES, name),
                score_lift=round(delta, 2),
                direction=direction,
            )
        )
    rows.sort(key=lambda row: row.score_lift, reverse=True)
    return rows
