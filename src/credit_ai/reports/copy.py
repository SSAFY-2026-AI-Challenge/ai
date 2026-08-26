from __future__ import annotations

from credit_ai.features.schema import CreditFeatures
from credit_ai.model.explain import FeatureContribution
from credit_ai.model.scorer import ScoreOutput

RECOMMENDATION_COPY: dict[str, dict[str, str]] = {
    "spending_trend": {
        "type": "REDUCE_SPENDING",
        "content": "사고 싶은 것과 꼭 필요한 것을 나눠 생각해 보세요. 일주일 용돈을 먼저 계획하면 지출이 차분해집니다.",
        "expected_effect": "과소비를 줄이면 신용점수가 오릅니다.",
    },
    "homework_rate": {
        "type": "FINISH_HOMEWORK",
        "content": "숙제는 미루지 말고 그날그날 끝내 보세요. 성실함은 신용평가에서 크게 반영됩니다.",
        "expected_effect": "숙제 완료율이 오르면 신용등급이 한 단계 올라갈 수 있습니다.",
    },
    "attendance_rate": {
        "type": "ATTEND_CLASS",
        "content": "학교에 꾸준히 나오는 친구는 믿을 수 있어요. 출석을 지키는 일부터 시작해 보세요.",
        "expected_effect": "출결률이 높아지면 기본 신뢰 점수가 안정적으로 올라갑니다.",
    },
}


def build_summary(
    name: str, result: ScoreOutput, contributions: list[FeatureContribution]
) -> str:
    hurts = [row for row in contributions if row.direction == "improve"]
    if result.score >= 850:
        return (
            f"{name} 친구의 신용점수는 {result.score}점, 등급은 {result.grade}입니다. "
            "약속을 잘 지키고 돈을 알뜰하게 다루고 있어요."
        )
    if hurts:
        focus = hurts[0]
        return (
            f"{name} 친구의 신용점수는 {result.score}점({result.grade})입니다. "
            f"지금은 {focus.label}을 조금만 더 신경 쓰면 점수가 오를 여지가 있어요."
        )
    return (
        f"{name} 친구의 신용점수는 {result.score}점, 등급은 {result.grade}입니다. "
        "지금처럼 꾸준히 지키면 됩니다."
    )


def build_recommendations(
    features: CreditFeatures,
    contributions: list[FeatureContribution],
) -> list[dict[str, str]]:
    recs: list[dict[str, str]] = []
    for row in contributions:
        if row.direction != "improve" or row.score_lift < 8:
            continue
        template = RECOMMENDATION_COPY[row.feature]
        recs.append(
            {
                "recommendation_type": template["type"],
                "content": template["content"],
                "expected_effect": f"{template['expected_effect']} (예상 +{int(round(row.score_lift))}점)",
            }
        )
        if len(recs) >= 3:
            break
    if not recs:
        recs.append(
            {
                "recommendation_type": "KEEP_HABIT",
                "content": "지금 습관을 그대로 유지해 보세요. 작은 약속을 계속 지키는 것이 가장 좋은 신용 관리입니다.",
                "expected_effect": "현재 등급을 안정적으로 유지할 수 있습니다.",
            }
        )
    return recs
