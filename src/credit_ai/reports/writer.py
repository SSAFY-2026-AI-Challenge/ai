from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from credit_ai.db.models import AiReport

TEXT_LIMIT = 512


def clip_text(value: str | None) -> str:
    return (value or "")[:TEXT_LIMIT]


def join_recommendation_fields(
    recommendations: list[dict[str, str]],
) -> tuple[str, str]:
    contents = [item["content"] for item in recommendations if item.get("content")]
    effects = [
        item["expected_effect"]
        for item in recommendations
        if item.get("expected_effect")
    ]
    return (
        clip_text("\n".join(contents) if contents else ""),
        clip_text("\n".join(effects) if effects else ""),
    )


def persist_report(
    session: Session,
    member_id: int,
    summary: str,
    features: dict[str, Any],
    recommendations: list[dict[str, str]],
    *,
    score: int,
) -> AiReport:
    content, expected_effect = join_recommendation_fields(recommendations)
    report = session.scalar(select(AiReport).where(AiReport.member_id == member_id))
    if report is None:
        report = AiReport(member_id=member_id)
        session.add(report)

    report.credit_score = score
    report.summary = clip_text(summary)
    report.content = content
    report.expected_effect = expected_effect
    report.features = features
    report.generated_at = datetime.now()
    session.flush()
    return report
