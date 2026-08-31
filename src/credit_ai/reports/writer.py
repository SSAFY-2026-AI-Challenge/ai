from __future__ import annotations

from datetime import date
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from credit_ai.db.models import AiReport


def join_recommendation_fields(
    recommendations: list[dict[str, str]],
) -> tuple[str | None, str | None]:
    contents = [item["content"] for item in recommendations if item.get("content")]
    effects = [
        item["expected_effect"]
        for item in recommendations
        if item.get("expected_effect")
    ]
    return (
        "\n".join(contents) if contents else None,
        "\n".join(effects) if effects else None,
    )


def persist_report(
    session: Session,
    member_id: int,
    summary: str,
    detail: dict[str, Any],
    recommendations: list[dict[str, str]],
) -> AiReport:
    content, expected_effect = join_recommendation_fields(recommendations)
    report = session.scalar(select(AiReport).where(AiReport.member_id == member_id))
    if report is None:
        report = AiReport(
            member_id=member_id,
            report_type="CREDIT_SCORE",
            summary=summary,
            content=content,
            expected_effect=expected_effect,
            detail_json=detail,
            generated_at=date.today(),
        )
        session.add(report)
        session.flush()
        return report

    report.report_type = "CREDIT_SCORE"
    report.summary = summary
    report.content = content
    report.expected_effect = expected_effect
    report.detail_json = detail
    report.generated_at = date.today()
    session.flush()
    return report
