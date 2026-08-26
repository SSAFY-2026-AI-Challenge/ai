from __future__ import annotations

from datetime import date
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from credit_ai.db.models import AiRecommendation, AiReport


def persist_report(
    session: Session,
    member_id: int,
    summary: str,
    detail: dict[str, Any],
    recommendations: list[dict[str, str]],
) -> AiReport:
    report = session.scalar(select(AiReport).where(AiReport.member_id == member_id))
    if report is None:
        report = AiReport(
            member_id=member_id,
            report_type="CREDIT_SCORE",
            summary=summary,
            detail_json=detail,
            generated_at=date.today(),
        )
        session.add(report)
        session.flush()
    else:
        report.report_type = "CREDIT_SCORE"
        report.summary = summary
        report.detail_json = detail
        report.generated_at = date.today()
        session.execute(delete(AiRecommendation).where(AiRecommendation.report_id == report.id))

    for item in recommendations:
        session.add(
            AiRecommendation(
                report_id=report.id,
                recommendation_type=item["recommendation_type"],
                content=item["content"],
                expected_effect=item["expected_effect"],
                is_applied=0,
            )
        )
    session.flush()
    return report
