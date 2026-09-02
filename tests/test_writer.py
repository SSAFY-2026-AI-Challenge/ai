from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from credit_ai.db.models import AiReport, Base, Member
from credit_ai.reports.writer import join_recommendation_fields, persist_report


def test_join_recommendation_fields_joins_and_clips() -> None:
    content, expected_effect = join_recommendation_fields(
        [
            {
                "content": "숙제는 그날그날 끝내 보세요.",
                "expected_effect": "숙제 완료율이 오르면 점수가 오릅니다.",
            },
            {
                "content": "출석을 지키는 일부터 시작해 보세요.",
                "expected_effect": "출결률이 높아지면 신뢰 점수가 안정됩니다.",
            },
        ]
    )
    assert "숙제" in content
    assert "출석" in content
    assert "\n" in content
    assert "점수가 오릅니다" in expected_effect
    assert len(content) <= 512
    assert len(expected_effect) <= 512


def test_ai_report_table_does_not_map_contributions() -> None:
    assert "contributions" not in {column.name for column in AiReport.__table__.columns}


def test_persist_report_writes_features_and_copy() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine, tables=[Member.__table__, AiReport.__table__])
    with Session(engine) as session:
        session.add(Member(id=1, login_id="minjun", pw="secret", name="민준"))
        session.commit()
        report = persist_report(
            session,
            1,
            "민준 친구의 신용점수는 820점입니다.",
            {
                "spending_trend": 0.2,
                "homework_rate": 0.95,
                "attendance_rate": 0.98,
            },
            [
                {
                    "content": "지금 습관을 그대로 유지해 보세요.",
                    "expected_effect": "현재 등급을 안정적으로 유지할 수 있습니다.",
                }
            ],
            score=820,
        )
        session.commit()

        stored = session.get(AiReport, report.id)
        assert stored is not None
        assert stored.credit_score == 820
        assert stored.features["homework_rate"] == 0.95
        assert "습관" in stored.content
        assert not hasattr(stored, "contributions")
