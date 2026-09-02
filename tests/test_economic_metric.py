from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from credit_ai.db.models import EconomicMetric


EXPECTED_COLUMNS = (
    "id",
    "class_id",
    "total_money",
    "average_asset",
    "weekly_transaction_volume",
    "average_consumption",
    "consumption_change_rate",
    "transaction_change_rate",
    "loan_change_rate",
    "saving_rate",
    "wealth_gap",
    "measured_at",
)


def test_economic_metric_maps_requested_columns() -> None:
    assert tuple(column.name for column in EconomicMetric.__table__.columns) == EXPECTED_COLUMNS


def test_economic_metric_persists_snapshot() -> None:
    engine = create_engine("sqlite://")
    EconomicMetric.__table__.create(engine)
    measured_at = datetime(2026, 9, 2, 9, 0, 0)
    with Session(engine) as session:
        session.add(
            EconomicMetric(
                class_id=31,
                total_money=54000,
                average_asset=Decimal("2160.00"),
                weekly_transaction_volume=12800,
                average_consumption=Decimal("430.50"),
                consumption_change_rate=Decimal("0.1200"),
                transaction_change_rate=Decimal("-0.0500"),
                loan_change_rate=Decimal("0.0000"),
                saving_rate=Decimal("0.1800"),
                wealth_gap=Decimal("0.3100"),
                measured_at=measured_at,
            )
        )
        session.commit()

        stored = session.scalars(select(EconomicMetric)).one()
        assert stored.id == 1
        assert stored.class_id == 31
        assert stored.total_money == 54000
        assert stored.average_asset == Decimal("2160.00")
        assert stored.weekly_transaction_volume == 12800
        assert stored.average_consumption == Decimal("430.50")
        assert stored.consumption_change_rate == Decimal("0.1200")
        assert stored.transaction_change_rate == Decimal("-0.0500")
        assert stored.loan_change_rate == Decimal("0.0000")
        assert stored.saving_rate == Decimal("0.1800")
        assert stored.wealth_gap == Decimal("0.3100")
        assert stored.measured_at == measured_at
