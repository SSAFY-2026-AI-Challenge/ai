from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from credit_ai.db.models import Account, EconomicMetric, Member, Payroll, Transaction
from credit_ai.metrics.compute import (
    AccountView,
    ClassEconomy,
    TransactionView,
    class_label_to_id,
    compute_class_metric,
    gini,
)
from credit_ai.metrics.refresh import collect_class_metrics, persist_metrics

EXPECTED_COLUMNS = (
    "id",
    "class_id",
    "total_money",
    "average_asset",
    "weekly_transaction_volume",
    "average_consumption",
    "consumption_change_rate",
    "transaction_change_rate",
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
        assert stored.saving_rate == Decimal("0.1800")
        assert stored.wealth_gap == Decimal("0.3100")
        assert stored.measured_at == measured_at


def test_class_label_to_id() -> None:
    assert class_label_to_id("3-1") == 301
    assert class_label_to_id("3-2") == 302
    assert class_label_to_id("3학년 1반") == 301
    assert class_label_to_id("12") == 12
    assert class_label_to_id("alpha") == class_label_to_id("alpha")
    assert class_label_to_id("alpha") != class_label_to_id("beta")


def test_gini_equal_and_unequal() -> None:
    assert gini([100, 100, 100]) == Decimal("0.0000")
    assert gini([0, 0, 0]) == Decimal("0.0000")
    assert gini([2000, 10000]) == Decimal("0.3333")


def test_compute_class_metric_week_over_week() -> None:
    as_of = datetime(2026, 9, 3, 12, 0, 0)
    this_week = as_of - timedelta(days=2)
    last_week = as_of - timedelta(days=9)
    economy = ClassEconomy(
        class_id=301,
        class_label="3-1",
        member_ids=(1, 2),
        accounts=(
            AccountView("chk-1", 1, "CHECKING", 8000),
            AccountView("sav-1", 1, "SAVING", 2000),
            AccountView("loan-1", 1, "LOAN", 1000),
            AccountView("chk-2", 2, "CHECKING", 2000),
        ),
        transactions=(
            TransactionView(10, "chk-1", 10000, "급여", this_week),
            TransactionView(11, "chk-1", 3000, "문구점", this_week),
            TransactionView(12, "chk-2", 1000, "매점", this_week),
            TransactionView(13, "loan-1", 500, "대출", this_week),
            TransactionView(20, "chk-1", 2000, "문구점", last_week),
            TransactionView(21, "chk-2", 500, "매점", last_week),
        ),
        payroll_transaction_ids=frozenset({10}),
    )

    metric = compute_class_metric(economy, as_of=as_of)
    assert metric.class_id == 301
    assert metric.class_label == "3-1"
    assert metric.total_money == 12000
    assert metric.average_asset == Decimal("6000.00")
    assert metric.weekly_transaction_volume == 14500
    assert metric.average_consumption == Decimal("2000.00")
    assert metric.consumption_change_rate == Decimal("0.6000")
    assert metric.transaction_change_rate == Decimal("4.8000")
    assert metric.saving_rate == Decimal("0.6000")
    assert metric.wealth_gap == Decimal("0.3333")
    assert metric.measured_at == as_of


def _seed_classroom(session: Session) -> None:
    as_of = datetime(2026, 9, 3, 12, 0, 0)
    this_week = as_of - timedelta(days=1)
    teacher = Member(id=99, login_id="teacher", pw="x", name="선생님", role="teacher", class_name="3-1")
    a = Member(id=1, login_id="a", pw="x", name="민준", role="student", class_name="3-1")
    b = Member(id=2, login_id="b", pw="x", name="서연", role="student", class_name="3-1")
    c = Member(id=3, login_id="c", pw="x", name="하준", role="student", class_name="3-2")
    session.add_all([teacher, a, b, c])
    session.flush()
    session.add_all(
        [
            Account(id="chk-1", member_id=1, account_type="CHECKING", balance=5000, created_at=as_of),
            Account(id="sav-1", member_id=1, account_type="SAVING", balance=1000, created_at=as_of),
            Account(id="chk-2", member_id=2, account_type="CHECKING", balance=3000, created_at=as_of),
            Account(id="chk-3", member_id=3, account_type="CHECKING", balance=8000, created_at=as_of),
            Account(id="t-chk", member_id=99, account_type="CHECKING", balance=99999, created_at=as_of),
        ]
    )
    session.add_all(
        [
            Transaction(id=1, account_id="chk-1", amount=4000, description="급여", occured_at=this_week),
            Transaction(id=2, account_id="chk-1", amount=1000, description="매점", occured_at=this_week),
            Transaction(id=3, account_id="chk-2", amount=500, description="문구점", occured_at=this_week),
            Transaction(id=4, account_id="chk-3", amount=2000, description="매점", occured_at=this_week),
        ]
    )
    session.add(Payroll(id=1, member_id=1, transaction_id=1))
    session.commit()


def _create_metric_tables(engine) -> None:
    Member.__table__.create(engine)
    Account.__table__.create(engine)
    Transaction.__table__.create(engine)
    Payroll.__table__.create(engine)
    EconomicMetric.__table__.create(engine)


def test_collect_and_persist_metrics_by_class() -> None:
    engine = create_engine("sqlite://")
    _create_metric_tables(engine)
    as_of = datetime(2026, 9, 3, 12, 0, 0)
    with Session(engine) as session:
        _seed_classroom(session)
        metrics = collect_class_metrics(session, as_of=as_of)
        by_label = {item.class_label: item for item in metrics}
        assert set(by_label) == {"3-1", "3-2"}
        assert by_label["3-1"].class_id == 301
        assert by_label["3-1"].total_money == 9000
        assert by_label["3-2"].total_money == 8000
        assert by_label["3-1"].weekly_transaction_volume == 5500

        rows = persist_metrics(session, metrics)
        session.commit()
        assert len(rows) == 2
        stored = list(session.scalars(select(EconomicMetric).order_by(EconomicMetric.class_id)).all())
        assert [row.class_id for row in stored] == [301, 302]
        assert stored[0].total_money == 9000
        assert stored[1].total_money == 8000
