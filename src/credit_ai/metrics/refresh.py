from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime

from sqlalchemy.orm import Session

from credit_ai.config import Settings
from credit_ai.db.models import EconomicMetric
from credit_ai.db.repositories import MemberBundle, fetch_all_student_bundles
from credit_ai.db.session import session_scope
from credit_ai.metrics.compute import (
    AccountView,
    ClassEconomy,
    ClassMetric,
    TransactionView,
    class_label_to_id,
    compute_class_metric,
)

logger = logging.getLogger("credit_ai.metrics")


def _group_by_class(bundles: list[MemberBundle]) -> list[ClassEconomy]:
    grouped: dict[str, list[MemberBundle]] = defaultdict(list)
    for bundle in bundles:
        label = (bundle.member.class_name or "").strip()
        if not label:
            continue
        grouped[label].append(bundle)

    economies: list[ClassEconomy] = []
    for label, class_bundles in grouped.items():
        accounts: list[AccountView] = []
        transactions: list[TransactionView] = []
        payroll_ids: set[int] = set()
        member_ids: list[int] = []
        for bundle in class_bundles:
            member_ids.append(bundle.member.id)
            for account in bundle.accounts:
                accounts.append(
                    AccountView(
                        id=account.id,
                        member_id=account.member_id,
                        account_type=account.account_type,
                        balance=int(account.balance),
                    )
                )
            for tx in bundle.transactions:
                transactions.append(
                    TransactionView(
                        id=tx.id,
                        account_id=tx.account_id,
                        amount=int(tx.amount),
                        description=tx.description,
                        occured_at=tx.occured_at,
                    )
                )
            payroll_ids.update(payroll.transaction_id for payroll in bundle.payrolls)
        economies.append(
            ClassEconomy(
                class_id=class_label_to_id(label),
                class_label=label,
                member_ids=tuple(member_ids),
                accounts=tuple(accounts),
                transactions=tuple(transactions),
                payroll_transaction_ids=frozenset(payroll_ids),
            )
        )
    economies.sort(key=lambda item: (item.class_id, item.class_label))
    return economies


def persist_metrics(session: Session, metrics: list[ClassMetric]) -> list[EconomicMetric]:
    rows: list[EconomicMetric] = []
    for metric in metrics:
        row = EconomicMetric(
            class_id=metric.class_id,
            total_money=metric.total_money,
            average_asset=metric.average_asset,
            weekly_transaction_volume=metric.weekly_transaction_volume,
            average_consumption=metric.average_consumption,
            consumption_change_rate=metric.consumption_change_rate,
            transaction_change_rate=metric.transaction_change_rate,
            saving_rate=metric.saving_rate,
            wealth_gap=metric.wealth_gap,
            measured_at=metric.measured_at,
        )
        session.add(row)
        rows.append(row)
    session.flush()
    return rows


def collect_class_metrics(
    session: Session,
    *,
    as_of: datetime | None = None,
) -> list[ClassMetric]:
    measured_at = as_of or datetime.now()
    bundles = fetch_all_student_bundles(session)
    return [
        compute_class_metric(economy, as_of=measured_at)
        for economy in _group_by_class(bundles)
    ]


def metric_to_dict(metric: ClassMetric, row_id: int | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "class_id": metric.class_id,
        "class_label": metric.class_label,
        "total_money": metric.total_money,
        "average_asset": str(metric.average_asset),
        "weekly_transaction_volume": metric.weekly_transaction_volume,
        "average_consumption": str(metric.average_consumption),
        "consumption_change_rate": str(metric.consumption_change_rate),
        "transaction_change_rate": str(metric.transaction_change_rate),
        "saving_rate": str(metric.saving_rate),
        "wealth_gap": str(metric.wealth_gap),
        "measured_at": metric.measured_at.isoformat(sep=" ", timespec="seconds"),
    }
    if row_id is not None:
        payload["id"] = row_id
    return payload


def refresh_economic_metrics(
    settings: Settings,
    *,
    as_of: datetime | None = None,
) -> list[dict[str, object]]:
    if not settings.db_configured:
        raise RuntimeError("DB 접속 정보가 없어 economic_metric 을 갱신할 수 없습니다.")

    with session_scope(settings) as session:
        metrics = collect_class_metrics(session, as_of=as_of)
        rows = persist_metrics(session, metrics)
        payload = [
            metric_to_dict(metric, row_id=row.id)
            for metric, row in zip(metrics, rows, strict=True)
        ]

    logger.info("economic_metric 스냅샷 %s개 저장", len(payload))
    return payload
