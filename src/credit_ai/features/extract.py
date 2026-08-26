from __future__ import annotations

import re
from datetime import datetime

import numpy as np

from credit_ai.db.models import Account, Member, Payroll, Transaction
from credit_ai.db.repositories import MemberBundle
from credit_ai.features.schema import CreditFeatures

INCOME_KEYWORDS = ("급여", "월급", "봉급", "salary", "payroll", "income", "용돈")


def parse_salary(raw: str | None) -> int:
    if not raw:
        return 0
    digits = re.sub(r"[^0-9-]", "", str(raw))
    if not digits or digits == "-":
        return 0
    return abs(int(digits))


def _income_and_expense(
    transactions: list[Transaction],
    payrolls: list[Payroll],
) -> tuple[int, int]:
    payroll_ids = {row.transaction_id for row in payrolls}
    income = 0
    expense = 0
    for tx in transactions:
        amount = int(tx.amount)
        description = (tx.description or "").lower()
        if tx.id in payroll_ids or any(key in description for key in INCOME_KEYWORDS):
            income += abs(amount)
        elif amount < 0:
            expense += abs(amount)
        else:
            expense += abs(amount)
    return income, expense


def _spending_slope(transactions: list[Transaction], payrolls: list[Payroll]) -> float:
    payroll_ids = {row.transaction_id for row in payrolls}
    points: list[tuple[datetime, int]] = []
    for tx in transactions:
        if tx.id in payroll_ids:
            continue
        if int(tx.amount) == 0:
            continue
        points.append((tx.occured_at, abs(int(tx.amount))))
    if len(points) < 3:
        return 0.0
    points.sort(key=lambda item: item[0])
    amounts = np.array([value for _, value in points], dtype=np.float64)
    xs = np.linspace(0.0, 1.0, num=len(amounts))
    slope = float(np.polyfit(xs, amounts, 1)[0])
    scale = float(np.mean(amounts) + 1.0)
    return float(np.clip(slope / scale, -1.0, 1.0))


def extract_spending_trend(
    member: Member,
    accounts: list[Account],
    transactions: list[Transaction],
    payrolls: list[Payroll],
) -> float:
    """소비 성향. 0에 가까울수록 검소, 1에 가까울수록 과소비. 거래 현금흐름만 사용합니다."""
    salary = parse_salary(member.salary)
    income, expense = _income_and_expense(transactions, payrolls)
    if income <= 0 and salary > 0:
        income = salary

    if income > 0:
        ratio = float(np.clip(float(expense) / float(income), 0.0, 2.0) / 2.0)
    else:
        ratio = 0.45

    slope = _spending_slope(transactions, payrolls)
    if slope > 0:
        ratio = float(np.clip(ratio + 0.12 * slope, 0.0, 1.0))
    elif slope < 0:
        ratio = float(np.clip(ratio + 0.08 * slope, 0.0, 1.0))
    return ratio


def extract_features(bundle: MemberBundle) -> CreditFeatures:
    member = bundle.member
    attendance = float(member.cs_attendance if member.cs_attendance is not None else 1.0)
    homework = float(member.cs_homework if member.cs_homework is not None else 1.0)
    return CreditFeatures(
        spending_trend=extract_spending_trend(
            member, bundle.accounts, bundle.transactions, bundle.payrolls
        ),
        homework_rate=homework,
        attendance_rate=attendance,
    ).clamped()
