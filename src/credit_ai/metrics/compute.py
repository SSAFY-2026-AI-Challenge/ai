from __future__ import annotations

import re
import zlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal

from credit_ai.features.extract import INCOME_KEYWORDS

WEEK = timedelta(days=7)
MONEY_QUANT = Decimal("0.01")
RATE_QUANT = Decimal("0.0001")
RATE_MIN = Decimal("-9999.9999")
RATE_MAX = Decimal("9999.9999")

_SAVING_TYPES = {"SAVING", "SAVINGS", "DEPOSIT", "적금", "예금", "저축"}
_LOAN_TYPES = {"LOAN", "DEBT", "CREDIT", "대출"}
_CLASS_PAIR_RE = re.compile(r"(\d+)\s*[-학년반_/.\s]+\s*(\d+)")


@dataclass(frozen=True, slots=True)
class AccountView:
    id: str
    member_id: int
    account_type: str
    balance: int


@dataclass(frozen=True, slots=True)
class TransactionView:
    id: int
    account_id: str
    amount: int
    description: str | None
    occured_at: datetime


@dataclass(frozen=True, slots=True)
class ClassEconomy:
    class_id: int
    class_label: str
    member_ids: tuple[int, ...]
    accounts: tuple[AccountView, ...]
    transactions: tuple[TransactionView, ...]
    payroll_transaction_ids: frozenset[int]


@dataclass(frozen=True, slots=True)
class ClassMetric:
    class_id: int
    class_label: str
    total_money: int
    average_asset: Decimal
    weekly_transaction_volume: int
    average_consumption: Decimal
    consumption_change_rate: Decimal
    transaction_change_rate: Decimal
    saving_rate: Decimal
    wealth_gap: Decimal
    measured_at: datetime


def class_label_to_id(label: str) -> int:
    text = label.strip()
    match = _CLASS_PAIR_RE.search(text)
    if match:
        return int(match.group(1)) * 100 + int(match.group(2))
    digits = re.findall(r"\d+", text)
    if len(digits) >= 2:
        return int(digits[0]) * 100 + int(digits[1])
    if len(digits) == 1:
        return int(digits[0])
    digest = zlib.crc32(text.encode("utf-8")) & 0x7FFFFFFF
    return digest or 1


def classify_account(account_type: str | None) -> str:
    raw = (account_type or "").strip()
    upper = raw.upper()
    if raw in _SAVING_TYPES or upper in _SAVING_TYPES or "SAV" in upper:
        return "SAVING"
    if raw in _LOAN_TYPES or upper in _LOAN_TYPES or "LOAN" in upper:
        return "LOAN"
    return "CHECKING"


def _to_decimal(value: Decimal | int | float) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        return Decimal(str(round(value, 8)))
    return Decimal(value)


def money2(value: Decimal | int | float) -> Decimal:
    return _to_decimal(value).quantize(MONEY_QUANT, rounding=ROUND_HALF_UP)


def rate4(value: Decimal | int | float) -> Decimal:
    quantized = _to_decimal(value).quantize(RATE_QUANT, rounding=ROUND_HALF_UP)
    return min(max(quantized, RATE_MIN), RATE_MAX)


def change_rate(current: int | float | Decimal, previous: int | float | Decimal) -> Decimal:
    current_d = Decimal(current)
    previous_d = Decimal(previous)
    if previous_d == 0:
        if current_d == 0:
            return rate4(0)
        return rate4(1)
    return rate4((current_d - previous_d) / previous_d)


def gini(values: list[int] | list[float]) -> Decimal:
    if not values:
        return rate4(0)
    xs = sorted(max(0.0, float(value)) for value in values)
    total = sum(xs)
    if total <= 0:
        return rate4(0)
    n = len(xs)
    weighted = sum((index + 1) * amount for index, amount in enumerate(xs))
    coefficient = (2.0 * weighted) / (n * total) - (n + 1) / n
    return rate4(min(max(coefficient, 0.0), 1.0))


def _in_window(
    tx: TransactionView,
    start: datetime,
    end: datetime,
) -> bool:
    return start <= tx.occured_at < end


def _is_income(tx: TransactionView, payroll_ids: frozenset[int]) -> bool:
    if tx.id in payroll_ids:
        return True
    description = (tx.description or "").lower()
    return any(key in description for key in INCOME_KEYWORDS)


def _volume(transactions: list[TransactionView]) -> int:
    return int(sum(abs(int(tx.amount)) for tx in transactions))


def _consumption(
    transactions: list[TransactionView],
    account_kind: dict[str, str],
    payroll_ids: frozenset[int],
) -> int:
    total = 0
    for tx in transactions:
        if account_kind.get(tx.account_id) != "CHECKING":
            continue
        if _is_income(tx, payroll_ids):
            continue
        total += abs(int(tx.amount))
    return total


def _income(
    transactions: list[TransactionView],
    account_kind: dict[str, str],
    payroll_ids: frozenset[int],
) -> int:
    total = 0
    for tx in transactions:
        if account_kind.get(tx.account_id) == "LOAN":
            continue
        if _is_income(tx, payroll_ids):
            total += abs(int(tx.amount))
    return total


def compute_class_metric(economy: ClassEconomy, *, as_of: datetime) -> ClassMetric:
    member_count = len(economy.member_ids) or 1
    account_kind = {account.id: classify_account(account.account_type) for account in economy.accounts}

    assets_by_member: dict[int, int] = {member_id: 0 for member_id in economy.member_ids}
    savings_total = 0
    total_money = 0
    for account in economy.accounts:
        kind = account_kind[account.id]
        if kind == "LOAN":
            continue
        balance = int(account.balance)
        total_money += balance
        assets_by_member[account.member_id] = assets_by_member.get(account.member_id, 0) + balance
        if kind == "SAVING":
            savings_total += balance

    this_start = as_of - WEEK
    prev_start = as_of - WEEK - WEEK
    this_week = [tx for tx in economy.transactions if _in_window(tx, this_start, as_of)]
    last_week = [tx for tx in economy.transactions if _in_window(tx, prev_start, this_start)]

    this_volume = _volume(this_week)
    last_volume = _volume(last_week)
    this_consumption = _consumption(this_week, account_kind, economy.payroll_transaction_ids)
    last_consumption = _consumption(last_week, account_kind, economy.payroll_transaction_ids)
    this_income = _income(this_week, account_kind, economy.payroll_transaction_ids)

    if this_income > 0:
        saving_rate = rate4((Decimal(this_income) - Decimal(this_consumption)) / Decimal(this_income))
    elif total_money > 0:
        saving_rate = rate4(Decimal(savings_total) / Decimal(total_money))
    else:
        saving_rate = rate4(0)

    return ClassMetric(
        class_id=economy.class_id,
        class_label=economy.class_label,
        total_money=total_money,
        average_asset=money2(Decimal(total_money) / Decimal(member_count)),
        weekly_transaction_volume=this_volume,
        average_consumption=money2(Decimal(this_consumption) / Decimal(member_count)),
        consumption_change_rate=change_rate(this_consumption, last_consumption),
        transaction_change_rate=change_rate(this_volume, last_volume),
        saving_rate=saving_rate,
        wealth_gap=gini(list(assets_by_member.values())),
        measured_at=as_of,
    )
