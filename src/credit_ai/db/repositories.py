from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from credit_ai.db.models import Account, Member, Payroll, Transaction


@dataclass(slots=True)
class MemberBundle:
    member: Member
    accounts: list[Account]
    transactions: list[Transaction]
    payrolls: list[Payroll]


def fetch_member_bundle(session: Session, member_id: int) -> MemberBundle | None:
    member = session.get(Member, member_id)
    if member is None:
        return None
    accounts = list(
        session.scalars(select(Account).where(Account.member_id == member_id)).all()
    )
    account_ids = [account.id for account in accounts]
    transactions: list[Transaction] = []
    if account_ids:
        transactions = list(
            session.scalars(
                select(Transaction)
                .where(Transaction.account_id.in_(account_ids))
                .order_by(Transaction.occured_at.asc())
            ).all()
        )
    payrolls = list(
        session.scalars(select(Payroll).where(Payroll.member_id == member_id)).all()
    )
    return MemberBundle(
        member=member,
        accounts=accounts,
        transactions=transactions,
        payrolls=payrolls,
    )


def fetch_all_student_bundles(session: Session) -> list[MemberBundle]:
    members = fetch_student_members(session)
    if not members:
        return []

    member_ids = [member.id for member in members]
    accounts_by_member: dict[int, list[Account]] = {
        member.id: list(member.accounts) for member in members
    }
    account_ids = [
        account.id
        for accounts in accounts_by_member.values()
        for account in accounts
    ]

    transactions_by_account: dict[str, list[Transaction]] = {
        account_id: [] for account_id in account_ids
    }
    if account_ids:
        loaded = session.scalars(
            select(Transaction).where(Transaction.account_id.in_(account_ids))
        ).all()
        for tx in loaded:
            transactions_by_account.setdefault(tx.account_id, []).append(tx)

    payrolls_by_member: dict[int, list[Payroll]] = {member_id: [] for member_id in member_ids}
    loaded_payrolls = session.scalars(
        select(Payroll).where(Payroll.member_id.in_(member_ids))
    ).all()
    for payroll in loaded_payrolls:
        payrolls_by_member.setdefault(payroll.member_id, []).append(payroll)

    bundles: list[MemberBundle] = []
    for member in members:
        accounts = accounts_by_member[member.id]
        transactions: list[Transaction] = []
        for account in accounts:
            transactions.extend(transactions_by_account.get(account.id, []))
        transactions.sort(key=lambda tx: tx.occured_at)
        bundles.append(
            MemberBundle(
                member=member,
                accounts=accounts,
                transactions=transactions,
                payrolls=payrolls_by_member.get(member.id, []),
            )
        )
    return bundles


def fetch_student_members(session: Session) -> list[Member]:
    stmt = (
        select(Member)
        .where(Member.role.is_(None) | (Member.role != "teacher"))
        .options(selectinload(Member.accounts))
        .order_by(Member.id.asc())
    )
    return list(session.scalars(stmt).all())
