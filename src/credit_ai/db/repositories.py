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


def fetch_student_members(session: Session) -> list[Member]:
    stmt = (
        select(Member)
        .where(Member.role.is_(None) | (Member.role != "teacher"))
        .options(selectinload(Member.accounts))
        .order_by(Member.id.asc())
    )
    return list(session.scalars(stmt).all())
