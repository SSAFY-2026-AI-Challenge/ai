from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Integer, JSON, Numeric, String, Text, UniqueConstraint
from sqlalchemy.dialects.mysql import SMALLINT, TINYINT
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Member(Base):
    __tablename__ = "member"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    login_id: Mapped[str] = mapped_column(String(45), unique=True)
    pw: Mapped[str] = mapped_column(String(45))
    name: Mapped[str] = mapped_column(String(45))
    role: Mapped[str | None] = mapped_column(String(45))
    avatar_url: Mapped[str | None] = mapped_column(String(45))
    status: Mapped[str | None] = mapped_column(String(45))
    class_name: Mapped[str | None] = mapped_column("class", String(45))
    job: Mapped[str | None] = mapped_column(String(45))
    salary: Mapped[str | None] = mapped_column(String(45))
    tax1_name: Mapped[str | None] = mapped_column(String(45))
    tax1_amount: Mapped[int | None] = mapped_column(Integer)
    tax2_name: Mapped[str | None] = mapped_column(String(45))
    tax2_amount: Mapped[int | None] = mapped_column(Integer)
    tax3_name: Mapped[str | None] = mapped_column(String(45))
    tax3_amount: Mapped[int | None] = mapped_column(Integer)
    tax4_name: Mapped[str | None] = mapped_column(String(45))
    tax4_amount: Mapped[int | None] = mapped_column(Integer)
    tax5_name: Mapped[str | None] = mapped_column(String(45))
    tax5_amount: Mapped[int | None] = mapped_column(Integer)
    cs_attendance: Mapped[Decimal | None] = mapped_column(
        Numeric(5, 4), default=Decimal("1.0000")
    )
    cs_overdue: Mapped[int | None] = mapped_column(Integer, default=0)
    cs_homework: Mapped[Decimal | None] = mapped_column(
        Numeric(5, 4), default=Decimal("1.0000")
    )

    accounts: Mapped[list[Account]] = relationship(back_populates="member")
    reports: Mapped[list[AiReport]] = relationship(back_populates="member")


class Account(Base):
    __tablename__ = "account"

    id: Mapped[str] = mapped_column(String(45), primary_key=True)
    member_id: Mapped[int] = mapped_column(ForeignKey("member.id"))
    account_type: Mapped[str] = mapped_column(String(20))
    balance: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime)

    member: Mapped[Member] = relationship(back_populates="accounts")
    transactions: Mapped[list[Transaction]] = relationship(back_populates="account")


class Transaction(Base):
    __tablename__ = "transaction"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[str] = mapped_column(ForeignKey("account.id"))
    amount: Mapped[int] = mapped_column(Integer)
    description: Mapped[str | None] = mapped_column(String(45))
    occured_at: Mapped[datetime] = mapped_column(DateTime)

    account: Mapped[Account] = relationship(back_populates="transactions")


class Payroll(Base):
    __tablename__ = "payroll"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    member_id: Mapped[int] = mapped_column(ForeignKey("member.id"), unique=True)
    transaction_id: Mapped[int] = mapped_column(ForeignKey("transaction.id"), unique=True)


class AiReport(Base):
    __tablename__ = "ai_report"
    __table_args__ = (
        UniqueConstraint("member_id", name="uk_ai_report_member"),
        CheckConstraint(
            "credit_score BETWEEN 300 AND 1000", name="chk_ai_report_score"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    member_id: Mapped[int] = mapped_column(
        ForeignKey("member.id", name="fk_ai_report_member")
    )
    credit_score: Mapped[int] = mapped_column(SMALLINT(unsigned=True))
    summary: Mapped[str] = mapped_column(String(512))
    content: Mapped[str] = mapped_column(String(512))
    expected_effect: Mapped[str] = mapped_column(String(512))
    features: Mapped[dict[str, Any]] = mapped_column(JSON)
    generated_at: Mapped[datetime] = mapped_column(DateTime)

    member: Mapped[Member] = relationship(back_populates="reports")


class AiRecommendation(Base):
    __tablename__ = "ai_recommendation"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    report_id: Mapped[int] = mapped_column(Integer)
    recommendation_type: Mapped[str | None] = mapped_column(String(45))
    content: Mapped[str | None] = mapped_column(Text)
    expected_effect: Mapped[str | None] = mapped_column(Text)
    is_applied: Mapped[int | None] = mapped_column(TINYINT, default=0)
