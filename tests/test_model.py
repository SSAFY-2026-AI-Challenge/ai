from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from credit_ai.db.models import Account, Member, Payroll, Transaction
from credit_ai.db.repositories import MemberBundle
from credit_ai.features.extract import extract_features
from credit_ai.features.schema import CreditFeatures
from credit_ai.model.explain import explain_score
from credit_ai.model.scorer import CreditScorer
from credit_ai.training.synthetic import teacher_score
from credit_ai.training.train import train_model


def test_teacher_score_stays_in_band() -> None:
    features = CreditFeatures(0.2, 1.0, 1.0)
    score = teacher_score(features)
    assert 300 <= score <= 1000


def test_better_habits_raise_teacher_score() -> None:
    weak = CreditFeatures(0.8, 0.5, 0.6)
    strong = CreditFeatures(0.15, 1.0, 1.0)
    assert teacher_score(strong) > teacher_score(weak) + 80


def test_extract_attendance_homework_and_spending() -> None:
    now = datetime(2026, 3, 1, 9, 0, 0)
    member = Member(
        id=1,
        login_id="minjun",
        pw="secret",
        name="민준",
        salary="20000",
        cs_attendance=Decimal("0.9700"),
        cs_overdue=2,
        cs_homework=Decimal("0.8800"),
    )
    checking = Account(
        id="chk-1",
        member_id=1,
        account_type="CHECKING",
        balance=3000,
        created_at=now,
    )
    payroll_tx = Transaction(
        id=10,
        account_id="chk-1",
        amount=20000,
        description="급여",
        occured_at=now,
    )
    snack = Transaction(
        id=11,
        account_id="chk-1",
        amount=4000,
        description="문구점",
        occured_at=now,
    )
    bundle = MemberBundle(
        member=member,
        accounts=[checking],
        transactions=[payroll_tx, snack],
        payrolls=[Payroll(id=1, member_id=1, transaction_id=10)],
    )
    features = extract_features(bundle)
    assert features.homework_rate == 0.88
    assert features.attendance_rate == 0.97
    assert 0 <= features.spending_trend <= 1
    assert not hasattr(features, "book_overdue")
    assert not hasattr(features, "savings_amount")


def test_mlp_follows_habit_direction(tmp_path) -> None:
    metrics = train_model(tmp_path, n_samples=900, epochs=12, batch_size=64, seed=7)
    assert metrics["val_mae"] < 55
    scorer = CreditScorer.load(tmp_path)
    low = CreditFeatures(0.85, 0.45, 0.55)
    high = CreditFeatures(0.12, 0.99, 0.99)
    assert scorer.score(high).score > scorer.score(low).score
    lifts = explain_score(scorer, low)
    assert lifts[0].score_lift > 0
