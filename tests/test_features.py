"""피처 엔지니어링 테스트.

DB 없이, 시뮬레이터가 만든 데이터를 그대로 프레임으로 넣어 검증한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fin_edu_credit import domain as D
from fin_edu_credit.config import ModelConfig, SimulationConfig
from fin_edu_credit.features import (
    FEATURE_NAMES,
    NO_LATE_SENTINEL,
    build_features,
    build_labels,
    build_panel,
    prepare_source,
)
from fin_edu_credit.seed import simulate


@pytest.fixture(scope="module")
def source():
    """작은 학급 하나로 축소한 시뮬레이션 결과를 DB 스키마 모양으로 변환한다."""
    cfg = SimulationConfig(grades=(3,), classes_per_grade=1, students_per_class=12)
    economy = simulate(cfg)

    members = economy.members.copy()
    members["id"] = np.arange(1, len(members) + 1)
    login_to_id = dict(zip(members["login_id"], members["id"]))

    accounts = economy.accounts.copy()
    accounts["member_id"] = accounts["login_id"].map(login_to_id)
    accounts = accounts[["account_id", "member_id", "account_type", "balance", "created_at"]]

    transactions = economy.transactions.copy().reset_index(drop=True)
    transactions["id"] = np.arange(1, len(transactions) + 1)
    transactions["member_id"] = transactions["login_id"].map(login_to_id)
    account_type = dict(zip(accounts["account_id"], accounts["account_type"]))
    transactions["account_type"] = transactions["account_id"].map(account_type)
    transactions = transactions[
        ["id", "account_id", "member_id", "account_type", "amount", "description", "occured_at"]
    ]

    return prepare_source(members, accounts, transactions)


def test_simulation_produces_all_account_types(source):
    assert set(source.accounts["account_type"]) == {
        D.ACCOUNT_CHECKING,
        D.ACCOUNT_SAVINGS,
        D.ACCOUNT_LOAN,
    }


def test_account_balance_equals_transaction_sum(source):
    """잔액은 거래액 누적합과 정확히 일치해야 한다(시점 잔액 복원의 전제)."""
    summed = source.transactions.groupby("account_id")["amount"].sum()
    stored = source.accounts.set_index("account_id")["balance"]
    aligned = stored.reindex(summed.index)
    assert (aligned - summed).abs().max() == 0


def test_build_features_returns_all_columns(source):
    as_of = source.max_date - pd.Timedelta(weeks=4)
    features = build_features(source, as_of, observation_weeks=8)
    assert list(features.columns) == ["as_of", *FEATURE_NAMES]
    assert len(features) > 0
    assert features[list(FEATURE_NAMES)].notna().all().all()


def test_features_use_only_past_data(source):
    """기준일 이후 거래를 지워도 피처가 그대로여야 한다(미래 정보 누수 방지)."""
    as_of = source.max_date - pd.Timedelta(weeks=6)
    full = build_features(source, as_of, observation_weeks=8)

    # build_features 는 기준일의 하루 끝까지를 관측 대상으로 삼는다.
    cutoff = as_of.normalize() + pd.Timedelta(hours=23, minutes=59)
    truncated = prepare_source(
        source.members,
        source.accounts[source.accounts["created_at"] <= cutoff],
        source.transactions[source.transactions["occured_at"] <= cutoff].drop(
            columns=[c for c in source.transactions.columns if c.startswith("is_")]
            + ["income_amount", "expense_amount", "late_fee_amount", "discretionary_amount",
               "savings_deposit_amount", "loan_repay_amount", "run_balance"]
        ),
    )
    partial = build_features(truncated, as_of, observation_weeks=8)
    common = partial.index.intersection(full.index)
    pd.testing.assert_frame_equal(
        full.loc[common, list(FEATURE_NAMES)],
        partial.loc[common, list(FEATURE_NAMES)],
        check_exact=False,
        rtol=1e-9,
    )


def test_days_since_last_late_sentinel(source):
    as_of = source.max_date - pd.Timedelta(weeks=4)
    features = build_features(source, as_of, observation_weeks=8)
    never_late = features["late_events_8w"] == 0
    assert (features.loc[never_late, "days_since_last_late"] == NO_LATE_SENTINEL).all()
    assert (features.loc[~never_late, "days_since_last_late"] < NO_LATE_SENTINEL).all()


def test_labels_match_late_transactions(source):
    as_of = source.max_date - pd.Timedelta(weeks=4)
    labels = build_labels(source, as_of, performance_weeks=4)
    window = source.transactions[
        (source.transactions["occured_at"] > as_of)
        & (source.transactions["occured_at"] <= as_of + pd.Timedelta(weeks=4))
        & source.transactions["is_late"]
    ]
    expected = set(window["member_id"].unique())
    assert set(labels[labels == 1].index) == expected


def test_ratio_features_stay_in_range(source):
    as_of = source.max_date - pd.Timedelta(weeks=4)
    features = build_features(source, as_of, observation_weeks=8)
    for column in ("savings_rate", "discretionary_ratio", "low_balance_ratio",
                   "repay_ontime_ratio", "tax_paid_ratio"):
        assert features[column].between(0, 1).all(), column


def test_build_panel_has_multiple_snapshots(source):
    panel = build_panel(source, ModelConfig())
    assert panel["as_of"].nunique() >= 3
    assert set(panel["y"].unique()) <= {0, 1}
    assert panel["member_id"].nunique() > 1
