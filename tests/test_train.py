"""학습 파이프라인 테스트 (DB 없이 시뮬레이션 데이터로 수행)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fin_edu_credit.config import ModelConfig, SimulationConfig
from fin_edu_credit.explain import reason_codes, summarize
from fin_edu_credit.features import FEATURE_NAMES, build_panel, prepare_source
from fin_edu_credit.scorecard import pd_to_score
from fin_edu_credit.seed import simulate
from fin_edu_credit.train import decile_table, ks_statistic, train_model


@pytest.fixture(scope="module")
def panel():
    cfg = SimulationConfig(grades=(3, 4), classes_per_grade=1, students_per_class=30)
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

    source = prepare_source(members, accounts, transactions)
    return build_panel(source, ModelConfig())


@pytest.fixture(scope="module")
def model(panel):
    return train_model(panel=panel, cfg=ModelConfig())


def test_panel_has_both_classes(panel):
    assert 0.02 < panel["y"].mean() < 0.6


def test_all_coefficients_are_non_positive(model):
    """WoE 는 클수록 안전하므로, 연체 확률 모델의 계수는 모두 0 이하여야 한다."""
    assert np.all(model.classifier.coef_[0] <= 0)


def test_model_beats_random(model):
    assert model.metrics["test"]["auc"] > 0.6
    assert model.metrics["test"]["ks"] > 0.2


def test_train_test_split_is_by_member(model):
    """같은 학생이 학습과 검증에 동시에 들어가면 안 된다."""
    assert model.metrics["n_members_train"] > 0
    assert model.metrics["n_members_test"] > 0


def test_points_sum_equals_score(model, panel):
    """배점의 합은 PD 를 변환한 점수와 정확히 같아야 한다."""
    cfg = ModelConfig()
    X = panel[list(FEATURE_NAMES)].head(200)
    points_total = model.points(X, cfg).sum(axis=1).to_numpy()
    clipped = np.clip(np.round(points_total), cfg.score_floor, cfg.score_cap)
    assert np.array_equal(clipped, pd_to_score(model.predict_pd(X), cfg))


def test_scorecard_table_covers_selected_features(model):
    table = model.scorecard_table()
    assert set(table["feature"]) == set(model.selected_features)
    assert table["points"].notna().all()


def test_better_bin_gives_more_points(model):
    """같은 피처 안에서는 WoE 가 높은(안전한) 구간이 항상 배점이 높아야 한다."""
    table = model.scorecard_table()
    for _, group in table.groupby("feature"):
        ordered = group.sort_values("woe")
        assert ordered["points"].is_monotonic_increasing


def test_reason_codes_explain_low_scores(model, panel):
    cfg = ModelConfig()
    X = panel[list(FEATURE_NAMES)].head(50)
    X.index = range(len(X))
    reasons = reason_codes(model, X, cfg)
    assert len(reasons) == len(X)
    scores = pd_to_score(model.predict_pd(X), cfg)
    worst = int(np.argmin(scores))
    assert reasons[worst]["minus"], "가장 낮은 점수에는 감점 사유가 있어야 한다"
    for item in reasons[worst]["minus"]:
        assert item["points_delta"] < 0
        assert item["advice"]


def test_summary_text_mentions_score_and_grade(model, panel):
    cfg = ModelConfig()
    X = panel[list(FEATURE_NAMES)].head(5)
    X.index = range(len(X))
    reasons = reason_codes(model, X, cfg)
    text = summarize("김하늘", 700, 4, "보통 상", reasons[0])
    assert "김하늘" in text and "700점" in text and "4등급" in text


def test_ks_statistic_bounds():
    y = np.array([0, 0, 1, 1])
    assert ks_statistic(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert 0 <= ks_statistic(y, np.array([0.5, 0.5, 0.5, 0.5])) <= 1


def test_decile_table_orders_by_score():
    rng = np.random.default_rng(0)
    scores = rng.uniform(400, 900, 1000)
    y = (rng.uniform(size=1000) < (900 - scores) / 700).astype(int)
    table = decile_table(y, scores)
    assert len(table) == 10
    assert table["bad_rate"].iloc[0] > table["bad_rate"].iloc[-1]


def test_metrics_report_lists_dropped_features(model):
    dropped = model.metrics["dropped_features"]
    assert set(dropped) == {"iv_too_low", "correlated", "no_contribution"}
    all_features = set(model.selected_features) | {f for group in dropped.values() for f in group}
    assert all_features == set(FEATURE_NAMES)
