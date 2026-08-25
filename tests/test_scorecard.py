"""점수 변환과 규칙기반 점수 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fin_edu_credit.config import GRADE_TABLE, ModelConfig
from fin_edu_credit.scorecard import (
    grade_label,
    pd_to_score,
    rule_based_score,
    rule_components,
    scaling_params,
    score_to_grade,
)


def test_pd_to_score_is_monotone_decreasing():
    probabilities = np.linspace(0.01, 0.95, 40)
    scores = pd_to_score(probabilities)
    assert np.all(np.diff(scores) <= 0), "연체 확률이 높아지면 점수는 내려가야 한다"


def test_base_score_matches_base_odds():
    cfg = ModelConfig()
    base_pd = 1 / (1 + cfg.base_odds)
    assert pd_to_score([base_pd], cfg)[0] == pytest.approx(cfg.base_score, abs=1)


def test_pdo_doubles_odds():
    """odds 가 2배가 되면 점수가 정확히 PDO 만큼 오른다."""
    cfg = ModelConfig()
    odds = 4.0
    score_low = pd_to_score([1 / (1 + odds)], cfg)[0]
    score_high = pd_to_score([1 / (1 + odds * 2)], cfg)[0]
    assert score_high - score_low == pytest.approx(cfg.pdo, abs=1)


def test_score_bounds():
    scores = pd_to_score([1e-9, 1 - 1e-9])
    cfg = ModelConfig()
    assert scores.max() <= cfg.score_cap
    assert scores.min() >= cfg.score_floor


def test_scaling_params_consistent():
    cfg = ModelConfig()
    factor, offset = scaling_params(cfg)
    assert factor == pytest.approx(cfg.pdo / np.log(2))
    assert offset + factor * np.log(cfg.base_odds) == pytest.approx(cfg.base_score)


def test_grade_cutoffs_are_descending():
    cutoffs = [cutoff for _, cutoff, _ in GRADE_TABLE]
    assert cutoffs == sorted(cutoffs, reverse=True)


@pytest.mark.parametrize(
    "score, expected",
    [(1000, 1), (GRADE_TABLE[0][1], 1), (GRADE_TABLE[0][1] - 1, 2), (0, 10)],
)
def test_score_to_grade(score, expected):
    assert int(score_to_grade([score])[0]) == expected


def test_grade_label_known():
    assert grade_label(1) == "최우수"
    assert grade_label(10) == "위험"


def _feature_row(**overrides) -> pd.DataFrame:
    base = {
        "cs_attendance": 0.9,
        "cs_homework": 0.85,
        "cs_overdue": 2,
        "late_events_8w": 1,
        "repay_ontime_ratio": 0.9,
        "tax_paid_ratio": 1.0,
        "savings_rate": 0.1,
        "has_savings": 1.0,
        "consumption_income_ratio": 0.9,
        "discretionary_ratio": 0.4,
        "liquidity_weeks": 1.5,
        "debt_to_income": 0.5,
    }
    base.update(overrides)
    return pd.DataFrame([base], index=[1])


def test_rule_components_within_unit_range():
    comp = rule_components(_feature_row())
    assert comp.to_numpy().min() >= 0
    assert comp.to_numpy().max() <= 1


def test_rule_score_rewards_good_habits():
    weak = rule_based_score(
        _feature_row(cs_attendance=0.75, cs_homework=0.6, cs_overdue=6, late_events_8w=4,
                     repay_ontime_ratio=0.2, savings_rate=0.0, has_savings=0.0,
                     debt_to_income=2.0, liquidity_weeks=0.0)
    ).iloc[0]
    strong = rule_based_score(
        _feature_row(cs_attendance=1.0, cs_homework=1.0, cs_overdue=0, late_events_8w=0,
                     repay_ontime_ratio=1.0, savings_rate=0.25, has_savings=1.0,
                     debt_to_income=0.0, liquidity_weeks=3.0)
    ).iloc[0]
    assert strong > weak
    assert 350 <= weak <= 1000 and 350 <= strong <= 1000


def test_rule_score_reacts_to_single_habit():
    before = rule_based_score(_feature_row(cs_attendance=0.80)).iloc[0]
    after = rule_based_score(_feature_row(cs_attendance=1.00)).iloc[0]
    assert after > before
