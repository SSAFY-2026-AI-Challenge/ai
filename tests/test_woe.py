"""WoE 구간화와 부호 제약 회귀 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fin_edu_credit.linear import NonPositiveLogisticRegression
from fin_edu_credit.woe import WoEBinner, iv_strength


@pytest.fixture
def toy_data() -> tuple[pd.DataFrame, np.ndarray]:
    """값이 클수록 안전한 피처와, 아무 정보 없는 피처를 섞은 데이터."""
    rng = np.random.default_rng(7)
    n = 2000
    good_signal = rng.uniform(0, 1, n)
    noise = rng.uniform(0, 1, n)
    probability = 1 / (1 + np.exp(-(2.5 - 5 * good_signal)))
    y = (rng.uniform(0, 1, n) < probability).astype(int)
    X = pd.DataFrame({"safe_when_high": good_signal, "pure_noise": noise})
    return X, y


def test_woe_is_monotone_in_declared_direction(toy_data):
    X, y = toy_data
    binner = WoEBinner(directions={"safe_when_high": True, "pure_noise": True}).fit(X, y)
    woe = binner.binnings_["safe_when_high"].woe_values()
    assert len(woe) >= 3
    assert np.all(np.diff(woe) >= 0)


def test_reversed_direction_collapses_to_no_information(toy_data):
    """방향을 반대로 선언하면 구간이 합쳐져 정보값이 사라져야 한다."""
    X, y = toy_data
    binner = WoEBinner(directions={"safe_when_high": False, "pure_noise": True}).fit(X, y)
    assert binner.iv_["safe_when_high"] < 0.02
    assert len(binner.binnings_["safe_when_high"].bins) == 1


def test_informative_feature_has_higher_iv(toy_data):
    X, y = toy_data
    binner = WoEBinner(directions={"safe_when_high": True, "pure_noise": True}).fit(X, y)
    assert binner.iv_["safe_when_high"] > binner.iv_["pure_noise"]
    assert binner.iv_["pure_noise"] < 0.1


def test_transform_maps_values_to_bin_woe(toy_data):
    X, y = toy_data
    binner = WoEBinner(directions={"safe_when_high": True, "pure_noise": True}).fit(X, y)
    transformed = binner.transform(X)
    assert transformed.shape == X.shape
    assert set(np.round(transformed["safe_when_high"].unique(), 6)) <= set(
        np.round(binner.binnings_["safe_when_high"].woe_values(), 6)
    )


def test_bins_respect_minimum_size(toy_data):
    X, y = toy_data
    binner = WoEBinner(directions={"safe_when_high": True, "pure_noise": True}, min_bin_frac=0.1).fit(X, y)
    counts = [b.count for b in binner.binnings_["safe_when_high"].bins]
    assert min(counts) >= 0.1 * len(y) * 0.5


def test_constant_feature_yields_single_bin():
    X = pd.DataFrame({"flat": np.ones(300)})
    y = np.random.default_rng(0).integers(0, 2, 300)
    binner = WoEBinner(directions={"flat": True}).fit(X, y)
    assert len(binner.binnings_["flat"].bins) == 1
    assert binner.iv_["flat"] == pytest.approx(0.0, abs=1e-9)


def test_iv_strength_labels():
    assert iv_strength(0.01) == "예측력 없음"
    assert iv_strength(0.2) == "보통"
    assert iv_strength(0.9) == "매우 강함"


def test_non_positive_logistic_never_returns_positive_coefficients():
    rng = np.random.default_rng(3)
    n = 800
    # 서로 강하게 상관된 두 피처를 넣어 일반 회귀라면 부호가 뒤집히는 상황을 만든다.
    a = rng.normal(size=n)
    b = a + rng.normal(scale=0.05, size=n)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.5 - 1.5 * a)))).astype(int)
    X = np.column_stack([a, b])
    model = NonPositiveLogisticRegression(C=1.0).fit(X, y)
    assert np.all(model.coef_[0] <= 0)


def test_non_positive_logistic_learns_direction():
    rng = np.random.default_rng(11)
    n = 1500
    x = rng.normal(size=n)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.2 - 2.0 * x)))).astype(int)
    model = NonPositiveLogisticRegression(C=1.0).fit(x.reshape(-1, 1), y)
    assert model.coef_[0][0] < -0.5
    probabilities = model.predict_proba(np.array([[-2.0], [2.0]]))[:, 1]
    assert probabilities[0] > probabilities[1]
