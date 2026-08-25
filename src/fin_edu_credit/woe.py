"""WoE(Weight of Evidence) 구간화.

신용평가에서 표준으로 쓰는 전처리다. 연속형 피처를 몇 개의 구간으로 나누고,
구간마다 '정상 비율 대비 연체 비율'을 로그로 바꿔 하나의 숫자(WoE)로 만든다.

    WoE_b = ln( (구간 b 의 정상 비중) / (구간 b 의 연체 비중) )

WoE 가 클수록 안전한 구간이다. 이렇게 바꾸면

* 비선형 관계를 구간으로 자연스럽게 흡수하고,
* 이상치의 영향이 사라지며,
* 무엇보다 **구간을 도메인 방향(출석률이 높을수록 안전)에 맞춰 단조롭게** 만들 수 있다.

원시 피처를 그대로 로지스틱 회귀에 넣으면 서로 상관된 피처끼리 계수 부호가
뒤집혀서 "적금을 많이 해서 점수가 깎였다" 같은 설명이 나온다. 교실에서 쓸
모델에서는 그런 설명이 나오면 안 되므로, 방향을 강제한 WoE 를 사용한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SMOOTHING = 0.5  # 빈 구간에서 로그가 발산하지 않도록 하는 보정값


@dataclass
class Bin:
    left: float
    right: float
    count: int
    bad: int
    woe: float

    @property
    def good(self) -> int:
        return self.count - self.bad

    @property
    def bad_rate(self) -> float:
        return self.bad / self.count if self.count else 0.0

    def label(self) -> str:
        if np.isinf(self.left) and np.isinf(self.right):
            return "전체"
        if np.isinf(self.left):
            return f"~ {self.right:,.4g}"
        if np.isinf(self.right):
            return f"{self.left:,.4g} 초과"
        return f"{self.left:,.4g} 초과 ~ {self.right:,.4g}"


@dataclass
class FeatureBinning:
    name: str
    edges: np.ndarray               # 길이 n_bins+1, 양끝은 ±inf
    bins: list[Bin] = field(default_factory=list)
    iv: float = 0.0

    def woe_values(self) -> np.ndarray:
        return np.array([b.woe for b in self.bins], dtype=float)

    def assign(self, x: np.ndarray) -> np.ndarray:
        inner = self.edges[1:-1]
        if inner.size == 0:
            return np.zeros(len(x), dtype=int)
        return np.digitize(x, inner, right=True)


def _woe_and_iv(counts_good: np.ndarray, counts_bad: np.ndarray) -> tuple[np.ndarray, float]:
    good = counts_good + SMOOTHING
    bad = counts_bad + SMOOTHING
    p_good = good / good.sum()
    p_bad = bad / bad.sum()
    woe = np.log(p_good / p_bad)
    iv = float(np.sum((p_good - p_bad) * woe))
    return woe, iv


class WoEBinner(BaseEstimator, TransformerMixin):
    """방향을 강제한 단조 WoE 구간화기.

    Parameters
    ----------
    directions:
        피처명 -> True(값이 클수록 안전) / False(값이 클수록 위험).
    max_bins:
        초기 분위수 구간 수. 단조성/최소 표본 조건에 따라 병합되며 줄어든다.
    min_bin_frac:
        한 구간이 가져야 하는 최소 표본 비율.
    """

    def __init__(
        self,
        directions: dict[str, bool],
        max_bins: int = 6,
        min_bin_frac: float = 0.05,
    ) -> None:
        self.directions = directions
        self.max_bins = max_bins
        self.min_bin_frac = min_bin_frac

    # --- fit -----------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: Sequence[int]) -> "WoEBinner":
        y = np.asarray(y).astype(int)
        self.feature_names_in_ = tuple(X.columns)
        self.binnings_: dict[str, FeatureBinning] = {}
        min_count = max(20, int(len(y) * self.min_bin_frac))
        for name in X.columns:
            increasing = bool(self.directions.get(name, True))
            self.binnings_[name] = self._fit_feature(
                name, X[name].to_numpy(dtype=float), y, increasing, min_count
            )
        self.iv_ = {name: b.iv for name, b in self.binnings_.items()}
        return self

    def _fit_feature(
        self,
        name: str,
        x: np.ndarray,
        y: np.ndarray,
        increasing: bool,
        min_count: int,
    ) -> FeatureBinning:
        edges = self._initial_edges(x)
        idx = np.digitize(x, edges[1:-1], right=True) if edges.size > 2 else np.zeros(len(x), int)

        n_bins = len(edges) - 1
        counts = np.array([(idx == i).sum() for i in range(n_bins)], dtype=float)
        bads = np.array([y[idx == i].sum() if (idx == i).any() else 0 for i in range(n_bins)], dtype=float)

        counts, bads, edges = self._merge_small(counts, bads, edges, min_count)
        counts, bads, edges = self._enforce_monotone(counts, bads, edges, increasing)

        woe, iv = _woe_and_iv(counts - bads, bads)
        bins = [
            Bin(left=float(edges[i]), right=float(edges[i + 1]), count=int(counts[i]),
                bad=int(bads[i]), woe=float(woe[i]))
            for i in range(len(counts))
        ]
        return FeatureBinning(name=name, edges=edges, bins=bins, iv=iv)

    def _initial_edges(self, x: np.ndarray) -> np.ndarray:
        finite = x[np.isfinite(x)]
        uniques = np.unique(finite)
        if uniques.size <= 1:
            return np.array([-np.inf, np.inf])
        if uniques.size <= self.max_bins:
            inner = uniques[:-1]
        else:
            qs = np.linspace(0, 1, self.max_bins + 1)[1:-1]
            inner = np.unique(np.quantile(finite, qs))
            inner = inner[inner < uniques[-1]]
        if inner.size == 0:
            return np.array([-np.inf, np.inf])
        return np.concatenate([[-np.inf], inner, [np.inf]])

    @staticmethod
    def _merge_at(
        counts: np.ndarray, bads: np.ndarray, edges: np.ndarray, i: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """i 번째와 i+1 번째 구간을 합친다."""
        counts = counts.copy()
        bads = bads.copy()
        counts[i] += counts[i + 1]
        bads[i] += bads[i + 1]
        counts = np.delete(counts, i + 1)
        bads = np.delete(bads, i + 1)
        edges = np.delete(edges, i + 1)
        return counts, bads, edges

    def _merge_small(self, counts, bads, edges, min_count):
        while len(counts) > 2:
            small = np.where(counts < min_count)[0]
            if small.size == 0:
                break
            i = int(small[0])
            merge_at = i - 1 if i == len(counts) - 1 else i
            counts, bads, edges = self._merge_at(counts, bads, edges, merge_at)
        return counts, bads, edges

    def _enforce_monotone(self, counts, bads, edges, increasing: bool):
        """WoE 가 도메인 방향대로 단조가 될 때까지 인접 구간을 병합한다.

        방향이 끝내 맞지 않으면 구간이 하나로 합쳐진다. 그러면 WoE 가 상수가 되어
        정보값(IV)이 0 이 되고, 이후 IV 필터에서 자동으로 탈락한다.
        '가정한 방향과 반대로 움직이는 피처는 아예 쓰지 않는다'는 뜻이다.
        """
        while len(counts) > 1:
            woe, _ = _woe_and_iv(counts - bads, bads)
            diffs = np.diff(woe)
            violations = diffs < 0 if increasing else diffs > 0
            if not violations.any():
                break
            candidates = np.where(violations)[0]
            i = int(candidates[np.argmin(np.abs(diffs[candidates]))])
            counts, bads, edges = self._merge_at(counts, bads, edges, i)
        return counts, bads, edges

    # --- transform -----------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for name in self.feature_names_in_:
            binning = self.binnings_[name]
            values = pd.to_numeric(X[name], errors="coerce").to_numpy(dtype=float)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            idx = binning.assign(values)
            out[name] = binning.woe_values()[idx]
        return pd.DataFrame(out, index=X.index)

    def bin_index(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for name in self.feature_names_in_:
            binning = self.binnings_[name]
            values = pd.to_numeric(X[name], errors="coerce").to_numpy(dtype=float)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            out[name] = binning.assign(values)
        return pd.DataFrame(out, index=X.index)

    # --- 리포트 --------------------------------------------------------------
    def iv_table(self) -> pd.DataFrame:
        rows = [{"feature": name, "iv": round(b.iv, 5), "n_bins": len(b.bins)}
                for name, b in self.binnings_.items()]
        return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)

    def bin_table(self, features: Sequence[str] | None = None) -> pd.DataFrame:
        names = features or self.feature_names_in_
        rows = []
        for name in names:
            for i, b in enumerate(self.binnings_[name].bins):
                rows.append(
                    {
                        "feature": name,
                        "bin": i,
                        "range": b.label(),
                        "count": b.count,
                        "bad": b.bad,
                        "bad_rate": round(b.bad_rate, 4),
                        "woe": round(b.woe, 4),
                    }
                )
        return pd.DataFrame(rows)


def iv_strength(iv: float) -> str:
    """정보값(IV) 관례적 해석."""
    if iv < 0.02:
        return "예측력 없음"
    if iv < 0.1:
        return "약함"
    if iv < 0.3:
        return "보통"
    if iv < 0.5:
        return "강함"
    return "매우 강함"
