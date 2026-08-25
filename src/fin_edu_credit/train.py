"""신용평가 스코어카드 학습.

전체 흐름
---------
1. DB 에서 member / account / transaction 을 읽는다.
2. 2주 간격 기준일마다 (관측 8주 피처, 성과 4주 연체 여부) 쌍을 만든다.
3. **학생 단위**로 학습/검증을 나눈다. 같은 학생이 여러 기준일에 등장하므로
   행 단위로 자르면 검증 성능이 부풀려진다.
4. 피처를 도메인 방향에 맞춰 단조 WoE 로 구간화하고, IV 로 걸러난 뒤,
   상관이 높은 쌍을 정리하고, 계수 부호를 0 이하로 제약해 적합한다.
5. 로지스틱 회귀 계수를 '구간별 점수표'로 환산해 저장한다.
   그래디언트 부스팅은 채택하지 않고, 스코어카드가 얼마나 손해인지 보기 위한
   비교 기준으로만 학습한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_val_score

from .config import ARTIFACT_DIR, REPORT_DIR, ModelConfig
from .features import FEATURE_BY_NAME, FEATURE_LABELS, FEATURE_NAMES, SourceData, build_panel
from .linear import NonPositiveLogisticRegression
from .scorecard import pd_to_score, score_to_grade, scaling_params
from .woe import WoEBinner, iv_strength

MIN_IV = 0.02               # 이보다 정보값이 낮으면 예측력이 없다고 보고 제외
MAX_WOE_CORRELATION = 0.80  # WoE 기준 상관이 이보다 높으면 IV 가 낮은 쪽을 제거
MIN_COEFFICIENT = 1e-4      # 이보다 작은 계수는 기여가 없다고 보고 제외


@dataclass
class TrainedModel:
    binner: WoEBinner
    classifier: NonPositiveLogisticRegression
    selected_features: tuple[str, ...]
    model_version: str
    base_points: float
    reference_points: dict[str, float]
    algorithm: str = "woe_logistic_scorecard"
    metrics: dict[str, Any] = field(default_factory=dict)

    # --- 예측 ---------------------------------------------------------------
    def _woe(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.binner.transform(X[list(self.binner.feature_names_in_)])[
            list(self.selected_features)
        ]

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        return self.classifier.predict_proba(self._woe(X).to_numpy())[:, 1]

    def points(self, X: pd.DataFrame, cfg: ModelConfig | None = None) -> pd.DataFrame:
        """피처별 배점. 모두 더하면 신용점수가 된다."""
        cfg = cfg or ModelConfig()
        factor, _ = scaling_params(cfg)
        woe = self._woe(X)
        coefs = self.classifier.coef_[0]
        data = {
            name: self.base_points - factor * coefs[i] * woe[name].to_numpy()
            for i, name in enumerate(self.selected_features)
        }
        return pd.DataFrame(data, index=X.index)

    def predict_score(self, X: pd.DataFrame, cfg: ModelConfig | None = None) -> np.ndarray:
        cfg = cfg or ModelConfig()
        return pd_to_score(self.predict_pd(X), cfg)

    def scorecard_table(self, cfg: ModelConfig | None = None) -> pd.DataFrame:
        """교실 게시용 점수표: 피처 x 구간 -> 배점."""
        cfg = cfg or ModelConfig()
        factor, _ = scaling_params(cfg)
        coefs = self.classifier.coef_[0]
        rows = []
        for i, name in enumerate(self.selected_features):
            binning = self.binner.binnings_[name]
            for b_idx, b in enumerate(binning.bins):
                rows.append(
                    {
                        "feature": name,
                        "label": FEATURE_LABELS[name],
                        "bin": b_idx,
                        "range": b.label(),
                        "count": b.count,
                        "bad_rate": round(b.bad_rate, 4),
                        "woe": round(b.woe, 4),
                        "points": round(self.base_points - factor * coefs[i] * b.woe, 1),
                    }
                )
        return pd.DataFrame(rows)


# --- 지표 --------------------------------------------------------------------

def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """정상/부실 두 누적분포의 최대 격차(KS)."""
    order = np.argsort(y_score)
    y = np.asarray(y_true)[order]
    n_bad = max(1, int(y.sum()))
    n_good = max(1, int((1 - y).sum()))
    cum_bad = np.cumsum(y) / n_bad
    cum_good = np.cumsum(1 - y) / n_good
    return float(np.max(np.abs(cum_bad - cum_good)))


def decile_table(y_true: np.ndarray, scores: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """점수 구간별 실제 연체율. 점수가 낮을수록 연체율이 높아야 한다."""
    df = pd.DataFrame({"y": y_true, "score": scores})
    df["band"] = pd.qcut(df["score"], q=bins, duplicates="drop")
    table = (
        df.groupby("band", observed=True)
        .agg(n=("y", "size"), bad=("y", "sum"))
        .reset_index()
    )
    table["bad_rate"] = (table["bad"] / table["n"]).round(4)
    table["band"] = table["band"].astype(str)
    return table


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, cfg: ModelConfig) -> dict[str, Any]:
    scores = pd_to_score(y_prob, cfg)
    auc = float(roc_auc_score(y_true, y_prob))
    return {
        "auc": round(auc, 5),
        "gini": round(2 * auc - 1, 5),
        "ks": round(ks_statistic(y_true, y_prob), 5),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 5),
        "brier": round(float(brier_score_loss(y_true, y_prob)), 6),
        "bad_rate": round(float(np.mean(y_true)), 5),
        "score_mean": int(np.mean(scores)),
        "score_p10": int(np.percentile(scores, 10)),
        "score_p50": int(np.percentile(scores, 50)),
        "score_p90": int(np.percentile(scores, 90)),
        "decile": decile_table(y_true, scores).to_dict(orient="records"),
        "grade_distribution": {
            int(k): int(v)
            for k, v in pd.Series(score_to_grade(scores)).value_counts().sort_index().items()
        },
    }


# --- 피처 선택 ----------------------------------------------------------------

def _drop_correlated(woe_df: pd.DataFrame, iv: dict[str, float], threshold: float) -> list[str]:
    """WoE 상관이 높은 쌍에서 정보값이 낮은 쪽을 버린다."""
    kept = sorted(woe_df.columns, key=lambda c: iv[c], reverse=True)
    corr = woe_df.corr().abs()
    selected: list[str] = []
    for name in kept:
        if all(corr.loc[name, other] < threshold for other in selected):
            selected.append(name)
    return selected


def _fit_scorecard(
    woe_train: pd.DataFrame, y_train: np.ndarray, cfg: ModelConfig
) -> tuple[NonPositiveLogisticRegression, list[str], list[str]]:
    """부호를 제약해 적합하고, 기여가 0 인 피처를 정리한 뒤 다시 적합한다."""
    features = list(woe_train.columns)
    clf = NonPositiveLogisticRegression(C=cfg.regularization_c).fit(
        woe_train[features].to_numpy(), y_train
    )
    contributing = [
        name for name, coef in zip(features, clf.coef_[0]) if abs(coef) >= MIN_COEFFICIENT
    ]
    no_contribution = [name for name in features if name not in contributing]
    if no_contribution and len(contributing) >= 2:
        clf = NonPositiveLogisticRegression(C=cfg.regularization_c).fit(
            woe_train[contributing].to_numpy(), y_train
        )
        features = contributing
    return clf, features, no_contribution


# --- 학습 --------------------------------------------------------------------

def train_model(
    src: SourceData | None = None,
    cfg: ModelConfig | None = None,
    panel: pd.DataFrame | None = None,
) -> TrainedModel:
    cfg = cfg or ModelConfig()
    if panel is None:
        if src is None:
            raise ValueError("src 또는 panel 중 하나는 필요합니다.")
        panel = build_panel(src, cfg)
    if panel.empty:
        raise RuntimeError("학습할 데이터가 없습니다. 먼저 더미 데이터를 생성하세요(seed).")

    X = panel[list(FEATURE_NAMES)]
    y = panel["y"].to_numpy()
    groups = panel["member_id"].to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    g_train = groups[train_idx]

    directions = {name: FEATURE_BY_NAME[name].higher_is_better for name in FEATURE_NAMES}
    binner = WoEBinner(directions=directions, max_bins=6, min_bin_frac=0.05)
    binner.fit(X_train, y_train)

    iv = binner.iv_
    by_iv = [name for name in FEATURE_NAMES if iv[name] >= MIN_IV]
    dropped_by_iv = [name for name in FEATURE_NAMES if name not in by_iv]

    woe_train_all = binner.transform(X_train)
    by_corr = _drop_correlated(woe_train_all[by_iv], iv, MAX_WOE_CORRELATION)
    dropped_by_corr = [name for name in by_iv if name not in by_corr]

    clf, selected, dropped_no_contribution = _fit_scorecard(woe_train_all[by_corr], y_train, cfg)

    factor, offset = scaling_params(cfg)
    base_points = (offset - factor * float(clf.intercept_[0])) / len(selected)

    model = TrainedModel(
        binner=binner,
        classifier=clf,
        selected_features=tuple(selected),
        model_version=f"scorecard-{datetime.now():%Y%m%d-%H%M%S}",
        base_points=base_points,
        reference_points={},
    )

    train_points = model.points(X_train, cfg)
    model.reference_points = {name: float(train_points[name].median()) for name in selected}

    prob_train = model.predict_pd(X_train)
    prob_test = model.predict_pd(X_test)

    # 비교 기준: 원시 피처를 그대로 쓰는 부스팅
    benchmark = HistGradientBoostingClassifier(
        max_depth=3, max_iter=250, learning_rate=0.06,
        l2_regularization=1.0, min_samples_leaf=40, random_state=cfg.random_state,
    )
    cv = StratifiedGroupKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)
    bench_cv = float(
        np.mean(cross_val_score(benchmark, X_train, y_train, groups=g_train, cv=cv, scoring="roc_auc"))
    )
    benchmark.fit(X_train, y_train)
    bench_test_auc = float(roc_auc_score(y_test, benchmark.predict_proba(X_test)[:, 1]))

    # 스코어카드 자신의 교차검증 점수(구간화는 폴드마다 다시 적합해야 정직하다)
    scorecard_cv = _cross_val_auc(X_train, y_train, g_train, cfg, directions, cv)

    model.metrics = {
        "algorithm": model.algorithm,
        "model_version": model.model_version,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_members_train": int(pd.unique(g_train).size),
        "n_members_test": int(pd.unique(groups[test_idx]).size),
        "n_snapshots": int(panel["as_of"].nunique()),
        "observation_weeks": cfg.observation_weeks,
        "performance_weeks": cfg.performance_weeks,
        "cv_auc": {
            "woe_scorecard": round(scorecard_cv, 5),
            "hist_gradient_boosting_benchmark": round(bench_cv, 5),
        },
        "benchmark_test_auc": round(bench_test_auc, 5),
        "selected_features": list(selected),
        "dropped_features": {
            "iv_too_low": dropped_by_iv,
            "correlated": dropped_by_corr,
            "no_contribution": dropped_no_contribution,
        },
        "information_value": [
            {
                "feature": name,
                "label": FEATURE_LABELS[name],
                "iv": round(iv[name], 5),
                "strength": iv_strength(iv[name]),
                "selected": name in selected,
            }
            for name in sorted(FEATURE_NAMES, key=lambda n: iv[n], reverse=True)
        ],
        "coefficients": [
            {
                "feature": name,
                "label": FEATURE_LABELS[name],
                "coefficient": round(float(clf.coef_[0][i]), 5),
                "point_range": round(
                    float(
                        factor
                        * abs(clf.coef_[0][i])
                        * (
                            binner.binnings_[name].woe_values().max()
                            - binner.binnings_[name].woe_values().min()
                        )
                    ),
                    1,
                ),
            }
            for i, name in enumerate(selected)
        ],
        "train": evaluate(y_train, prob_train, cfg),
        "test": evaluate(y_test, prob_test, cfg),
    }
    model.metrics["coefficients"].sort(key=lambda r: r["point_range"], reverse=True)
    return model


def _cross_val_auc(X, y, groups, cfg: ModelConfig, directions, cv) -> float:
    """구간화까지 폴드 안에서 다시 적합하는 정직한 교차검증."""
    aucs = []
    for tr, va in cv.split(X, y, groups):
        binner = WoEBinner(directions=directions, max_bins=6, min_bin_frac=0.05)
        binner.fit(X.iloc[tr], y[tr])
        keep = [n for n in X.columns if binner.iv_[n] >= MIN_IV]
        woe_tr = binner.transform(X.iloc[tr])[keep]
        keep = _drop_correlated(woe_tr, binner.iv_, MAX_WOE_CORRELATION)
        clf, feats, _ = _fit_scorecard(woe_tr[keep], y[tr], cfg)
        woe_va = binner.transform(X.iloc[va])[feats]
        aucs.append(roc_auc_score(y[va], clf.predict_proba(woe_va.to_numpy())[:, 1]))
    return float(np.mean(aucs))


# --- 저장/적재 ----------------------------------------------------------------

def save_model(model: TrainedModel, artifact_dir: Path | None = None) -> Path:
    artifact_dir = artifact_dir or ARTIFACT_DIR
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / "credit_model.joblib"
    joblib.dump(model, path)
    (artifact_dir / "model_version.txt").write_text(model.model_version, encoding="utf-8")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "training_metrics.json").write_text(
        json.dumps(model.metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    model.scorecard_table().to_csv(REPORT_DIR / "scorecard.csv", index=False, encoding="utf-8-sig")
    return path


def load_model(artifact_dir: Path | None = None) -> TrainedModel:
    artifact_dir = artifact_dir or ARTIFACT_DIR
    path = artifact_dir / "credit_model.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"학습된 모델이 없습니다: {path}\n먼저 `python -m fin_edu_credit.cli train` 을 실행하세요."
        )
    return joblib.load(path)


def record_run(conn, model: TrainedModel) -> None:
    """학습 이력을 credit_model_run 테이블에 남긴다."""
    from .db import execute

    test = model.metrics["test"]
    execute(
        conn,
        """
        INSERT INTO credit_model_run
            (model_version, algorithm, trained_at, n_train, n_test, bad_rate, auc, ks, gini, brier, metrics_json)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            algorithm = VALUES(algorithm), trained_at = VALUES(trained_at),
            auc = VALUES(auc), ks = VALUES(ks), gini = VALUES(gini),
            brier = VALUES(brier), metrics_json = VALUES(metrics_json)
        """,
        (
            model.model_version,
            model.algorithm,
            datetime.now(),
            model.metrics["n_train"],
            model.metrics["n_test"],
            test["bad_rate"],
            test["auc"],
            test["ks"],
            test["gini"],
            test["brier"],
            json.dumps(model.metrics, ensure_ascii=False, default=str),
        ),
    )
    conn.commit()
