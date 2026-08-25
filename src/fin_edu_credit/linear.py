"""부호를 제약한 로지스틱 회귀.

WoE 는 '값이 클수록 안전' 하도록 만들었으므로, 연체 확률을 예측하는 모델에서
모든 계수는 0 이하여야 한다. 그런데 서로 상관된 피처를 함께 넣으면 일반적인
로지스틱 회귀는 한쪽 계수를 양수로 뒤집어 버린다. 그러면
"적금을 많이 해서 점수가 깎였다" 같은 설명이 나온다.

흔한 대처는 부호가 틀린 피처를 하나씩 빼면서 다시 적합하는 것인데,
그러면 출석률이나 저축처럼 교육적으로 꼭 보여 주고 싶은 항목이
연쇄적으로 탈락한다.

그래서 아예 계수 상한을 0 으로 두고 최적화한다. 기여할 것이 없는 피처는
탈락하는 대신 계수가 0 으로 수렴한다(= 배점 0점).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


class NonPositiveLogisticRegression:
    """계수를 0 이하로 제약한 L2 정규화 로지스틱 회귀.

    목적함수는 scikit-learn 과 같은 형태를 쓴다.

        0.5 * ||w||^2 + C * Σ [ log(1 + exp(z)) - y * z ],   z = Xw + b
    """

    def __init__(self, C: float = 1.0, max_iter: int = 500) -> None:
        self.C = C
        self.max_iter = max_iter

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NonPositiveLogisticRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_features = X.shape[1]

        def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
            w, b = theta[:n_features], theta[n_features]
            z = X @ w + b
            loss = float(np.sum(np.logaddexp(0.0, z) - y * z))
            value = 0.5 * float(w @ w) + self.C * loss
            residual = 1.0 / (1.0 + np.exp(-z)) - y
            grad_w = w + self.C * (X.T @ residual)
            grad_b = self.C * float(residual.sum())
            return value, np.concatenate([grad_w, [grad_b]])

        bounds = [(None, 0.0)] * n_features + [(None, None)]
        start = np.zeros(n_features + 1)
        result = minimize(
            objective,
            start,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.max_iter},
        )
        self.coef_ = result.x[:n_features].reshape(1, -1)
        self.intercept_ = np.array([result.x[n_features]])
        self.n_iter_ = int(result.nit)
        self.converged_ = bool(result.success)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_[0] + self.intercept_[0]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = self.decision_function(X)
        p = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
