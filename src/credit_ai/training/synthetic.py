from __future__ import annotations

import numpy as np

from credit_ai.features.schema import CreditFeatures

RNG_SEED = 42


def teacher_score(features: CreditFeatures, rng: np.random.Generator | None = None) -> float:
    """비선형 교사 함수. 출결·숙제·소비 상호작용을 넣습니다."""
    item = features.clamped()
    spending = item.spending_trend
    homework = item.homework_rate
    attendance = item.attendance_rate

    score = 470.0
    score += attendance * 180.0
    score += homework * 150.0
    score -= spending * 130.0

    score += 60.0 * attendance * homework
    score -= 35.0 * spending * (1.0 - homework)

    if attendance < 0.72:
        score -= 55.0
    if homework >= 0.95 and attendance >= 0.95:
        score += 32.0
    if spending > 0.75:
        score -= 40.0

    noise = 0.0
    if rng is not None:
        noise = float(rng.normal(0.0, 16.0))
    return float(np.clip(score + noise, 300.0, 1000.0))


def sample_features(rng: np.random.Generator) -> CreditFeatures:
    attendance = float(np.clip(rng.beta(8.0, 1.35), 0.35, 1.0))
    homework = float(np.clip(rng.beta(5.2, 1.7), 0.2, 1.0))
    spending = float(np.clip(rng.beta(2.1, 3.4), 0.0, 1.0))
    if rng.random() < 0.08:
        spending = float(np.clip(spending + 0.35, 0.0, 1.0))
    return CreditFeatures(
        spending_trend=spending,
        homework_rate=homework,
        attendance_rate=attendance,
    ).clamped()


def generate_dataset(
    n_samples: int,
    seed: int = RNG_SEED,
    with_noise: bool = True,
) -> tuple[list[CreditFeatures], np.ndarray]:
    rng = np.random.default_rng(seed)
    features = [sample_features(rng) for _ in range(n_samples)]
    labels = np.array(
        [teacher_score(item, rng if with_noise else None) for item in features],
        dtype=np.float32,
    )
    return features, labels
