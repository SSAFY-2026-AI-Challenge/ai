from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

FEATURE_ORDER = (
    "spending_trend",
    "homework_rate",
    "attendance_rate",
)

FEATURE_LABELS_KO = {
    "spending_trend": "지출 추이",
    "homework_rate": "숙제 완료율",
    "attendance_rate": "출결률",
}

FEATURE_DIM = len(FEATURE_ORDER)


@dataclass(slots=True)
class CreditFeatures:
    spending_trend: float
    homework_rate: float
    attendance_rate: float

    def clamped(self) -> CreditFeatures:
        return replace(
            self,
            spending_trend=float(np.clip(self.spending_trend, 0.0, 1.0)),
            homework_rate=float(np.clip(self.homework_rate, 0.0, 1.0)),
            attendance_rate=float(np.clip(self.attendance_rate, 0.0, 1.0)),
        )

    def as_dict(self) -> dict[str, float | int]:
        value = self.clamped()
        return {
            "spending_trend": round(value.spending_trend, 4),
            "homework_rate": round(value.homework_rate, 4),
            "attendance_rate": round(value.attendance_rate, 4),
        }

    def to_raw_vector(self) -> np.ndarray:
        value = self.clamped()
        return np.array(
            [
                value.spending_trend,
                value.homework_rate,
                value.attendance_rate,
            ],
            dtype=np.float32,
        )

    def to_model_vector(self) -> np.ndarray:
        value = self.clamped()
        return np.array(
            [
                value.spending_trend,
                value.homework_rate,
                value.attendance_rate,
            ],
            dtype=np.float32,
        )


def features_from_mapping(payload: dict[str, float | int]) -> CreditFeatures:
    return CreditFeatures(
        spending_trend=float(payload["spending_trend"]),
        homework_rate=float(payload["homework_rate"]),
        attendance_rate=float(payload["attendance_rate"]),
    ).clamped()


def asdict_features(features: CreditFeatures) -> dict[str, float | int]:
    return asdict(features.clamped())
