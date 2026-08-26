from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from credit_ai.features.schema import CreditFeatures

DEMO_PROFILES: list[dict[str, Any]] = [
    {"name": "민준", "job": "은행원", "salary": 25000, "features": (0.22, 0.98, 1.00)},
    {"name": "서연", "job": "도서관 사서", "salary": 22000, "features": (0.18, 1.00, 0.99)},
    {"name": "하준", "job": "환경미화원", "salary": 20000, "features": (0.35, 0.88, 0.96)},
    {"name": "지우", "job": "우체부", "salary": 21000, "features": (0.41, 0.91, 0.94)},
    {"name": "서준", "job": "매점 직원", "salary": 18000, "features": (0.78, 0.72, 0.90)},
    {"name": "하윤", "job": "요리사", "salary": 23000, "features": (0.48, 0.84, 0.97)},
    {"name": "도윤", "job": "소방관", "salary": 24000, "features": (0.30, 0.95, 0.88)},
    {"name": "수아", "job": "기자", "salary": 22000, "features": (0.55, 0.93, 0.92)},
    {"name": "은우", "job": "농부", "salary": 19000, "features": (0.28, 0.80, 0.85)},
    {"name": "지아", "job": "화가", "salary": 17000, "features": (0.62, 0.70, 0.83)},
    {"name": "시우", "job": "운동선수", "salary": 21000, "features": (0.44, 0.77, 0.99)},
    {"name": "예준", "job": "선생님 도우미", "salary": 20000, "features": (0.25, 0.99, 1.00)},
    {"name": "주원", "job": "우체부", "salary": 20000, "features": (0.81, 0.64, 0.78)},
    {"name": "하은", "job": "사서", "salary": 22000, "features": (0.33, 0.90, 0.95)},
    {"name": "유준", "job": "은행원", "salary": 25000, "features": (0.19, 0.86, 0.91)},
    {"name": "소율", "job": "요리사", "salary": 23000, "features": (0.52, 0.58, 0.89)},
    {"name": "지호", "job": "소방관", "salary": 24000, "features": (0.37, 0.82, 0.67)},
    {"name": "채원", "job": "기자", "salary": 22000, "features": (0.46, 0.96, 0.98)},
    {"name": "건우", "job": "환경미화원", "salary": 20000, "features": (0.29, 0.74, 0.93)},
    {"name": "다은", "job": "화가", "salary": 18000, "features": (0.70, 0.61, 0.80)},
]


def _to_features(raw: tuple[float, float, float]) -> CreditFeatures:
    spending, homework, attendance = raw
    return CreditFeatures(
        spending_trend=spending,
        homework_rate=homework,
        attendance_rate=attendance,
    ).clamped()


@dataclass
class DemoStudent:
    member_id: int
    name: str
    class_name: str
    job: str
    salary: int
    features: CreditFeatures
    report: dict[str, Any] | None = None


@dataclass
class DemoStore:
    students: dict[int, DemoStudent] = field(default_factory=dict)

    def seed(self) -> None:
        self.students.clear()
        for index, profile in enumerate(DEMO_PROFILES, start=1):
            self.students[index] = DemoStudent(
                member_id=index,
                name=profile["name"],
                class_name="3-2",
                job=profile["job"],
                salary=profile["salary"],
                features=_to_features(profile["features"]),
            )

    def get(self, member_id: int) -> DemoStudent | None:
        return self.students.get(member_id)

    def all_students(self) -> list[DemoStudent]:
        return [self.students[key] for key in sorted(self.students)]
