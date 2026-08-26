from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class FeaturePayload(BaseModel):
    spending_trend: float = Field(ge=0, le=1, description="지출 추이. 0=검소, 1=과소비")
    homework_rate: float = Field(ge=0, le=1, description="숙제 완료율")
    attendance_rate: float = Field(ge=0, le=1, description="출결률")


class RecommendationOut(BaseModel):
    recommendation_type: str
    content: str
    expected_effect: str


class ContributionOut(BaseModel):
    feature: str
    label: str
    current_value: float | int
    ideal_value: float | int
    score_lift: float
    direction: str


class ScoreResponse(BaseModel):
    member_id: int | None = None
    name: str
    class_name: str | None = None
    job: str | None = None
    credit_score: int
    grade: str
    grade_code: str
    features: dict[str, Any]
    contributions: list[ContributionOut]
    summary: str
    recommendations: list[RecommendationOut]
    report_id: int | None = None


class ClassRow(BaseModel):
    member_id: int
    name: str
    class_name: str | None
    job: str | None
    credit_score: int
    grade: str
    grade_code: str
    features: dict[str, Any]
    summary: str


class StatusResponse(BaseModel):
    ok: bool
    mode: str
    db_configured: bool
    using_database: bool
    model_loaded: bool
