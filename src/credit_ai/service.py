from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from credit_ai.config import Settings
from credit_ai.db.repositories import fetch_member_bundle, fetch_student_members
from credit_ai.db.session import session_scope
from credit_ai.demo import DemoStore, DemoStudent
from credit_ai.features.extract import extract_features
from credit_ai.features.schema import CreditFeatures
from credit_ai.model.explain import FeatureContribution, explain_score
from credit_ai.model.scorer import CreditScorer, ScoreOutput
from credit_ai.reports.copy import build_recommendations, build_summary
from credit_ai.reports.writer import persist_report


@dataclass(slots=True)
class ScoreBundle:
    member_id: int | None
    name: str
    class_name: str | None
    job: str | None
    result: ScoreOutput
    contributions: list[FeatureContribution]
    summary: str
    recommendations: list[dict[str, str]]
    report_id: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "member_id": self.member_id,
            "name": self.name,
            "class_name": self.class_name,
            "job": self.job,
            "credit_score": self.result.score,
            "grade": self.result.grade,
            "grade_code": self.result.grade_code,
            "features": self.result.features,
            "contributions": [asdict(row) for row in self.contributions],
            "summary": self.summary,
            "recommendations": self.recommendations,
            "report_id": self.report_id,
        }


class CreditService:
    def __init__(
        self,
        settings: Settings,
        scorer: CreditScorer | None,
        store: DemoStore,
    ) -> None:
        self.settings = settings
        self.scorer = scorer
        self.store = store

    def require_scorer(self) -> CreditScorer:
        if self.scorer is None:
            raise RuntimeError(
                "학습된 모델이 없습니다. 프로젝트 루트에서 `uv run credit-ai train` 을 실행하세요."
            )
        return self.scorer

    def evaluate(
        self,
        features: CreditFeatures,
        *,
        member_id: int | None = None,
        name: str = "학생",
        class_name: str | None = None,
        job: str | None = None,
        persist: bool = False,
    ) -> ScoreBundle:
        scorer = self.require_scorer()
        result = scorer.score(features)
        contributions = explain_score(scorer, features)
        summary = build_summary(name, result, contributions)
        recommendations = build_recommendations(features, contributions)
        report_id = None
        if persist and member_id is not None and self.settings.use_database:
            with session_scope(self.settings) as session:
                report = persist_report(
                    session,
                    member_id,
                    summary,
                    result.features,
                    [asdict(row) for row in contributions],
                    recommendations,
                    score=result.score,
                )
                report_id = report.id
        return ScoreBundle(
            member_id=member_id,
            name=name,
            class_name=class_name,
            job=job,
            result=result,
            contributions=contributions,
            summary=summary,
            recommendations=recommendations,
            report_id=report_id,
        )

    def score_features(self, features: CreditFeatures) -> ScoreBundle:
        return self.evaluate(features, persist=False)

    def score_member(self, member_id: int) -> ScoreBundle:
        if self.settings.use_database:
            with session_scope(self.settings) as session:
                bundle = fetch_member_bundle(session, member_id)
                if bundle is None:
                    raise KeyError(f"member_id={member_id} 를 찾을 수 없습니다.")
                member = bundle.member
                features = extract_features(bundle)
                name = member.name
                class_name = member.class_name
                job = member.job
            scored = self.evaluate(
                features,
                member_id=member_id,
                name=name,
                class_name=class_name,
                job=job,
                persist=True,
            )
            return scored

        student = self.store.get(member_id)
        if student is None:
            raise KeyError(f"member_id={member_id} 를 찾을 수 없습니다.")
        scored = self.evaluate(
            student.features,
            member_id=student.member_id,
            name=student.name,
            class_name=student.class_name,
            job=student.job,
            persist=False,
        )
        student.report = scored.as_dict()
        return scored

    def score_class(self) -> list[ScoreBundle]:
        if self.settings.use_database:
            with session_scope(self.settings) as session:
                members = fetch_student_members(session)
                ids = [member.id for member in members]
            return [self.score_member(member_id) for member_id in ids]
        return [self.score_member(student.member_id) for student in self.store.all_students()]

    def list_class(self) -> list[dict[str, Any]]:
        if not self.settings.use_database:
            rows: list[dict[str, Any]] = []
            for student in self.store.all_students():
                if student.report is None:
                    self.score_member(student.member_id)
                assert student.report is not None
                rows.append(_class_row(student, student.report))
            rows.sort(key=lambda item: item["credit_score"], reverse=True)
            return rows

        bundles = self.score_class()
        rows = []
        for bundle in bundles:
            rows.append(
                {
                    "member_id": bundle.member_id,
                    "name": bundle.name,
                    "class_name": bundle.class_name,
                    "job": bundle.job,
                    "credit_score": bundle.result.score,
                    "grade": bundle.result.grade,
                    "grade_code": bundle.result.grade_code,
                    "features": bundle.result.features,
                    "summary": bundle.summary,
                }
            )
        rows.sort(key=lambda item: item["credit_score"], reverse=True)
        return rows


def _class_row(student: DemoStudent, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "member_id": student.member_id,
        "name": student.name,
        "class_name": student.class_name,
        "job": student.job,
        "credit_score": report["credit_score"],
        "grade": report["grade"],
        "grade_code": report["grade_code"],
        "features": report["features"],
        "summary": report["summary"],
    }
