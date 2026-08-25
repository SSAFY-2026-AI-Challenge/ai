"""배치 스코어링: 전교생 신용점수를 계산해 DB 에 적재한다."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pandas as pd

from .config import ModelConfig
from .explain import reason_codes, summarize
from .features import FEATURE_NAMES, SourceData, build_features
from .scorecard import grade_label, pd_to_score, rule_based_score, rule_components, score_to_grade


def score_members(
    model,
    src: SourceData,
    as_of: pd.Timestamp | None = None,
    cfg: ModelConfig | None = None,
) -> pd.DataFrame:
    """기준일 시점의 학생별 신용점수/등급/사유코드를 계산한다."""
    cfg = cfg or ModelConfig()
    as_of = pd.Timestamp(as_of).normalize() if as_of is not None else src.max_date

    features = build_features(src, as_of, cfg.observation_weeks)
    features = features[features["income_stability"] > 0]
    if features.empty:
        return pd.DataFrame()

    X = features[list(FEATURE_NAMES)]
    probabilities = model.predict_pd(X)
    scores = pd_to_score(probabilities, cfg)
    grades = score_to_grade(scores)
    rule_scores = rule_based_score(X)
    rule_grades = score_to_grade(rule_scores.to_numpy())
    reasons = reason_codes(model, X, cfg)
    components = rule_components(X)

    members = src.members.set_index("id")
    result = pd.DataFrame(
        {
            "member_id": features.index,
            "name": members.loc[features.index, "name"].to_numpy(),
            "class": members.loc[features.index, "class"].to_numpy(),
            "job": members.loc[features.index, "job"].to_numpy(),
            "evaluated_at": as_of.date(),
            "model_version": model.model_version,
            "pd": probabilities.round(6),
            "score": scores,
            "grade": grades,
            "grade_label": [grade_label(g) for g in grades],
            "rule_score": rule_scores.to_numpy(),
            "rule_grade": rule_grades,
        }
    ).set_index("member_id", drop=False)

    result["reason"] = [reasons[mid] for mid in result["member_id"]]
    result["rule_components"] = [
        {k: round(float(v), 4) for k, v in components.loc[mid].items()} for mid in result["member_id"]
    ]
    result["features"] = [
        {k: round(float(v), 4) for k, v in X.loc[mid].items()} for mid in result["member_id"]
    ]
    result["summary"] = [
        summarize(row["name"], int(row["score"]), int(row["grade"]), row["grade_label"], row["reason"])
        for _, row in result.iterrows()
    ]
    return result.sort_values("score", ascending=False)


def save_scores(conn, scored: pd.DataFrame) -> int:
    """credit_score 테이블에 upsert 한다."""
    from .db import executemany

    now = datetime.now()
    rows = [
        (
            int(r.member_id),
            r.evaluated_at,
            r.model_version,
            int(r.score),
            int(r.grade),
            r.grade_label,
            float(r.pd),
            int(r.rule_score),
            int(r.rule_grade),
            json.dumps(
                {"summary": r.summary, **r.reason, "rule_components": r.rule_components},
                ensure_ascii=False,
            ),
            json.dumps(r.features, ensure_ascii=False),
            now,
        )
        for r in scored.itertuples(index=False)
    ]
    n = executemany(
        conn,
        """
        INSERT INTO credit_score
            (member_id, evaluated_at, model_version, score, grade, grade_label, pd,
             rule_score, rule_grade, reason_json, feature_json, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            score = VALUES(score), grade = VALUES(grade), grade_label = VALUES(grade_label),
            pd = VALUES(pd), rule_score = VALUES(rule_score), rule_grade = VALUES(rule_grade),
            reason_json = VALUES(reason_json), feature_json = VALUES(feature_json),
            created_at = VALUES(created_at)
        """,
        rows,
    )
    conn.commit()
    return n


def save_ai_reports(conn, scored: pd.DataFrame) -> int:
    """학생별 요약을 ai_report 테이블에도 반영한다(학생당 1건)."""
    from .db import executemany

    rows = [
        (
            int(r.member_id),
            "CREDIT",
            r.summary,
            json.dumps(
                {
                    "score": int(r.score),
                    "grade": int(r.grade),
                    "grade_label": r.grade_label,
                    "rule_score": int(r.rule_score),
                    "model_version": r.model_version,
                    **r.reason,
                },
                ensure_ascii=False,
            ),
            r.evaluated_at,
        )
        for r in scored.itertuples(index=False)
    ]
    n = executemany(
        conn,
        """
        INSERT INTO ai_report (member_id, report_type, summary, detail_json, generated_at)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            report_type = VALUES(report_type), summary = VALUES(summary),
            detail_json = VALUES(detail_json), generated_at = VALUES(generated_at)
        """,
        rows,
    )
    conn.commit()
    return n


def class_summary(scored: pd.DataFrame) -> list[dict[str, Any]]:
    """반별 평균 점수와 등급 분포."""
    out = []
    for class_name, group in scored.groupby("class"):
        out.append(
            {
                "class": class_name,
                "students": int(len(group)),
                "score_mean": int(group["score"].mean()),
                "score_min": int(group["score"].min()),
                "score_max": int(group["score"].max()),
                "grade_mean": round(float(group["grade"].mean()), 2),
                "high_risk": int((group["grade"] >= 8).sum()),
            }
        )
    return sorted(out, key=lambda r: r["class"])
