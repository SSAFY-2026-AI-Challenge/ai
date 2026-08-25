"""신용평가 조회 API 와 교사용 대시보드.

    python -m fin_edu_credit.cli serve

- ``/``          : 학급 신용 현황 대시보드(HTML)
- ``/docs``      : 자동 생성 API 문서
- ``/api/...``   : 학급 서비스(백엔드)에서 호출할 JSON 엔드포인트
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from .config import GRADE_TABLE, ModelConfig
from .db import connect, fetch_df
from .explain import reason_codes, summarize
from .features import FEATURE_BY_NAME, FEATURE_NAMES, build_features, load_source_frames
from .predict import class_summary, score_members
from .scorecard import (
    RULE_COMPONENT_LABELS,
    grade_label,
    pd_to_score,
    rule_based_score,
    score_to_grade,
)

app = FastAPI(
    title="fin_edu 학급 신용평가 API",
    description="초등학교 모의 경제의 거래·계좌·생활 기록으로 학생 신용점수를 산출한다.",
    version="0.1.0",
)

STATIC_DIR = Path(__file__).resolve().parent / "static"


@lru_cache(maxsize=1)
def _model():
    from .train import load_model

    return load_model()


def _source():
    with connect() as conn:
        return load_source_frames(conn)


@lru_cache(maxsize=1)
def _scored_cache() -> pd.DataFrame:
    return score_members(_model(), _source(), None, ModelConfig())


def _reload() -> pd.DataFrame:
    _scored_cache.cache_clear()
    return _scored_cache()


def _record(row: pd.Series) -> dict[str, Any]:
    return {
        "member_id": int(row["member_id"]),
        "name": row["name"],
        "class": row["class"],
        "job": row["job"],
        "evaluated_at": str(row["evaluated_at"]),
        "score": int(row["score"]),
        "grade": int(row["grade"]),
        "grade_label": row["grade_label"],
        "pd": float(row["pd"]),
        "rule_score": int(row["rule_score"]),
        "rule_grade": int(row["rule_grade"]),
        "summary": row["summary"],
        "reason": row["reason"],
        "rule_components": {
            RULE_COMPONENT_LABELS[k]: v for k, v in row["rule_components"].items()
        },
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    model = _model()
    return {
        "status": "ok",
        "model_version": model.model_version,
        "algorithm": model.algorithm,
        "features": list(model.selected_features),
    }


@app.get("/api/grades")
def grades() -> list[dict[str, Any]]:
    return [{"grade": g, "min_score": cutoff, "label": label} for g, cutoff, label in GRADE_TABLE]


@app.get("/api/scores")
def scores(
    class_name: str | None = Query(default=None, alias="class"),
    limit: int = Query(default=500, ge=1, le=2000),
) -> dict[str, Any]:
    scored = _scored_cache()
    if class_name:
        scored = scored[scored["class"] == class_name]
    rows = [_record(row) for _, row in scored.head(limit).iterrows()]
    return {
        "evaluated_at": str(scored["evaluated_at"].iloc[0]) if len(scored) else None,
        "count": len(rows),
        "average_score": int(scored["score"].mean()) if len(scored) else None,
        "grade_distribution": {int(k): int(v) for k, v in scored["grade"].value_counts().sort_index().items()},
        "items": rows,
    }


@app.get("/api/classes")
def classes() -> list[dict[str, Any]]:
    return class_summary(_scored_cache())


@app.get("/api/members/{member_id}")
def member_detail(member_id: int) -> dict[str, Any]:
    scored = _scored_cache()
    if member_id not in scored.index:
        raise HTTPException(status_code=404, detail="해당 학생의 평가 결과가 없습니다.")
    row = scored.loc[member_id]
    detail = _record(row)
    detail["features"] = {
        FEATURE_BY_NAME[k].label: v for k, v in row["features"].items() if k in FEATURE_BY_NAME
    }
    detail["points"] = {
        FEATURE_BY_NAME[k].label: round(float(v), 1)
        for k, v in _model()
        .points(pd.DataFrame([row["features"]], index=[member_id]))
        .loc[member_id]
        .items()
    }
    return detail


@app.get("/api/members/{member_id}/history")
def member_history(member_id: int) -> list[dict[str, Any]]:
    with connect() as conn:
        df = fetch_df(
            conn,
            "SELECT evaluated_at, score, grade, grade_label, pd, rule_score "
            "FROM credit_score WHERE member_id = %s ORDER BY evaluated_at",
            (member_id,),
        )
    if df.empty:
        raise HTTPException(status_code=404, detail="적재된 평가 이력이 없습니다.")
    df["evaluated_at"] = df["evaluated_at"].astype(str)
    return df.to_dict(orient="records")


@app.post("/api/simulate")
def simulate_what_if(member_id: int, overrides: dict[str, float]) -> dict[str, Any]:
    """'적금을 500 더 넣으면 점수가 어떻게 될까?' 같은 what-if 계산."""
    scored = _scored_cache()
    if member_id not in scored.index:
        raise HTTPException(status_code=404, detail="해당 학생의 평가 결과가 없습니다.")
    unknown = set(overrides) - set(FEATURE_NAMES)
    if unknown:
        raise HTTPException(status_code=400, detail=f"알 수 없는 피처: {sorted(unknown)}")

    base = dict(scored.loc[member_id, "features"])
    changed = {**base, **overrides}
    X = pd.DataFrame([changed], index=[member_id])[list(FEATURE_NAMES)]

    cfg = ModelConfig()
    probability = float(_model().predict_pd(X)[0])
    score = int(pd_to_score([probability], cfg)[0])
    grade = int(score_to_grade([score])[0])
    reasons = reason_codes(_model(), X, cfg)[member_id]
    return {
        "member_id": member_id,
        "before": {"score": int(scored.loc[member_id, "score"]), "grade": int(scored.loc[member_id, "grade"])},
        "after": {
            "score": score,
            "grade": grade,
            "grade_label": grade_label(grade),
            "pd": round(probability, 6),
            "rule_score": int(rule_based_score(X).iloc[0]),
        },
        "delta": score - int(scored.loc[member_id, "score"]),
        "reason": reasons,
        "summary": summarize(scored.loc[member_id, "name"], score, grade, grade_label(grade), reasons),
    }


@app.get("/api/scorecard")
def scorecard() -> list[dict[str, Any]]:
    """구간별 배점표. 교실 게시용."""
    table = _model().scorecard_table()
    return table.to_dict(orient="records")


@app.post("/api/refresh")
def refresh() -> dict[str, Any]:
    scored = _reload()
    return {"reloaded": True, "count": int(len(scored))}


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return (STATIC_DIR / "dashboard.html").read_text(encoding="utf-8")


def evaluate_member_at(member_id: int, as_of: str) -> dict[str, Any]:
    """특정 시점 기준으로 한 학생을 다시 평가한다(배치 외 조회용)."""
    src = _source()
    cfg = ModelConfig()
    features = build_features(src, pd.Timestamp(as_of), cfg.observation_weeks, member_ids=[member_id])
    if features.empty:
        raise HTTPException(status_code=404, detail="해당 시점의 데이터가 없습니다.")
    X = features[list(FEATURE_NAMES)]
    probability = float(_model().predict_pd(X)[0])
    score = int(pd_to_score([probability], cfg)[0])
    grade = int(score_to_grade([score])[0])
    return {"member_id": member_id, "as_of": as_of, "score": score, "grade": grade, "pd": probability}
