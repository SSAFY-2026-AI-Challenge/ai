"""명령줄 진입점.

    python -m fin_edu_credit.cli migrate   # 스키마 정리 + 결과 테이블 생성
    python -m fin_edu_credit.cli seed      # 더미 모의 경제 생성
    python -m fin_edu_credit.cli train     # 신용평가 스코어카드 학습
    python -m fin_edu_credit.cli score     # 전교생 점수 산출 후 DB 적재
    python -m fin_edu_credit.cli report    # 최근 학습 리포트 출력
    python -m fin_edu_credit.cli serve     # 조회 API/대시보드 실행
    python -m fin_edu_credit.cli all       # migrate -> seed -> train -> score
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

from .config import REPORT_DIR, ModelConfig, SimulationConfig
from .db import apply_migrations, connect
from .features import build_panel, load_source_frames
from .predict import class_summary, save_ai_reports, save_scores, score_members
from .seed import load_into_db, simulate
from .train import load_model, record_run, save_model, train_model


def _print(title: str, payload) -> None:
    print(f"\n== {title} ==")
    if isinstance(payload, (dict, list)):
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(payload)


def cmd_migrate(args) -> None:
    with connect() as conn:
        applied = apply_migrations(conn)
    _print("적용한 마이그레이션", applied)


def cmd_seed(args) -> None:
    sim_cfg = SimulationConfig(
        students_per_class=args.students_per_class,
        random_seed=args.seed,
    )
    economy = simulate(sim_cfg)
    _print("시뮬레이션 결과", economy.summary())
    if args.dry_run:
        print("\n--dry-run 이므로 DB 에 적재하지 않았습니다.")
        return
    with connect() as conn:
        counts = load_into_db(conn, economy, reset=not args.append)
    _print("적재 건수", counts)


def cmd_train(args) -> None:
    cfg = ModelConfig()
    with connect() as conn:
        src = load_source_frames(conn)
        panel = build_panel(src, cfg)
        model = train_model(src, cfg, panel)
        record_run(conn, model)
    path = save_model(model)
    metrics = model.metrics
    _print(
        "학습 요약",
        {
            "model_version": model.model_version,
            "algorithm": model.algorithm,
            "학습/검증 표본": f"{metrics['n_train']} / {metrics['n_test']}",
            "학습/검증 학생 수": f"{metrics['n_members_train']} / {metrics['n_members_test']}",
            "기준일 수": metrics["n_snapshots"],
            "연체율": metrics["test"]["bad_rate"],
            "교차검증 AUC": metrics["cv_auc"],
            "검증 AUC": metrics["test"]["auc"],
            "검증 KS": metrics["test"]["ks"],
            "검증 Gini": metrics["test"]["gini"],
            "Brier": metrics["test"]["brier"],
            "채택 피처 수": len(metrics["selected_features"]),
        },
    )
    _print("피처별 배점 폭(점)", metrics["coefficients"])
    _print("점수 구간별 실제 연체율(검증)", metrics["test"]["decile"])
    print(f"\n모델 저장: {path}")
    print(f"리포트 저장: {REPORT_DIR}")


def cmd_score(args) -> None:
    model = load_model()
    cfg = ModelConfig()
    with connect() as conn:
        src = load_source_frames(conn)
        as_of = pd.Timestamp(args.as_of) if args.as_of else None
        scored = score_members(model, src, as_of, cfg)
        if scored.empty:
            print("점수를 낼 대상 학생이 없습니다.")
            return
        n = save_scores(conn, scored)
        n_report = save_ai_reports(conn, scored) if args.write_report else 0

    _print(
        "스코어링 요약",
        {
            "기준일": str(scored["evaluated_at"].iloc[0]),
            "대상 학생": int(len(scored)),
            "평균 점수": int(scored["score"].mean()),
            "최고/최저": f"{int(scored['score'].max())} / {int(scored['score'].min())}",
            "등급 분포": scored["grade"].value_counts().sort_index().to_dict(),
            "credit_score 적재": n,
            "ai_report 적재": n_report,
        },
    )
    _print("반별 요약", class_summary(scored))
    top = scored.head(3)[["name", "class", "score", "grade_label"]].to_dict(orient="records")
    bottom = scored.tail(3)[["name", "class", "score", "grade_label"]].to_dict(orient="records")
    _print("점수 상위 3명", top)
    _print("점수 하위 3명", bottom)
    _print("예시 설명", scored.iloc[-1]["summary"])


def cmd_report(args) -> None:
    path = REPORT_DIR / "training_metrics.json"
    if not path.exists():
        print("학습 리포트가 없습니다. 먼저 train 을 실행하세요.")
        return
    metrics = json.loads(path.read_text(encoding="utf-8"))
    _print("정보값(IV)", metrics["information_value"])
    _print("제외된 피처", metrics["dropped_features"])
    _print("검증 지표", {k: v for k, v in metrics["test"].items() if k not in ("decile",)})
    scorecard = REPORT_DIR / "scorecard.csv"
    if scorecard.exists():
        table = pd.read_csv(scorecard)
        _print("구간별 배점표(일부)", table.head(30).to_string(index=False))


def cmd_serve(args) -> None:
    import uvicorn

    host = os.environ.get("FIN_EDU_API_HOST", "0.0.0.0")
    port = int(os.environ.get("FIN_EDU_API_PORT", "8899"))
    uvicorn.run("fin_edu_credit.api:app", host=args.host or host, port=args.port or port, reload=False)


def cmd_all(args) -> None:
    cmd_migrate(args)
    cmd_seed(args)
    cmd_train(args)
    cmd_score(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fin-edu-credit", description="초등학교 모의 경제 신용평가 모델"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("migrate", help="스키마 정리와 결과 테이블 생성")
    p.set_defaults(func=cmd_migrate)

    p = sub.add_parser("seed", help="더미 모의 경제 데이터 생성")
    p.add_argument("--students-per-class", type=int, default=25)
    p.add_argument("--seed", type=int, default=SimulationConfig().random_seed)
    p.add_argument("--append", action="store_true", help="기존 데이터를 지우지 않고 추가")
    p.add_argument("--dry-run", action="store_true", help="DB 적재 없이 통계만 출력")
    p.set_defaults(func=cmd_seed)

    p = sub.add_parser("train", help="신용평가 스코어카드 학습")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("score", help="전교생 신용점수 산출 및 적재")
    p.add_argument("--as-of", type=str, default=None, help="평가 기준일 (YYYY-MM-DD)")
    p.add_argument("--write-report", action="store_true", help="ai_report 테이블에도 요약 저장")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("report", help="최근 학습 리포트 출력")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("serve", help="조회 API / 대시보드 실행")
    p.add_argument("--host", type=str, default=None)
    p.add_argument("--port", type=int, default=None)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("all", help="migrate -> seed -> train -> score 를 한 번에")
    p.add_argument("--students-per-class", type=int, default=25)
    p.add_argument("--seed", type=int, default=SimulationConfig().random_seed)
    p.add_argument("--append", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--as-of", type=str, default=None)
    p.add_argument("--write-report", action="store_true")
    p.set_defaults(func=cmd_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
