from __future__ import annotations

import argparse
import json
import sys

import uvicorn

from credit_ai.config import get_settings
from credit_ai.features.schema import CreditFeatures
from credit_ai.model.scorer import CreditScorer
from credit_ai.training.train import train_model


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="credit-ai", description="교실 경제 신용평가 AI")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="합성 데이터로 MLP 신용점수 모델을 학습합니다")
    train_p.add_argument("--samples", type=int, default=8000)
    train_p.add_argument("--epochs", type=int, default=45)
    train_p.add_argument("--batch-size", type=int, default=128)

    serve_p = sub.add_parser("serve", help="추론 API와 교사 대시보드를 실행합니다")
    serve_p.add_argument("--host", default=None)
    serve_p.add_argument("--port", type=int, default=None)

    score_p = sub.add_parser("score", help="피처를 직접 넣어 점수를 계산합니다")
    score_p.add_argument("--spending", type=float, required=True, help="지출 추이 0~1")
    score_p.add_argument("--homework", type=float, required=True, help="숙제 완료율 0~1")
    score_p.add_argument("--attendance", type=float, required=True, help="출결률 0~1")

    sub.add_parser(
        "verify-secrets",
        help="Secret Manager 의 DB_* 시크릿을 읽을 수 있는지 확인합니다",
    )

    args = parser.parse_args(argv)

    if args.command == "verify-secrets":
        from credit_ai.gcp_secrets import probe_secret_manager

        report = probe_secret_manager()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    settings = get_settings()

    if args.command == "train":
        metrics = train_model(
            out_dir=settings.resolved_model_dir,
            n_samples=args.samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return

    if args.command == "serve":
        host = args.host or settings.app_host
        port = args.port or settings.app_port
        uvicorn.run(
            "credit_ai.api.app:app",
            host=host,
            port=port,
            reload=False,
        )
        return

    if args.command == "score":
        scorer = CreditScorer.load(settings.resolved_model_dir)
        features = CreditFeatures(
            spending_trend=args.spending,
            homework_rate=args.homework,
            attendance_rate=args.attendance,
        )
        result = scorer.score(features)
        print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
        return

    parser.print_help()
    sys.exit(2)
