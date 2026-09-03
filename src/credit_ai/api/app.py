from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from credit_ai.api.routes import router
from credit_ai.config import Settings, get_settings
from credit_ai.demo import DemoStore
from credit_ai.model.scorer import CreditScorer
from credit_ai.service import CreditService

logger = logging.getLogger("credit_ai.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    scorer = CreditScorer.try_load(settings.resolved_model_dir)
    store = DemoStore()
    if not settings.use_database:
        store.seed()
    app.state.settings = settings
    app.state.service = CreditService(settings, scorer, store)

    metric_task: asyncio.Task[None] | None = None
    if settings.use_database and settings.metric_interval_seconds > 0:
        from credit_ai.metrics.scheduler import run_metric_scheduler

        metric_task = asyncio.create_task(
            run_metric_scheduler(settings),
            name="economic-metric",
        )
        logger.info("economic_metric 백그라운드 갱신을 시작합니다")

    try:
        yield
    finally:
        if metric_task is not None:
            metric_task.cancel()
            with suppress(asyncio.CancelledError):
                await metric_task


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is not None:
        get_settings.cache_clear()
    app = FastAPI(
        title="교실 경제 신용평가 AI",
        description="초등 교실 경제활동 시뮬레이션을 위한 딥러닝 신용점수 API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


create_app = create_app
app = create_app()
