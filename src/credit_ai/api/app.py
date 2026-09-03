from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from credit_ai.api.routes import router
from credit_ai.config import Settings, get_settings
from credit_ai.demo import DemoStore
from credit_ai.model.scorer import CreditScorer
from credit_ai.service import CreditService


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    scorer = CreditScorer.try_load(settings.resolved_model_dir)
    store = DemoStore()
    if not settings.use_database:
        store.seed()
    app.state.settings = settings
    app.state.service = CreditService(settings, scorer, store)
    yield


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
