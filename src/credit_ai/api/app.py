from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from credit_ai.api.routes import router
from credit_ai.config import Settings, get_settings
from credit_ai.demo import DemoStore
from credit_ai.model.scorer import CreditScorer
from credit_ai.service import CreditService

PACKAGE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(PACKAGE_DIR / "templates"))


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
    static_dir = PACKAGE_DIR / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/favicon.ico")
    def favicon() -> RedirectResponse:
        return RedirectResponse(url="/static/favicon.svg", status_code=307)

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "index.html",
            {"title": "교실신용"},
        )

    return app


create_app = create_app
app = create_app()
