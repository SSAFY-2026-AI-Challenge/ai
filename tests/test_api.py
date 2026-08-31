from __future__ import annotations

from fastapi.testclient import TestClient

from credit_ai.api.app import create_app
from credit_ai.config import get_settings
from credit_ai.demo import DemoStore
from credit_ai.model.scorer import CreditScorer
from credit_ai.service import CreditService
from credit_ai.training.train import train_model


def test_health_and_score_flow(tmp_path, monkeypatch) -> None:
    train_model(tmp_path, n_samples=700, epochs=8, batch_size=64, seed=3)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("APP_MODE", "offline")
    get_settings.cache_clear()

    settings = get_settings()
    app = create_app(settings)
    scorer = CreditScorer.load(tmp_path)
    store = DemoStore()
    store.seed()
    app.state.service = CreditService(settings, scorer, store)

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200

        scored = client.post(
            "/v1/score",
            json={
                "spending_trend": 0.2,
                "homework_rate": 0.95,
                "attendance_rate": 0.98,
            },
        )
        assert scored.status_code == 200
        body = scored.json()
        assert 300 <= body["credit_score"] <= 1000
        assert body["grade"]
        assert len(body["recommendations"]) >= 1

        roster = client.get("/v1/class")
        assert roster.status_code == 200
        assert len(roster.json()) >= 10

        detail = client.post("/v1/score/member/1")
        assert detail.status_code == 200
        assert detail.json()["name"]
