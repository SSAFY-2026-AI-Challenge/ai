from __future__ import annotations

from types import SimpleNamespace

import pytest

from credit_ai.config import Settings, get_settings
from credit_ai.gcp_secrets import (
    GcpSecretError,
    fetch_db_secrets,
    overlay_db_settings,
    probe_secret_manager,
)


class FakeClient:
    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping
        self.requested: list[str] = []

    def access_secret_version(self, request: dict[str, str]) -> object:
        name = request["name"]
        self.requested.append(name)
        secret_id = name.split("/")[3]
        if secret_id not in self.mapping:
            raise KeyError(secret_id)
        return SimpleNamespace(
            payload=SimpleNamespace(data=self.mapping[secret_id].encode("utf-8"))
        )


SAMPLE = {
    "DB_HOST": "10.0.0.8\n",
    "DB_PORT": "3306",
    "DB_USER": "credit",
    "DB_PASSWORD": "s3cret",
    "DB_NAME": "classroom",
}


def test_fetch_db_secrets_reads_named_secrets_and_strips() -> None:
    client = FakeClient(SAMPLE)
    values = fetch_db_secrets("ssafy-2026-ai", client=client)
    assert values["DB_HOST"] == "10.0.0.8"
    assert values["DB_PASSWORD"] == "s3cret"
    assert all(
        f"projects/ssafy-2026-ai/secrets/{key}/versions/latest" in client.requested
        for key in SAMPLE
    )


def test_fetch_db_secrets_lists_missing_names() -> None:
    client = FakeClient({"DB_HOST": "127.0.0.1"})
    with pytest.raises(GcpSecretError, match="DB_PASSWORD"):
        fetch_db_secrets("ssafy-2026-ai", client=client)


def test_overlay_uses_secret_manager_in_db_mode() -> None:
    settings = Settings(
        app_mode="db",
        secrets_source="gcp",
        gcp_project_id="ssafy-2026-ai",
    )
    loaded = overlay_db_settings(
        settings,
        fetch=lambda _pid: {
            "DB_HOST": "10.0.0.8",
            "DB_PORT": "3306",
            "DB_USER": "credit",
            "DB_PASSWORD": "s3cret",
            "DB_NAME": "classroom",
        },
    )
    assert loaded.use_database
    assert loaded.db_secrets_from == "gcp"
    assert loaded.db_host == "10.0.0.8"
    assert loaded.db_port == 3306
    assert "s3cret" in loaded.sqlalchemy_url()
    assert "credit" in loaded.sqlalchemy_url()


def test_offline_mode_skips_secret_manager() -> None:
    settings = Settings(app_mode="offline", secrets_source="auto")

    def boom(_pid: str) -> dict[str, str]:
        raise AssertionError("offline 에서는 Secret Manager 를 호출하면 안 됩니다")

    loaded = overlay_db_settings(settings, fetch=boom)
    assert loaded.db_secrets_from == "env"
    assert not loaded.use_database


def test_auto_falls_back_to_env_when_already_configured() -> None:
    settings = Settings(
        app_mode="db",
        secrets_source="auto",
        gcp_project_id="ssafy-2026-ai",
        db_host="127.0.0.1",
        db_user="local",
        db_password="local-pass",
        db_name="local_db",
    )

    def fail(_pid: str) -> dict[str, str]:
        raise GcpSecretError("denied")

    loaded = overlay_db_settings(settings, fetch=fail)
    assert loaded.db_secrets_from == "env"
    assert loaded.db_user == "local"


def test_get_settings_offline_default(monkeypatch) -> None:
    monkeypatch.delenv("APP_MODE", raising=False)
    monkeypatch.setenv("APP_MODE", "offline")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.app_mode == "offline"
    assert settings.db_secrets_from == "env"
    get_settings.cache_clear()


def test_probe_secret_manager_omits_values() -> None:
    client = FakeClient(SAMPLE)
    report = probe_secret_manager("ssafy-2026-ai", client=client)
    assert report["ok"] is True
    assert report["project_id"] == "ssafy-2026-ai"
    assert report["secrets"]["DB_PASSWORD"]["readable"] is True
    assert report["secrets"]["DB_PASSWORD"]["chars"] == 6
    dumped = str(report)
    assert "s3cret" not in dumped
    assert "10.0.0.8" not in dumped


def test_probe_secret_manager_reports_unread_secrets() -> None:
    client = FakeClient({"DB_HOST": "10.0.0.8", "DB_PORT": "3306"})
    report = probe_secret_manager("ssafy-2026-ai", client=client)
    assert report["ok"] is False
    assert report["secrets"]["DB_HOST"]["readable"] is True
    assert report["secrets"]["DB_PASSWORD"]["readable"] is False
    assert report["secrets"]["DB_PASSWORD"]["error"] == "KeyError"
    dumped = str(report)
    assert "s3cret" not in dumped
    assert "10.0.0.8" not in dumped
