"""GCP Secret Manager 에서 DB 접속 정보를 읽습니다."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Callable, Protocol

if TYPE_CHECKING:
    from credit_ai.config import Settings

DB_SECRET_IDS = (
    "DB_HOST",
    "DB_PORT",
    "DB_USER",
    "DB_PASSWORD",
    "DB_NAME",
)


class GcpSecretError(RuntimeError):
    """Secret Manager 에서 필수 시크릿을 읽지 못했을 때 발생합니다."""


class SecretVersionClient(Protocol):
    def access_secret_version(self, request: dict[str, str]) -> object: ...


def resolve_gcp_project_id(explicit: str = "") -> str:
    if explicit.strip():
        return explicit.strip()
    for key in ("GCP_PROJECT_ID", "GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    try:
        import google.auth

        _, project = google.auth.default()
    except Exception:
        return ""
    return str(project).strip() if project else ""


def _payload_text(response: object) -> str:
    payload = getattr(response, "payload", None)
    data = getattr(payload, "data", b"")
    if isinstance(data, bytes):
        return data.decode("utf-8").strip()
    return str(data).strip()


def fetch_db_secrets(
    project_id: str,
    client: SecretVersionClient | None = None,
) -> dict[str, str]:
    if not project_id:
        raise GcpSecretError(
            "GCP 프로젝트 ID를 찾지 못했습니다. GCP_PROJECT_ID 를 설정하거나 "
            "GCE 서비스 계정으로 실행하세요."
        )
    if client is None:
        from google.cloud import secretmanager

        client = secretmanager.SecretManagerServiceClient()

    values: dict[str, str] = {}
    missing: list[str] = []
    for secret_id in DB_SECRET_IDS:
        name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
        try:
            response = client.access_secret_version(request={"name": name})
            text = _payload_text(response)
        except Exception as exc:
            missing.append(f"{secret_id} ({exc.__class__.__name__})")
            continue
        if not text:
            missing.append(f"{secret_id} (empty)")
            continue
        values[secret_id] = text

    if missing:
        joined = ", ".join(missing)
        raise GcpSecretError(
            "Secret Manager 에서 DB 시크릿을 읽지 못했습니다: "
            f"{joined}. 시크릿 이름과 서비스 계정 권한"
            "(roles/secretmanager.secretAccessor)을 확인하세요."
        )
    return values


def overlay_db_settings(
    settings: Settings,
    *,
    fetch: Callable[[str], dict[str, str]] | None = None,
) -> Settings:
    """APP_MODE=db 이면 Secret Manager 값을 Settings 에 덮어쓰니다."""
    source = settings.secrets_source
    if source == "env":
        return settings.model_copy(update={"db_secrets_from": "env"})
    if settings.app_mode != "db" and source != "gcp":
        return settings.model_copy(update={"db_secrets_from": "env"})

    project_id = resolve_gcp_project_id(settings.gcp_project_id)
    if source == "auto" and not project_id:
        return settings.model_copy(update={"db_secrets_from": "env"})

    loader = fetch or (lambda pid: fetch_db_secrets(pid))
    try:
        secrets = loader(project_id)
    except GcpSecretError:
        if source == "auto" and settings.db_configured:
            return settings.model_copy(update={"db_secrets_from": "env"})
        raise

    update = {
        "gcp_project_id": project_id,
        "db_host": secrets["DB_HOST"],
        "db_port": int(secrets["DB_PORT"]),
        "db_user": secrets["DB_USER"],
        "db_password": secrets["DB_PASSWORD"],
        "db_name": secrets["DB_NAME"],
        "db_secrets_from": "gcp",
    }
    return settings.model_copy(update=update)
