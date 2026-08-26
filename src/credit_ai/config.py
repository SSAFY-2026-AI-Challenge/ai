from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env",),
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    app_host: str = "0.0.0.0"
    app_port: int = 48765
    app_mode: Literal["offline", "db"] = "offline"

    db_host: str = ""
    db_port: int = 3306
    db_user: str = ""
    db_password: str = ""
    db_name: str = ""
    db_charset: str = "utf8mb4"

    model_dir: Path = Field(default=Path("models"))

    @property
    def project_root(self) -> Path:
        return find_project_root()

    @property
    def resolved_model_dir(self) -> Path:
        path = self.model_dir
        if not path.is_absolute():
            path = self.project_root / path
        return path

    @property
    def db_configured(self) -> bool:
        return bool(self.db_host and self.db_user and self.db_name)

    @property
    def use_database(self) -> bool:
        return self.app_mode == "db" and self.db_configured

    def sqlalchemy_url(self) -> str:
        if not self.db_configured:
            raise RuntimeError(
                "DB 접속 정보가 없습니다. 배포 서버의 env에 "
                "DB_HOST, DB_USER, DB_PASSWORD, DB_NAME 을 설정하세요."
            )
        from urllib.parse import quote_plus

        user = quote_plus(self.db_user)
        password = quote_plus(self.db_password)
        return (
            f"mysql+pymysql://{user}:{password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
            f"?charset={self.db_charset}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
