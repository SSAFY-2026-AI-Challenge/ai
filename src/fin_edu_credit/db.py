"""MySQL 접근 헬퍼."""

from __future__ import annotations

import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import pandas as pd
import pymysql
from pymysql.cursors import DictCursor

from .config import DBConfig, SQL_DIR


@contextmanager
def connect(cfg: DBConfig | None = None, autocommit: bool = False) -> Iterator[pymysql.connections.Connection]:
    cfg = cfg or DBConfig.from_env()
    if not cfg.password:
        raise RuntimeError(
            "DB 비밀번호가 비어 있습니다. `cp .env.example .env` 후 FIN_EDU_DB_PASSWORD 를 설정하세요."
        )
    conn = pymysql.connect(
        host=cfg.host,
        port=cfg.port,
        user=cfg.user,
        password=cfg.password,
        database=cfg.database,
        charset=cfg.charset,
        autocommit=autocommit,
        cursorclass=DictCursor,
    )
    try:
        yield conn
    finally:
        conn.close()


def fetch_df(conn: pymysql.connections.Connection, sql: str, params: Sequence[Any] | None = None) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params or ())
        rows = cur.fetchall()
        columns = [d[0] for d in cur.description] if cur.description else []
    return pd.DataFrame(rows, columns=columns)


def execute(conn: pymysql.connections.Connection, sql: str, params: Sequence[Any] | None = None) -> int:
    with conn.cursor() as cur:
        return cur.execute(sql, params or ())


def executemany(
    conn: pymysql.connections.Connection,
    sql: str,
    rows: Iterable[Sequence[Any]],
    batch_size: int = 2000,
) -> int:
    """대량 INSERT 를 배치로 나눠 실행한다."""
    total = 0
    batch: list[Sequence[Any]] = []
    with conn.cursor() as cur:
        for row in rows:
            batch.append(row)
            if len(batch) >= batch_size:
                cur.executemany(sql, batch)
                total += len(batch)
                batch.clear()
        if batch:
            cur.executemany(sql, batch)
            total += len(batch)
    return total


_STATEMENT_SPLIT = re.compile(r";\s*(?:\n|$)")


def run_sql_file(conn: pymysql.connections.Connection, path: Path) -> list[str]:
    """세미콜론으로 구분된 SQL 스크립트를 순차 실행한다.

    PREPARE/EXECUTE 를 쓰는 조건부 DDL 을 지원하기 위해 statement 단위로 나눠 실행한다.
    """
    sql_text = path.read_text(encoding="utf-8")
    executed: list[str] = []
    with conn.cursor() as cur:
        for raw in _STATEMENT_SPLIT.split(sql_text):
            stmt = "\n".join(
                line for line in raw.splitlines() if not line.strip().startswith("--")
            ).strip()
            if not stmt:
                continue
            cur.execute(stmt)
            # PREPARE 된 SELECT 결과가 남아 있으면 비워 준다.
            try:
                cur.fetchall()
            except Exception:  # noqa: BLE001 - 결과 없는 statement
                pass
            executed.append(stmt.splitlines()[0][:70])
    conn.commit()
    return executed


def apply_migrations(conn: pymysql.connections.Connection, sql_dir: Path | None = None) -> list[str]:
    sql_dir = sql_dir or SQL_DIR
    applied: list[str] = []
    for path in sorted(sql_dir.glob("*.sql")):
        run_sql_file(conn, path)
        applied.append(path.name)
    return applied
