"""Engine and connection helpers.

Phase -1 uses psycopg directly — no ORM yet, because the only table is
articles_raw and the collector must not wait on the full model layer.
SQLAlchemy models arrive in Step 1.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from swing.common.settings import get_settings

SCHEMA = Path(__file__).parent / "schema.sql"


def dsn() -> str:
    """psycopg wants a bare libpq DSN; strip the SQLAlchemy driver prefix."""
    url = get_settings().database_url
    return url.replace("postgresql+psycopg://", "postgresql://")


@contextmanager
def connect(autocommit: bool = True):
    with psycopg.connect(dsn(), autocommit=autocommit, row_factory=dict_row) as conn:
        yield conn


def apply_schema() -> None:
    sql = SCHEMA.read_text()
    with connect() as conn:
        conn.execute(sql)  # type: ignore[arg-type]
