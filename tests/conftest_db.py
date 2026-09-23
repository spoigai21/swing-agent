"""Fixtures for tests that run real SQL against a real Postgres.

⚠️ NEVER against the developer's own database. These tests INSERT and DELETE, so
they run only when `SWING_TEST_DATABASE_URL` is set and names a database ending
in `_test`. Everything else skips. Pointing them at a working corpus would
destroy months of collection that cannot be re-collected.
"""
from __future__ import annotations

import os

import pytest

TEST_DB_ENV = "SWING_TEST_DATABASE_URL"


def _test_dsn() -> str | None:
    dsn = os.getenv(TEST_DB_ENV, "").strip()
    if not dsn:
        return None
    name = dsn.rsplit("/", 1)[-1].split("?")[0]
    if not name.endswith("_test"):
        pytest.exit(f"{TEST_DB_ENV} must name a database ending in '_test', got {name!r}",
                    returncode=2)
    return dsn


@pytest.fixture(scope="session")
def db_dsn() -> str:
    dsn = _test_dsn()
    if not dsn:
        pytest.skip(f"set {TEST_DB_ENV} to a *_test database to run SQL tests")
    return dsn


@pytest.fixture(scope="session")
def db(db_dsn, tmp_path_factory):
    """A schema-applied test database, with settings pointed at it."""
    from swing.common import settings as settings_mod
    from swing.store import session as session_mod

    cfg = settings_mod.get_settings().model_copy(update={"database_url": db_dsn})
    settings_mod.get_settings.cache_clear()
    original = settings_mod.get_settings
    settings_mod.get_settings = lambda: cfg
    session_mod.get_settings = lambda: cfg

    session_mod.apply_schema()
    yield session_mod
    settings_mod.get_settings = original
    settings_mod.get_settings.cache_clear()


@pytest.fixture
def clean_db(db):
    """Empty the analysis tables around each test."""
    tables = ("cluster_members", "clusters", "attributions", "annotations",
              "swings", "daily_factors", "articles", "articles_raw",
              "intraday_bars", "bars", "source_health", "edgar_cursor")
    def wipe():
        with db.connect() as conn:
            for t in tables:
                conn.execute(f"TRUNCATE {t} CASCADE")
    wipe()
    yield db
    wipe()
