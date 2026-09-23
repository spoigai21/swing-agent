from __future__ import annotations

from datetime import UTC, datetime

import pytest


@pytest.fixture
def aware() -> datetime:
    return datetime(2026, 8, 14, 14, 35, tzinfo=UTC)


@pytest.fixture
def naive() -> datetime:
    # Deliberately naive: this is the input assert_utc must reject.
    return datetime(2026, 8, 14, 14, 35)  # noqa: DTZ001

# Database fixtures live next door so this file stays importable without a DB.
pytest_plugins = ["tests.conftest_db"]
