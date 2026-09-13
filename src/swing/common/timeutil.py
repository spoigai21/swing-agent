"""Timezone discipline.

The failure mode this guards against does not announce itself: a naive datetime
slipping through reclassifies pre-move articles as post-move, which silently
corrupts the core signal. agent-plan.md §0.4 / TROUBLESHOOTING.

Call assert_utc() on every timestamp write and every read.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from zoneinfo import ZoneInfo


def from_wallclock_epoch(ts: float, tz: str) -> datetime:
    """Decode a 'Unix timestamp' that actually encodes LOCAL wall-clock time.

    ⚠️ Finnhub company-news `datetime` is Eastern wall-clock time encoded as if
    it were UTC. Verified 2026-09-13: the same CNBC stories via CNBC's own RSS
    ('... GMT', fetched seconds after publication) sit exactly 4h after
    Finnhub's stamp in summer, and winter earnings releases need 5h -- a flat
    +4h still puts SBUX and QCOM reaction stories 45 minutes BEFORE the 8-K.
    Read naively, every Finnhub article lands 4-5h early, which files post-move
    commentary as pre-move evidence.

    fold=1 resolves the repeated fall-back hour to standard time, as Postgres
    AT TIME ZONE does, so the correcting migration and live ingest agree.
    """
    wall = datetime.fromtimestamp(ts, UTC).replace(tzinfo=None)
    return wall.replace(tzinfo=ZoneInfo(tz), fold=1).astimezone(UTC)


def assert_utc(ts: datetime) -> datetime:
    """Reject naive datetimes at the boundary; normalise aware ones to UTC."""
    if not isinstance(ts, datetime):
        raise TypeError(f"expected datetime, got {type(ts).__name__}: {ts!r}")
    if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
        raise ValueError(f"naive datetime: {ts!r}")
    return ts.astimezone(UTC)


def now_utc() -> datetime:
    return datetime.now(UTC)


def parse_rss_datetime(value: str | None, struct_time=None) -> datetime | None:
    """RSS pubDate arrives in several formats. Return UTC or None, never naive.

    feedparser gives us a parsed struct_time in UTC when it succeeds; fall back
    to RFC-2822 parsing of the raw string when it doesn't.
    """
    if struct_time is not None:
        try:
            return datetime(*struct_time[:6], tzinfo=UTC)
        except (TypeError, ValueError):
            pass
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        # RFC-2822 without an offset. Treat as UTC but do not pretend to know.
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def parse_iso(value: str) -> datetime:
    """Parse an ISO-8601 string, treating a bare timestamp as UTC.

    SEC acceptanceDateTime arrives as '2025-08-29T16:31:05.000Z' or without a
    zone; either way the result must be aware.
    """
    v = value.strip().replace("Z", "+00:00")
    ts = datetime.fromisoformat(v)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def days_ago(n: int) -> datetime:
    return now_utc() - timedelta(days=n)
