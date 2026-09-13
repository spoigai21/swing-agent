"""Timezone discipline.

The failure this guards against is silent: a naive datetime slipping through
reclassifies pre-move articles as post-move and corrupts the core signal with
no visible symptom. So the tests assert it RAISES, not that it copes.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from swing.common.timeutil import (
    assert_utc,
    days_ago,
    from_wallclock_epoch,
    now_utc,
    parse_iso,
    parse_rss_datetime,
)

ET = timezone(timedelta(hours=-4))


def test_naive_datetime_is_rejected(naive):
    with pytest.raises(ValueError, match="naive datetime"):
        assert_utc(naive)


def test_non_datetime_is_rejected():
    with pytest.raises(TypeError):
        assert_utc("2026-08-14T14:35:00Z")


def test_aware_utc_passes_through(aware):
    assert assert_utc(aware) == aware
    assert assert_utc(aware).tzinfo == UTC


def test_offset_is_converted_not_relabelled():
    et = datetime(2026, 8, 14, 10, 35, tzinfo=ET)   # 14:35 UTC
    out = assert_utc(et)
    assert out.tzinfo == UTC
    assert (out.hour, out.minute) == (14, 35)


def test_now_is_aware():
    assert now_utc().tzinfo is not None
    assert days_ago(2) < now_utc()


class TestParseIso:
    def test_z_suffix(self):
        # SEC acceptanceDateTime format
        assert parse_iso("2026-08-29T16:31:05.000Z") == datetime(
            2026, 8, 29, 16, 31, 5, tzinfo=UTC
        )

    def test_explicit_offset_is_converted(self):
        assert parse_iso("2026-08-29T12:31:05-04:00").hour == 16

    def test_bare_timestamp_treated_as_utc(self):
        assert parse_iso("2026-08-29T16:31:05").tzinfo == UTC

    def test_date_only(self):
        assert parse_iso("2026-08-29").tzinfo == UTC


class TestParseRss:
    def test_rfc2822_with_offset(self):
        got = parse_rss_datetime("Fri, 29 Aug 2026 12:31:05 -0400", None)
        assert got.tzinfo == UTC and got.hour == 16

    def test_struct_time_preferred(self):
        got = parse_rss_datetime(None, (2026, 8, 29, 16, 31, 5, 0, 0, 0))
        assert got == datetime(2026, 8, 29, 16, 31, 5, tzinfo=UTC)

    def test_unparseable_returns_none_rather_than_guessing(self):
        # No timestamp means no timing evidence. Returning None lets the caller
        # drop the article; guessing would silently corrupt the pre/post split.
        assert parse_rss_datetime("not a date", None) is None
        assert parse_rss_datetime(None, None) is None

    def test_result_is_never_naive(self):
        for raw in ["Fri, 29 Aug 2026 12:31:05 GMT", "Fri, 29 Aug 2026 12:31:05 -0400"]:
            got = parse_rss_datetime(raw, None)
            assert got is not None and got.tzinfo is not None


def _epoch(*wall) -> int:
    """The epoch Finnhub sends: Eastern wall-clock digits labelled as UTC."""
    return int(datetime(*wall, tzinfo=UTC).timestamp())


class TestWallclockEpoch:
    """Finnhub company-news stamps Eastern wall-clock time as if it were UTC.
    Read naively, every article lands 4-5h early and post-move commentary is
    filed as pre-move evidence."""

    def test_summer_is_four_hours_later(self):
        # CNBC 'Morning Squawk': CNBC's RSS says 13:06:11 GMT; Finnhub sent 09:06:11.
        got = from_wallclock_epoch(_epoch(2026, 9, 11, 9, 6, 11), "America/New_York")
        assert got == datetime(2026, 9, 11, 13, 6, 11, tzinfo=UTC)

    def test_winter_is_five_hours_not_a_flat_offset(self):
        # A flat +4h left SBUX/QCOM winter reaction stories BEFORE their 8-K.
        got = from_wallclock_epoch(_epoch(2026, 1, 28, 7, 59), "America/New_York")
        assert got == datetime(2026, 1, 28, 12, 59, tzinfo=UTC)

    def test_result_is_aware_utc(self):
        got = from_wallclock_epoch(_epoch(2025, 10, 21, 16, 27), "America/New_York")
        assert got.tzinfo == UTC and assert_utc(got) == got
