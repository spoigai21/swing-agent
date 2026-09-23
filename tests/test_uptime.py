"""An hour the collector misses is coverage gone for good.

RSS serves only the most recent 20-50 items, so downtime cannot be backfilled.
On a laptop that makes uptime the largest single input to catalyst coverage —
and it was invisible: measured on 2026-09-22, 144 of the previous 337 hours
collected nothing, including one stretch of 101 hours.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from swing.ingest import uptime as U


def hours(pattern: str, start=datetime(2026, 9, 1, 12, tzinfo=UTC)):
    """'#' collected, '.' silent."""
    return [U.Hour(start + timedelta(hours=i), 0 if ch == "." else 5)
            for i, ch in enumerate(pattern)]


class TestGapDetection:
    def test_the_longest_silent_run_is_found(self):
        n, start = U.longest_gap(hours("##...#.#....#"))
        assert n == 4
        assert start == datetime(2026, 9, 1, 20, tzinfo=UTC)   # index 8 of the pattern

    def test_a_gap_at_the_very_end_still_counts(self):
        """A collector that is down right now is the case that matters most."""
        n, _ = U.longest_gap(hours("####...."))
        assert n == 4

    def test_no_gaps_reports_zero_rather_than_failing(self):
        assert U.longest_gap(hours("####")) == (0, None)

    def test_an_empty_history_is_not_a_crash(self):
        assert U.longest_gap([]) == (0, None)


class TestSummary:
    def test_uptime_is_the_share_of_hours_that_collected(self, monkeypatch):
        monkeypatch.setattr(U, "hourly", lambda days=14: hours("###."))
        s = U.summary()
        assert s["hours"] == 4 and s["silent"] == 1
        assert abs(s["uptime"] - 0.75) < 1e-9

    def test_a_silent_hour_is_one_with_no_rows(self):
        assert U.Hour(datetime(2026, 9, 1, tzinfo=UTC), 0).silent
        assert not U.Hour(datetime(2026, 9, 1, tzinfo=UTC), 1).silent

    def test_naturally_quiet_hours_are_marked_separately(self):
        """3am on a Sunday is not an outage. The report separates them instead
        of declaring downtime from an absence of news."""
        quiet = U.Hour(datetime(2026, 9, 1, 7, tzinfo=UTC), 0)
        busy = U.Hour(datetime(2026, 9, 1, 14, tzinfo=UTC), 0)
        assert quiet.quiet_by_nature and not busy.quiet_by_nature

    def test_the_sparkline_is_one_character_per_hour(self):
        assert U.sparkline(hours("#.#.")) == "#.#."


class TestTheCollectorRecordsItsOwnDowntime:
    def test_startup_warns_about_a_gap(self):
        """A collector that has been asleep looks identical to one that has been
        running, once it is back up."""
        import inspect

        from swing.ingest import collector

        src = inspect.getsource(collector.run)
        assert "gap_since_last_row" in src
        assert "swing uptime" in src

    def test_a_failed_diagnostic_never_stops_collection(self):
        import inspect

        from swing.ingest import collector

        src = inspect.getsource(collector.run)
        gap_block = src[src.index("gap_since_last_row"):]
        assert "except Exception" in gap_block


class TestWiredIntoTheCli:
    def test_uptime_is_a_command_with_a_window(self):
        from swing.cli import build_parser

        assert build_parser().parse_args(["uptime"]).days == 14
        assert build_parser().parse_args(["uptime", "--days", "30"]).days == 30
