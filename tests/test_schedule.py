"""The collector's once-a-day jobs: post-close run + alerts, nightly placebo."""
from __future__ import annotations

from datetime import date, datetime

from swing.interface import schedule as s


def _at(zone, *args) -> datetime:
    return datetime(*args, tzinfo=zone)


class TestDue:
    def test_not_before_the_time(self):
        assert s.due(_at(s.ET, 2026, 9, 14, 16, 30), s.ET, s.DAILY_AT_ET, None, True) is None

    def test_runs_after_the_time_for_that_local_day(self):
        now = _at(s.ET, 2026, 9, 14, 16, 50)
        assert s.due(now, s.ET, s.DAILY_AT_ET, None, True) == date(2026, 9, 14)

    def test_once_per_day(self):
        now = _at(s.ET, 2026, 9, 14, 20, 0)
        assert s.due(now, s.ET, s.DAILY_AT_ET, date(2026, 9, 14), True) is None

    def test_post_close_run_skips_weekends(self):
        assert s.due(_at(s.ET, 2026, 9, 13, 18, 0), s.ET, s.DAILY_AT_ET, None, True) is None

    def test_placebo_runs_every_night_in_pacific_time(self):
        # 07:40 UTC on a Sunday is 00:40 Pacific.
        now = _at(s.ET, 2026, 9, 13, 3, 40)
        assert s.due(now, s.PT, s.PLACEBO_AT_PT, None, False) == date(2026, 9, 13)

    def test_marker_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(s, "DATA", tmp_path)
        assert s.last_run("daily") is None
        s.mark("daily", date(2026, 9, 14))
        assert s.last_run("daily") == date(2026, 9, 14)


def test_placebo_stops_spending_quota_once_gate_4_has_its_200(tmp_path, monkeypatch):
    import pytest

    from swing.eval import placebo

    monkeypatch.setattr(s, "DATA", tmp_path)
    monkeypatch.setattr(placebo, "cumulative", lambda: {"n": 200, "confabulated": 3, "rate": 0.015})
    monkeypatch.setattr(placebo, "run", lambda **k: pytest.fail("must not spend quota"))
    assert s.placebo_if_due(_at(s.PT, 2026, 9, 15, 0, 40)) == 0


def test_last_night_tops_up_to_exactly_200(tmp_path, monkeypatch):
    from swing.eval import placebo

    asked = {}

    def fake_run(n, **kwargs):
        asked["n"] = n
        return {"n": n}

    monkeypatch.setattr(s, "DATA", tmp_path)
    monkeypatch.setattr(placebo, "cumulative", lambda: {"n": 190, "confabulated": 1, "rate": 0.005})
    monkeypatch.setattr(placebo, "run", fake_run)
    s.placebo_if_due(_at(s.PT, 2026, 9, 15, 0, 40))
    assert asked["n"] == 10


def test_the_collector_runs_both_jobs():
    from swing.ingest.collector import build_jobs

    names = {j.name for j in build_jobs()}
    assert {"scheduled-daily", "scheduled-placebo"} <= names


def test_every_source_has_a_collector_job():
    # A job referencing a module the collector never imported raises NameError
    # at startup, and the whole collector fails to start.
    from swing.ingest.collector import build_jobs

    names = {j.name for j in build_jobs()}
    assert {"sec-edgar", "sec-edgar-text", "finnhub-news", "analyst-ratings",
            "gdelt", "normalize", "health-check"} <= names


def test_scheduled_attribution_spends_quota_only_on_recent_swings():
    import inspect

    from swing.interface import batch

    assert "s.d >=" in inspect.getsource(batch._attribute)
