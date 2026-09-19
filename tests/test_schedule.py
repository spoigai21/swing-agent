"""The collector's once-a-day jobs: post-close run + alerts, nightly placebo."""
from __future__ import annotations

from datetime import date, datetime, timedelta

from swing.interface import schedule as s


def _at(zone, *args) -> datetime:
    return datetime(*args, tzinfo=zone)


def _after_placebo_time(year: int, month: int, day: int) -> datetime:
    """Ten minutes past whatever PLACEBO_AT_PT currently is.

    The slot moves (00:30 normally, 21:00 during the Gate 4 push); a fixture
    that hard-codes 00:40 silently stops exercising the code it was written for.
    """
    return datetime(year, month, day, s.PLACEBO_AT_PT.hour, s.PLACEBO_AT_PT.minute,
                    tzinfo=s.PT) + timedelta(minutes=10)


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

    def test_placebo_runs_every_night_including_weekends(self):
        """Time-of-day is a knob (00:30 normally, 21:00 during the Gate 4 push),
        so the fixture is built FROM the constant rather than hard-coding it.
        What must not change is that placebo runs on weekends too."""
        sunday = datetime(2026, 9, 13, s.PLACEBO_AT_PT.hour, s.PLACEBO_AT_PT.minute,
                          tzinfo=s.PT) + timedelta(minutes=10)
        assert s.due(sunday, s.PT, s.PLACEBO_AT_PT, None, False) == date(2026, 9, 13)

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
    assert s.placebo_if_due(_after_placebo_time(2026, 9, 15)) == 0


def test_last_night_tops_up_to_exactly_200(tmp_path, monkeypatch):
    from swing.agent import llm
    from swing.eval import placebo

    asked = {}

    def fake_run(n, **kwargs):
        asked["n"] = n
        return {"n": n}

    monkeypatch.setattr(s, "DATA", tmp_path)
    monkeypatch.setattr(placebo, "cumulative", lambda: {"n": 190, "confabulated": 1, "rate": 0.005})
    monkeypatch.setattr(placebo, "run", fake_run)
    monkeypatch.setattr(llm, "remaining_today", lambda: llm.DAILY_QUOTA)
    s.placebo_if_due(_after_placebo_time(2026, 9, 15))
    assert asked["n"] == 10


def test_a_burnt_day_shortens_the_batch_instead_of_spending_the_tail(tmp_path, monkeypatch):
    """Failed calls spend quota without storing a row. If the night has only 8
    requests left, the batch must be 8 - INTERACTIVE_RESERVE, not the constant
    12, or the tail of the run walks into 429 RESOURCE_EXHAUSTED."""
    from swing.agent import llm
    from swing.eval import placebo

    asked = {}

    def fake_run(n, **kwargs):
        asked["n"] = n
        return {"n": n}

    monkeypatch.setattr(s, "DATA", tmp_path)
    monkeypatch.setattr(placebo, "cumulative", lambda: {"n": 0, "confabulated": 0, "rate": None})
    monkeypatch.setattr(placebo, "run", fake_run)
    monkeypatch.setattr(llm, "remaining_today", lambda: 8)
    s.placebo_if_due(_after_placebo_time(2026, 9, 15))
    assert asked["n"] == s.NIGHTLY_PLACEBO_CASES, "the ledger no longer caps the batch"


def test_a_ledger_reading_zero_does_not_skip_the_night(tmp_path, monkeypatch):
    """The ledger is advisory. It has read 23 on a 20-request day and 7 when the
    cap was already reached, so trusting it to skip costs 20 real cases on a bad
    estimate. Attempting when the quota is gone costs two refused calls, which
    Google does not charge and placebo.run stops after."""
    from swing.agent import llm
    from swing.eval import placebo

    asked = {}

    def fake_run(n, **kwargs):
        asked["n"] = n
        return {"n": 0, "note": "quota"}

    monkeypatch.setattr(s, "DATA", tmp_path)
    monkeypatch.setattr(placebo, "cumulative", lambda: {"n": 0, "confabulated": 0, "rate": None})
    monkeypatch.setattr(placebo, "run", fake_run)
    monkeypatch.setattr(llm, "remaining_today", lambda: 0)
    s.placebo_if_due(_after_placebo_time(2026, 9, 15))
    assert asked["n"] == s.NIGHTLY_PLACEBO_CASES, "must still attempt; the API decides"


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


class TestQuotaLeavesRoomForQuestions:
    """The scheduler and the user draw on ONE 20-request daily budget."""

    def test_scheduled_work_never_consumes_the_whole_day(self):
        assert (s.NIGHTLY_PLACEBO_CASES + s.DAILY_ATTRIBUTIONS
                + s.INTERACTIVE_RESERVE) <= s.DAILY_MODEL_QUOTA

    def test_questions_are_protected_by_reserve_or_by_timing(self):
        """Either hold quota back, or run the batch after the day is over.

        The original bug was placebo 17 + daily 3 == 20 fired at 00:30, so every
        question answered `model_unavailable` by breakfast. During the Gate 4
        push the reserve is 0 and the protection is the 21:00 slot instead; both
        arrangements are acceptable, having neither is not.
        """
        late_enough = s.PLACEBO_AT_PT.hour >= 17
        assert s.INTERACTIVE_RESERVE >= 3 or late_enough, (
            "with no reserve, placebo must run after the trading day")


class TestQuotaLedgerCountsAttempts:
    """The attributions table stores only SUCCESSFUL calls, so it undercounts
    the day: a failed call plus retries spends quota and leaves no row. On
    2026-09-16 that made the DB read "10 left" when 0 remained, and the run
    walked straight into 429 RESOURCE_EXHAUSTED."""

    def _ledger(self, tmp_path, monkeypatch):
        from swing.agent import llm

        monkeypatch.setattr(llm, "QUOTA_PATH", tmp_path / "gemini_usage.json")
        return llm

    def test_a_failed_attempt_still_costs_quota(self, tmp_path, monkeypatch):
        llm = self._ledger(tmp_path, monkeypatch)
        assert llm.spent_today() == 0
        for _ in range(3):           # one call that failed and retried twice
            llm.record_call()
        assert llm.spent_today() == 3
        assert llm.remaining_today() == llm.DAILY_QUOTA - 3

    def test_remaining_never_goes_negative(self, tmp_path, monkeypatch):
        llm = self._ledger(tmp_path, monkeypatch)
        llm.record_call(llm.DAILY_QUOTA + 5)
        assert llm.remaining_today() == 0

    def test_invoke_counts_the_attempt_before_calling(self):
        import inspect

        from swing.agent import llm

        src = inspect.getsource(llm.invoke_with_retry)
        assert "record_call()" in src
        assert src.index("record_call()") < src.index("runnable.invoke")

    def test_the_ledger_is_advisory_not_a_gate(self):
        """This test previously pinned the opposite rule — that the batch was
        capped by a `budget` derived from the ledger. That rule was reversed
        (16.29) after the ledger proved wrong in both directions: it read 23 on
        a 20-request day, then 7 when the cap was already reached. It is now
        read for the log only; Google's 429 decides what is left."""
        import inspect

        src = inspect.getsource(s.placebo_if_due)
        assert "remaining_today" in src, "still worth logging what the ledger thinks"
        assert "advisory" in src, "its advisory status must be stated at the call site"
        assert "budget" not in src, "the ledger must not cap or skip the batch"


class TestRefusedRequestsAreNotCharged:
    """Google rejects over-quota calls without charging them. Counting those as
    spend read 36 attempts on a 20-request day, and the nightly batch sizes
    itself from remaining_today() — so it would skip capacity Gate 4 had."""

    def _ledger(self, tmp_path, monkeypatch):
        from swing.agent import llm

        monkeypatch.setattr(llm, "QUOTA_PATH", tmp_path / "usage.json")
        return llm

    def test_a_quota_rejection_is_recognised(self):
        from swing.agent import llm

        assert llm.is_quota_rejection(RuntimeError("429 RESOURCE_EXHAUSTED ... limit: 20"))
        assert not llm.is_quota_rejection(RuntimeError("503 UNAVAILABLE high demand"))

    def test_a_refused_call_leaves_the_ledger_unchanged(self, tmp_path, monkeypatch):
        import pytest

        llm = self._ledger(tmp_path, monkeypatch)

        class _Refused:
            def invoke(self, _prompt):
                raise RuntimeError("429 RESOURCE_EXHAUSTED for gemini-3.6-flash")

        before = llm.spent_today()
        with pytest.raises(RuntimeError):
            llm.invoke_with_retry(_Refused(), "x")
        assert llm.spent_today() == before, "a refused request must not count as spend"

    def test_a_served_call_counts(self, tmp_path, monkeypatch):
        llm = self._ledger(tmp_path, monkeypatch)

        class _Ok:
            def invoke(self, _prompt):
                return "answer"

        before = llm.spent_today()
        assert llm.invoke_with_retry(_Ok(), "x") == "answer"
        assert llm.spent_today() == before + 1

    def test_the_ledger_never_goes_negative(self, tmp_path, monkeypatch):
        llm = self._ledger(tmp_path, monkeypatch)
        llm.record_call(-5)
        assert llm.spent_today() == 0


class TestDailyCapIsNotWorthRetrying:
    """A per-minute refusal clears in seconds; a per-day one clears at midnight.
    Treating them alike burned three attempts per failure on a 20-request
    budget — 2026-09-19 stored 7 cases from a full day."""

    def test_a_daily_cap_is_not_transient(self):
        from swing.agent import llm

        exc = RuntimeError(
            "429 RESOURCE_EXHAUSTED ... 'quotaId': "
            "'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaValue': '20'")
        assert llm.is_daily_cap(exc)
        assert not llm._is_transient(exc), "retrying until midnight is futile"

    def test_a_per_minute_refusal_is_still_retried(self):
        from swing.agent import llm

        exc = RuntimeError(
            "429 RESOURCE_EXHAUSTED ... 'quotaId': "
            "'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', 'quotaValue': '5'")
        assert not llm.is_daily_cap(exc)
        assert llm._is_transient(exc), "RPM clears in seconds and is worth waiting for"

    def test_overload_is_still_retried(self):
        from swing.agent import llm

        assert llm._is_transient(RuntimeError("503 UNAVAILABLE high demand"))


class TestAbstentionIsQueuedBeforePlacebo:
    """13 calls converts the only gate metric that has never produced a number.
    It must get first claim on the fresh quota, or the 21:00 placebo batch takes
    the whole day and it waits forever."""

    def test_it_runs_before_the_placebo_batch(self):
        assert s.ABSTENTION_AT_PT < s.PLACEBO_AT_PT

    def test_it_is_a_no_op_once_nothing_is_pending(self, tmp_path, monkeypatch):
        import pytest

        from swing.eval import abstention

        monkeypatch.setattr(s, "DATA", tmp_path)
        monkeypatch.setattr(abstention, "pending", list)
        monkeypatch.setattr(abstention, "run",
                            lambda **k: pytest.fail("must not spend quota when done"))
        now = datetime(2026, 9, 20, s.ABSTENTION_AT_PT.hour, s.ABSTENTION_AT_PT.minute,
                       tzinfo=s.PT) + timedelta(minutes=10)
        assert s.abstention_if_due(now) == 0

    def test_it_runs_when_swings_are_pending(self, tmp_path, monkeypatch):
        from swing.eval import abstention

        called = {}
        monkeypatch.setattr(s, "DATA", tmp_path)
        monkeypatch.setattr(abstention, "pending", lambda: [1, 2, 3])
        monkeypatch.setattr(abstention, "run",
                            lambda **k: called.setdefault("ran", True) or {"attributed": 3})
        now = datetime(2026, 9, 20, s.ABSTENTION_AT_PT.hour, s.ABSTENTION_AT_PT.minute,
                       tzinfo=s.PT) + timedelta(minutes=10)
        s.abstention_if_due(now)
        assert called.get("ran"), "pending swings must be attributed"

    def test_the_collector_actually_runs_it(self):
        from swing.ingest.collector import build_jobs

        assert "scheduled-abstention" in {j.name for j in build_jobs()}
