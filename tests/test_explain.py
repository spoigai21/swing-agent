"""The one-question path: `swing` -> "why is NVDA down?" -> the reason.

Parsing and wording are pure, so they are tested exhaustively. The pipeline
itself (prices, news, Gemini) is exercised live, not mocked.
"""
from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from swing.analysis.decompose import Decomposition
from swing.interface import explain as ex

FRI = date(2026, 9, 11)


def _et(*args) -> datetime:
    return datetime(*args, tzinfo=ex.ET)


class TestParse:
    @pytest.mark.parametrize("q, ticker", [
        ("why is NVDA down?", "NVDA"),
        ("why is nvda going up", "NVDA"),
        ("what happened to Tesla", "TSLA"),
        ("why did Micron drop", "MU"),
        ("why did Google rise", "GOOGL"),
        ("MU", "MU"),
    ])
    def test_finds_the_stock(self, q, ticker):
        assert ex.parse(q, FRI).ticker == ticker

    def test_the_word_mu_is_not_micron(self):
        assert ex.parse("the mu variant is spreading", FRI).ticker is None

    def test_unknown_symbol_is_named(self):
        q = ex.parse("why is AMD down", FRI)
        assert q.ticker is None and q.unknown_symbol == "AMD"

    def test_ordinary_capitals_are_not_symbols(self):
        assert ex.parse("why is AI hot", FRI).unknown_symbol is None

    def test_no_date_means_latest_session(self):
        assert ex.parse("why is NVDA down", FRI).day is None

    def test_iso_date(self):
        assert ex.parse("why did MU drop on 2026-08-27", FRI).day == date(2026, 8, 27)

    def test_month_and_day(self):
        assert ex.parse("why did AAPL drop on Aug 27", FRI).day == date(2026, 8, 27)

    def test_yesterday_skips_the_weekend(self):
        monday = date(2026, 9, 14)
        assert ex.parse("why did NVDA fall yesterday", monday).day == FRI

    def test_weekday_name_is_the_most_recent_one(self):
        assert ex.parse("what happened to TSLA on tuesday", FRI).day == date(2026, 9, 8)


class TestSessions:
    def test_weekend_means_friday(self):
        assert ex.last_completed_session(_et(2026, 9, 13, 12, 0)) == FRI

    def test_before_the_bar_settles_means_the_previous_day(self):
        assert ex.last_completed_session(_et(2026, 9, 11, 15, 0)) == date(2026, 9, 10)

    def test_after_the_close_means_today(self):
        assert ex.last_completed_session(_et(2026, 9, 11, 16, 30)) == FRI

    def test_monday_morning_means_friday(self):
        assert ex.last_completed_session(_et(2026, 9, 14, 9, 0)) == FRI

    def test_market_hours(self):
        assert ex.market_open(_et(2026, 9, 11, 10, 0))
        assert not ex.market_open(_et(2026, 9, 13, 10, 0))


def _dec(ret, mkt, sec, resid, z, ticker="NVDA"):
    return Decomposition(ticker, FRI, ret, mkt, sec, resid, z, 1.8, 1.1, 0.6, 0.5, "ok")


class TestWording:
    def test_header_splits_the_move(self):
        h = ex.header(_dec(-0.031, -0.004, -0.009, -0.018, -2.4), "SMH")
        assert "NVDA fell 3.1% on Fri Sep 11." in h
        assert "market -0.4% · chip stocks -0.9% · NVDA on its own -1.8%" in h

    def test_normal_day_driven_by_the_market_says_so(self):
        t = ex.normal_day(_dec(-0.02, -0.012, -0.006, -0.002, -0.3), "SMH")
        assert "mostly the market and chip stocks" in t

    def test_normal_day_on_its_own_says_nothing_unusual(self):
        t = ex.normal_day(_dec(0.012, 0.001, 0.001, 0.010, 1.1), "SMH")
        assert "nothing unusual" in t

    def test_a_flat_day_is_called_flat(self):
        assert "NVDA was flat on Fri Sep 11." in ex.header(_dec(-0.0001, 0.016, -0.002,
                                                               -0.0141, -0.8), "SMH")

    def test_lagging_a_rising_market_is_not_called_mostly_the_market(self):
        # NVDA 2026-09-11: flat while the market rose 1.6%. The first version said
        # "mostly the market", which read as the market dragging it down.
        t = ex.normal_day(_dec(-0.0001, 0.016, -0.002, -0.0141, -0.8), "SMH")
        assert "NVDA lagged the market and chip stocks by 1.4%" in t
        assert "mostly" not in t

    def test_unexplained_never_offers_a_reason(self):
        from swing.agent.abstention import NOTE_BASE

        t = ex.render_verdict("unexplained", {"candidates": []},
                              NOTE_BASE + "Volume was unremarkable.", {})
        assert "no news published before the move explains it" in t
        assert NOTE_BASE.strip() not in t and "Volume was unremarkable." in t

    def test_explained_shows_when_and_where_the_evidence_came_from(self):
        payload = {"candidates": [{"catalyst": "Q3 earnings miss", "confidence": "high",
                                   "evidence": [{"cluster_id": 7}]}]}
        details = {7: {"earliest_published": datetime(2025, 10, 21, 20, 2, tzinfo=UTC),
                       "source": "sec-edgar", "headline": "NFLX 8-K — Item 2.02",
                       "url": "https://www.sec.gov/x"}}
        t = ex.render_verdict("explained", payload, None, details)
        assert "Why: Q3 earnings miss" in t
        assert "Tue Oct 21, 4:02pm ET · SEC filing · NFLX 8-K" in t
        assert "Confidence: high" in t

    def test_evidence_we_do_not_hold_is_never_shown(self):
        payload = {"candidates": [{"catalyst": "x", "confidence": "low",
                                   "evidence": [{"cluster_id": 999}]}]}
        t = ex.render_verdict("partially_explained", payload, None, {})
        assert "•" not in t and "Partly explained" in t


class TestRouting:
    def test_forecast_is_refused_before_anything_runs(self, monkeypatch):
        monkeypatch.setattr(ex, "explain", lambda *a, **k: pytest.fail("must not run"))
        assert "not a price predictor" in ex.respond("should I buy NVDA?")

    def test_stock_question_is_explained(self, monkeypatch):
        monkeypatch.setattr(ex, "explain", lambda t, d, **k: f"explained {t} {d}")
        now = datetime(2026, 9, 13, 16, 0, tzinfo=UTC)
        assert ex.respond("why is NVDA down?", now=now) == "explained NVDA None"

    def test_uncovered_stock_says_what_is_covered(self):
        assert "only cover these stocks" in ex.respond("why is AMD down?")


def test_model_calls_cannot_hang_a_question():
    """The Gemini client defaults to no timeout and 6 retries; when the model
    stopped answering, a question hung instead of failing cleanly."""
    import inspect

    from swing.agent import llm

    src = inspect.getsource(llm.get_llm)
    assert "timeout=60" in src and "max_retries=1" in src


def test_overload_is_retried_like_a_rate_limit():
    # One 503 "high demand" spike used to fail the whole question.
    from swing.agent import llm

    assert llm._is_transient(RuntimeError("503 UNAVAILABLE. high demand"))
    assert llm._is_transient(RuntimeError("429 RESOURCE_EXHAUSTED"))
    assert not llm._is_transient(ValueError("schema validation failed"))


def test_failed_model_call_is_not_stored_as_a_verdict(monkeypatch):
    """A quota error used to be persisted as `unexplained`, which scored as a
    correct abstention in the placebo test and was served as a real answer."""
    from swing.agent import nodes
    from swing.store import session

    monkeypatch.setattr(session, "connect", lambda *a, **k: pytest.fail("must not write"))
    assert nodes.persist({"attribution": object(), "verdict_reason": "llm_error"}) == {}


def test_no_sampling_parameters_are_sent_to_gemini():
    """Gemini 3.x ignores temperature/top_p/top_k and later models 400 on them.

    Passing them also made the code claim a determinism it never had: repeated
    questions can be worded differently, so attribution reuse is a quota cache.
    """
    import inspect

    from swing.agent import llm

    src = inspect.getsource(llm.get_llm)
    body = src.split('"""')[-1]          # ignore the docstring, which explains why
    for param in ("temperature=", "top_p=", "top_k="):
        assert param not in body, f"{param} is ignored by Gemini 3.x; do not send it"


def test_the_question_path_scopes_the_sec_poll():
    """Unfiltered, edgar.poll makes one HTTP request per CIK and was ~11s of a
    question's latency, re-polling companies the question is not about. The
    collector still polls everything every 600s, so nothing loses coverage."""
    import inspect

    from swing.ingest import edgar
    from swing.interface import explain

    assert "tickers=[ticker, *related(ticker)]" in inspect.getsource(explain._refresh_news)
    assert "tickers" in inspect.signature(edgar.poll).parameters


class TestDriftDaysAreNotCalledOrdinary:
    """Detection flags a drift on a CUMULATIVE z over 3-5 days
    (drift_z_threshold 2.5); explain judges the SINGLE day (z_threshold 2.0).
    Both are right, and together they let the product say "nothing unusual"
    about a date that is in the swings table and may already have alerted.
    MRVL 2026-09-02: 1.2x typical on the day, z=-2.55 across the run.
    """

    def _patch(self, monkeypatch, row):
        import contextlib

        from swing.store import session

        class _Conn:
            def execute(self, *a, **k):
                return self

            def fetchone(self):
                return row

        @contextlib.contextmanager
        def _c(*a, **k):
            yield _Conn()

        monkeypatch.setattr(session, "connect", _c)

    def test_a_day_inside_a_flagged_run_says_so(self, monkeypatch):
        from datetime import date

        from swing.interface import explain

        self._patch(monkeypatch, {"d": date(2026, 9, 2), "drift_window": 5,
                                  "total_return": -0.0188, "residual_z": -2.55})
        out = explain._drift_context("MRVL", date(2026, 9, 2))
        assert out
        assert "5-day run" in out and "-1.9% cumulative" in out and "sigma" in out

    def test_an_ordinary_day_outside_any_run_adds_nothing(self, monkeypatch):
        from datetime import date

        from swing.interface import explain

        self._patch(monkeypatch, None)
        assert explain._drift_context("MRVL", date(2026, 9, 2)) is None

    def test_the_normal_day_branch_consults_it(self):
        import inspect

        from swing.interface import explain

        assert "_drift_context" in inspect.getsource(explain.explain)
