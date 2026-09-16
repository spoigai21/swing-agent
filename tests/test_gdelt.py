"""GDELT parsing. Fixtures are real shapes from gdelt-bq.gdeltv2.gkg_partitioned."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from swing.ingest.gdelt import domains, org_pattern, page_title, parse_ts


def test_timestamp_is_utc_to_the_quarter_hour():
    # CNBC's Qualcomm AI-chip story, as GDELT recorded it.
    assert parse_ts(20251027153000) == datetime(2025, 10, 27, 15, 30, tzinfo=UTC)
    assert parse_ts("20260423200000") == datetime(2026, 4, 23, 20, 0, tzinfo=UTC)


class TestPageTitle:
    def test_plain_title(self):
        extras = "<PAGE_PRECISEPUBTIMESTAMP>2025</PAGE_PRECISEPUBTIMESTAMP>" \
                 "<PAGE_TITLE>Qualcomm announces AI chips to compete with AMD and Nvidia</PAGE_TITLE>"
        assert page_title(extras) == "Qualcomm announces AI chips to compete with AMD and Nvidia"

    def test_cdata_and_html_entities_are_unwrapped(self):
        extras = "<PAGE_TITLE><![CDATA[Netflix &amp; WBD amend deal\n  to all-cash]]></PAGE_TITLE>"
        assert page_title(extras) == "Netflix & WBD amend deal to all-cash"

    def test_missing_title_is_none(self):
        assert page_title("<PAGE_LINKS>http://x</PAGE_LINKS>") is None
        assert page_title(None) is None


class TestOrgPattern:
    def test_uses_the_short_company_name_not_the_ticker(self):
        # GDELT names organisations, never tickers, and writes "Qualcomm" —
        # matching "QUALCOMM INC" returned 0 rows for its own launch story.
        assert org_pattern("QCOM") == "QUALCOMM"
        assert org_pattern("NVDA") == "NVIDIA"
        assert org_pattern("AAPL") == "APPLE"

    def test_punctuation_is_dropped_so_take_two_matches(self):
        # GDELT writes "Take Two Interactive", the watchlist "Take-Two".
        pattern = org_pattern("TTWO")
        assert "-" not in pattern and pattern.isupper()


def test_only_configured_publishers_are_ingested():
    mapping = domains()
    assert mapping["reuters.com"] == "reuters" and mapping["bloomberg.com"] == "bloomberg"
    assert "wsj.com" in mapping and "ft.com" in mapping
    # Aggregators GDELT indexes but this stack excludes.
    assert "finance.yahoo.com" not in mapping and "benzinga.com" not in mapping


class TestCostControl:
    """BigQuery bills bytes scanned, so an unguarded loop spends real money."""

    def test_budget_stops_querying_once_the_month_is_spent(self, tmp_path, monkeypatch):
        from swing.ingest import gdelt

        monkeypatch.setattr(gdelt, "USAGE", tmp_path / "usage.json")
        monkeypatch.setattr(gdelt, "budget_gb", lambda: 10.0)
        assert gdelt.affordable() is True
        gdelt.record(9.0)
        assert gdelt.affordable() is True
        gdelt.record(2.0)
        assert gdelt.affordable() is False
        assert gdelt.spent_gb("2099-01") == 0.0   # a new month starts at zero

    def test_one_query_covers_every_ticker(self):
        # A 12-ticker query dry-runs at the same 0.40 GB as a 1-ticker query, so
        # a per-ticker loop billed 12x for identical bytes.
        import inspect

        from swing.ingest import gdelt

        assert "REGEXP_CONTAINS" in inspect.getsource(gdelt.fetch_window)
        assert "for ticker in stocks()" not in inspect.getsource(gdelt.poll)


class TestRowMapping:
    def _row(self, orgs, title="Nvidia and Qualcomm strike a deal"):
        return {"DATE": 20251027153000, "SourceCommonName": "reuters.com",
                "DocumentIdentifier": "https://www.reuters.com/x",
                "Extras": f"<PAGE_TITLE>{title}</PAGE_TITLE>", "orgs": orgs}

    def _patterns(self, *tickers):
        from swing.ingest.gdelt import org_pattern

        return {t: org_pattern(t) for t in tickers}

    def test_a_story_about_two_companies_is_tagged_with_both(self):
        # insert_many dedupes on URL: under the old per-ticker loop whichever
        # ticker queried first claimed this story and the other never saw it.
        from swing.ingest.gdelt import _to_raw

        raw = _to_raw(self._row("NVIDIA;QUALCOMM INC;FOO CORP"),
                      self._patterns("NVDA", "QCOM", "AAPL"))
        assert raw.raw["feed_tickers"] == ["NVDA", "QCOM"]
        # normalize.resolve_tickers treats a bare `ticker` as the ONLY company.
        assert "ticker" not in raw.raw

    def test_publisher_name_is_stored_so_tiering_works(self):
        from swing.ingest.gdelt import _to_raw

        raw = _to_raw(self._row("NVIDIA"), self._patterns("NVDA"))
        assert raw.source == "reuters"
        assert raw.raw["timestamp_precision"] == "15min"

    def test_a_row_naming_no_watchlist_company_is_dropped(self):
        from swing.ingest.gdelt import _to_raw

        assert _to_raw(self._row("TESLA INC"), self._patterns("NVDA")) is None


def test_related_companies_get_a_real_name_not_the_bare_ticker():
    # Retrieval searches related companies too. GDELT writes "ALPHABET INC" and
    # never "GOOGL", so falling back to the ticker matches nothing at all.
    from swing.ingest.gdelt import org_pattern

    pattern = org_pattern("GOOGL")
    assert pattern != "GOOGL"
    assert pattern.replace(" ", "").isalpha() and pattern.isupper()


class TestOnDemandRefresh:
    """The interactive path asks per question; bytes are billed per query."""

    def test_an_already_stored_window_costs_nothing(self, monkeypatch):
        from swing.ingest import gdelt

        monkeypatch.setattr(gdelt, "covered", lambda s, e: True)
        monkeypatch.setattr(gdelt, "fetch_window",
                            lambda *a, **k: pytest.fail("must not rescan a stored window"))
        assert gdelt.refresh(["NVDA"], datetime(2026, 3, 5, tzinfo=UTC),
                             datetime(2026, 3, 7, tzinfo=UTC)) == 0

    def test_an_uncovered_window_is_fetched(self, monkeypatch):
        from swing.ingest import gdelt

        seen = {}
        monkeypatch.setattr(gdelt, "covered", lambda s, e: False)
        monkeypatch.setattr(gdelt, "fetch_window",
                            lambda tickers, s, e: seen.setdefault("tickers", tickers) and 0 or 7)
        assert gdelt.refresh(["NVDA", "GOOGL"], datetime(2026, 3, 5, tzinfo=UTC),
                             datetime(2026, 3, 7, tzinfo=UTC)) == 7
        assert seen["tickers"] == ["NVDA", "GOOGL"]


def test_the_question_path_refreshes_wire_copy():
    # Gate 2's misses are missing sources; the interactive path must not answer
    # from news that predates the collector's last 4-hourly GDELT pass.
    import inspect

    from swing.interface import explain

    assert "gdelt.refresh" in inspect.getsource(explain._refresh_news)
