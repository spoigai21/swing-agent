"""Analyst actions and other companies' news: the two source gaps measured on
2026-09-13, when 15 of 33 researched causes were news swing never held."""
from __future__ import annotations

from datetime import UTC, datetime

from swing.ingest import analyst
from swing.ingest.config import (
    company_name,
    edgar_targets,
    related,
    related_companies,
    stocks,
)


class TestAnalystHeadline:
    def test_upgrade_names_both_grades(self):
        h = analyst.headline("TSLA", "Tesla", "UBS", "up", "Neutral", "Sell",
                             "Maintains", 352.0, 352.0)
        assert h == "UBS upgrades Tesla (TSLA) to Neutral from Sell; price target $352"

    def test_target_change_shows_new_and_old(self):
        h = analyst.headline("TSLA", "Tesla", "TD Cowen", "main", "Buy", "Buy",
                             "Lowers", 490.0, 519.0)
        assert h == "TD Cowen maintains Buy on Tesla (TSLA); lowers price target to $490 from $519"

    def test_initiation(self):
        h = analyst.headline("TTWO", "Take-Two", "Piper Sandler", "init", "Overweight", "",
                             "Announces", 280.0, 0.0)
        assert h == "Piper Sandler initiates Take-Two (TTWO) at Overweight; price target $280"

    def test_missing_target_is_left_out(self):
        h = analyst.headline("AAPL", "Apple", "Needham", "reit", "Hold", "Hold", "",
                             0.0, float("nan"))
        assert h == "Needham reiterates Hold on Apple (AAPL)"

    def test_cents_are_kept(self):
        h = analyst.headline("AAPL", "Apple", "Jefferies", "main", "Hold", "Hold",
                             "Raises", 286.54, 276.47)
        assert h.endswith("raises price target to $286.54 from $276.47")


class TestWireCopy:
    """Reuters and Bloomberg stories reach us relabelled "Yahoo" (tier 4). The
    wire's own dateline identifies them; nothing weaker is trusted."""

    @staticmethod
    def _row(summary):
        return {"source": "yahoo", "summary": summary,
                "raw": {"publisher": "Yahoo", "via": "finnhub"}}

    def test_reuters_dateline_is_tier_two(self):
        from swing.ingest.normalize import resolve_tier, wire_publisher

        row = self._row("Dec 17 (Reuters) - Republican lawmakers accused Intel this week")
        assert wire_publisher(row) == "reuters" and resolve_tier(row) == 2

    def test_bloomberg_dateline_is_tier_two(self):
        from swing.ingest.normalize import resolve_tier

        assert resolve_tier(self._row("(Bloomberg) -- Apple Inc. is delaying Siri")) == 2

    def test_merely_mentioning_a_wire_stays_excluded(self):
        from swing.ingest.normalize import resolve_tier, wire_publisher

        row = self._row("Shares slid after Bloomberg reported the delay, analysts said")
        assert wire_publisher(row) is None and resolve_tier(row) == 4


class TestRelatedConfig:
    def test_every_related_ticker_is_defined(self):
        known = set(stocks()) | set(related_companies())
        for t in stocks():
            assert set(related(t)) <= known, f"{t}: {set(related(t)) - known}"

    def test_no_stock_is_related_to_itself(self):
        assert all(t not in related(t) for t in stocks())

    def test_related_companies_sit_outside_the_watchlist(self):
        assert not set(related_companies()) & set(stocks())

    def test_every_related_company_is_used_by_some_stock(self):
        used = {r for t in stocks() for r in related(t)}
        assert set(related_companies()) <= used

    def test_edgar_polls_related_companies_with_padded_ciks(self):
        targets = edgar_targets()
        assert {"INTC", "AMZN", "NVDA"} <= set(targets)
        assert all(len(c) == 10 and c.isdigit() for c in targets.values())

    def test_company_names(self):
        assert company_name("INTC") == "Intel"
        assert company_name("QCOM") == "QUALCOMM Inc"


class TestTagging:
    def test_a_related_company_named_in_text_is_tagged(self):
        from swing.ingest.normalize import resolve_tickers

        row = {"headline": "Intel forecasts second-quarter revenue above estimates", "raw": {}}
        assert "INTC" in resolve_tickers(row)

    def test_short_symbols_are_never_matched_as_bare_words(self):
        from swing.ingest.normalize import resolve_tickers

        row = {"headline": "The GM of the U unit said F grades fell", "raw": {}}
        assert resolve_tickers(row) == []

    def test_short_symbol_companies_are_found_by_name(self):
        from swing.ingest.normalize import resolve_tickers

        assert "GM" in resolve_tickers({"headline": "General Motors raises guidance", "raw": {}})


class TestRetrieval:
    def test_a_stock_also_searches_its_related_companies(self):
        from swing.analysis.retrieval import own_tickers, retrieval_tickers

        swing = {"ticker": "QCOM", "entity_type": "stock"}
        assert own_tickers(swing) == ["QCOM"]
        tickers = retrieval_tickers(swing)
        assert tickers[0] == "QCOM" and "INTC" in tickers

    def test_a_sector_searches_only_its_constituents(self):
        from swing.analysis.retrieval import retrieval_tickers

        tickers = retrieval_tickers({"ticker": "SMH", "entity_type": "sector"})
        assert "NVDA" in tickers and "INTC" not in tickers

    def test_related_only_news_ranks_below_equal_own_news(self):
        from swing.analysis.dedup import ClusterView
        from swing.analysis.rank import score_clusters

        onset = datetime(2026, 4, 24, 13, 30, tzinfo=UTC)
        seen = datetime(2026, 4, 23, 21, 0, tzinfo=UTC)

        def cluster(article_id, tickers):
            return ClusterView("pre_move", [article_id], article_id, "h", "cnbc", 1, 1,
                               seen, 3, members=[{"embedding": [1.0, 0.0], "tickers": tickers}])

        ranked = score_clusters([cluster(1, ["INTC"]), cluster(2, ["QCOM"])], onset,
                                None, None, own={"QCOM"})
        assert [c.canonical_article for c in ranked] == [2, 1]


class TestPrompt:
    @staticmethod
    def _cluster(tickers):
        return {"id": 5, "timing": "pre_move", "best_tier": 1, "distinct_sources": 1,
                "earliest_published": datetime(2026, 4, 23, 20, 5, tzinfo=UTC),
                "source": "sec-edgar", "headline": "INTC 8-K — Item 2.02", "tickers": tickers}

    def test_related_company_news_is_labelled(self):
        from swing.agent.prompts import _fmt_clusters

        out = _fmt_clusters([self._cluster(["INTC"])], {"QCOM"})
        assert "about: Intel (INTC). This is news about a RELATED company, not QCOM" in out

    def test_own_news_carries_no_label(self):
        from swing.agent.prompts import _fmt_clusters

        assert "about:" not in _fmt_clusters([self._cluster(["QCOM", "INTC"])], {"QCOM"})

    def test_the_active_prompt_renders(self):
        from swing.agent.prompts import render

        swing = {"ticker": "QCOM", "entity_type": "stock", "d": "2026-04-24",
                 "total_return": 0.105, "market_component": 0.01, "sector_component": 0.008,
                 "residual": 0.087, "residual_z": 6.9, "swing_type": "gap",
                 "onset_ts": datetime(2026, 4, 24, 13, 30, tzinfo=UTC), "volume_z": 12.7,
                 "earnings_mode": False}
        version, text = render(swing, "QCOM rose 10.5%.", [self._cluster(["INTC"])], [])
        assert version == "attribution_v3"
        assert "about: Intel (INTC)" in text and "RELATED company" in text
        # Intel's 8-K at 20:05 UTC, move started 13:30 UTC next day.
        assert "(17h before the move started)" in text

    def test_stale_evidence_says_how_stale(self):
        # Gate 3, 2026-09-14: a Broadcom 8-K from weeks earlier was offered as
        # the cause. The distance must be stated, not left to date arithmetic.
        from swing.agent.prompts import _relative

        onset = datetime(2026, 4, 24, 13, 30, tzinfo=UTC)
        assert _relative(datetime(2026, 4, 5, 13, 30, tzinfo=UTC), onset) == \
            "  (19 days before the move started)"
        assert _relative(datetime(2026, 4, 24, 15, 30, tzinfo=UTC), onset) == \
            "  (2h after the move started)"
