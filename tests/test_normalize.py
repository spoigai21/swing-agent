"""Tiering and ticker tagging — both silently corrupt retrieval when wrong."""
from __future__ import annotations

from swing.ingest.normalize import _ticker_patterns, resolve_tickers, resolve_tier


def _raw(**kw):
    base = {"source": "wsj", "headline": "h", "summary": "", "raw": {}}
    return {**base, **kw}


class TestTiering:
    def test_publisher_beats_feed(self):
        # Finnhub carries many publishers through one pipe; tiering by feed
        # would admit Tier 4 aggregator content as evidence.
        row = _raw(source="finnhub", raw={"publisher": "SeekingAlpha", "via": "finnhub"})
        assert resolve_tier(row) == 4

    def test_wire_publisher_is_tier_2(self):
        assert resolve_tier(_raw(source="finnhub", raw={"publisher": "Reuters"})) == 2

    def test_unlisted_publisher_defaults_to_excluded(self):
        assert resolve_tier(_raw(source="finnhub", raw={"publisher": "SomeSEOBlog"})) == 4

    def test_edgar_is_tier_1(self):
        assert resolve_tier(_raw(source="sec-edgar", raw={"form": "8-K"})) == 1

    def test_case_and_space_insensitive(self):
        assert resolve_tier(_raw(source="x", raw={"publisher": "  seeking alpha "})) == 4


class TestTickerTagging:
    def test_edgar_ticker_is_authoritative(self):
        assert resolve_tickers(_raw(source="sec-edgar", raw={"ticker": "nvda"})) == ["NVDA"]

    def test_alias_match(self):
        row = _raw(headline="Grand Theft Auto 6 preview signals demand")
        assert "TTWO" in resolve_tickers(row)

    def test_generic_word_does_not_tag(self):
        # 'Interactive' came from deriving aliases off "Take-Two Interactive".
        assert "TTWO" not in resolve_tickers(_raw(headline="An interactive dashboard"))

    def test_ungrouped_alternation_bug_stays_fixed(self):
        # The old pattern was `(?<!x)MU|Micron(?!x)`: the lookbehind bound only
        # to MU and the lookahead only to Micron, so "Musk", "Multiple" and
        # "Munich" all tagged MU. 28 of 68 MU articles were mistagged.
        for text in ["Elon Musk sells Tesla shares", "Multiple analysts cut targets",
                     "Munich summit on chips"]:
            assert "MU" not in resolve_tickers(_raw(headline=text)), text

    def test_real_mentions_still_tag(self):
        assert "MU" in resolve_tickers(_raw(headline="Micron guides higher"))
        assert "MU" in resolve_tickers(_raw(headline="MU rallies on memory pricing"))

    def test_every_pattern_is_grouped(self):
        # A bare '|' outside a group re-introduces the bug for that ticker.
        for ticker, pat in _ticker_patterns().items():
            assert pat.pattern.startswith("(?<![A-Za-z0-9])(?:"), ticker
            assert pat.pattern.endswith(")(?![A-Za-z0-9])"), ticker
