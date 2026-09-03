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

    def test_every_alias_pattern_is_grouped(self):
        # A bare '|' outside a group re-introduces the ungrouped-alternation bug
        # (lookbehind binds to the first alternative, lookahead to the last).
        for ticker, (sym, alias) in _ticker_patterns().items():
            assert alias.pattern.startswith("(?<![A-Za-z0-9])(?:"), ticker
            assert alias.pattern.endswith(")(?![A-Za-z0-9])"), ticker
            assert sym.pattern.startswith("(?<![A-Za-z0-9])"), ticker
            assert sym.pattern.endswith("(?![A-Za-z0-9])"), ticker


class TestSymbolCaseSensitivity:
    """A two-letter ticker matched case-insensitively is a magnet for ordinary
    words in any language. 'MU' tagged the Czech word 'mu' in a PR Newswire
    release before symbols were made case-sensitive."""

    def test_lowercase_symbol_does_not_tag(self):
        assert "MU" not in resolve_tickers(_raw(headline="the mu variant spread quickly"))

    def test_foreign_language_word_does_not_tag(self):
        assert "MU" not in resolve_tickers(
            _raw(headline="dokoncily v Japonsku mu demonstraci"))

    def test_uppercase_symbol_tags(self):
        assert "MU" in resolve_tickers(_raw(headline="MU rallies on memory pricing"))

    def test_company_name_tags_case_insensitively(self):
        assert "MU" in resolve_tickers(_raw(headline="micron guides higher"))
        assert "MU" in resolve_tickers(_raw(headline="Micron guides higher"))

    def test_symbol_pattern_is_not_ignorecase(self):
        import re

        from swing.ingest.normalize import _ticker_patterns

        for ticker, (sym, alias) in _ticker_patterns().items():
            assert not (sym.flags & re.IGNORECASE), f"{ticker} symbol must be case-sensitive"
            assert alias.flags & re.IGNORECASE, f"{ticker} alias should be case-insensitive"


class TestTierConfigIntegrity:
    def test_publisher_tiers_have_no_duplicate_keys(self):
        """YAML silently keeps the LAST duplicate. prnewswire was listed at both
        tier 1 and tier 2, so the tier-1 intent was overridden invisibly."""
        import re

        from swing.paths import SOURCES

        block = SOURCES.read_text().split("publisher_tiers:")[1]
        keys = []
        for line in block.splitlines():
            m = re.match(r'\s{2}"?([\w .\-/]+)"?\s*:\s*\d+\s*$', line)
            if m:
                keys.append(m.group(1).strip().strip('"').lower())
        dupes = {k for k in keys if keys.count(k) > 1}
        assert not dupes, f"duplicate publisher_tiers keys: {sorted(dupes)}"

    def test_feed_tier_agrees_with_publisher_tier(self):
        """resolve_tier falls back to the feed tier, so a disagreement makes the
        resolved tier depend on which lookup happens to hit."""
        from swing.ingest.config import feeds, sources

        pub = {k.lower(): v for k, v in sources()["publisher_tiers"].items()}
        for f in feeds(include_disabled=True):
            if f.source.lower() in pub:
                assert pub[f.source.lower()] == f.tier, (
                    f"{f.id}: feed tier {f.tier} != publisher tier "
                    f"{pub[f.source.lower()]} for {f.source}")
