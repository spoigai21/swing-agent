"""Config invariants that break things far downstream if violated."""
from __future__ import annotations

import pytest

from swing.ingest.config import cik_map, feeds, stocks, watchlist


def test_twelve_stocks_and_four_sectors():
    assert len(stocks()) == 12
    assert set(watchlist()["sectors"]) == {"SMH", "XLK", "XLC", "XLY"}


def test_every_cik_is_zero_padded_to_ten():
    # An unpadded CIK 404s and looks like a missing company, not a format bug.
    for ticker, cik in cik_map().items():
        assert len(cik) == 10, f"{ticker}: {cik}"
        assert cik.isdigit(), f"{ticker}: {cik}"


def test_sector_etfs_referenced_by_stocks_are_defined_as_entities():
    defined = set(watchlist()["sectors"])
    used = {m["sector_etf"] for m in stocks().values() if m.get("sector_etf")}
    assert used <= defined, f"undefined sector ETFs: {used - defined}"


def test_sndk_history_start_is_pinned():
    # Pre-2016 SanDisk was a different company; splicing them silently corrupts
    # every beta fitted across the join. data-sources.md A.2.
    assert stocks()["SNDK"]["history_starts"] == "2025-02-24"


def test_no_tier_four_feed_is_enabled():
    # Tier 4 is never ingested.
    assert not [f for f in feeds() if f.tier == 4]


def test_enabled_feeds_have_urls_and_sane_intervals():
    for f in feeds():
        assert f.url.startswith("http"), f.id
        assert 60 <= f.poll_seconds <= 86400, f.id


@pytest.mark.parametrize("field", ["market_etf", "min_history_days"])
def test_defaults_present(field):
    assert field in watchlist()["defaults"]


def test_every_ticker_has_a_revenue_tag():
    # One tag does NOT work across the watchlist: the split is 6/6 between
    # Revenues and RevenueFromContractWithCustomerExcludingAssessedTax.
    # Assuming a single tag makes revenue extraction silently return nothing
    # for half the tickers. data-sources.md D.4.
    for ticker, meta in stocks().items():
        assert meta.get("revenue_tag"), f"{ticker} has no revenue_tag"
