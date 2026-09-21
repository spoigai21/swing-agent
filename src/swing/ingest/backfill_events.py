"""Day-one history, from the two sources that actually have a past.

The cold-start problem: attribution needs news published BEFORE a move, and a
fresh install has none. RSS cannot fix it — a feed serves the last 20-50 items,
so everything older has already scrolled away permanently. That is why
`collector.py` says every hour it is not running is coverage that is gone.

Two sources are different, and they are the two this pulls:

    SEC EDGAR   every filing a company has ever made, free, no lookback cap.
                8-K Item 2.02 is an earnings release — the single largest
                driver of idiosyncratic moves.
    GDELT       Reuters/Bloomberg/WSJ/FT coverage back to 2015, via BigQuery's
                public dataset.

They are also the two that are unambiguously redistributable: SEC filings are
public domain and GDELT is published for reuse. Everything else stays local.

⚠️ GDELT is billed by bytes SCANNED, against a 1 TB/month free allowance. One
month of one watchlist costs roughly 4-5 GB, so a year is ~55 GB. This walks
month by month and stops the moment `gdelt.affordable()` says the budget is
gone, rather than discovering it inside a single enormous query.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from swing.common import logging as log

logger = log.get("ingest.backfill_events")

# One query per calendar month. Small enough that a budget stop loses at most
# one month's scan, large enough that a year is 12 queries and not 365.
CHUNK_DAYS = 31

# EDGAR's submissions feed returns ~1000 filings per company in one request.
# Ask for all of them: the point of a backfill is the years, not the fortnight.
EDGAR_BACKFILL_LIMIT = 10_000


@dataclass
class Result:
    filings: int = 0
    articles: int = 0
    months_done: int = 0
    months_asked: int = 0
    gb_spent: float = 0.0
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        gdelt = (f"{self.articles:,} GDELT articles "
                 f"({self.months_done}/{self.months_asked} months, "
                 f"{self.gb_spent:.1f} GB scanned)")
        line = f"{self.filings:,} SEC filings · {gdelt}"
        return line + ("\n  " + "\n  ".join(self.notes) if self.notes else "")


def edgar_history(tickers: list[str] | None = None) -> int:
    """Every 8-K the watchlist has filed that the submissions feed still lists.

    Re-running is cheap and safe: `edgar.poll` keeps a per-company cursor, so a
    second pass stops at the newest filing it already stored.
    """
    from swing.ingest import edgar

    n = edgar.poll(first_run_limit=EDGAR_BACKFILL_LIMIT, tickers=tickers)
    logger.info("edgar backfill: %d filings", n)
    return n


def gdelt_history(months: int, tickers: list[str] | None = None) -> tuple[int, int, int, float, list[str]]:
    """Walk backwards a month at a time. Returns (articles, done, asked, gb, notes)."""
    from swing.ingest import gdelt
    from swing.ingest.config import stocks

    notes: list[str] = []
    symbols = [t.upper() for t in (tickers or stocks())]
    if not gdelt.affordable():
        return 0, 0, months, 0.0, [
            "GDELT skipped: this month's BigQuery budget is already spent."]

    before = gdelt.spent_gb()
    total = done = 0
    end = datetime.now(UTC)
    for _ in range(months):
        start = end - timedelta(days=CHUNK_DAYS)
        if not gdelt.affordable():
            notes.append(
                f"GDELT stopped at {start.date()}: BigQuery budget reached. "
                "It resets monthly — re-run `swing backfill-events` then.")
            break
        try:
            total += gdelt.fetch_window(symbols, start, end)
            done += 1
        except Exception as exc:   # noqa: BLE001 — one bad month must not end the walk
            logger.warning("gdelt %s..%s failed: %s", start.date(), end.date(), exc)
            notes.append(f"GDELT {start.date()}..{end.date()} failed: {exc}")
        end = start
    return total, done, months, gdelt.spent_gb() - before, notes


def run(months: int = 12, tickers: list[str] | None = None,
        *, skip_gdelt: bool = False) -> Result:
    """Both sources, then normalize so the rows are actually retrievable."""
    res = Result()
    res.filings = edgar_history(tickers)

    if skip_gdelt:
        res.notes.append("GDELT skipped (--no-gdelt).")
        res.months_asked = 0
    else:
        from swing.common.settings import get_settings

        if not get_settings().google_cloud_project:
            res.notes.append(
                "GDELT skipped: no GOOGLE_CLOUD_PROJECT set. It needs a Google "
                "Cloud project and a service-account key; the BigQuery free tier "
                "(1 TB/month scanned) covers this comfortably.")
            res.months_asked = months
        else:
            (res.articles, res.months_done, res.months_asked,
             res.gb_spent, notes) = gdelt_history(months, tickers)
            res.notes.extend(notes)

    # ⚠️ Without this the rows sit in articles_raw and retrieval never sees
    # them, so a backfill that "worked" leaves every swing unexplained.
    from swing.ingest.normalize import normalize_all

    written = normalize_all()["written"]
    logger.info("backfill normalized %d articles", written)
    return res
