"""One function per subcommand. Each returns a process exit code.

Commands whose backing tables do not exist yet raise NotImplementedError with
the build step that unlocks them, rather than failing with a SQL error.
"""
from __future__ import annotations

from typing import Any

from swing.store.session import connect

# --------------------------------------------------------------------------
# Available now (Step 0) — these run against articles_raw
# --------------------------------------------------------------------------

def coverage_data() -> dict[str, Any]:
    from swing.ingest.config import stocks

    with connect() as conn:
        arts = conn.execute(
            "SELECT count(*) n, min(published_at)::date lo, max(published_at)::date hi "
            "FROM articles_raw"
        ).fetchone()
        by_source = conn.execute(
            "SELECT source, count(*) n FROM articles_raw GROUP BY source ORDER BY n DESC"
        ).fetchall()
        has_attr = conn.execute(
            "SELECT to_regclass('public.attributions') IS NOT NULL AS ok"
        ).fetchone()["ok"]
        attributions = 0
        if has_attr:
            attributions = conn.execute("SELECT count(*) n FROM attributions").fetchone()["n"]
    return {
        "tickers": sorted(stocks().keys()),
        "articles": arts["n"] or 0,
        "articles_from": arts["lo"],
        "articles_to": arts["hi"],
        "sources": [(r["source"], r["n"]) for r in by_source],
        "attributions": attributions,
    }


def coverage() -> int:
    d = coverage_data()
    print(
        f"articles from {d['articles_from']} to {d['articles_to']} · "
        f"{d['articles']:,} articles · {len(d['sources'])} sources · "
        f"{d['attributions']} attributions"
    )
    print()
    for source, n in d["sources"]:
        print(f"  {source:<16} {n:>6,}")
    print()
    print(f"  {len(d['tickers'])} tickers: {', '.join(d['tickers'])}")
    return 0


def collect(daemon: bool = False) -> int:
    # collector.main() configures logging; going through the CLI bypasses it,
    # so set it up here or every poll line is silently discarded.
    from swing.common import logging as log
    from swing.ingest.collector import run
    from swing.paths import DATA

    log.setup(logfile=DATA / "collector.log")
    return run(once=not daemon)


def health() -> int:
    from swing.ingest.health import check_dead_feeds, print_report

    print_report()
    broken = check_dead_feeds()
    print()
    print(f"broken feeds: {', '.join(broken) if broken else 'none'}")
    return 0


def dbinit() -> int:
    from swing.store.session import apply_schema

    apply_schema()
    print("schema applied")
    return 0


# --------------------------------------------------------------------------
# Not built yet — each names the step that unlocks it
# --------------------------------------------------------------------------

def _needs(step: str, what: str):
    raise NotImplementedError(f"{what} — lands in {step}")


def why(ticker: str, date: str | None) -> int:
    _needs("Step 5 (the attribution agent)", "`swing why` needs the attributions table")


def stats(ticker: str, days: int) -> int:
    _needs("Step 3 (decomposition)", "`swing stats` needs daily_factors and swings")


def compare(tickers: list[str], days: int) -> int:
    _needs("Step 3 (decomposition)", "`swing compare` needs daily_factors.idio_share")


def unexplained(ticker: str | None, days: int) -> int:
    _needs("Step 5 (the attribution agent)", "`swing unexplained` needs attributions.verdict")


def ask(question: str) -> int:
    _needs("Step 5+ (insight agent)", "`swing ask` needs the insight agent and its guardrails")


def batch(date: str | None) -> int:
    _needs("Step 5 (the attribution agent)", "`swing batch` needs the full pipeline")
