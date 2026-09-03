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


def backfill(years: int = 2, symbols: list[str] | None = None) -> int:
    from swing.common import logging as log
    from swing.ingest.prices import backfill_daily, enforce_history_start

    log.setup()
    written = backfill_daily(symbols or None, years=years)
    print(f"\n{sum(written.values()):,} bars across {len(written)} symbols")
    for ticker, ok, msg in enforce_history_start(purge=True):
        print(f"  {ticker}: {'ok' if ok else 'PROBLEM'} — {msg}")
    return 0


def backfill_news(months: int = 12, tickers: list[str] | None = None) -> int:
    from datetime import UTC, datetime, timedelta

    from swing.common import logging as log
    from swing.ingest.news_finnhub import backfill

    log.setup()
    end = datetime.now(UTC).date()
    start = end - timedelta(days=int(months * 30.44))
    n = backfill(start, end, tickers=tickers or None)
    print(f"{n:,} historical articles stored ({start} -> {end}). "
          "Run `swing normalize` next.")
    return 0


def normalize(limit: int | None = None) -> int:
    from swing.common import logging as log
    from swing.ingest.normalize import normalize_all, normalize_batch

    log.setup()
    got = normalize_batch(limit) if limit else normalize_all()
    print(f"read {got['read']}, wrote {got['written']}, "
          f"dropped {got['dropped_tier4']} as tier 4")
    return 0


def prices() -> int:
    from swing.ingest.prices import coverage_report, enforce_history_start

    rows = coverage_report()
    if not rows:
        print("no bars yet — run `swing backfill`")
        return 0
    print(f"{'ticker':<8}{'bars':>7}  {'first':<12}{'last':<12}")
    for r in rows:
        print(f"{r['ticker']:<8}{r['n']:>7}  {r['first_day']!s:<12}{r['last_day']!s:<12}")
    for ticker, ok, msg in enforce_history_start(purge=False):
        print(f"\n  {ticker}: {'ok' if ok else 'PROBLEM'} — {msg}")
    return 0


def factors(ticker: str | None = None) -> int:
    from swing.analysis.factors import compute, entities, rebuild_all
    from swing.common import logging as log

    log.setup()
    if ticker:
        e = next((x for x in entities() if x.ticker == ticker.upper()), None)
        if not e:
            print(f"unknown ticker: {ticker}")
            return 1
        print(f"{e.ticker}: {len(compute(e))} rows (not written; use without --ticker)")
        return 0
    out = rebuild_all()
    print(f"{sum(out.values()):,} factor rows across {len(out)} entities")
    return 0


def detect(capture: bool = False) -> int:
    from swing.analysis.swings import detect_all
    from swing.common import logging as log

    log.setup()
    out = detect_all(capture_intraday=capture)
    print(f"{sum(out.values())} swings across {len(out)} entities")
    return 0


def onsets(limit: int | None = None) -> int:
    from swing.analysis.swings import backfill_onsets
    from swing.common import logging as log

    log.setup()
    r = backfill_onsets(limit)
    print(f"onset backfill: {r['fixed']} fixed, {r['failed']} still without intraday "
          f"(of {r['considered']} considered)")
    return 0


def retrieve(swing_id: int | None = None, limit: int | None = None) -> int:
    from swing.analysis.retrieval import build_all, build_for_swing
    from swing.common import logging as log

    log.setup()
    if swing_id:
        g = build_for_swing(swing_id)
        for timing in ("pre_move", "post_move"):
            print(f"\n{timing}: {len(g[timing])} clusters")
            for c in g[timing][:10]:
                print(f"  #{c.rank} score={c.rank_score:.2f} tier={c.best_tier} "
                      f"srcs={c.distinct_sources} {c.earliest_published:%m-%d %H:%M}Z  "
                      f"{c.headline[:52]}")
        return 0
    r = build_all(limit)
    print(f"built clusters for {r['swings']} swings; "
          f"{r['with_pre_move']} have pre-move coverage")
    return 0


def annotate(blind: bool = False, ticker: str | None = None, limit: int = 10,
             show_progress: bool = False) -> int:
    from swing.eval.annotate import _candidates, annotate_one, progress

    if show_progress:
        progress()
        return 0
    rows = _candidates(blind, ticker, limit)
    if not rows:
        print("nothing left to annotate matching that filter")
        return 0
    print(f"{len(rows)} swings queued ({'BLIND' if blind else 'assisted'} mode). "
          "Ctrl-C to stop.")
    for s in rows:
        annotate_one(s, blind)
    progress()
    return 0


def metrics() -> int:
    from swing.eval.harness import print_report

    print_report()
    return 0


# --------------------------------------------------------------------------
# Not built yet — each names the step that unlocks it
# --------------------------------------------------------------------------

def _needs(step: str, what: str):
    raise NotImplementedError(f"{what} — lands in {step}")


def why(ticker: str, date: str | None) -> int:
    _needs("Step 5 (the attribution agent)", "`swing why` needs the attributions table")


def stats(ticker: str, days: int = 90) -> int:
    from swing.store import queries

    t = ticker.upper()
    rows = queries.factors_for(t, days)
    if not rows:
        print(f"no factor rows for {t} in the last {days} days")
        return 1
    ok = [r for r in rows if r["status"] == "ok"]
    if not ok:
        print(f"{t}: {rows[0]['status']}")
        return 0
    import statistics as st
    z = [abs(float(r["residual_z"])) for r in ok]
    sw = [r for r in ok if abs(float(r["residual_z"])) >= 2]
    print(f"{t} — last {days} days, n={len(ok)}")
    print(f"  beta_mkt      {st.mean(float(r['beta_mkt']) for r in ok):>7.2f}")
    if ok[0]["beta_sector"] is not None:
        print(f"  beta_sector   {st.mean(float(r['beta_sector']) for r in ok):>7.2f}")
    print(f"  R^2           {st.mean(float(r['r_squared']) for r in ok):>7.3f}")
    print(f"  residual vol  {st.mean(float(r['residual_vol_60']) for r in ok) * 100:>7.2f}%")
    print(f"  mean |z|      {st.mean(z):>7.2f}")
    print(f"  swing days    {len(sw):>7}  ({len(sw) / len(ok):.1%})")
    return 0


def compare(tickers: list[str], days: int = 90) -> int:
    from swing.store import queries

    rows = queries.idio_share_summary([t.upper() for t in tickers], days)
    if not rows:
        print("no factor rows for those tickers")
        return 1
    print(f"{'ticker':<8}{'n':>5}{'idio_share':>12}{'R^2':>8}{'resid_vol':>11}{'swings':>8}")
    for r in rows:
        print(f"{r['ticker']:<8}{r['n']:>5}{float(r['avg_idio_share'] or 0):>12.2f}"
              f"{float(r['avg_r2'] or 0):>8.3f}{float(r['residual_vol'] or 0) * 100:>10.2f}%"
              f"{r['swing_days']:>8}")
    return 0


def unexplained(ticker: str | None, days: int) -> int:
    _needs("Step 5 (the attribution agent)", "`swing unexplained` needs attributions.verdict")


def ask(question: str) -> int:
    _needs("Step 5+ (insight agent)", "`swing ask` needs the insight agent and its guardrails")


def batch(date: str | None) -> int:
    _needs("Step 5 (the attribution agent)", "`swing batch` needs the full pipeline")
