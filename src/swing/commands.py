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


def placebo(n: int = 30, seed: int = 0) -> int:
    from swing.common import logging as log
    from swing.eval.placebo import run

    log.setup()
    from swing.eval.placebo import cumulative

    r = run(n=n, seed=seed)
    if not r["n"]:
        print(r.get("note", "no cases"))
    else:
        print(f"\n  this run: {r['n']} cases, {r['confabulated']} confabulated "
              f"({r['rate']:.1%})")
    c = cumulative()
    if c["n"]:
        status = "PASS" if c["rate"] < 0.10 else "FAIL"
        print(f"  CUMULATIVE on this version: {c['n']} cases, "
              f"{c['confabulated']} confabulated ({c['rate']:.1%})  "
              f"target < 10%  {status}")
        print("  (Gate 4 wants 200; free tier is 20 requests/day per model, "
              "so this accumulates across runs)")
    return 0


def alert(days: int = 3, min_z: float | None = None, dry_run: bool = False) -> int:
    from swing.common import logging as log
    from swing.interface.alert import send

    log.setup()
    n = send(days, min_z, dry_run)
    print(f"{n} alert(s)" + (" (dry run)" if dry_run else " sent"))
    return 0


def monitor(days: int = 90) -> int:
    from swing.interface.monitor import dashboard

    dashboard(days)
    return 0


def daily(date: str | None = None, limit: int = 15, skip_prices: bool = False,
          skip_attribution: bool = False) -> int:
    from datetime import date as _date

    from swing.common import logging as log
    from swing.interface.batch import run

    log.setup()
    res = run(_date.fromisoformat(date) if date else None,
              attribution_limit=limit, skip_prices=skip_prices,
              skip_attribution=skip_attribution)
    print(f"\n  {res.summary()}")
    for e in res.errors:
        print(f"  ERROR {e}")
    return 1 if res.errors else 0


def metrics() -> int:
    from swing.eval.harness import print_report

    print_report()
    return 0


# --------------------------------------------------------------------------
# Not built yet — each names the step that unlocks it
# --------------------------------------------------------------------------

def _needs(step: str, what: str):
    raise NotImplementedError(f"{what} — lands in {step}")


def why(ticker: str, date: str | None = None) -> int:
    """Why `ticker` moved on `date` (default: the latest completed session)."""
    from datetime import date as _date

    from swing.common import logging as log
    from swing.ingest.config import stocks
    from swing.interface.explain import explain
    from swing.paths import DATA

    log.setup(logfile=DATA / "swing.log", console=False)
    t = ticker.upper()
    if t not in stocks():
        print(f"I only cover these stocks for now: {', '.join(sorted(stocks()))}.")
        return 1
    print(explain(t, _date.fromisoformat(date) if date else None))
    return 0


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
    flagged = False
    for r in rows:
        share = float(r["avg_idio_share"] or 0)
        flagged |= share > 1.0
        print(f"{r['ticker']:<8}{r['n']:>5}{share:>12.2f}"
              f"{float(r['avg_r2'] or 0):>8.3f}{float(r['residual_vol'] or 0) * 100:>10.2f}%"
              f"{r['swing_days']:>8}{'  <- factor adds variance' if share > 1 else ''}")
    if flagged:
        print("\n  idio_share > 1.0 is possible: betas are fitted out-of-sample, so a")
        print("  poorly-matched sector factor can add variance instead of removing it.")
    return 0


def unexplained(ticker: str | None = None, days: int = 30) -> int:
    """Unexplained swings mark gaps in source coverage — worth reviewing."""
    from swing.store import queries
    from swing.store.session import connect

    sql = """
        SELECT s.ticker, s.d, s.residual_z, s.volume_z, s.swing_type, a.unexplained_note
        FROM attributions a JOIN swings s ON s.id = a.swing_id
        WHERE a.verdict='unexplained' AND a.run_kind='production'
          AND s.d > current_date - %s::int
    """
    params: list = [days]
    if ticker:
        sql += " AND s.ticker=%s"
        params.append(ticker.upper())
    sql += " ORDER BY abs(s.residual_z) DESC"
    with connect() as conn:
        rows = conn.execute(sql, tuple(params)).fetchall()
    if not rows:
        print(f"no unexplained swings in the last {days} days")
    else:
        print(f"{'ticker':<8}{'date':<12}{'z':>7}{'vol_z':>7}  {'type':<10}note")
        for r in rows:
            print(f"{r['ticker']:<8}{r['d']!s:<12}{float(r['residual_z']):>7.2f}"
                  f"{float(r['volume_z'] or 0):>7.1f}  {r['swing_type']:<10}"
                  f"{(r['unexplained_note'] or '')[:44]}")
    rates = queries.unexplained_rate_by_ticker(days * 3)
    if rates:
        print("\n  unexplained rate by ticker (the coverage diagnostic):")
        for r in rates:
            print(f"    {r['ticker']:<8}{float(r['unexplained_rate'] or 0):>6.0%}  n={r['n']}")
    return 0


def ask(question: str) -> int:
    """One question: why a stock moved, or a question about the stored history.

    The forecast guardrail runs first, before anything is fetched or any model
    is called.
    """
    from swing.common import logging as log
    from swing.interface.explain import respond
    from swing.paths import DATA

    log.setup(logfile=DATA / "swing.log", console=False)
    print(respond(question))
    return 0


def batch(date: str | None = None, limit: int | None = None) -> int:
    """Daily run: attribute every swing that lacks a production attribution.

    Sector entities are processed FIRST so a stock attribution can reference an
    already-computed sector story (agent-plan.md 0.1b).
    """
    from datetime import date as _date

    from swing.agent.graph import attribute_swing
    from swing.common import logging as log
    from swing.store.session import connect

    log.setup()
    sql = """
        SELECT s.id, s.ticker, s.d, s.residual_z, s.entity_type
        FROM swings s
        WHERE NOT EXISTS (SELECT 1 FROM attributions a
                          WHERE a.swing_id=s.id AND a.run_kind='production')
          AND s.onset_ts IS NOT NULL
    """
    params: list = []
    if date:
        sql += " AND s.d = %s"
        params.append(_date.fromisoformat(date))
    sql += " ORDER BY (s.entity_type='sector') DESC, abs(s.residual_z) DESC"
    if limit:
        sql += f" LIMIT {int(limit)}"
    with connect() as conn:
        rows = conn.execute(sql, tuple(params)).fetchall()
    if not rows:
        print("nothing to attribute")
        return 0
    print(f"attributing {len(rows)} swings "
          "(free-tier daily caps are small; use --limit to pace)")
    counts: dict[str, int] = {}
    for r in rows:
        try:
            out = attribute_swing(r["id"])
        except Exception as e:  # noqa: BLE001 - one swing must not stop the batch
            print(f"  {r['ticker']} {r['d']}: FAILED {type(e).__name__}: {str(e)[:60]}")
            counts["error"] = counts.get("error", 0) + 1
            continue
        if out.get("verdict_reason") == "llm_error":
            # Usually the daily quota: nothing was stored, and the rest would fail too.
            print(f"  {r['ticker']} {r['d']}: model unavailable (daily quota?); stopping")
            counts["stopped"] = counts.get("stopped", 0) + 1
            break
        attr = out.get("attribution")
        v = attr.verdict if attr else "error"
        counts[v] = counts.get(v, 0) + 1
        print(f"  {r['ticker']:<7}{r['d']!s:<12}z={float(r['residual_z']):+6.2f}  {v}")
    print("\n  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 0
