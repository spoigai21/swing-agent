"""Operational monitoring. agent-plan.md 6.4.

Three things, and the third is the most useful number in the system:

  * per-source article counts   -> catches a silently dead feed
  * abstention rate over time   -> a sudden DROP almost always means the timing
                                   filter broke, not that the world got more
                                   explicable
  * unexplained rate per ticker -> the coverage diagnostic. A ticker at 40%
                                   while others sit at 10% is a SOURCE GAP, not
                                   a model problem, and it names the feed to add
"""
from __future__ import annotations

from swing.store.session import connect


def unexplained_by_ticker(days: int = 90) -> list[dict]:
    with connect() as conn:
        return conn.execute(
            """
            SELECT s.ticker, count(*) AS n,
                   round((count(*) FILTER (WHERE a.verdict='unexplained'))::numeric
                         / nullif(count(*),0), 3) AS rate
            FROM attributions a JOIN swings s ON s.id=a.swing_id
            WHERE a.run_kind='production'
              AND a.created_at > now() - (%s::int || ' days')::interval
            GROUP BY s.ticker ORDER BY rate DESC NULLS LAST
            """,
            (days,),
        ).fetchall()


def unexplained_by_swing_type(days: int = 90) -> list[dict]:
    """A high rate on `gap` swings means the overnight window or the 8-K poller
    is broken — gap moves should be the EASIEST cases."""
    with connect() as conn:
        return conn.execute(
            """
            SELECT s.swing_type, count(*) AS n,
                   round((count(*) FILTER (WHERE a.verdict='unexplained'))::numeric
                         / nullif(count(*),0), 3) AS rate
            FROM attributions a JOIN swings s ON s.id=a.swing_id
            WHERE a.run_kind='production'
              AND a.created_at > now() - (%s::int || ' days')::interval
            GROUP BY s.swing_type ORDER BY rate DESC NULLS LAST
            """,
            (days,),
        ).fetchall()


def abstention_trend(weeks: int = 8) -> list[dict]:
    with connect() as conn:
        return conn.execute(
            """
            SELECT date_trunc('week', a.created_at)::date AS wk, count(*) AS n,
                   round((count(*) FILTER (WHERE a.verdict='unexplained'))::numeric
                         / nullif(count(*),0), 3) AS rate
            FROM attributions a
            WHERE a.run_kind='production'
              AND a.created_at > now() - (%s::int || ' weeks')::interval
            GROUP BY 1 ORDER BY 1
            """,
            (weeks,),
        ).fetchall()


def source_counts(days: int = 7) -> list[dict]:
    with connect() as conn:
        return conn.execute(
            """
            SELECT source,
                   count(*) FILTER (WHERE published_at > now() - interval '24 hours') AS d1,
                   count(*) FILTER (WHERE published_at > now() - (%s::int || ' days')::interval) AS dn,
                   count(*) AS total
            FROM articles_raw GROUP BY source ORDER BY dn DESC
            """,
            (days,),
        ).fetchall()


def dashboard(days: int = 90) -> None:
    from swing.ingest.health import check_dead_feeds

    print("═══ SOURCE HEALTH ═══")
    print(f"  {'source':<22}{'24h':>7}{'7d':>8}{'total':>9}")
    for r in source_counts():
        print(f"  {r['source']:<22}{r['d1']:>7}{r['dn']:>8}{r['total']:>9,}")
    broken = check_dead_feeds()
    print(f"\n  broken feeds: {', '.join(broken) if broken else 'none'}")

    rows = unexplained_by_ticker(days)
    if rows:
        print(f"\n═══ UNEXPLAINED RATE BY TICKER (last {days}d) ═══")
        print("  A high rate is a SOURCE GAP, not a model problem.")
        for r in rows:
            print(f"  {r['ticker']:<8}{float(r['rate'] or 0):>7.0%}  n={r['n']}")

    rows = unexplained_by_swing_type(days)
    if rows:
        print(f"\n═══ UNEXPLAINED RATE BY SWING TYPE (last {days}d) ═══")
        print("  A high rate on `gap` means the overnight window or 8-K poller is broken.")
        for r in rows:
            print(f"  {r['swing_type']:<10}{float(r['rate'] or 0):>7.0%}  n={r['n']}")

    rows = abstention_trend()
    if rows:
        print("\n═══ ABSTENTION TREND ═══")
        print("  A sudden DROP usually means the timing filter broke.")
        for r in rows:
            print(f"  {r['wk']}  {float(r['rate'] or 0):>6.0%}  n={r['n']}")
