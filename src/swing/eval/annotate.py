"""Annotation CLI. The asset the whole project rests on. agent-plan.md 4.1.

⚠️ TWO MODES, AND YOU NEED BOTH.

  assisted (~170 cases)  show retrieved clusters, pick the right one
                         -> measures attribution accuracy. Fast.
  blind    (>=30 cases)  determine the cause INDEPENDENTLY, commit an answer,
                         and only THEN see what retrieval returned
                         -> measures recall. The only honest measurement.

The trap: if you label by picking from the retrieved list, you can never record
a catalyst retrieval MISSED. recall@10 then looks perfect by construction and
Gate 2 measures nothing. Do the blind set FIRST, before tuning ranking weights,
so you are not anchored.

recall@10 is computed ONLY over blind=true rows.
"""
from __future__ import annotations

import argparse
import textwrap

from swing.analysis.decompose import load as load_decomp
from swing.store.session import connect

EVENT_TYPES = ["earnings", "guidance", "analyst_action", "m_and_a", "regulatory",
               "litigation", "product", "macro", "management", "other"]

#: Sources that state what a company or an analyst SAID, as opposed to what a
#: journalist concluded. Safe to show a blind annotator: agent-plan 4.1 tells
#: them to check primary sources, and none of these is our ranked cluster list.
#: ⚠️ Journalism (bloomberg, cnbc, marketwatch…) is deliberately absent. Reading
#: the day's coverage is encouraged — but they should go find it themselves, or
#: the blind set stops measuring what our corpus missed.
PRIMARY_SOURCES = ("sec-edgar", "analyst-ratings", "prnewswire", "businesswire")

#: 8-K Item numbers are free event labels. Offered as a DEFAULT, never applied
#: silently — the annotator can overrule it with one keystroke.
ITEM_EVENT_TYPE = {
    "1.01": "m_and_a", "1.02": "m_and_a", "2.01": "m_and_a",
    "2.02": "earnings", "2.05": "management", "2.06": "other",
    "3.02": "other", "5.02": "management", "5.03": "other",
    "7.01": "other", "8.01": "other",
}


def event_type_hint(items: str | None) -> str | None:
    """The event type an 8-K's Item numbers imply, if any."""
    if not items:
        return None
    for item in (i.strip() for i in items.split(",")):
        if item in ITEM_EVENT_TYPE:
            return ITEM_EVENT_TYPE[item]
    return None


def _one_swing(swing_id: int) -> list[dict]:
    with connect() as conn:
        row = conn.execute("SELECT * FROM swings WHERE id=%s", (swing_id,)).fetchone()
    return [row] if row else []


def _candidates(blind: bool, ticker: str | None, limit: int,
                since: str | None = None) -> list[dict]:
    """Swings not yet annotated, biggest |z| first.

    Blind mode prefers swings that actually have retrievable coverage,
    otherwise you spend ten minutes researching a swing whose window we hold no
    articles for.
    """
    sql = """
        SELECT s.*, (SELECT count(*) FROM clusters c
                     WHERE c.swing_id = s.id AND c.timing='pre_move') AS pre_clusters
        FROM swings s
        LEFT JOIN annotations a ON a.swing_id = s.id
        WHERE a.swing_id IS NULL AND s.onset_ts IS NOT NULL
    """
    params: list = []
    if ticker:
        sql += " AND s.ticker = %s"
        params.append(ticker.upper())
    if since:
        # Labels from the period the collector was actually running are the ones
        # that say what the product does now, rather than what it could not have
        # known before it existed.
        sql += " AND s.d >= %s::date"
        params.append(since)
    # Blind cases are researched by hand, so offer the ones a human can actually
    # verify: single names (a sector move needs the whole industry checked),
    # with real coverage, biggest moves first.
    if blind:
        sql += " AND s.entity_type = 'stock' AND s.kind = 'daily'"
    sql += """
        ORDER BY (SELECT count(*) FROM clusters c
                  WHERE c.swing_id=s.id AND c.timing='pre_move') DESC,
                 abs(s.residual_z) DESC
        LIMIT %s
    """
    params.append(limit)
    with connect() as conn:
        return conn.execute(sql, tuple(params)).fetchall()


def _show_swing(s: dict) -> None:
    d = load_decomp(s["ticker"], s["d"])
    print("\n" + "=" * 78)
    print(f"  {s['ticker']}  {s['d']}   swing #{s['id']}")
    print("=" * 78)
    if d:
        print(textwrap.fill(d.sentence(), 76, initial_indent="  ", subsequent_indent="  "))
    print(f"  swing_type={s['swing_type']}  onset={s['onset_ts']:%Y-%m-%d %H:%M}Z "
          f"({s['onset_source']})")
    print(f"  residual_z={float(s['residual_z']):+.2f}  "
          f"volume_z={float(s['volume_z'] or 0):+.1f}  "
          f"earnings_mode={s['earnings_mode']}")


def _research_pane(swing: dict) -> None:
    """Independent evidence for the blind researcher.

    ⚠️ This is NOT our retrieval. It lists EVERY filing we hold for the ticker
    around the move, unranked and unfiltered, plus the price shape and search
    strings. The circularity the plan warns about is constraining your answer to
    our RANKED CLUSTER LIST; checking primary sources is exactly what agent-plan
    4.1 tells you to do ("read that day's coverage, check the 8-K, search the
    web"), and it is what makes a blind case take minutes instead of forever.
    """
    from datetime import timedelta

    from swing.analysis.retrieval import own_tickers

    d, ticker = swing["d"], swing["ticker"]
    # A sector ETF files nothing itself; its constituents do.
    filing_tickers = own_tickers(dict(swing))
    with connect() as conn:
        # ⚠️ Was: source='sec-edgar' AND raw->>'ticker' = ANY(...). That missed
        # every IR press release, every analyst action and every newswire
        # release we hold — for AVGO 2025-11-24 it showed 0 items while the
        # corpus held primary sources for the window. A blind annotator who
        # cannot see what we have spends minutes rediscovering it.
        filings = conn.execute(
            """
            SELECT a.published_at, a.headline, a.url, a.source,
                   r.raw->>'items' AS items, r.raw->>'form' AS form
            FROM articles a LEFT JOIN articles_raw r ON r.id = a.raw_id
            WHERE (a.source = ANY(%s) OR a.source LIKE %s)
              AND a.tickers && %s
              AND a.published_at::date BETWEEN %s AND %s
            ORDER BY a.published_at
            """,
            (list(PRIMARY_SOURCES), "%-ir", filing_tickers,
             d - timedelta(days=4), d + timedelta(days=1)),
        ).fetchall()
        bars = conn.execute(
            """
            SELECT ts::date d, open, close, volume FROM bars
            WHERE ticker=%s AND ts::date BETWEEN %s AND %s ORDER BY ts
            """,
            (ticker, d - timedelta(days=2), d + timedelta(days=1)),
        ).fetchall()

    print("\n  --- INDEPENDENT RESEARCH (not our retrieval) ---")
    if bars:
        print("  price action:")
        for b in bars:
            mark = "  <-- swing day" if b["d"] == d else ""
            chg = (float(b["close"]) / float(b["open"]) - 1) * 100
            print(f"    {b['d']}  open {float(b['open']):>9.2f}  close {float(b['close']):>9.2f}"
                  f"  ({chg:+.1f}%)  vol {b['volume']:>13,}{mark}")
    print(f"\n  PRIMARY SOURCES, {d - timedelta(days=4)} .. {d + timedelta(days=1)} "
          f"(filings, IR releases, analyst actions — ALL of them, unranked):")
    if not filings:
        print("    (none — nothing the company or an analyst said is in the corpus)")
    for f in filings:
        item = f" Item {f['items']}" if f["items"] else ""
        label = f["form"] or f["source"]
        hint = event_type_hint(f["items"])
        suffix = f"   [looks like: {hint}]" if hint else ""
        print(f"    [{f['published_at']:%m-%d %H:%M}Z] {label}{item}{suffix}")
        print(f"        {f['headline'][:88]}")
        print(f"        {f['url']}")
    print("\n  suggested searches:")
    print(f"    \"{ticker}\" stock {d}")
    print(f"    {ticker} news {d.strftime('%B %d, %Y')}")
    print("  ---------------------------------------------")


def _suggested_event_type(swing: dict) -> str | None:
    """Best guess from any 8-K filed by the company in the pre-move window."""
    from datetime import timedelta

    from swing.analysis.retrieval import own_tickers

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT raw->>'items' AS items FROM articles_raw
            WHERE source='sec-edgar' AND raw->>'ticker' = ANY(%s)
              AND published_at::date BETWEEN %s AND %s
            ORDER BY published_at DESC
            """,
            (own_tickers(dict(swing)), swing["d"] - timedelta(days=2), swing["d"]),
        ).fetchall()
    for r in rows:
        if hint := event_type_hint(r["items"]):
            return hint
    return None


def _show_clusters(swing_id: int) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT c.id, c.rank, c.best_tier, c.distinct_sources, c.member_count,
                   c.earliest_published, c.rank_score, a.headline, a.source, a.url
            FROM clusters c JOIN articles a ON a.id = c.canonical_article
            WHERE c.swing_id=%s AND c.timing='pre_move'
            ORDER BY c.rank LIMIT 10
            """,
            (swing_id,),
        ).fetchall()
    if not rows:
        print("\n  (no pre-move clusters retrieved)")
        return []
    print(f"\n  {'#':<4}{'tier':<6}{'srcs':<6}{'published':<18}headline")
    print("  " + "-" * 74)
    for r in rows:
        print(f"  {r['rank']:<4}{r['best_tier']:<6}{r['distinct_sources']:<6}"
              f"{r['earliest_published']:%m-%d %H:%M}Z     {r['headline'][:44]}")
    return rows


def _prompt(msg: str, default: str = "") -> str:
    try:
        v = input(msg).strip()
    except (EOFError, KeyboardInterrupt):
        raise SystemExit("\naborted") from None
    return v or default


def _save(swing_id: int, blind: bool, catalyst: str, cluster_id: int | None,
          event_type: str | None, no_catalyst: bool, note: str) -> None:
    """Store the chosen cluster's ARTICLES as the label, not just its id.

    Cluster ids change when retrieval is rebuilt; article ids never do. Without
    this, retuning ranking after annotating could not move recall@10.
    """
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO annotations (swing_id, blind, true_catalyst, true_cluster_id,
                                     true_article_ids, true_event_type, no_catalyst,
                                     annotator_note)
            VALUES (%(swing_id)s, %(blind)s, %(catalyst)s, %(cluster_id)s,
                    (SELECT array_agg(article_id ORDER BY article_id)
                     FROM cluster_members WHERE cluster_id = %(cluster_id)s),
                    %(event_type)s, %(no_catalyst)s, %(note)s)
            ON CONFLICT (swing_id) DO UPDATE SET
              blind=EXCLUDED.blind, true_catalyst=EXCLUDED.true_catalyst,
              true_cluster_id=EXCLUDED.true_cluster_id,
              true_article_ids=EXCLUDED.true_article_ids,
              true_event_type=EXCLUDED.true_event_type,
              no_catalyst=EXCLUDED.no_catalyst, annotator_note=EXCLUDED.annotator_note
            """,
            {"swing_id": swing_id, "blind": blind, "catalyst": catalyst or None,
             "cluster_id": cluster_id, "event_type": event_type,
             "no_catalyst": no_catalyst, "note": note or None},
        )


def annotate_one(s: dict, blind: bool) -> bool:
    _show_swing(s)
    if blind:
        print("\n  BLIND MODE — retrieval is hidden until you commit an answer.")
        print("  Research this independently: read that day's coverage, check the")
        print("  8-K on EDGAR, search the web. Then answer.")
        _research_pane(s)
        print()
        catalyst = _prompt("  True catalyst (blank = none found): ")
        no_cat = not catalyst
        etype = ""
        if not no_cat:
            suggested = _suggested_event_type(s) or "other"
            etype = _prompt(f"  Event type {EVENT_TYPES}\n    [{suggested}]: ", suggested)
        note = _prompt("  Note (optional): ")
        rows = _show_clusters(s["id"])          # revealed only now
        cid = None
        if rows and not no_cat:
            pick = _prompt("\n  Which rank matches your answer? (number / 'none'): ", "none")
            if pick.isdigit():
                cid = next((r["id"] for r in rows if r["rank"] == int(pick)), None)
                if cid is None:
                    print("  (rank not in the retrieved list — recorded as a MISS)")
        _save(s["id"], True, catalyst, cid, etype or None, no_cat, note)
    else:
        rows = _show_clusters(s["id"])
        pick = _prompt("\n  Correct cluster rank (number / 'none' / 's' to skip): ", "none")
        if pick.lower() == "s":
            return False
        cid, no_cat = None, True
        catalyst, etype = "", ""
        if pick.isdigit():
            row = next((r for r in rows if r["rank"] == int(pick)), None)
            if row:
                cid, no_cat = row["id"], False
                catalyst = row["headline"]
                suggested = _suggested_event_type(s) or "other"
                etype = _prompt(f"  Event type {EVENT_TYPES}\n    [{suggested}]: ",
                                suggested)
        note = _prompt("  Note (optional): ")
        _save(s["id"], False, catalyst, cid, etype or None, no_cat, note)
    print("  saved.")
    return True


def progress() -> None:
    with connect() as conn:
        r = conn.execute(
            """
            SELECT count(*) FILTER (WHERE blind) AS blind,
                   count(*) FILTER (WHERE NOT blind) AS assisted,
                   count(*) FILTER (WHERE no_catalyst) AS no_catalyst,
                   count(*) AS total
            FROM annotations
            """).fetchone()
    print(f"  annotations: {r['total']}  (blind {r['blind']}/30 target, "
          f"assisted {r['assisted']}/170 target, no-catalyst {r['no_catalyst']})")


def main() -> int:
    ap = argparse.ArgumentParser(description="Annotate swings with the true catalyst")
    ap.add_argument("--blind", action="store_true",
                    help="hide retrieval until you commit an answer (measures recall)")
    ap.add_argument("--ticker")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--swing", type=int,
                    help="annotate this swing id specifically (re-labels if needed)")
    ap.add_argument("--since",
                    help="only swings on or after this date, e.g. 2026-08-01")
    a = ap.parse_args()

    if a.progress:
        progress()
        return 0

    rows = (_one_swing(a.swing) if a.swing
            else _candidates(a.blind, a.ticker, a.limit, a.since))
    if not rows:
        print("nothing left to annotate matching that filter")
        return 0
    print(f"{len(rows)} swings queued ({'BLIND' if a.blind else 'assisted'} mode). "
          "Ctrl-C to stop.")
    for s in rows:
        annotate_one(s, a.blind)
    progress()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
