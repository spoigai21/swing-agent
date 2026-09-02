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


def _candidates(blind: bool, ticker: str | None, limit: int) -> list[dict]:
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
    sql += " ORDER BY (SELECT count(*) FROM clusters c WHERE c.swing_id=s.id) DESC, "
    sql += "abs(s.residual_z) DESC LIMIT %s"
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
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO annotations (swing_id, blind, true_catalyst, true_cluster_id,
                                     true_event_type, no_catalyst, annotator_note)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (swing_id) DO UPDATE SET
              blind=EXCLUDED.blind, true_catalyst=EXCLUDED.true_catalyst,
              true_cluster_id=EXCLUDED.true_cluster_id,
              true_event_type=EXCLUDED.true_event_type,
              no_catalyst=EXCLUDED.no_catalyst, annotator_note=EXCLUDED.annotator_note
            """,
            (swing_id, blind, catalyst or None, cluster_id, event_type, no_catalyst,
             note or None),
        )


def annotate_one(s: dict, blind: bool) -> bool:
    _show_swing(s)
    if blind:
        print("\n  BLIND MODE — retrieval is hidden until you commit an answer.")
        print("  Research this independently: read that day's coverage, check the")
        print("  8-K on EDGAR, search the web. Then answer.\n")
        catalyst = _prompt("  True catalyst (blank = none found): ")
        no_cat = not catalyst
        etype = ""
        if not no_cat:
            etype = _prompt(f"  Event type {EVENT_TYPES}: ", "other")
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
                etype = _prompt(f"  Event type {EVENT_TYPES}: ", "other")
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
    a = ap.parse_args()

    if a.progress:
        progress()
        return 0

    rows = _candidates(a.blind, a.ticker, a.limit)
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
