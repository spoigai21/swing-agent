"""Re-check uncovered labels against a corpus that has since grown.

`catalyst_coverage` counts annotation rows whose `true_article_ids` is non-empty.
That column was written when a human annotated, so **new articles cannot change
it**: the 12-month GDELT backfill added 2,742 wire stories and coverage stayed at
0.709 to three decimals, because the metric has no way to notice.

An annotator recorded `true_catalyst` as free text even when no article for it
existed at the time. Those sentences are the query: for each uncovered swing this
searches the pre-move window of today's corpus for articles that match, and
prints them for a human to confirm.

    swing recheck-coverage                  # show candidates
    swing recheck-coverage --link 447=12345,12346

⚠️ IT NEVER LINKS ANYTHING BY ITSELF. These rows are the held-out answer key.
Auto-linking on similarity would let the retrieval stack choose its own ground
truth — the catalyst it ranks highest would become the catalyst it is scored
against, and coverage would rise by construction. A wrong link is worse than a
missing one, because it is invisible afterwards.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from swing.common import logging as log

logger = log.get("eval.recheck")

TOP_N = 6
#: Below this cosine the match is not worth a human's attention.
MIN_SIMILARITY = 0.35


@dataclass
class Candidate:
    article_id: int
    similarity: float
    published_at: object
    source: str
    tier: int
    headline: str

    def line(self) -> str:
        return (f"      [{self.article_id:>6}] {self.similarity:.2f}  "
                f"{self.published_at:%Y-%m-%d %H:%M}Z  tier {self.tier}  "
                f"{self.source:<16} {self.headline[:70]}")


@dataclass
class Uncovered:
    swing_id: int
    ticker: str
    d: object
    true_catalyst: str
    candidates: list[Candidate] = field(default_factory=list)
    note: str = ""

    def block(self) -> str:
        head = (f"\n  swing {self.swing_id}  {self.ticker}  {self.d}\n"
                f"    labelled catalyst: {self.true_catalyst[:110]}")
        if self.note:
            return f"{head}\n      {self.note}"
        if not self.candidates:
            return f"{head}\n      no article in the window matches — still uncovered"
        return head + "\n" + "\n".join(c.line() for c in self.candidates)


def uncovered_rows() -> list[dict]:
    from swing.store.session import connect

    with connect() as conn:
        return [dict(r) for r in conn.execute(
            """
            SELECT a.swing_id, a.true_catalyst, s.ticker, s.d
            FROM annotations a JOIN swings s ON s.id = a.swing_id
            WHERE a.blind AND NOT a.no_catalyst
              AND coalesce(cardinality(a.true_article_ids), 0) = 0
              AND a.true_catalyst IS NOT NULL
            ORDER BY s.d DESC
            """).fetchall()]


def _window_articles(swing_id: int) -> list[dict]:
    """The same pre-move window retrieval would use, unclustered."""
    from datetime import UTC, datetime, time, timedelta

    from swing.analysis.retrieval import (
        _articles_in,
        _prev_close_ts,
        _swing_row,
        admissible,
        own_tickers,
        retrieval_tickers,
        windows_for,
    )
    from swing.ingest.config import thresholds

    swing = _swing_row(swing_id)
    if not swing or swing["onset_ts"] is None:
        return []
    swing = dict(swing)
    onset = swing["onset_ts"]
    drift_start = (onset - timedelta(days=int(swing["drift_window"]))
                   if swing["kind"] == "drift" and swing["drift_window"] else None)
    pre_w, _ = windows_for(
        swing["swing_type"], onset, _prev_close_ts(swing["ticker"], swing["d"]),
        datetime.combine(swing["d"], time(13, 30), tzinfo=UTC), drift_start)
    max_tier = int(thresholds()["retrieval"]["exclude_tier"]) - 1
    own = own_tickers(swing)
    return [a for a in _articles_in(retrieval_tickers(swing), pre_w.start,
                                    pre_w.end, max_tier)
            if admissible(a, own)]


def match(catalyst: str, articles: list[dict], top_n: int = TOP_N,
          min_similarity: float = MIN_SIMILARITY) -> list[Candidate]:
    """Rank window articles by similarity to the annotator's own sentence."""
    from swing.common.vectors import as_array
    from swing.ingest.normalize import _embed_texts

    if not articles or not catalyst.strip():
        return []
    q = np.asarray(_embed_texts([catalyst])[0], dtype=float)
    qn = np.linalg.norm(q)
    if qn == 0:
        return []
    out: list[Candidate] = []
    for a in articles:
        v = as_array(a["embedding"])
        vn = np.linalg.norm(v)
        if vn == 0:
            continue
        sim = float(v @ q / (vn * qn))
        if sim >= min_similarity:
            out.append(Candidate(a["id"], sim, a["published_at"], a["source"],
                                 int(a["source_tier"] or 4), a["headline"] or ""))
    out.sort(key=lambda c: -c.similarity)
    return out[:top_n]


def scan(min_similarity: float = MIN_SIMILARITY) -> list[Uncovered]:
    results: list[Uncovered] = []
    for row in uncovered_rows():
        item = Uncovered(row["swing_id"], row["ticker"], row["d"],
                         row["true_catalyst"] or "")
        try:
            articles = _window_articles(row["swing_id"])
        except Exception as exc:   # noqa: BLE001 — one bad swing must not stop the sweep
            item.note = f"window unavailable: {exc}"
            results.append(item)
            continue
        if not articles:
            item.note = "no admissible articles in the pre-move window at all"
        else:
            item.candidates = match(item.true_catalyst, articles,
                                    min_similarity=min_similarity)
        results.append(item)
    return results


def link(swing_id: int, article_ids: list[int]) -> str:
    """Attach confirmed articles to a label. Human-confirmed ids only.

    Records that it was a re-check rather than an original annotation, so a
    later reader can tell which labels were written against which corpus.
    """
    from swing.store.session import connect

    with connect() as conn:
        found = conn.execute(
            "SELECT id FROM articles WHERE id = ANY(%s)", (article_ids,)).fetchall()
        missing = sorted(set(article_ids) - {r["id"] for r in found})
        if missing:
            return f"no such article id(s): {missing} — nothing written"
        row = conn.execute(
            """
            UPDATE annotations
               SET true_article_ids = %s,
                   annotator_note = coalesce(annotator_note || ' | ', '')
                                    || 'recheck: linked against the enlarged corpus'
             WHERE swing_id = %s
               AND coalesce(cardinality(true_article_ids), 0) = 0
            RETURNING swing_id
            """, (sorted(article_ids), swing_id)).fetchone()
    if not row:
        return (f"swing {swing_id}: not updated — either it has no annotation, or it "
                "already has articles linked (this never overwrites an existing label)")
    return f"swing {swing_id}: linked {len(article_ids)} article(s)"


def report(min_similarity: float = MIN_SIMILARITY) -> str:
    items = scan(min_similarity)
    if not items:
        return "\nNo uncovered labels: every annotated catalyst already has articles."
    with_any = [i for i in items if i.candidates]
    lines = [
        f"\n{len(items)} labelled swing(s) whose catalyst is not linked to any article.",
        "Candidates below come from TODAY's corpus, which has grown since they were",
        "annotated. Confirm one yourself before linking it.\n",
    ]
    lines += [i.block() for i in items]
    lines += [
        "",
        f"  {len(with_any)} of {len(items)} now have at least one plausible match.",
        ("  Link a confirmed one with:  swing recheck-coverage --link "
         "<swing_id>=<article_id>[,<id>...]"),
        "",
        "  ⚠ Read the article before linking. These rows are the held-out answer key,",
        "    and a wrong link is invisible once written.",
    ]
    return "\n".join(lines)


def main(link_spec: str | None = None, min_similarity: float = MIN_SIMILARITY) -> int:
    if link_spec:
        swing_part, _, ids_part = link_spec.partition("=")
        if not ids_part:
            print("expected --link <swing_id>=<article_id>[,<article_id>...]")
            return 2
        try:
            ids = [int(x) for x in ids_part.split(",") if x.strip()]
            print(link(int(swing_part), ids))
        except ValueError:
            print("swing id and article ids must be integers")
            return 2
        return 0
    print(report(min_similarity))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
