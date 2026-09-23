"""Find swings likely to carry the event types nothing has ever labelled.

Phase 5.2 shipped TF-IDF because DistilBERT could not beat it, and §16.25 said
why: the win is in labels, not architecture. Three of the ten event types have
**zero** examples and two more have almost none:

    regulatory 0   litigation 0   management 0   macro 2   guidance 3
    (earnings 16, product 16, other 9, analyst_action 5, m_and_a 4)

A classifier cannot learn a class it has never seen, and macro-F1 weights every
class equally, so those five are most of what is holding 5.2 back. There are 312
unlabelled swings; hunting through them by hand for a litigation case is the
expensive part, so this ranks them by what their pre-move coverage actually says.

⚠️ LABEL THESE IN ASSISTED MODE, NOT BLIND. Choosing which swings to label by
keyword is a biased sample — fine for training a classifier, wrong for measuring
recall. `recall@10` is computed only over `blind=true` rows, so a keyword-picked
swing must not join that set, or the measurement quietly becomes "recall over
cases we could already describe".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from swing.common import logging as log

logger = log.get("eval.classgaps")

#: Phrases that suggest an event type, used only to ORDER the labelling queue.
#: The annotator decides the label; these decide who gets looked at first.
CLASS_PATTERNS: dict[str, tuple[str, ...]] = {
    "regulatory": (
        r"\bFTC\b", r"\bDOJ\b", r"\bSEC\b charges?", r"antitrust", r"regulat\w+",
        r"export controls?", r"sanction\w*", r"tariffs?", r"probe", r"investigat\w+",
        r"\bEU\b (?:fine|charges|complaint)", r"Digital Markets Act", r"subpoena",
    ),
    "litigation": (
        r"lawsuit", r"\bsued?\b", r"sues\b", r"verdict", r"settlement", r"settles?\b",
        r"patent (?:suit|infringement|dispute)", r"class action", r"court\b",
        r"judge\b", r"appeal\w*", r"damages",
    ),
    "management": (
        r"\bCEO\b", r"\bCFO\b", r"\bCTO\b", r"chief executive", r"steps? down",
        r"resign\w*", r"appoint\w+", r"names? .{0,20}(?:chief|president)",
        r"succession", r"departure", r"ousted", r"interim",
    ),
    "macro": (
        r"\bFed\b", r"Federal Reserve", r"inflation", r"\bCPI\b", r"rate (?:cut|hike)",
        r"jobs report", r"payrolls", r"yields?\b", r"recession", r"selloff",
        r"market[- ]wide", r"stocks? (?:slide|rally|tumble)", r"treasury",
    ),
    "guidance": (
        r"guidance", r"outlook", r"forecast\w*", r"raises? .{0,20}(?:target|estimate)",
        r"cuts? .{0,20}(?:outlook|forecast)", r"warns?\b", r"expects? .{0,20}revenue",
        r"full[- ]year", r"next quarter",
    ),
}


@dataclass
class Suggestion:
    swing_id: int
    ticker: str
    d: object
    residual_z: float
    hits: int
    evidence: list[str] = field(default_factory=list)

    def line(self) -> str:
        head = (f"    swing {self.swing_id:<5} {self.ticker:<6} {self.d}  "
                f"z={self.residual_z:+.1f}  ({self.hits} matching headline"
                f"{'s' if self.hits != 1 else ''})")
        return head + "\n" + "\n".join(f"        {e[:96]}" for e in self.evidence[:2])


def label_counts() -> dict[str, int]:
    from swing.eval.annotate import EVENT_TYPES
    from swing.store.session import connect

    with connect() as conn:
        rows = conn.execute(
            "SELECT true_event_type t, count(*) n FROM annotations "
            "WHERE NOT no_catalyst AND true_event_type IS NOT NULL GROUP BY 1").fetchall()
    have = {r["t"]: r["n"] for r in rows}
    return {t: have.get(t, 0) for t in EVENT_TYPES}


def _unlabelled_with_coverage() -> list[dict]:
    """Unlabelled swings and the text of their pre-move coverage."""
    from swing.store.session import connect

    with connect() as conn:
        return [dict(r) for r in conn.execute(
            """
            SELECT s.id, s.ticker, s.d, s.residual_z,
                   array_agg(a.headline ORDER BY c.rank) AS headlines
            FROM swings s
            LEFT JOIN annotations n ON n.swing_id = s.id
            JOIN clusters c ON c.swing_id = s.id AND c.timing = 'pre_move'
            JOIN articles a ON a.id = c.canonical_article
            WHERE n.swing_id IS NULL AND s.onset_ts IS NOT NULL AND c.rank <= 10
            GROUP BY s.id, s.ticker, s.d, s.residual_z
            """).fetchall()]


def suggest(event_type: str, limit: int = 8,
            rows: list[dict] | None = None) -> list[Suggestion]:
    """Unlabelled swings whose pre-move coverage reads like `event_type`."""
    patterns = [re.compile(p, re.IGNORECASE) for p in CLASS_PATTERNS[event_type]]
    out: list[Suggestion] = []
    for row in (rows if rows is not None else _unlabelled_with_coverage()):
        matched = [h for h in (row["headlines"] or [])
                   if h and any(p.search(h) for p in patterns)]
        if matched:
            out.append(Suggestion(row["id"], row["ticker"], row["d"],
                                  float(row["residual_z"] or 0), len(matched), matched))
    # Most matches first, then biggest move: a clear-cut case is faster to label.
    out.sort(key=lambda s: (-s.hits, -abs(s.residual_z)))
    return out[:limit]


def report(limit: int = 6, only: str | None = None) -> str:
    counts = label_counts()
    targets = [t for t in CLASS_PATTERNS if (only is None or t == only)]
    rows = _unlabelled_with_coverage()
    lines = [(f"\nEvent-type labels — {sum(counts.values())} across "
              f"{len(counts)} types\n")]
    for t, n in sorted(counts.items(), key=lambda kv: kv[1]):
        flag = "   <- nothing to learn from" if n == 0 else ""
        lines.append(f"  {t:<16}{n:>4}{flag}")
    lines.append(f"\n{len(rows)} unlabelled swings have pre-move coverage. "
                 "Candidates for the thin classes:")
    for t in targets:
        picks = suggest(t, limit, rows)
        lines.append(f"\n  {t.upper()}  (have {counts.get(t, 0)})")
        lines += [p.line() for p in picks] or ["    (nothing matched)"]
    lines += [
        "",
        "  Label one with:  swing annotate --swing <id>",
        "",
        "  ⚠ ASSISTED mode, not --blind. Swings picked by keyword are a biased",
        "    sample: fine for training the 5.2 classifier, wrong for recall@10,",
        "    which is measured only over blind rows.",
    ]
    return "\n".join(lines)


def main(limit: int = 6, only: str | None = None) -> int:
    print(report(limit, only))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
