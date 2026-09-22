"""Which sources actually explain moves?

The publisher tier table in sources.yaml is hand-written: someone decided
Reuters is tier 1 and Yahoo is tier 4, and every ranking decision leans on it.
Nothing has ever checked those judgements against outcomes, even though the
system records exactly what it needs to:

    cited      the source appeared in the evidence for a chosen catalyst
    confirmed  a human annotation agreed that catalyst was the real one

That ratio is a source's precision, measured rather than assumed. It is the one
learning signal that compounds: the corpus grows, the estimate sharpens, and it
needs no model calls and no labels beyond the annotations already collected.

    swing source-precision

⚠️ REPORTS ONLY. Rewriting tiers from these numbers would be a feedback loop —
tiers drive ranking, ranking drives what gets cited, citations drive these
numbers — and at n=12 citations it would be fitting noise into the config that
every future measurement depends on.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from swing.common import logging as log

logger = log.get("eval.sources")

#: Below this, a source's precision is a coin flip dressed as a number.
MIN_CITATIONS_TO_JUDGE = 10


@dataclass
class SourceStat:
    source: str
    tier: int
    cited: int = 0
    confirmed: int = 0

    @property
    def precision(self) -> float | None:
        return self.confirmed / self.cited if self.cited else None

    @property
    def judgeable(self) -> bool:
        return self.cited >= MIN_CITATIONS_TO_JUDGE

    def line(self) -> str:
        p = "  n/a" if self.precision is None else f"{self.precision:5.2f}"
        note = "" if self.judgeable else "   (too few to judge)"
        return (f"  tier {self.tier}  {self.source:<22} cited {self.cited:>4}  "
                f"confirmed {self.confirmed:>4}  precision {p}{note}")


def _payload(raw) -> dict:
    return raw if isinstance(raw, dict) else json.loads(raw)


def collect() -> list[SourceStat]:
    """Per-source cited/confirmed counts over annotated swings."""
    from swing.eval.harness import top_candidate_hit
    from swing.store.session import connect

    stats: dict[str, SourceStat] = {}
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT a.swing_id, a.true_cluster_id, a.true_article_ids, at.payload
            FROM annotations a
            JOIN LATERAL (SELECT payload FROM attributions
                          WHERE swing_id=a.swing_id AND run_kind='production'
                          ORDER BY created_at DESC LIMIT 1) at ON true
            WHERE NOT a.no_catalyst
              AND (a.true_article_ids IS NOT NULL OR a.true_cluster_id IS NOT NULL)
            """).fetchall()
        for r in rows:
            payload = _payload(r["payload"])
            cands = payload.get("candidates") or []
            if not cands:
                continue
            cited_ids = [e["cluster_id"] for e in (cands[0].get("evidence") or [])]
            if not cited_ids:
                continue
            members: dict[int, set[int]] = {}
            for m in conn.execute(
                    "SELECT cluster_id, article_id FROM cluster_members "
                    "WHERE cluster_id = ANY(%s)", (cited_ids,)).fetchall():
                members.setdefault(m["cluster_id"], set()).add(m["article_id"])
            correct = bool(top_candidate_hit(payload, r["true_cluster_id"],
                                             r["true_article_ids"], members))
            # One vote per (source, case): a story carried by six outlets must
            # not count as six pieces of evidence for that source's precision.
            seen: set[str] = set()
            for cl in conn.execute(
                    "SELECT c.id, a.source, a.source_tier FROM clusters c "
                    "JOIN articles a ON a.id = c.canonical_article "
                    "WHERE c.id = ANY(%s)", (cited_ids,)).fetchall():
                if cl["source"] in seen:
                    continue
                seen.add(cl["source"])
                st = stats.setdefault(cl["source"],
                                      SourceStat(cl["source"], int(cl["source_tier"] or 4)))
                st.cited += 1
                st.confirmed += correct
    return sorted(stats.values(), key=lambda s: (s.tier, -s.cited, s.source))


def tier_disagreements(stats: list[SourceStat]) -> list[str]:
    """Where measured precision contradicts the hand-written tier.

    Only sources with enough citations to judge; everything else is noise.
    """
    out = []
    judgeable = [s for s in stats if s.judgeable]
    for s in judgeable:
        if s.tier <= 2 and s.precision is not None and s.precision < 0.5:
            out.append(f"{s.source} is tier {s.tier} but confirmed only "
                       f"{s.precision:.0%} of {s.cited} citations")
        if s.tier >= 3 and s.precision is not None and s.precision >= 0.9:
            out.append(f"{s.source} is tier {s.tier} but confirmed "
                       f"{s.precision:.0%} of {s.cited} citations")
    return out


def report() -> str:
    stats = collect()
    total = sum(s.cited for s in stats)
    if not total:
        return ("\nNo cited-and-annotated evidence yet. This needs annotated swings the "
                "agent answered with a catalyst — run `swing annotate` and the accuracy "
                "backlog first.")
    lines = [f"\nSource precision — {total} citation(s) over annotated swings\n"]
    lines += [s.line() for s in stats]
    judgeable = [s for s in stats if s.judgeable]
    lines.append("")
    if not judgeable:
        lines.append(f"  No source has the {MIN_CITATIONS_TO_JUDGE} citations needed to "
                     "judge it yet. These counts are a starting point, not a verdict.")
    else:
        dis = tier_disagreements(stats)
        lines += ([f"  ⚠ {d}" for d in dis] if dis
                  else ["  No source with enough citations contradicts its tier."])
    lines.append("\n  ⚠ Reporting only. Rewriting tiers from these numbers is a feedback "
                 "loop:\n    tiers drive ranking, ranking drives citations, citations "
                 "drive these numbers.")
    return "\n".join(lines)


def main() -> int:
    print(report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
