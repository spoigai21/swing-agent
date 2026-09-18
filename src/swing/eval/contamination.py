"""Step 4.4 — flag evaluation cases the model may simply remember.

agent-plan.md: "Your LLM was trained on data covering your historical test
period. When you ask it to explain a well-known 2024 move, part of the answer may
be recall rather than inference from the articles you supplied. Two mitigations:
weight evaluation toward obscure tickers and minor moves unlikely to be
memorized, and treat the placebo test — which cannot be passed by recall — as
your primary honesty signal."

The second mitigation was already in place: the placebo test is the primary
signal and cannot be passed by recall, because the evidence shown belongs to a
different week. This module adds the first, as a DIAGNOSTIC rather than a filter.

⚠️ **This is a heuristic proxy, not a measurement.** Nothing here observes what
the model memorised; it cannot. It ranks cases by how likely they are to have
been widely written about — mega-cap name, large move, earnings day — so that a
metric can be read on the obscure tail as well as the whole set. If accuracy
holds up on the low-risk subset, recall is not doing the work. If it collapses,
that is worth knowing and no single number would have shown it.

⚠️ Deliberately NOT in `eval/placebo.py`: that file is hashed into `eval_hash`,
so changing it discards every accumulated Gate 4 case. Contamination scoring
must not cost the run.
"""
from __future__ import annotations

from dataclasses import dataclass

# How heavily a name is written about. Not market cap — press volume. A model
# has read far more about Apple and Tesla than about Sandisk or Take-Two.
PROMINENCE = {
    "AAPL": 1.0, "TSLA": 1.0, "NVDA": 1.0, "GOOGL": 0.9, "AMZN": 0.9, "MSFT": 0.9,
    "META": 0.9, "NFLX": 0.7, "AMD": 0.7, "INTC": 0.7, "MU": 0.5, "AVGO": 0.5,
    "QCOM": 0.5, "SBUX": 0.5, "DIS": 0.5, "MRVL": 0.3, "TTWO": 0.3, "SNDK": 0.2,
    "RBLX": 0.3, "U": 0.2, "RIVN": 0.3, "GM": 0.4, "F": 0.4, "SONY": 0.4,
}
DEFAULT_PROMINENCE = 0.3        # an unlisted ticker is, by construction, obscure
LOW_RISK_MAX = 0.45             # at or below this, memorisation is unlikely


@dataclass(frozen=True, slots=True)
class Risk:
    swing_id: int
    ticker: str
    score: float
    reasons: tuple[str, ...]

    @property
    def low(self) -> bool:
        return self.score <= LOW_RISK_MAX


def score(ticker: str, residual_z: float | None, earnings_mode: bool = False) -> Risk:
    """0 = obscure and unmemorable, 1 = the move everyone wrote about."""
    prominence = PROMINENCE.get((ticker or "").upper(), DEFAULT_PROMINENCE)
    reasons: list[str] = []
    if prominence >= 0.7:
        reasons.append("mega-cap name")

    # A 6-sigma move gets wall-to-wall coverage; a 2-sigma one barely registers.
    # float() at the boundary: Postgres hands back Decimal, which will not mix
    # with float arithmetic.
    z = abs(float(residual_z or 0.0))
    magnitude = min(1.0, max(0.0, (z - 2.0) / 4.0))
    if z >= 4.0:
        reasons.append(f"|z|={z:.1f} move")

    earnings = 0.0
    if earnings_mode:
        earnings = 1.0
        reasons.append("earnings day")

    # Prominence dominates: an obscure ticker's big move still gets little ink.
    value = 0.55 * prominence + 0.30 * magnitude + 0.15 * earnings
    return Risk(swing_id=0, ticker=(ticker or "").upper(), score=round(value, 3),
                reasons=tuple(reasons))


def annotated_risks() -> list[Risk]:
    """Contamination risk for every blind-annotated swing."""
    from swing.store.session import connect

    with connect() as conn:
        rows = conn.execute(
            "SELECT a.swing_id, s.ticker, s.residual_z, s.earnings_mode "
            "FROM annotations a JOIN swings s ON s.id = a.swing_id "
            "WHERE a.blind ORDER BY a.swing_id").fetchall()
    out = []
    for r in rows:
        base = score(r["ticker"], r["residual_z"], bool(r["earnings_mode"]))
        out.append(Risk(swing_id=r["swing_id"], ticker=base.ticker,
                        score=base.score, reasons=base.reasons))
    return out


def summary() -> dict[str, object]:
    risks = annotated_risks()
    if not risks:
        return {"n": 0}
    low = [r for r in risks if r.low]
    return {
        "n": len(risks),
        "low_risk": len(low),
        "high_risk": len(risks) - len(low),
        "mean_score": round(sum(r.score for r in risks) / len(risks), 3),
        "low_risk_swing_ids": [r.swing_id for r in low],
    }


def main() -> int:
    s = summary()
    if not s.get("n"):
        print("  no blind annotations to score")
        return 0
    print(f"  {s['n']} annotated swings   mean contamination risk {s['mean_score']}")
    print(f"  low risk (<= {LOW_RISK_MAX}): {s['low_risk']}    high risk: {s['high_risk']}")
    print("\n  ⚠️ A heuristic proxy for press volume, not a measurement of what the")
    print("     model memorised. Read metrics on the low-risk subset as a check that")
    print("     recall is not doing the work the evidence should be doing.")
    for r in sorted(annotated_risks(), key=lambda x: -x.score)[:8]:
        why = ", ".join(r.reasons) or "—"
        print(f"    #{r.swing_id:<5} {r.ticker:<6} {r.score:.2f}  {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
