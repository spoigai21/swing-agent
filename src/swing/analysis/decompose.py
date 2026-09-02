"""Single-day decomposition, and the sentence a user actually reads.

This output feeds the explanation directly and is what prevents the system's
most common failure: blaming a company headline for a sector-wide selloff.
agent-plan.md 1.1.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from swing.store.session import connect


@dataclass(slots=True)
class Decomposition:
    ticker: str
    d: date
    total_return: float
    market_component: float
    sector_component: float
    residual: float
    residual_z: float
    beta_mkt: float | None
    beta_sector: float | None
    r_squared: float | None
    volume_z: float | None
    status: str

    def sentence(self) -> str:
        """The user-facing decomposition line."""
        if self.status != "ok":
            return f"{self.ticker} on {self.d}: {self.status.replace('_', ' ')}."
        direction = "rose" if self.total_return >= 0 else "fell"
        parts = [
            f"{self.ticker} {direction} {abs(self.total_return) * 100:.1f}% on {self.d}."
        ]
        parts.append(f"Broad market accounts for {self.market_component * 100:+.1f}%")
        if self.beta_sector is not None:
            parts.append(f"sector for {self.sector_component * 100:+.1f}%")
        parts.append(f"The stock-specific residual is {self.residual * 100:+.1f}% "
                     f"(z = {self.residual_z:+.1f}).")
        return " ".join([parts[0], ", ".join(parts[1:-1]) + ".", parts[-1]])


def load(ticker: str, d: date) -> Decomposition | None:
    with connect() as conn:
        r = conn.execute(
            "SELECT * FROM daily_factors WHERE ticker=%s AND d=%s", (ticker, d)
        ).fetchone()
    if not r:
        return None
    def f(k):
        return float(r[k]) if r[k] is not None else None

    return Decomposition(
        ticker=r["ticker"], d=r["d"], total_return=f("ret") or 0.0,
        market_component=f("market_component") or 0.0,
        sector_component=f("sector_component") or 0.0,
        residual=f("residual") or 0.0, residual_z=f("residual_z") or 0.0,
        beta_mkt=f("beta_mkt"), beta_sector=f("beta_sector"),
        r_squared=f("r_squared"), volume_z=f("volume_z"), status=r["status"],
    )
