"""Rolling factor decomposition. The highest-leverage file in the repo.

r_stock = alpha + beta_mkt * r_market + beta_sector * r_sector_orth + residual

⚠️ The sector factor is ORTHOGONALISED against the market before fitting.

agent-plan.md 1.1 regresses on market and sector returns directly. On real data
those factors are heavily collinear (corr(SPY, SMH) = 0.80, corr(SPY, XLK) =
0.89), which leaves the market/sector split jointly identified but individually
meaningless — MRVL fits beta_mkt = -1.20, which is not credible for a
semiconductor. That is not numerical instability: the betas are stable across
refits and R^2 is 0.57. It is textbook multicollinearity.

It matters because the split IS the user-facing output ("Broad market accounts
for -1.1%, semiconductor weakness for -1.9%"). Orthogonalising fixes the split
and leaves the residual bit-identical, so swing detection is unaffected.

Two stages per window:
  1. r_sector = g0 + g1 * r_market + sector_orth     (purge the market)
  2. r_stock  = alpha + beta_mkt * r_market + beta_sector * sector_orth
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from swing.common import logging as log
from swing.ingest.config import market_symbol, sectors, stocks, thresholds
from swing.store.session import connect

logger = log.get("analysis.factors")


@dataclass(slots=True)
class Entity:
    """One thing we decompose. Stocks and sector ETFs share this path."""
    ticker: str
    entity_type: str            # 'stock' | 'sector'
    sector_etf: str | None
    market_etf: str
    min_history_days: int


def entities() -> list[Entity]:
    """Sector entities FIRST, so a stock attribution can reference an
    already-computed sector story. agent-plan.md 0.1b."""
    wl_defaults = thresholds()["decomposition"]
    mkt = market_symbol()
    out = [
        Entity(t, "sector", None, mkt, wl_defaults["min_history_days"])
        for t in sectors()
    ]
    out += [
        Entity(t, "stock", m.get("sector_etf"), mkt,
               int(m.get("min_history_days", wl_defaults["min_history_days"])))
        for t, m in stocks().items()
    ]
    return out


def _series(ticker: str) -> tuple[list[date], np.ndarray, np.ndarray]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT ts, close, volume FROM bars WHERE ticker=%s ORDER BY ts", (ticker,)
        ).fetchall()
    d = [r["ts"].date() for r in rows]
    close = np.array([float(r["close"]) for r in rows], dtype=float)
    vol = np.array([float(r["volume"] or 0) for r in rows], dtype=float)
    return d, close, vol


def _fit(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float]:
    """Least squares plus R^2. Fails loudly rather than returning a silent NaN."""
    if np.isnan(X).any() or np.isnan(y).any():
        raise ValueError("NaN in regression inputs")
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    denom = float(np.var(y))
    r2 = 1.0 - float(np.var(resid)) / denom if denom > 0 else 0.0
    return coef, r2


def compute(entity: Entity) -> list[dict]:
    """One row per (ticker, day). Empty list when history is insufficient."""
    cfg = thresholds()["decomposition"]
    lb = int(cfg["lookback_days"])
    volw = int(cfg["vol_window"])

    d_s, c_s, v_s = _series(entity.ticker)
    if len(d_s) < entity.min_history_days:
        logger.warning("%s: %d bars < min_history_days %d — insufficient_history",
                       entity.ticker, len(d_s), entity.min_history_days)
        return [{"ticker": entity.ticker, "d": d_s[-1] if d_s else None,
                 "status": "insufficient_history"}] if d_s else []

    d_m, c_m, _ = _series(entity.market_etf)
    common = sorted(set(d_s) & set(d_m))
    if entity.sector_etf:
        d_sec, c_sec, _ = _series(entity.sector_etf)
        common = sorted(set(common) & set(d_sec))
        sec_map = dict(zip(d_sec, c_sec, strict=True))
    s_map, m_map = dict(zip(d_s, c_s, strict=True)), dict(zip(d_m, c_m, strict=True))
    v_map = dict(zip(d_s, v_s, strict=True))

    px_s = np.array([s_map[x] for x in common])
    px_m = np.array([m_map[x] for x in common])
    r_s, r_m = np.diff(np.log(px_s)), np.diff(np.log(px_m))
    r_sec = (np.diff(np.log(np.array([sec_map[x] for x in common])))
             if entity.sector_etf else None)
    dates = common[1:]
    vol = np.array([v_map[x] for x in common[1:]])

    rows = []
    for i in range(len(dates)):
        # The fit window ends the day BEFORE the event. Never include the event
        # day in its own beta estimate. agent-plan.md 1.1.
        if i < lb + 1:
            continue
        sl = slice(i - lb, i)
        rm_w, rs_w = r_m[sl], r_s[sl]

        if r_sec is not None:
            # Stage 1: purge the market from the sector factor.
            g, _ = _fit(r_sec[sl], np.column_stack([np.ones(lb), rm_w]))
            sec_orth_w = r_sec[sl] - (g[0] + g[1] * rm_w)
            sec_orth_i = r_sec[i] - (g[0] + g[1] * r_m[i])
            X = np.column_stack([np.ones(lb), rm_w, sec_orth_w])
            coef, r2 = _fit(rs_w, X)
            alpha, b_mkt, b_sec = coef
            sec_c = float(b_sec * sec_orth_i)
        else:
            X = np.column_stack([np.ones(lb), rm_w])
            coef, r2 = _fit(rs_w, X)
            alpha, b_mkt = coef
            b_sec, sec_c = None, 0.0

        resid_hist = rs_w - X @ coef
        mkt_c = float(b_mkt * r_m[i])
        resid = float(r_s[i] - alpha - mkt_c - sec_c)

        vol_hist = resid_hist[-volw:] if len(resid_hist) >= volw else resid_hist
        rvol = float(np.std(vol_hist, ddof=1))
        z = resid / rvol if rvol > 0 else 0.0

        vslice = vol[max(0, i - 20):i]
        vstd = float(np.std(vslice, ddof=1)) if len(vslice) > 1 else 0.0
        vz = float((vol[i] - np.mean(vslice)) / vstd) if vstd > 0 else 0.0

        ret = float(r_s[i])
        rows.append({
            "ticker": entity.ticker, "d": dates[i], "ret": ret,
            "alpha": float(alpha), "beta_mkt": float(b_mkt),
            "beta_sector": float(b_sec) if b_sec is not None else None,
            "r_squared": float(r2), "market_component": mkt_c,
            "sector_component": sec_c, "residual": resid,
            "residual_vol_60": rvol, "residual_z": float(z), "volume_z": vz,
            # Raw per-day ratio. Can exceed 1 when components offset, so
            # `swing compare` uses a variance-based measure instead.
            "idio_share": abs(resid) / abs(ret) if ret != 0 else None,
            "status": "ok",
        })
    return rows


UPSERT = """
INSERT INTO daily_factors (ticker, d, ret, alpha, beta_mkt, beta_sector, r_squared,
                           market_component, sector_component, residual,
                           residual_vol_60, residual_z, volume_z, idio_share, status)
VALUES (%(ticker)s, %(d)s, %(ret)s, %(alpha)s, %(beta_mkt)s, %(beta_sector)s,
        %(r_squared)s, %(market_component)s, %(sector_component)s, %(residual)s,
        %(residual_vol_60)s, %(residual_z)s, %(volume_z)s, %(idio_share)s, %(status)s)
ON CONFLICT (ticker, d) DO UPDATE SET
  ret=EXCLUDED.ret, alpha=EXCLUDED.alpha, beta_mkt=EXCLUDED.beta_mkt,
  beta_sector=EXCLUDED.beta_sector, r_squared=EXCLUDED.r_squared,
  market_component=EXCLUDED.market_component, sector_component=EXCLUDED.sector_component,
  residual=EXCLUDED.residual, residual_vol_60=EXCLUDED.residual_vol_60,
  residual_z=EXCLUDED.residual_z, volume_z=EXCLUDED.volume_z,
  idio_share=EXCLUDED.idio_share, status=EXCLUDED.status
"""


def rebuild_all() -> dict[str, int]:
    """Recompute daily_factors for every entity, sectors first."""
    out = {}
    for e in entities():
        rows = [r for r in compute(e) if r.get("d")]
        if rows:
            with connect() as conn, conn.cursor() as cur:
                cur.executemany(UPSERT, [{**{k: None for k in (
                    "ret", "alpha", "beta_mkt", "beta_sector", "r_squared",
                    "market_component", "sector_component", "residual",
                    "residual_vol_60", "residual_z", "volume_z", "idio_share")}, **r}
                    for r in rows])
        out[e.ticker] = len(rows)
        logger.info("%s (%s): %d factor rows", e.ticker, e.entity_type, len(rows))
    return out
