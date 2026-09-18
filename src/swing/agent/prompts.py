"""Prompt loading and rendering. Versioned, because the version is recorded on
every stored attribution (agent-plan.md 3.1b)."""
from __future__ import annotations

from datetime import datetime
from functools import lru_cache

from swing.paths import CONFIG

PROMPT_DIR = CONFIG / "prompts"

EARNINGS_INSTRUCTION = """
## Earnings mode

This swing coincides with an earnings release (8-K Item 2.02), so the catalyst
is already known. Your job is not to find it but to CHARACTERISE it: what in the
release moved the stock? Guidance changes typically move a stock more than the
reported quarter does, so say which one drove it if the evidence supports that.
Use `event_type` "earnings" only for the reported results; use "guidance" when
forward-looking commentary is the driver.
""".strip()


def _earnings_block(ticker: str, day) -> str:
    """The earnings instruction, plus the company's own filed figures.

    Step 1.7 asks the agent to CHARACTERISE a known catalyst, which it cannot do
    from an instruction alone — it needs the numbers. These come from SEC XBRL
    (`analysis.earnings`) and are the company's own filings, never a consensus
    comparison, which is why the line says so explicitly.

    ⚠️ Never raises: an SEC outage must not fail an attribution. Without figures
    the model still gets the instruction, which is what it had before.
    """
    try:
        from swing.analysis.earnings import characterize

        filed = characterize(ticker, day)
    except Exception:  # noqa: BLE001 - a prompt build must never lose an answer
        # Narrowing is the house style where failures are enumerable, but this
        # path can raise httpx transport errors, _raise_for_retryable, JSON
        # ValueError, cache OSError or an unmapped-CIK KeyError. Missing one
        # member of that list would turn a prompt build into a lost attribution,
        # so breadth is deliberate here — as in batch.py and cli.py.
        filed = ""
    if not filed:
        return EARNINGS_INSTRUCTION
    return f"{EARNINGS_INSTRUCTION}\n\nFiled figures for this quarter: {filed}"


@lru_cache(maxsize=8)
def load(version: str | None = None) -> tuple[str, str]:
    """Return (version, template). Defaults to the highest-numbered version."""
    files = sorted(PROMPT_DIR.glob("attribution_v*.md"))
    if not files:
        raise FileNotFoundError(f"no attribution prompt in {PROMPT_DIR}")
    chosen = next((f for f in files if f.stem == version), files[-1]) if version else files[-1]
    return chosen.stem, chosen.read_text()


def _about(cluster: dict, own: set[str] | None) -> str:
    """An `about:` line for news that is only about a related company.

    Without it, "INTC 8-K — Item 2.02" in QCOM's evidence reads as if it were
    QCOM's own disclosure."""
    tickers = set(cluster.get("tickers") or [])
    if not own or not tickers or tickers & own:
        return ""
    from swing.ingest.config import company_name

    names = ", ".join(f"{company_name(t)} ({t})" for t in sorted(tickers))
    return (f"\n    about: {names}. This is news about a RELATED company, "
            f"not {'/'.join(sorted(own))}")


def _relative(published: datetime, onset: datetime | None) -> str:
    """'26h before the move started' / '19 days before ...' / '2h after ...'.

    ⚠️ Raw UTC stamps left the model to do date arithmetic. On 2026-09-14 Gate 3
    failed because a Broadcom 8-K from weeks earlier was offered as the cause of
    a move; stating the distance makes staleness impossible to overlook
    (agent-plan.md 3.5).
    """
    if onset is None:
        return ""
    hours = (onset - published).total_seconds() / 3600
    side = "before" if hours >= 0 else "after"
    h = abs(hours)
    span = f"{h:.0f}h" if h < 48 else f"{h / 24:.0f} days"
    return f"  ({span} {side} the move started)"


def _fmt_clusters(clusters: list[dict], own: set[str] | None = None,
                  onset: datetime | None = None) -> str:
    """Numbered list with timing impossible to overlook. agent-plan.md 3.5."""
    if not clusters:
        return "(none)"
    out = []
    for c in clusters:
        published = c["earliest_published"]
        is_dt = isinstance(published, datetime)
        ts = published.strftime("%Y-%m-%d %H:%M UTC") if is_dt else str(published)
        when = _relative(published, onset) if is_dt else ""
        out.append(
            f"[cluster_id={c['id']}]  timing={c['timing']}  tier={c['best_tier']}  "
            f"published={ts}{when}  distinct_sources={c['distinct_sources']}\n"
            f"    source: {c['source']}\n"
            f"    headline: {c['headline']}"
            + (f"\n    summary: {c['summary'][:300]}" if c.get("summary") else "")
            + _about(c, own)
        )
    return "\n\n".join(out)


def render(swing: dict, decomposition_sentence: str,
           pre: list[dict], post: list[dict], version: str | None = None) -> tuple[str, str]:
    from swing.analysis.retrieval import own_tickers

    own = set(own_tickers(swing))
    ver, template = load(version)
    return ver, template.format(
        decomposition=decomposition_sentence,
        ticker=swing["ticker"], date=swing["d"],
        total_return=float(swing["total_return"]),
        market_component=float(swing["market_component"]),
        sector_component=float(swing["sector_component"]),
        residual=float(swing["residual"]), residual_z=float(swing["residual_z"]),
        swing_type=swing["swing_type"],
        onset_ts=swing["onset_ts"].strftime("%Y-%m-%d %H:%M UTC") if swing["onset_ts"] else "unknown",
        volume_z=float(swing["volume_z"] or 0.0),
        earnings_mode=swing["earnings_mode"],
        pre_clusters=_fmt_clusters(pre, own, swing["onset_ts"]),
        post_clusters=_fmt_clusters(post, own, swing["onset_ts"]),
        earnings_instruction=(_earnings_block(swing["ticker"], swing["d"])
                              if swing["earnings_mode"] else ""),
    )
