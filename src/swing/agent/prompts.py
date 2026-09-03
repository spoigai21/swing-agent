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


@lru_cache(maxsize=8)
def load(version: str | None = None) -> tuple[str, str]:
    """Return (version, template). Defaults to the highest-numbered version."""
    files = sorted(PROMPT_DIR.glob("attribution_v*.md"))
    if not files:
        raise FileNotFoundError(f"no attribution prompt in {PROMPT_DIR}")
    chosen = next((f for f in files if f.stem == version), files[-1]) if version else files[-1]
    return chosen.stem, chosen.read_text()


def _fmt_clusters(clusters: list[dict]) -> str:
    """Numbered list with timing impossible to overlook. agent-plan.md 3.5."""
    if not clusters:
        return "(none)"
    out = []
    for c in clusters:
        published = c["earliest_published"]
        ts = published.strftime("%Y-%m-%d %H:%M UTC") if isinstance(published, datetime) \
            else str(published)
        out.append(
            f"[cluster_id={c['id']}]  timing={c['timing']}  tier={c['best_tier']}  "
            f"published={ts}  distinct_sources={c['distinct_sources']}\n"
            f"    source: {c['source']}\n"
            f"    headline: {c['headline']}"
            + (f"\n    summary: {c['summary'][:300]}" if c.get("summary") else "")
        )
    return "\n\n".join(out)


def render(swing: dict, decomposition_sentence: str,
           pre: list[dict], post: list[dict], version: str | None = None) -> tuple[str, str]:
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
        pre_clusters=_fmt_clusters(pre),
        post_clusters=_fmt_clusters(post),
        earnings_instruction=EARNINGS_INSTRUCTION if swing["earnings_mode"] else "",
    )
