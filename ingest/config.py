"""Load config/*.yaml once, with the shapes the pollers expect."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import yaml

from common.settings import CONFIG_DIR


@dataclass(slots=True)
class FeedSpec:
    id: str
    source: str
    tier: int
    url: str
    tickers: list[str]
    poll_seconds: int
    enabled: bool = True
    block: str | None = None


@lru_cache(maxsize=1)
def watchlist() -> dict[str, Any]:
    return yaml.safe_load((CONFIG_DIR / "watchlist.yaml").read_text())


@lru_cache(maxsize=1)
def sources() -> dict[str, Any]:
    return yaml.safe_load((CONFIG_DIR / "sources.yaml").read_text())


def stocks() -> dict[str, dict[str, Any]]:
    return watchlist()["stocks"]


def cik_map() -> dict[str, str]:
    """ticker -> zero-padded 10-digit CIK. Padding is asserted, not assumed:
    an unpadded CIK 404s and looks like a missing company (data-sources D.2.3)."""
    out = {}
    for ticker, meta in stocks().items():
        cik = str(meta["cik"])
        if len(cik) != 10 or not cik.isdigit():
            raise ValueError(f"{ticker}: CIK {cik!r} is not zero-padded to 10 digits")
        out[ticker] = cik
    return out


def feeds(include_disabled: bool = False) -> list[FeedSpec]:
    out = []
    for f in sources()["feeds"]:
        spec = FeedSpec(
            id=f["id"],
            source=f["source"],
            tier=int(f["tier"]),
            url=f["url"],
            tickers=list(f.get("tickers") or []),
            poll_seconds=int(f.get("poll_seconds", 900)),
            enabled=bool(f.get("enabled", True)),
            block=f.get("block"),
        )
        if spec.enabled or include_disabled:
            out.append(spec)
    return out


def edgar_config() -> dict[str, Any]:
    return sources()["edgar"]
