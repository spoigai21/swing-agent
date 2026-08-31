"""System-version fingerprinting for stored attributions.

Without these, the system is unfalsifiable: when a metric moves you cannot tell
whether it was your prompt edit, your threshold change, or Google rotating the
Flash model underneath you. And Phase 5.3's training pairs silently become a
blend of system versions you cannot separate. agent-plan.md 3.1b.

Computed from the config files at runtime, never by hand.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from swing.paths import CONFIG

# Files whose contents change how an attribution is produced. Adding a file
# here changes every future config_hash, which is the intent.
HASHED_CONFIGS = ("thresholds.yaml", "sources.yaml", "watchlist.yaml")


def _canonical(path: Path) -> str:
    """Hash semantic content, not formatting: reindenting a YAML file must not
    invalidate every prior attribution."""
    import yaml

    data = yaml.safe_load(path.read_text()) if path.exists() else None
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


@lru_cache(maxsize=1)
def config_hash() -> str:
    h = hashlib.sha256()
    for name in HASHED_CONFIGS:
        h.update(name.encode())
        h.update(_canonical(CONFIG / name).encode())
    return h.hexdigest()[:16]


@lru_cache(maxsize=1)
def prompt_version() -> str:
    """Stem of the highest-numbered prompt file, e.g. 'attribution_v2'."""
    prompts = sorted((CONFIG / "prompts").glob("attribution_v*.md"))
    if not prompts:
        return "none"
    return prompts[-1].stem


def model_id() -> str:
    """The exact model string used, not an alias."""
    from swing.common.settings import get_settings

    return get_settings().attribution_model


def stamp() -> dict[str, str]:
    """The three columns every attribution row must carry."""
    return {
        "prompt_version": prompt_version(),
        "model_id": model_id(),
        "config_hash": config_hash(),
    }
