"""Project paths, resolved absolutely.

Once `swing` is installed globally you will run it from anywhere, and every
relative path then resolves against the wrong directory. The symptom is
misleading: it works inside the project and fails everywhere else, so it looks
like an install problem when it is a path problem. swing-cli.md Part 4.

Never use a bare relative path anywhere in this codebase.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def project_root() -> Path:
    """SWING_HOME if set, else inferred from this file's location.

    Layout is src/swing/paths.py, so the repo root is three parents up.
    """
    if env := os.getenv("SWING_HOME"):
        root = Path(env).expanduser().resolve()
        if not (root / "config").is_dir():
            raise RuntimeError(
                f"SWING_HOME={root} has no config/ directory. "
                "Point it at the repository root."
            )
        return root

    root = Path(__file__).resolve().parent.parent.parent
    if not (root / "config").is_dir():
        # Non-editable install: the package lives in site-packages and the
        # repo is elsewhere. Fail with the fix rather than a FileNotFoundError
        # from three frames deeper.
        raise RuntimeError(
            f"cannot locate the project root from {__file__} (tried {root}). "
            "Set SWING_HOME to the repository root, or reinstall editable "
            "with: pipx install -e /path/to/stock-agent-analysis --force"
        )
    return root


ROOT = project_root()
CONFIG = ROOT / "config"
DATA = ROOT / "data"
ENV_FILE = ROOT / ".env"
WATCHLIST = CONFIG / "watchlist.yaml"
SOURCES = CONFIG / "sources.yaml"
THRESHOLDS = CONFIG / "thresholds.yaml"
