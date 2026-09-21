"""Project paths, resolved absolutely.

Once `swing` is installed globally you will run it from anywhere, and every
relative path then resolves against the wrong directory. The symptom is
misleading: it works inside the project and fails everywhere else, so it looks
like an install problem when it is a path problem. swing-cli.md Part 4.

Never use a bare relative path anywhere in this codebase.

⚠️ The 0.1.0 release shipped without this file's third case and was unusable by
anyone who was not the author. `config/` lives at the REPO root, outside the
package, so it is absent from the wheel — and `swing coverage` on a clean
install died with "cannot locate the project root". It looked fine in testing
only because `uvx` resolved `swing` to the local editable checkout rather than
to the package it had just downloaded.

Resolution order, most explicit first:

    SWING_HOME          set it and that is the answer, no inference
    a repo checkout     ../../.. has config/, so this is a working tree
    ~/.swing            what `swing init` creates for an installed user

The last case never raises, because `swing init` has to be able to run before
the directory it creates exists.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

#: Where an installed (non-checkout) user's config and data live.
DEFAULT_HOME = Path.home() / ".swing"

#: Config shipped inside the wheel. Seeds DEFAULT_HOME; never read directly, so
#: that a user's edits are never silently overridden by a package upgrade.
PACKAGED_DEFAULTS = Path(__file__).resolve().parent / "defaults"


def _checkout_root() -> Path | None:
    """The repo root when running from a source tree, else None."""
    root = Path(__file__).resolve().parent.parent.parent   # src/swing/paths.py
    return root if (root / "config").is_dir() else None


@lru_cache(maxsize=1)
def project_root() -> Path:
    if env := os.getenv("SWING_HOME"):
        root = Path(env).expanduser().resolve()
        if not (root / "config").is_dir():
            raise RuntimeError(
                f"SWING_HOME={root} has no config/ directory. "
                "Point it at a swing home, or run `swing init` to create one."
            )
        return root
    return _checkout_root() or DEFAULT_HOME


ROOT = project_root()
CONFIG = ROOT / "config"
DATA = ROOT / "data"
ENV_FILE = ROOT / ".env"
WATCHLIST = CONFIG / "watchlist.yaml"
SOURCES = CONFIG / "sources.yaml"
THRESHOLDS = CONFIG / "thresholds.yaml"


def is_initialised() -> bool:
    """True when ROOT holds the config the commands need."""
    return WATCHLIST.is_file() and SOURCES.is_file() and THRESHOLDS.is_file()


def require_initialised() -> None:
    """Fail with the fix rather than a FileNotFoundError three frames deeper."""
    if not is_initialised():
        raise SystemExit(
            f"swing is not set up yet — no config found in {ROOT}.\n"
            "Run `swing init` to create it."
        )
