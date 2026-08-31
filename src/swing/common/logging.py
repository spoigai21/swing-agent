"""One structured line per ingest event. Readable in a terminal, greppable in a file."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False


def setup(level: int = logging.INFO, logfile: Path | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)-22s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if logfile:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logfile))
    root = logging.getLogger()
    root.setLevel(level)
    for h in handlers:
        h.setFormatter(fmt)
        root.addHandler(h)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    _CONFIGURED = True


def get(name: str) -> logging.Logger:
    return logging.getLogger(name)
