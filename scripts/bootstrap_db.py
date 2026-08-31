#!/usr/bin/env python
"""Apply the Phase -1 schema. Idempotent."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import logging as log  # noqa: E402
from store.session import apply_schema, connect  # noqa: E402


def main() -> int:
    log.setup()
    logger = log.get("bootstrap")
    apply_schema()
    with connect() as conn:
        rows = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='public' ORDER BY table_name"
        ).fetchall()
    logger.info("schema applied; tables: %s", ", ".join(r["table_name"] for r in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
