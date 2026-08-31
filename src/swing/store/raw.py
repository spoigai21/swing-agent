"""Writes into articles_raw. The only write path for Phase -1."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from swing.common.timeutil import assert_utc, now_utc
from swing.store.session import connect

INSERT = """
INSERT INTO articles_raw (url, source, headline, summary, body, published_at, retrieved_at, raw)
VALUES (%(url)s, %(source)s, %(headline)s, %(summary)s, %(body)s,
        %(published_at)s, %(retrieved_at)s, %(raw)s)
ON CONFLICT (url) DO NOTHING
RETURNING id
"""


@dataclass(slots=True)
class RawArticle:
    url: str
    source: str
    headline: str
    published_at: datetime
    summary: str | None = None
    body: str | None = None
    retrieved_at: datetime = field(default_factory=now_utc)
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Timezone discipline at the boundary, not deep in the pipeline.
        self.published_at = assert_utc(self.published_at)
        self.retrieved_at = assert_utc(self.retrieved_at)
        if not self.url or not self.headline:
            raise ValueError(f"url and headline are required: {self!r}")


def insert_many(articles: list[RawArticle]) -> int:
    """Insert, skipping URLs already stored. Returns the count actually new."""
    if not articles:
        return 0
    inserted = 0
    with connect() as conn, conn.cursor() as cur:
        for a in articles:
            cur.execute(
                INSERT,
                {
                    "url": a.url,
                    "source": a.source,
                    "headline": a.headline[:2000],
                    "summary": a.summary,
                    "body": a.body,
                    "published_at": a.published_at,
                    "retrieved_at": a.retrieved_at,
                    # Store the full payload unparsed: reparsing is free, refetching
                    # is impossible. agent-plan.md Step -1.1
                    "raw": json.dumps(a.raw, default=str),
                },
            )
            if cur.fetchone():
                inserted += 1
    return inserted


def bump_health(source: str, count: int) -> None:
    """Record per-source daily counts, including zero-count polls."""
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO source_health (source, d, article_count, last_seen_at)
            VALUES (%s, (now() AT TIME ZONE 'utc')::date, %s, now())
            ON CONFLICT (source, d) DO UPDATE
              SET article_count = source_health.article_count + EXCLUDED.article_count,
                  last_seen_at  = EXCLUDED.last_seen_at
            """,
            (source, count),
        )
