"""Correct Finnhub company-news timestamps: Eastern wall clock stored as UTC.

Finnhub's `datetime` encodes Eastern wall-clock time as if it were a UTC epoch,
and ingest read it as UTC. Every Finnhub article -- about half the evidence
pool -- was therefore stamped 4h early (5h in winter). That is the timezone
failure agent-plan.md warns "does not announce itself": post-move commentary
published within hours of the onset was filed as PRE-move evidence, e.g. a
MarketWatch "Dow and S&P 500 open flat" item sat before the NFLX 2025-10-22 open
and a Q3-results story sat four hours before Netflix released the results.

Verified 2026-09-13 two ways:
  * the same CNBC stories via CNBC's own RSS ('... GMT', fetched seconds after
    publication) are exactly 4.00h after Finnhub's stamp (11/11, summer);
  * against EDGAR 8-K Item 2.02 acceptance times, a flat +4h still leaves
    winter reaction stories 45 minutes BEFORE the release, while re-reading the
    stamp as America/New_York puts every one after it.

articles_raw is the archive of record, so the value as received is preserved
in raw.finnhub_ts_as_received and the change is reversible. Clusters built on
the old stamps are stale: run `swing retrieve` after upgrading.

Revision ID: 0006_finnhub_eastern_time
Revises: 0005_annotation_articles
"""
from alembic import op

revision = "0006_finnhub_eastern_time"
down_revision = "0005_annotation_articles"
branch_labels = None
depends_on = None

_PROPAGATE = """
    UPDATE articles a SET published_at = r.published_at
    FROM articles_raw r
    WHERE r.id = a.raw_id AND r.raw->>'via' = 'finnhub'
      AND a.published_at IS DISTINCT FROM r.published_at
"""


def upgrade() -> None:
    # Rows already carrying finnhub_ts_basis were ingested by corrected code.
    op.execute("""
        UPDATE articles_raw
        SET raw = raw || jsonb_build_object(
                'finnhub_ts_as_received', published_at,
                'finnhub_ts_basis', 'America/New_York'),
            published_at = (published_at AT TIME ZONE 'UTC') AT TIME ZONE 'America/New_York'
        WHERE raw->>'via' = 'finnhub' AND NOT raw ? 'finnhub_ts_basis'
    """)
    op.execute(_PROPAGATE)


def downgrade() -> None:
    op.execute("""
        UPDATE articles_raw
        SET published_at = (raw->>'finnhub_ts_as_received')::timestamptz,
            raw = raw - 'finnhub_ts_as_received' - 'finnhub_ts_basis'
        WHERE raw->>'via' = 'finnhub' AND raw ? 'finnhub_ts_as_received'
    """)
    op.execute(_PROPAGATE)
