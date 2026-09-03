"""Fix the swings uniqueness constraint for NULL drift_window.

UNIQUE (ticker, d, kind, drift_window) never fires for daily swings because
drift_window is NULL and NULL != NULL in Postgres. Every detect_all() re-run
therefore INSERTed instead of upserting: 101 duplicates out of 472 rows, and
the daily batch attributed the same swing twice.

Replaced with two partial unique indexes that handle NULL explicitly.

Revision ID: 0002_swing_unique
Revises: 0001_baseline
"""
from alembic import op

revision = "0002_swing_unique"
down_revision = "0001_baseline"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Repoint children at the surviving (lowest-id) row before deleting.
    op.execute("""
        CREATE TEMP TABLE swing_dedup AS
        SELECT id AS dup_id,
               min(id) OVER (PARTITION BY ticker, d, kind, coalesce(drift_window, -1))
                   AS keep_id
        FROM swings
    """)
    for child, col in [("clusters", "swing_id"), ("attributions", "swing_id"),
                       ("annotations", "swing_id")]:
        op.execute(f"""
            UPDATE {child} c SET {col} = d.keep_id
            FROM swing_dedup d
            WHERE c.{col} = d.dup_id AND d.dup_id <> d.keep_id
        """)
    op.execute("UPDATE swings s SET superseded_by = NULL "
               "WHERE superseded_by IN (SELECT dup_id FROM swing_dedup "
               "WHERE dup_id <> keep_id)")
    op.execute("DELETE FROM swings WHERE id IN "
               "(SELECT dup_id FROM swing_dedup WHERE dup_id <> keep_id)")

    op.execute("ALTER TABLE swings DROP CONSTRAINT IF EXISTS "
               "swings_ticker_d_kind_drift_window_key")
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS swings_daily_unique
        ON swings (ticker, d, kind) WHERE drift_window IS NULL
    """)
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS swings_drift_unique
        ON swings (ticker, d, kind, drift_window) WHERE drift_window IS NOT NULL
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS swings_daily_unique")
    op.execute("DROP INDEX IF EXISTS swings_drift_unique")
