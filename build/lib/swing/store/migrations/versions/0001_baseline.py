"""Baseline: the schema as created by store/schema.sql.

schema.sql remains the source of truth for a fresh install (`swing dbinit`).
Alembic exists to migrate EXISTING databases forward from here — most
importantly the vector(768) column, which cannot be changed without
re-embedding every article.
"""

revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # No-op: an existing database is stamped at this revision, and a fresh one
    # is created by schema.sql. Real migrations start at 0002.
    pass


def downgrade() -> None:
    pass
