"""Persist the cluster ids actually shown to the model.

citation_validity() compared cited ids against the swing's OWN clusters. That is
wrong for any run with an injected cluster set: a placebo run shows the DONOR's
clusters, so every valid citation was scored invalid and the metric read 0.000.

Storing what was actually passed makes the check correct for every run kind, and
is the only way to audit a historical attribution at all.

Revision ID: 0004_allowed_clusters
Revises: 0003_verdict_reason
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0004_allowed_clusters"
down_revision = "0003_verdict_reason"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("attributions",
                  sa.Column("shown_cluster_ids", postgresql.ARRAY(sa.BigInteger()),
                            nullable=True))


def downgrade() -> None:
    op.drop_column("attributions", "shown_cluster_ids")
