"""Persist verdict_reason on attributions.

The graph computes WHY it reached a verdict (residual_below_threshold,
no_pre_move_clusters, llm_error, or a model judgement) but never stored it.
Without it you cannot separate "abstained because we held no evidence" from
"abstained because the model judged the evidence insufficient" — and abstention
precision means something different in each case.

Revision ID: 0003_verdict_reason
Revises: 0002_swing_unique
"""
import sqlalchemy as sa
from alembic import op

revision = "0003_verdict_reason"
down_revision = "0002_swing_unique"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("attributions", sa.Column("verdict_reason", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("attributions", "verdict_reason")
