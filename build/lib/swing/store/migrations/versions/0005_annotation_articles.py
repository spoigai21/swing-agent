"""Make annotation labels survive a cluster rebuild.

retrieval._persist rebuilt a swing's clusters by DELETE + re-INSERT, which gave
every cluster a new id on each rebuild. annotations.true_cluster_id referenced
clusters(id) with no ON DELETE, so once a swing was labelled its clusters could
never be rebuilt again: the DELETE raised a foreign-key violation, build_all()
logged it and moved on, and the labelled swing kept its OLD ranking forever.
Retuning ranking weights after Gate 2 would then leave recall@10 unchanged no
matter what the weights were -- a metric frozen while appearing to work.

Three changes:

* annotations.true_article_ids -- the label is now the ARTICLES in the chosen
  cluster, which are stable forever. recall@10 resolves them against whatever
  clusters currently exist, so a re-cluster that merges or splits the story
  still scores correctly.
* true_cluster_id becomes ON DELETE SET NULL: kept as provenance, no longer a
  lock on the clusters table.
* A unique index on (swing_id, timing, canonical_article) so _persist can upsert
  and a cluster keeps its id when only its rank changes (a weight retune).

Revision ID: 0005_annotation_articles
Revises: 0004_allowed_clusters
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0005_annotation_articles"
down_revision = "0004_allowed_clusters"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("annotations",
                  sa.Column("true_article_ids", postgresql.ARRAY(sa.BigInteger()),
                            nullable=True))
    op.execute("""
        UPDATE annotations a SET true_article_ids = (
            SELECT array_agg(m.article_id ORDER BY m.article_id)
            FROM cluster_members m WHERE m.cluster_id = a.true_cluster_id)
        WHERE a.true_cluster_id IS NOT NULL
    """)
    op.drop_constraint("annotations_true_cluster_id_fkey", "annotations",
                       type_="foreignkey")
    op.create_foreign_key("annotations_true_cluster_id_fkey", "annotations", "clusters",
                          ["true_cluster_id"], ["id"], ondelete="SET NULL")
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS clusters_identity_idx "
               "ON clusters (swing_id, timing, canonical_article)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS clusters_identity_idx")
    op.drop_constraint("annotations_true_cluster_id_fkey", "annotations",
                       type_="foreignkey")
    op.create_foreign_key("annotations_true_cluster_id_fkey", "annotations", "clusters",
                          ["true_cluster_id"], ["id"])
    op.drop_column("annotations", "true_article_ids")
