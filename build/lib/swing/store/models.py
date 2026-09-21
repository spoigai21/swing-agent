"""SQLAlchemy models mirroring store/schema.sql.

schema.sql stays the source of truth; these exist for typed reads and writes.
If you change one, change both — `swing dbinit` applies the SQL, not this.
"""
from __future__ import annotations

from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    LargeBinary,
    Numeric,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ArticleRaw(Base):
    __tablename__ = "articles_raw"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    url: Mapped[str] = mapped_column(Text, unique=True)
    source: Mapped[str] = mapped_column(Text)
    headline: Mapped[str] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text)
    body: Mapped[str | None] = mapped_column(Text)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    raw: Mapped[dict | None] = mapped_column(JSONB)
    normalized: Mapped[bool] = mapped_column(Boolean, default=False)


class Article(Base):
    __tablename__ = "articles"
    __table_args__ = (
        CheckConstraint("source_tier BETWEEN 1 AND 4", name="articles_source_tier_check"),
    )
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    raw_id: Mapped[int | None] = mapped_column(ForeignKey("articles_raw.id"))
    url: Mapped[str] = mapped_column(Text, unique=True)
    source: Mapped[str] = mapped_column(Text)
    source_tier: Mapped[int] = mapped_column(Integer)
    headline: Mapped[str] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text)
    body: Mapped[str | None] = mapped_column(Text)          # nullable by design
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    tickers: Mapped[list[str]] = mapped_column(ARRAY(Text))
    event_hint: Mapped[str | None] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(768))
    minhash: Mapped[bytes | None] = mapped_column(LargeBinary)
    dup_group_id: Mapped[int | None] = mapped_column(BigInteger)


class Bar(Base):
    __tablename__ = "bars"
    ticker: Mapped[str] = mapped_column(Text, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    open: Mapped[float | None] = mapped_column(Numeric)
    high: Mapped[float | None] = mapped_column(Numeric)
    low: Mapped[float | None] = mapped_column(Numeric)
    close: Mapped[float | None] = mapped_column(Numeric)
    volume: Mapped[int | None] = mapped_column(BigInteger)
    source: Mapped[str] = mapped_column(Text, default="yfinance")


class IntradayBar(Base):
    __tablename__ = "intraday_bars"
    ticker: Mapped[str] = mapped_column(Text, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    interval_sec: Mapped[int] = mapped_column(Integer, primary_key=True, default=300)
    open: Mapped[float | None] = mapped_column(Numeric)
    high: Mapped[float | None] = mapped_column(Numeric)
    low: Mapped[float | None] = mapped_column(Numeric)
    close: Mapped[float | None] = mapped_column(Numeric)
    volume: Mapped[int | None] = mapped_column(BigInteger)


class CorporateAction(Base):
    __tablename__ = "corporate_actions"
    __table_args__ = (
        CheckConstraint("kind IN ('split', 'dividend')", name="corporate_actions_kind_check"),
    )
    ticker: Mapped[str] = mapped_column(Text, primary_key=True)
    ex_date: Mapped[date] = mapped_column(Date, primary_key=True)
    kind: Mapped[str] = mapped_column(Text, primary_key=True)
    ratio: Mapped[float | None] = mapped_column(Numeric)
    amount: Mapped[float | None] = mapped_column(Numeric)


class DailyFactor(Base):
    __tablename__ = "daily_factors"
    ticker: Mapped[str] = mapped_column(Text, primary_key=True)
    d: Mapped[date] = mapped_column(Date, primary_key=True)
    ret: Mapped[float | None] = mapped_column(Numeric)
    alpha: Mapped[float | None] = mapped_column(Numeric)
    beta_mkt: Mapped[float | None] = mapped_column(Numeric)
    beta_sector: Mapped[float | None] = mapped_column(Numeric)
    r_squared: Mapped[float | None] = mapped_column(Numeric)
    market_component: Mapped[float | None] = mapped_column(Numeric)
    sector_component: Mapped[float | None] = mapped_column(Numeric)
    residual: Mapped[float | None] = mapped_column(Numeric)
    residual_vol_60: Mapped[float | None] = mapped_column(Numeric)
    residual_z: Mapped[float | None] = mapped_column(Numeric)
    volume_z: Mapped[float | None] = mapped_column(Numeric)
    idio_share: Mapped[float | None] = mapped_column(Numeric)
    status: Mapped[str] = mapped_column(Text, default="ok")


class Swing(Base):
    __tablename__ = "swings"
    __table_args__ = (
        CheckConstraint("kind IN ('daily', 'drift')", name="swings_kind_check"),
        CheckConstraint(
            "swing_type IN ('gap','intraday','mixed','drift','unknown')",
            name="swings_swing_type_check",
        ),
        CheckConstraint(
            "onset_source IN ('intraday','fallback_48h')", name="swings_onset_source_check"
        ),
        CheckConstraint(
            "entity_type IN ('stock','sector')", name="swings_entity_type_check"
        ),
    )
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    ticker: Mapped[str] = mapped_column(Text)
    d: Mapped[date] = mapped_column(Date)
    kind: Mapped[str] = mapped_column(Text)
    drift_window: Mapped[int | None] = mapped_column(Integer)
    residual: Mapped[float] = mapped_column(Numeric)
    residual_z: Mapped[float] = mapped_column(Numeric)
    total_return: Mapped[float] = mapped_column(Numeric)
    market_component: Mapped[float] = mapped_column(Numeric)
    sector_component: Mapped[float] = mapped_column(Numeric)
    volume_z: Mapped[float | None] = mapped_column(Numeric)
    swing_type: Mapped[str] = mapped_column(Text)
    onset_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    onset_source: Mapped[str | None] = mapped_column(Text)
    earnings_mode: Mapped[bool] = mapped_column(Boolean, default=False)
    entity_type: Mapped[str] = mapped_column(Text, default="stock")
    superseded_by: Mapped[int | None] = mapped_column(ForeignKey("swings.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Cluster(Base):
    __tablename__ = "clusters"
    __table_args__ = (
        CheckConstraint("timing IN ('pre_move','post_move')", name="clusters_timing_check"),
    )
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    swing_id: Mapped[int] = mapped_column(ForeignKey("swings.id", ondelete="CASCADE"))
    timing: Mapped[str] = mapped_column(Text)
    canonical_article: Mapped[int] = mapped_column(ForeignKey("articles.id"))
    member_count: Mapped[int] = mapped_column(Integer)
    distinct_sources: Mapped[int] = mapped_column(Integer)
    earliest_published: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    best_tier: Mapped[int] = mapped_column(Integer)
    semantic_score: Mapped[float | None] = mapped_column(Numeric)
    timing_score: Mapped[float | None] = mapped_column(Numeric)
    novelty_score: Mapped[float | None] = mapped_column(Numeric)
    rank_score: Mapped[float | None] = mapped_column(Numeric)
    rank: Mapped[int | None] = mapped_column(Integer)


class ClusterMember(Base):
    __tablename__ = "cluster_members"
    cluster_id: Mapped[int] = mapped_column(
        ForeignKey("clusters.id", ondelete="CASCADE"), primary_key=True
    )
    article_id: Mapped[int] = mapped_column(ForeignKey("articles.id"), primary_key=True)


class Attribution(Base):
    __tablename__ = "attributions"
    __table_args__ = (
        CheckConstraint(
            "verdict IN ('explained','partially_explained','unexplained')",
            name="attributions_verdict_check",
        ),
        CheckConstraint(
            "run_kind IN ('production','placebo','eval')", name="attributions_run_kind_check"
        ),
    )
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    swing_id: Mapped[int] = mapped_column(ForeignKey("swings.id"))
    verdict: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict] = mapped_column(JSONB)
    unexplained_note: Mapped[str | None] = mapped_column(Text)
    verdict_reason: Mapped[str | None] = mapped_column(Text)
    shown_cluster_ids: Mapped[list[int] | None] = mapped_column(ARRAY(BigInteger))
    prompt_version: Mapped[str] = mapped_column(Text)
    model_id: Mapped[str] = mapped_column(Text)
    config_hash: Mapped[str] = mapped_column(Text)
    run_kind: Mapped[str] = mapped_column(Text, default="production")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Annotation(Base):
    __tablename__ = "annotations"
    swing_id: Mapped[int] = mapped_column(ForeignKey("swings.id"), primary_key=True)
    blind: Mapped[bool] = mapped_column(Boolean)
    true_catalyst: Mapped[str | None] = mapped_column(Text)
    true_cluster_id: Mapped[int | None] = mapped_column(
        ForeignKey("clusters.id", ondelete="SET NULL"))
    true_article_ids: Mapped[list[int] | None] = mapped_column(ARRAY(BigInteger))
    true_event_type: Mapped[str | None] = mapped_column(Text)
    no_catalyst: Mapped[bool] = mapped_column(Boolean, default=False)
    annotator_note: Mapped[str | None] = mapped_column(Text)
    annotated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class SourceHealth(Base):
    __tablename__ = "source_health"
    source: Mapped[str] = mapped_column(Text, primary_key=True)
    d: Mapped[date] = mapped_column(Date, primary_key=True)
    article_count: Mapped[int] = mapped_column(Integer, default=0)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class EdgarCursor(Base):
    __tablename__ = "edgar_cursor"
    cik: Mapped[str] = mapped_column(Text, primary_key=True)
    ticker: Mapped[str] = mapped_column(Text)
    last_accession: Mapped[str | None] = mapped_column(Text)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
