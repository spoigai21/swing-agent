-- Canonical schema. Source of truth; store/models.py mirrors it.
-- Idempotent: safe to re-apply. Apply with `swing dbinit`.

CREATE EXTENSION IF NOT EXISTS vector;

-- ===========================================================================
-- INGEST TIER
-- ===========================================================================

-- The archive of record. Write-only; never mutated except to flip `normalized`.
CREATE TABLE IF NOT EXISTS articles_raw (
  id           bigserial PRIMARY KEY,
  url          text UNIQUE NOT NULL,
  source       text NOT NULL,
  headline     text NOT NULL,
  summary      text,
  body         text,
  published_at timestamptz NOT NULL,
  retrieved_at timestamptz NOT NULL DEFAULT now(),
  raw          jsonb,
  normalized   boolean NOT NULL DEFAULT false
);
CREATE INDEX IF NOT EXISTS articles_raw_unnormalized_idx ON articles_raw (id) WHERE normalized = false;
CREATE INDEX IF NOT EXISTS articles_raw_source_published_idx ON articles_raw (source, published_at DESC);
CREATE INDEX IF NOT EXISTS articles_raw_published_idx ON articles_raw (published_at DESC);

CREATE TABLE IF NOT EXISTS source_health (
  source        text NOT NULL,
  d             date NOT NULL,
  article_count int  NOT NULL DEFAULT 0,
  last_seen_at  timestamptz,
  PRIMARY KEY (source, d)
);

CREATE TABLE IF NOT EXISTS edgar_cursor (
  cik            text PRIMARY KEY,
  ticker         text NOT NULL,
  last_accession text,
  last_seen_at   timestamptz
);

-- ===========================================================================
-- PRICE TIER
-- ===========================================================================

-- UNADJUSTED OHLCV. Adjusted series get silently rewritten on every split or
-- dividend, so backtests change results for no visible reason. Apply
-- adjustments at query time from corporate_actions. agent-plan.md 0.2.
CREATE TABLE IF NOT EXISTS bars (
  ticker text   NOT NULL,
  ts     timestamptz NOT NULL,
  open   numeric, high numeric, low numeric, close numeric,
  volume bigint,
  source text NOT NULL DEFAULT 'yfinance',
  PRIMARY KEY (ticker, ts)
);
CREATE INDEX IF NOT EXISTS bars_ticker_ts_idx ON bars (ticker, ts DESC);

CREATE TABLE IF NOT EXISTS corporate_actions (
  ticker  text NOT NULL,
  ex_date date NOT NULL,
  kind    text NOT NULL CHECK (kind IN ('split', 'dividend')),
  ratio   numeric,
  amount  numeric,
  PRIMARY KEY (ticker, ex_date, kind)
);

-- 5-minute bars, captured permanently the moment a swing is detected.
-- yfinance serves these for the last 60 days only (verified 2026-08-31), so
-- capture-on-detect means the cap limits backfill but never forward operation.
CREATE TABLE IF NOT EXISTS intraday_bars (
  ticker       text NOT NULL,
  ts           timestamptz NOT NULL,
  interval_sec int  NOT NULL DEFAULT 300,
  open numeric, high numeric, low numeric, close numeric,
  volume bigint,
  PRIMARY KEY (ticker, ts, interval_sec)
);
CREATE INDEX IF NOT EXISTS intraday_bars_ticker_ts_idx ON intraday_bars (ticker, ts);

-- ===========================================================================
-- ARTICLE TIER
-- ===========================================================================

CREATE TABLE IF NOT EXISTS articles (
  id           bigserial PRIMARY KEY,
  raw_id       bigint REFERENCES articles_raw(id),
  url          text UNIQUE NOT NULL,
  source       text NOT NULL,
  source_tier  int  NOT NULL CHECK (source_tier BETWEEN 1 AND 4),
  headline     text NOT NULL,
  summary      text,
  body         text,                       -- nullable BY DESIGN (agent-plan.md 0.6)
  published_at timestamptz NOT NULL,
  retrieved_at timestamptz NOT NULL,
  tickers      text[] NOT NULL DEFAULT '{}',
  event_hint   text,                       -- 8-K Item number, e.g. '2.02'
  embedding    vector(768),                -- headline + summary ONLY
  minhash      bytea,
  dup_group_id bigint                      -- global, incremental, MinHash-derived
);
CREATE INDEX IF NOT EXISTS articles_published_idx ON articles (published_at DESC);
CREATE INDEX IF NOT EXISTS articles_tickers_idx ON articles USING gin (tickers);
CREATE INDEX IF NOT EXISTS articles_dup_group_idx ON articles (dup_group_id);
-- hnsw, not ivfflat: ivfflat builds its partition from existing rows, so
-- creating it before load gives a useless index. hnsw has no such
-- build-order dependency. agent-plan.md TROUBLESHOOTING.
CREATE INDEX IF NOT EXISTS articles_embedding_idx ON articles USING hnsw (embedding vector_cosine_ops);

-- ===========================================================================
-- ANALYSIS TIER
-- ===========================================================================

-- One row per (ticker, day), produced by one vectorized rolling pass so the
-- z-score denominator is consistent, cheap and reproducible.
CREATE TABLE IF NOT EXISTS daily_factors (
  ticker           text NOT NULL,
  d                date NOT NULL,
  ret              numeric,
  alpha            numeric,
  beta_mkt         numeric,
  beta_sector      numeric,
  r_squared        numeric,        -- per-ticker fit quality; drives the MU/SMH check
  market_component numeric,
  sector_component numeric,
  residual         numeric,
  residual_vol_60  numeric,
  residual_z       numeric,
  volume_z         numeric,
  idio_share       numeric,        -- |residual| / |ret|: share of the move that is idiosyncratic
  status           text NOT NULL DEFAULT 'ok',
  PRIMARY KEY (ticker, d)
);
CREATE INDEX IF NOT EXISTS daily_factors_d_idx ON daily_factors (d DESC);

CREATE TABLE IF NOT EXISTS swings (
  id               bigserial PRIMARY KEY,
  ticker           text NOT NULL,
  d                date NOT NULL,
  kind             text NOT NULL CHECK (kind IN ('daily', 'drift')),
  drift_window     int,
  residual         numeric NOT NULL,
  residual_z       numeric NOT NULL,
  total_return     numeric NOT NULL,
  market_component numeric NOT NULL,
  sector_component numeric NOT NULL,
  volume_z         numeric,
  swing_type       text NOT NULL CHECK (swing_type IN ('gap','intraday','mixed','drift','unknown')),
  onset_ts         timestamptz,   -- NULL only when swing_type='unknown'
  onset_source     text CHECK (onset_source IN ('intraday','fallback_48h')),
  earnings_mode    boolean NOT NULL DEFAULT false,
  entity_type      text NOT NULL DEFAULT 'stock' CHECK (entity_type IN ('stock','sector')),
  superseded_by    bigint REFERENCES swings(id),
  created_at       timestamptz NOT NULL DEFAULT now(),
  UNIQUE (ticker, d, kind, drift_window)
);
CREATE INDEX IF NOT EXISTS swings_ticker_d_idx ON swings (ticker, d DESC);

-- Window-scoped semantic clusters, persisted so an attribution's citations
-- stay resolvable forever even after dedup thresholds are retuned.
CREATE TABLE IF NOT EXISTS clusters (
  id                 bigserial PRIMARY KEY,
  swing_id           bigint NOT NULL REFERENCES swings(id) ON DELETE CASCADE,
  timing             text NOT NULL CHECK (timing IN ('pre_move','post_move')),
  canonical_article  bigint NOT NULL REFERENCES articles(id),
  member_count       int NOT NULL,
  distinct_sources   int NOT NULL,      -- the corroboration count
  earliest_published timestamptz NOT NULL,
  best_tier          int NOT NULL,
  semantic_score numeric, timing_score numeric, novelty_score numeric,
  rank_score     numeric, rank int
);
CREATE INDEX IF NOT EXISTS clusters_swing_rank_idx ON clusters (swing_id, rank);

CREATE TABLE IF NOT EXISTS cluster_members (
  cluster_id bigint NOT NULL REFERENCES clusters(id) ON DELETE CASCADE,
  article_id bigint NOT NULL REFERENCES articles(id),
  PRIMARY KEY (cluster_id, article_id)
);

-- ===========================================================================
-- OUTPUT + EVALUATION TIER
-- ===========================================================================

CREATE TABLE IF NOT EXISTS attributions (
  id               bigserial PRIMARY KEY,
  swing_id         bigint NOT NULL REFERENCES swings(id),
  verdict          text NOT NULL CHECK (verdict IN ('explained','partially_explained','unexplained')),
  payload          jsonb NOT NULL,
  unexplained_note text,
  -- Populated by us, never by the model. Without these the system is
  -- unfalsifiable: you cannot tell whether a metric moved because of your
  -- prompt edit or because Google rotated the model. agent-plan.md 3.1b.
  prompt_version   text NOT NULL,
  model_id         text NOT NULL,
  config_hash      text NOT NULL,
  run_kind         text NOT NULL DEFAULT 'production'
                     CHECK (run_kind IN ('production','placebo','eval')),
  created_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS attributions_swing_idx ON attributions (swing_id, created_at DESC);
CREATE INDEX IF NOT EXISTS attributions_verdict_idx ON attributions (verdict, created_at DESC);

CREATE TABLE IF NOT EXISTS annotations (
  swing_id        bigint PRIMARY KEY REFERENCES swings(id),
  blind           boolean NOT NULL,   -- recall@10 is computed ONLY over blind=true
  true_catalyst   text,
  true_cluster_id bigint REFERENCES clusters(id),
  true_event_type text,
  no_catalyst     boolean NOT NULL DEFAULT false,
  annotator_note  text,
  annotated_at    timestamptz NOT NULL DEFAULT now()
);
