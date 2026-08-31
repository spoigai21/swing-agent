-- Phase -1 schema. Deliberately minimal: only what the collector needs.
-- The full schema (bars, articles, swings, clusters, attributions, ...) lands
-- in Step 1. Nothing here blocks starting the clock.
-- agent-plan.md Step -1.1

CREATE EXTENSION IF NOT EXISTS vector;

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
  -- normalize.py's work queue (CODEBASE-PLAN G2). articles_raw is the archive
  -- of record and is never mutated except to flip this flag.
  normalized   boolean NOT NULL DEFAULT false
);

CREATE INDEX IF NOT EXISTS articles_raw_unnormalized_idx
  ON articles_raw (id) WHERE normalized = false;
CREATE INDEX IF NOT EXISTS articles_raw_source_published_idx
  ON articles_raw (source, published_at DESC);
CREATE INDEX IF NOT EXISTS articles_raw_published_idx
  ON articles_raw (published_at DESC);

-- Per-source daily counts. This is how a silently dead feed gets caught.
-- agent-plan.md Step -1.3 / §6.4
CREATE TABLE IF NOT EXISTS source_health (
  source        text NOT NULL,
  d             date NOT NULL,
  article_count int  NOT NULL DEFAULT 0,
  last_seen_at  timestamptz,
  PRIMARY KEY (source, d)
);

-- Cursor per EDGAR CIK so the poller only diffs what is new.
CREATE TABLE IF NOT EXISTS edgar_cursor (
  cik            text PRIMARY KEY,
  ticker         text NOT NULL,
  last_accession text,
  last_seen_at   timestamptz
);
