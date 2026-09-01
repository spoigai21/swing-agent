"""articles_raw -> articles.

The archive is never mutated, so this stage is fully re-runnable: changing the
tier map or the embedding model is a replay, not a refetch. CODEBASE-PLAN G2.

Three things happen here, and each has a failure mode worth naming:

1. **Tier assignment, per publisher.** Finnhub returns many publishers through
   one pipe. Tiering by feed would admit Tier 4 aggregator content as evidence.
   Unlisted publishers default to tier 4 and are dropped.
2. **Embedding of headline + summary ONLY, never body.** Full-text sources would
   otherwise systematically outrank headline-only ones like WSJ purely on token
   count, silently inverting the source tiers. agent-plan.md 0.6.
3. **MinHash for the cheap dedup pass**, giving a stable global dup_group_id.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from swing.common import logging as log
from swing.common.timeutil import assert_utc
from swing.ingest.config import feeds, sources, stocks, thresholds
from swing.store.session import connect

logger = log.get("ingest.normalize")

DEFAULT_TIER = 4  # unlisted publisher -> excluded


@lru_cache(maxsize=1)
def _publisher_tiers() -> dict[str, int]:
    return {k.lower().strip(): int(v) for k, v in sources().get("publisher_tiers", {}).items()}


@lru_cache(maxsize=1)
def _feed_tickers() -> dict[str, list[str]]:
    """feed source -> tickers, for single-company IR feeds."""
    out: dict[str, list[str]] = {}
    for f in feeds(include_disabled=True):
        if f.tickers:
            out[f.source] = f.tickers
    return out


@lru_cache(maxsize=1)
def _ticker_patterns() -> dict[str, re.Pattern]:
    """Ticker symbol plus curated aliases from config.

    Aliases are curated, never derived from the company name. Deriving them
    produced `Take-Two Interactive` -> the bare word "Interactive", which tags
    any article mentioning an interactive anything.

    The alternation MUST be grouped: `(?<!x)A|B|C(?!x)` applies the lookbehind
    only to A and the lookahead only to C, which silently defeats both guards.
    """
    pats = {}
    for ticker, meta in stocks().items():
        alts = [ticker, *(meta.get("aliases") or [])]
        joined = "|".join(re.escape(a) for a in alts)
        pats[ticker] = re.compile(rf"(?<![A-Za-z0-9])(?:{joined})(?![A-Za-z0-9])", re.IGNORECASE)
    return pats


def resolve_tier(row: dict[str, Any]) -> int:
    """Publisher first, feed tier as fallback, tier 4 (excluded) as default."""
    raw = row.get("raw") or {}
    tiers = _publisher_tiers()

    # Finnhub company-news carries the real publisher in raw.source.
    for key in ("publisher", "source"):
        v = (raw.get(key) or "").strip().lower()
        if v and v in tiers:
            return tiers[v]

    src = (row.get("source") or "").strip().lower()
    if src in tiers:
        return tiers[src]
    if isinstance(raw.get("source_tier"), int):
        return int(raw["source_tier"])
    return DEFAULT_TIER


def resolve_tickers(row: dict[str, Any]) -> list[str]:
    raw = row.get("raw") or {}
    if raw.get("ticker"):                       # EDGAR: authoritative
        return [str(raw["ticker"]).upper()]
    if raw.get("feed_tickers"):                 # single-company IR feed
        return [t.upper() for t in raw["feed_tickers"]]
    if row.get("source") in _feed_tickers():
        return [t.upper() for t in _feed_tickers()[row["source"]]]

    text = f"{row.get('headline') or ''} {row.get('summary') or ''}"
    return sorted(t for t, pat in _ticker_patterns().items() if pat.search(text))


def event_hint(row: dict[str, Any]) -> str | None:
    """8-K Item number: free labelled data for the Phase 5.2 classifier."""
    raw = row.get("raw") or {}
    return (raw.get("items") or None) if raw.get("form") else None


def _embed_texts(texts: list[str]) -> list[list[float]]:

    cfg = thresholds()["embedding"]
    model = _load_model(cfg["model"], cfg.get("device"))
    vecs = model.encode(texts, batch_size=int(cfg.get("batch_size", 32)),
                        normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vecs]


@lru_cache(maxsize=2)
def _load_model(name: str, device: str | None):
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(name, device=device)
    except Exception:  # noqa: BLE001 - any device failure must fall back to CPU
        logger.warning("device %s unavailable; falling back to CPU", device)
        return SentenceTransformer(name)


def _minhash(text: str, num_perm: int) -> tuple[bytes, int]:
    from datasketch import MinHash

    m = MinHash(num_perm=num_perm)
    for tok in set(re.findall(r"[a-z0-9]+", text.lower())):
        m.update(tok.encode())
    return m.hashvalues.tobytes(), int(m.hashvalues[0])


INSERT = """
INSERT INTO articles (raw_id, url, source, source_tier, headline, summary, body,
                      published_at, retrieved_at, tickers, event_hint, embedding,
                      minhash, dup_group_id)
VALUES (%(raw_id)s, %(url)s, %(source)s, %(source_tier)s, %(headline)s, %(summary)s,
        %(body)s, %(published_at)s, %(retrieved_at)s, %(tickers)s, %(event_hint)s,
        %(embedding)s, %(minhash)s, %(dup_group_id)s)
ON CONFLICT (url) DO NOTHING
"""


def normalize_batch(limit: int = 256) -> dict[str, int]:
    """Drain a batch of articles_raw. Idempotent and safe to re-run."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM articles_raw WHERE normalized = false ORDER BY id LIMIT %s",
            (limit,),
        ).fetchall()
    if not rows:
        return {"read": 0, "written": 0, "dropped_tier4": 0}

    cfg = thresholds()["dedup"]
    keep, dropped = [], 0
    for r in rows:
        tier = resolve_tier(r)
        if tier >= DEFAULT_TIER:
            dropped += 1
            continue
        keep.append((r, tier))

    written = 0
    if keep:
        # Headline + summary only. Never body.
        texts = [f"{r['headline']} {r.get('summary') or ''}".strip() for r, _ in keep]
        vecs = _embed_texts(texts)
        payload = []
        for (r, tier), text, vec in zip(keep, texts, vecs, strict=True):
            mh, group = _minhash(text, int(cfg.get("minhash_num_perm", 128)))
            payload.append({
                "raw_id": r["id"], "url": r["url"], "source": r["source"],
                "source_tier": tier, "headline": r["headline"], "summary": r.get("summary"),
                "body": r.get("body"),
                "published_at": assert_utc(r["published_at"]),
                "retrieved_at": assert_utc(r["retrieved_at"]),
                "tickers": resolve_tickers(r), "event_hint": event_hint(r),
                "embedding": str(vec), "minhash": mh, "dup_group_id": group,
            })
        with connect() as conn, conn.cursor() as cur:
            cur.executemany(INSERT, payload)
            written = len(payload)

    with connect() as conn:
        conn.execute("UPDATE articles_raw SET normalized = true WHERE id = ANY(%s)",
                     ([r["id"] for r in rows],))
    return {"read": len(rows), "written": written, "dropped_tier4": dropped}


def normalize_all(batch: int = 256, max_batches: int = 1000) -> dict[str, int]:
    total = {"read": 0, "written": 0, "dropped_tier4": 0}
    for _ in range(max_batches):
        got = normalize_batch(batch)
        if got["read"] == 0:
            break
        for k in total:
            total[k] += got[k]
        logger.info("normalized %(read)d (wrote %(written)d, dropped %(dropped_tier4)d)", got)
    return total
