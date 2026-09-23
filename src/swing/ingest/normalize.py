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
from swing.ingest.language import demote_non_english
from swing.store.session import connect

logger = log.get("ingest.normalize")

DEFAULT_TIER = 4  # unlisted publisher -> excluded


def _norm_publisher(name: str) -> str:
    """Fold a publisher name to a comparable key.

    Publishers arrive spelled inconsistently across pipes: Finnhub sends
    "DowJones" while the config lists "dow jones". That gap silently dropped
    genuine Dow Jones wire copy to tier 4. Strip everything but alphanumerics
    so spelling variants collapse to one key.
    """
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


@lru_cache(maxsize=1)
def _publisher_tiers() -> dict[str, int]:
    return {_norm_publisher(k): int(v) for k, v in sources().get("publisher_tiers", {}).items()}


@lru_cache(maxsize=1)
def _feed_tickers() -> dict[str, list[str]]:
    """feed source -> tickers, for single-company IR feeds."""
    out: dict[str, list[str]] = {}
    for f in feeds(include_disabled=True):
        if f.tickers:
            out[f.source] = f.tickers
    return out


@lru_cache(maxsize=1)
def _ticker_patterns() -> dict[str, tuple[re.Pattern, re.Pattern]]:
    """(symbol_pattern, alias_pattern) per ticker.

    ⚠️ The symbol is matched CASE-SENSITIVELY; aliases are case-insensitive.

    A two-letter ticker matched case-insensitively is a magnet for ordinary
    words in any language: "MU" tagged the Czech word "mu" in a PR Newswire
    release, and previously matched "Musk", "Multiple" and "Munich". Company
    names do not have that problem, so they stay case-insensitive.

    Aliases are curated, never derived from the company name — deriving them
    produced `Take-Two Interactive` -> the bare word "Interactive", which tags
    any article mentioning an interactive anything.

    The alternation MUST be grouped: `(?<!x)A|B|C(?!x)` binds the lookbehind
    only to A and the lookahead only to C, silently defeating both guards.
    """
    from swing.ingest.config import related_companies

    # Keeps the standard guards so every symbol pattern has the same shape.
    never = re.compile(r"(?<![A-Za-z0-9])(?!x)x(?![A-Za-z0-9])")
    # Related companies are tagged too, so a story that names only Intel still
    # reaches QCOM's retrieval. Their 1-2 letter symbols (U, F, GM) are ordinary
    # words and are matched by alias only.
    companies = [(t, m, True) for t, m in stocks().items()]
    companies += [(t, m, len(t) >= 3) for t, m in related_companies().items()]
    pats = {}
    for ticker, meta, match_symbol in companies:
        sym = (re.compile(rf"(?<![A-Za-z0-9]){re.escape(ticker)}(?![A-Za-z0-9])")
               if match_symbol else never)
        aliases = meta.get("aliases") or []
        joined = "|".join(re.escape(a) for a in aliases) or r"(?!x)x"
        alias = re.compile(rf"(?<![A-Za-z0-9])(?:{joined})(?![A-Za-z0-9])", re.IGNORECASE)
        pats[ticker] = (sym, alias)
    return pats


_WIRE_DATELINE = re.compile(r"\((Reuters|Bloomberg)\) (?:--|-)")


def wire_publisher(row: dict[str, Any]) -> str | None:
    """The real publisher of syndicated wire copy, or None.

    Finnhub labels everything Yahoo carries as "Yahoo", which is tier 4 — and
    that includes genuine Reuters and Bloomberg stories, the tier-2 wire copy
    this stack otherwise lacks (CODEBASE-PLAN §12). Their text keeps the wire's
    own dateline ("Dec 17 (Reuters) - ...", "(Bloomberg) -- ..."). Only that
    dateline is trusted; an article that merely mentions Bloomberg stays tier 4.
    """
    m = _WIRE_DATELINE.search(row.get("summary") or "")
    return m.group(1).lower() if m else None


def resolve_tier(row: dict[str, Any]) -> int:
    """Wire dateline first, then publisher, feed tier as fallback, tier 4 default."""
    raw = row.get("raw") or {}
    tiers = _publisher_tiers()
    if wire := wire_publisher(row):
        return tiers.get(_norm_publisher(wire), DEFAULT_TIER)

    # Finnhub company-news carries the real publisher in raw.source.
    for key in ("publisher", "source"):
        v = _norm_publisher(raw.get(key) or "")
        if v and v in tiers:
            return tiers[v]

    src = _norm_publisher(row.get("source") or "")
    if src in tiers:
        return tiers[src]
    if isinstance(raw.get("source_tier"), int):
        return int(raw["source_tier"])
    return DEFAULT_TIER


def resolve_tickers(row: dict[str, Any]) -> list[str]:
    """Provenance tags UNIONED with the companies the text actually names.

    ⚠️ These three used to return early, and that lost evidence. An article
    pulled from NVDA's Finnhub company feed was tagged NVDA and nothing else, so
    "Apple stock drops 6% on MacBook and iPad price hikes" never carried AAPL
    and could not be retrieved for an Apple swing at all. 1,259 tagged articles
    (7%) named a watchlist company their tags omitted, including an Apple IR
    release about Broadcom and a CNBC piece headlined "AI concerns hit Alphabet"
    tagged MSFT.

    Provenance is still trusted — an EDGAR filer and a single-company IR feed are
    facts about who published, not guesses — it simply no longer suppresses what
    the headline says. `_ticker_patterns` keeps this from over-tagging: a short
    symbol must appear in capitals, and aliases match on word boundaries.
    """
    raw = row.get("raw") or {}
    tags: set[str] = set()
    if raw.get("ticker"):                       # EDGAR: the filer
        tags.add(str(raw["ticker"]).upper())
    if raw.get("feed_tickers"):                 # GDELT orgs, single-company feed
        tags.update(t.upper() for t in raw["feed_tickers"])
    if row.get("source") in _feed_tickers():
        tags.update(t.upper() for t in _feed_tickers()[row["source"]])

    text = f"{row.get('headline') or ''} {row.get('summary') or ''}"
    tags.update(t for t, (sym, alias) in _ticker_patterns().items()
                if sym.search(text) or alias.search(text))
    return sorted(tags)


def retag_all(batch: int = 2000) -> int:
    """Re-derive `articles.tickers` from the archive after tagging config changes
    (new related companies or aliases). Rows are updated in place, so cluster and
    annotation references to article ids survive. Returns rows changed."""
    changed, last_id = 0, 0
    while True:
        with connect() as conn:
            rows = conn.execute(
                "SELECT a.id, a.tickers, r.source, r.headline, r.summary, r.raw "
                "FROM articles a JOIN articles_raw r ON r.id = a.raw_id "
                "WHERE a.id > %s ORDER BY a.id LIMIT %s", (last_id, batch)).fetchall()
        if not rows:
            return changed
        updates = [(tags, r["id"]) for r in rows
                   if set(tags := resolve_tickers(r)) != set(r["tickers"] or [])]
        if updates:
            with connect() as conn, conn.cursor() as cur:
                cur.executemany("UPDATE articles SET tickers = %s WHERE id = %s", updates)
            changed += len(updates)
        last_id = rows[-1]["id"]


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
        # ⚠️ A foreign-language wire release keeps its publisher's tier and so
        # outranks every newsroom story, while embedding as noise: the model is
        # bge-base-en. GOOGL 2026-09-21 was offered two German/Spanish Artprice
        # releases as its best pre-move evidence. Demoted, never dropped.
        tier = demote_non_english(r["headline"], r.get("summary"), tier)
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
                # Wire copy is credited to the wire, not to the pipe that carried it.
                "raw_id": r["id"], "url": r["url"], "source": wire_publisher(r) or r["source"],
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
