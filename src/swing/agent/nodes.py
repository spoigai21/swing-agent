"""One function per graph node, each independently testable.

Only `llm_attribute` touches the network. Everything else is pure Python over a
fully-materialised state, which is what lets the placebo harness replay
retrieval without re-fetching.
"""
from __future__ import annotations

from typing import Any

from swing.agent.abstention import enforce_abstention, unexplained_note
from swing.agent.schema import Attribution
from swing.agent.state import AttributionState
from swing.agent.validate import validate_citations
from swing.common import logging as log
from swing.ingest.config import thresholds

logger = log.get("agent.nodes")


def load_swing(state: AttributionState) -> dict[str, Any]:
    """Pull the swing, its decomposition sentence, and its persisted clusters."""
    from swing.analysis.decompose import load as load_decomp
    from swing.store.session import connect

    swing_id = state["swing_id"]
    with connect() as conn:
        swing = conn.execute("SELECT * FROM swings WHERE id=%s", (swing_id,)).fetchone()
        if not swing:
            raise ValueError(f"no swing {swing_id}")
        clusters = conn.execute(
            """
            SELECT c.id, c.timing, c.best_tier, c.distinct_sources, c.member_count,
                   c.earliest_published, c.rank, a.headline, a.summary, a.source
            FROM clusters c JOIN articles a ON a.id = c.canonical_article
            WHERE c.swing_id=%s ORDER BY c.timing, c.rank
            """,
            (swing_id,),
        ).fetchall()

    top_k = int(thresholds()["retrieval"]["max_clusters_to_llm"])
    override = state.get("cluster_override")
    if override is not None:
        # A real swing paired with someone else's articles: Gate 3's synthetic
        # no-news cases and the Phase 4 placebo harness. The correct answer is
        # always `unexplained`.
        pre = list(override.get("pre_move", []))[:top_k]
        post = list(override.get("post_move", []))[:top_k]
    else:
        pre = [dict(c) for c in clusters if c["timing"] == "pre_move"][:top_k]
        post = [dict(c) for c in clusters if c["timing"] == "post_move"][:top_k]
    decomp = load_decomp(swing["ticker"], swing["d"])

    return {
        "swing": dict(swing),
        "decomposition": decomp.sentence() if decomp else "",
        "pre_clusters": pre,
        "post_clusters": post,
        # Only what we actually PASSED may be cited.
        "allowed_cluster_ids": [c["id"] for c in pre + post],
    }


def route_after_load(state: AttributionState) -> str:
    """Two cheap exits before spending an LLM call."""
    swing = state["swing"]
    if abs(float(swing["residual_z"])) < float(thresholds()["swings"]["z_threshold"]):
        return "no_move"
    if not state["pre_clusters"]:
        return "no_evidence"
    return "attribute"


def _skeleton(swing: dict, verdict: str, note: str | None) -> Attribution:
    return Attribution(
        ticker=swing["ticker"], date=str(swing["d"]),
        total_return=float(swing["total_return"]),
        market_component=float(swing["market_component"]),
        sector_component=float(swing["sector_component"]),
        residual=float(swing["residual"]), residual_z=float(swing["residual_z"]),
        swing_type=swing["swing_type"],
        onset_ts=swing["onset_ts"].isoformat() if swing["onset_ts"] else "",
        volume_z=float(swing["volume_z"] or 0.0),
        earnings_mode=bool(swing["earnings_mode"]),
        verdict=verdict, candidates=[], unexplained_note=note,
    )


def emit_no_move(state: AttributionState) -> dict[str, Any]:
    swing = state["swing"]
    return {
        "attribution": _skeleton(swing, "unexplained",
                                 "No significant idiosyncratic move: the residual is "
                                 "within normal range for this name."),
        "verdict_reason": "residual_below_threshold",
    }


def emit_no_evidence(state: AttributionState) -> dict[str, Any]:
    """Zero pre-move clusters. Abstain WITHOUT calling the model.

    There is nothing for it to reason over, and asking anyway is exactly how a
    confabulation gets produced.
    """
    swing = state["swing"]
    return {
        "attribution": _skeleton(swing, "unexplained",
                                 unexplained_note(float(swing["volume_z"] or 0.0))),
        "verdict_reason": "no_pre_move_clusters",
    }


def llm_attribute(state: AttributionState) -> dict[str, Any]:
    """The only node that touches the network."""
    from swing.agent.llm import get_attributor, invoke_with_retry
    from swing.agent.prompts import render

    swing = state["swing"]
    version, prompt = render(swing, state["decomposition"],
                             state["pre_clusters"], state["post_clusters"])
    attributor = get_attributor()
    try:
        out = invoke_with_retry(attributor, prompt)
    except Exception:
        logger.exception("attribution call failed for swing %s", swing["id"])
        return {"raw_attribution": None, "prompt_version": version,
                "verdict_reason": "llm_error"}
    return {"raw_attribution": out, "prompt_version": version}


def apply_guards(state: AttributionState) -> dict[str, Any]:
    """Citation validation THEN abstention, in that order.

    Order matters: dropping a hallucinated citation can leave a candidate with
    no evidence, and abstention must then see that candidate as unsupported.
    Running abstention first would let it pass on evidence that does not exist.
    """
    swing = state["swing"]
    raw = state.get("raw_attribution")
    if raw is None:
        return {
            "attribution": _skeleton(swing, "unexplained",
                                     unexplained_note(float(swing["volume_z"] or 0.0))),
            "validation": {"ok": True, "invalid_ids": [], "dropped_candidates": 0},
        }

    attr, result = validate_citations(raw, set(state["allowed_cluster_ids"]))
    if not result.ok:
        logger.warning("swing %s cited unknown cluster ids: %s",
                       swing["id"], result.invalid_ids)
    attr = enforce_abstention(attr)
    return {
        "attribution": attr,
        "validation": {"ok": result.ok, "invalid_ids": result.invalid_ids,
                       "dropped_candidates": result.dropped_candidates},
    }


def persist(state: AttributionState) -> dict[str, Any]:
    """Store the attribution with its full version stamp."""
    from swing.common.versioning import config_hash, model_id
    from swing.store.session import connect

    attr = state["attribution"]
    if attr is None or not state.get("persist", True):
        return {}
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO attributions (swing_id, verdict, payload, unexplained_note,
                                      prompt_version, model_id, config_hash, run_kind)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (state["swing_id"], attr.verdict, attr.model_dump_json(),
             attr.unexplained_note, state.get("prompt_version", "none"),
             model_id(), config_hash(), state.get("run_kind", "production")),
        )
    return {}
