"""The StateGraph. agent-plan.md 3.3.

    load_swing
        |
        +-- residual_z below threshold --> emit_no_move ------> persist -> END
        |
        +-- zero pre-move clusters ------> emit_no_evidence --> persist -> END
        |
    llm_attribute        (structured output bound to Attribution)
        |
    apply_guards         (validate_citations then enforce_abstention; no LLM)
        |
    persist
        |
       END

`StateGraph` rather than the prebuilt `create_agent` loop, because this flow has
a real branch (abstain vs explain) and a mandatory non-LLM validation node.
`AgentExecutor` is deprecated; nothing here uses it.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from langgraph.graph import END, StateGraph

from swing.agent import nodes
from swing.agent.state import AttributionState


@lru_cache(maxsize=1)
def build_graph():
    g = StateGraph(AttributionState)

    g.add_node("load_swing", nodes.load_swing)
    g.add_node("emit_no_move", nodes.emit_no_move)
    g.add_node("emit_no_evidence", nodes.emit_no_evidence)
    g.add_node("llm_attribute", nodes.llm_attribute)
    g.add_node("apply_guards", nodes.apply_guards)
    g.add_node("persist", nodes.persist)

    g.set_entry_point("load_swing")
    g.add_conditional_edges("load_swing", nodes.route_after_load, {
        "no_move": "emit_no_move",
        "no_evidence": "emit_no_evidence",
        "attribute": "llm_attribute",
    })
    g.add_edge("emit_no_move", "persist")
    g.add_edge("emit_no_evidence", "persist")
    g.add_edge("llm_attribute", "apply_guards")
    g.add_edge("apply_guards", "persist")
    g.add_edge("persist", END)
    return g.compile()


def attribute_swing(swing_id: int, run_kind: str = "production",
                    persist: bool = True,
                    cluster_override: dict[str, list] | None = None,
                    verdict_reason_tag: str | None = None) -> dict[str, Any]:
    """Run one swing through the graph. Returns the final state."""
    state: AttributionState = {
        "swing_id": swing_id,
        "run_kind": run_kind,
        # Dry runs (Gate 3, weight tuning) must not pollute the metrics tables.
        "persist": persist,
        "cluster_override": cluster_override,
        "verdict_reason_tag": verdict_reason_tag,
        "messages": [],
    }
    return build_graph().invoke(state)
