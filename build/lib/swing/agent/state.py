"""Graph state. TypedDict, not Pydantic.

agent-plan.md 3.3: TypedDict has lower overhead and plays cleanly with
checkpointing. The `add_messages` reducer on any message list is mandatory —
forgetting it is the most common LangGraph bug, and the symptom is the agent
losing context between nodes.
"""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from swing.agent.schema import Attribution


class AttributionState(TypedDict, total=False):
    swing_id: int
    swing: dict[str, Any]
    decomposition: str
    pre_clusters: list[dict[str, Any]]
    post_clusters: list[dict[str, Any]]
    allowed_cluster_ids: list[int]
    cluster_override: dict[str, list[dict[str, Any]]] | None

    raw_attribution: Attribution | None
    attribution: Attribution | None

    prompt_version: str
    verdict_reason: str | None
    verdict_reason_tag: str | None
    validation: dict[str, Any] | None
    run_kind: str
    persist: bool

    messages: Annotated[list[AnyMessage], add_messages]
