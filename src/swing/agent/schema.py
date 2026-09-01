"""The Attribution schema. This is what enforces honesty; the prompt only encourages it.

⚠️ Nesting depth: `Attribution -> candidates[] -> evidence[]` is three levels,
which agent-plan.md 3.1 warns sits at Gemini's boundary and may need flattening.

VERIFIED 2026-08-31 against gemini-3.7-flash and gemini-flash-latest: the nested
schema is accepted and returns correct structured output in both the
explain and abstain directions. **No flattening is required.** The flat variant
below is kept only as a fallback if a future model rejects the nested form —
do not use it by default, it exists to avoid rediscovering the workaround.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, Field

EventType = Literal[
    "earnings", "guidance", "analyst_action", "m_and_a",
    "regulatory", "litigation", "product", "macro", "management", "other",
]
Confidence = Literal["high", "medium", "low"]
Verdict = Literal["explained", "partially_explained", "unexplained"]
Timing = Literal["pre_move", "post_move"]


class Evidence(BaseModel):
    cluster_id: int
    headline: str
    source: str
    source_tier: int
    published_at: str
    timing: Timing
    distinct_sources: int
    relevance_note: str = Field(description="Why this connects to the move")


class Candidate(BaseModel):
    catalyst: str = Field(description="One sentence: what happened")
    event_type: EventType
    direction_consistent: bool = Field(
        description="Would this plausibly move the stock in the observed direction?"
    )
    magnitude_plausible: bool = Field(
        description="Is a move of this size plausible for this event type?"
    )
    evidence: list[Evidence]
    confidence: Confidence


class Attribution(BaseModel):
    ticker: str
    date: str
    total_return: float
    market_component: float
    sector_component: float
    residual: float
    residual_z: float

    # swing context, from Phase 1
    swing_type: Literal["gap", "intraday", "mixed", "drift", "unknown"]
    onset_ts: str
    volume_z: float
    earnings_mode: bool

    verdict: Verdict
    candidates: list[Candidate]
    unexplained_note: str | None = None
    source_disagreement: str | None = Field(
        default=None, description="Where outlets frame the same event differently"
    )
    reactive_coverage_note: str | None = Field(
        default=None, description="Post-move commentary, explicitly flagged as after-the-fact"
    )


# --------------------------------------------------------------------------
# Fallback only. Unused while the nested schema works (verified 2026-08-31).
# --------------------------------------------------------------------------

class CandidateFlat(BaseModel):
    candidate_id: str
    catalyst: str
    event_type: EventType
    direction_consistent: bool
    magnitude_plausible: bool
    confidence: Confidence


class EvidenceFlat(Evidence):
    candidate_id: str  # foreign key back to CandidateFlat


class AttributionFlat(BaseModel):
    ticker: str
    date: str
    total_return: float
    market_component: float
    sector_component: float
    residual: float
    residual_z: float
    swing_type: Literal["gap", "intraday", "mixed", "drift", "unknown"]
    onset_ts: str
    volume_z: float
    earnings_mode: bool
    verdict: Verdict
    candidates: list[CandidateFlat]
    evidence: list[EvidenceFlat]
    unexplained_note: str | None = None
    source_disagreement: str | None = None
    reactive_coverage_note: str | None = None


def rehydrate(flat: AttributionFlat) -> Attribution:
    """Join the two flat lists back into the nested shape."""
    by_candidate: dict[str, list[Evidence]] = defaultdict(list)
    for e in flat.evidence:
        by_candidate[e.candidate_id].append(Evidence(**e.model_dump(exclude={"candidate_id"})))
    candidates = [
        Candidate(**c.model_dump(exclude={"candidate_id"}), evidence=by_candidate[c.candidate_id])
        for c in flat.candidates
    ]
    scalars = flat.model_dump(exclude={"candidates", "evidence"})
    return Attribution(**scalars, candidates=candidates)
