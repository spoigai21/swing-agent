"""Attribution schema invariants, including the flat<->nested round trip."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from swing.agent.schema import (
    Attribution,
    AttributionFlat,
    Candidate,
    CandidateFlat,
    Evidence,
    EvidenceFlat,
    rehydrate,
)

SCALARS = {
    "ticker": "MRVL", "date": "2026-08-28", "total_return": -0.06,
    "market_component": -0.004, "sector_component": -0.011, "residual": -0.045,
    "residual_z": -3.2, "swing_type": "gap", "onset_ts": "2026-08-28T13:30:00Z",
    "volume_z": 2.8, "earnings_mode": True,
}


def _evidence(**kw):
    base = {
        "cluster_id": 1, "headline": "h", "source": "sec-edgar", "source_tier": 1,
        "published_at": "2026-08-27T20:05:00Z", "timing": "pre_move",
        "distinct_sources": 1, "relevance_note": "r",
    }
    return {**base, **kw}


def test_unexplained_with_no_candidates_is_valid():
    a = Attribution(**SCALARS, verdict="unexplained", candidates=[], unexplained_note="none found")
    assert a.candidates == []


def test_verdict_is_constrained():
    with pytest.raises(ValidationError):
        Attribution(**SCALARS, verdict="probably", candidates=[])


def test_timing_is_constrained():
    with pytest.raises(ValidationError):
        Evidence(**_evidence(timing="during_move"))


def test_event_type_is_constrained():
    with pytest.raises(ValidationError):
        Candidate(catalyst="c", event_type="vibes", direction_consistent=True,
                  magnitude_plausible=True, evidence=[], confidence="high")


def test_rehydrate_joins_evidence_to_the_right_candidate():
    flat = AttributionFlat(
        **SCALARS, verdict="explained",
        candidates=[
            CandidateFlat(candidate_id="c1", catalyst="guidance cut", event_type="guidance",
                          direction_consistent=True, magnitude_plausible=True, confidence="high"),
            CandidateFlat(candidate_id="c2", catalyst="downgrade", event_type="analyst_action",
                          direction_consistent=True, magnitude_plausible=False, confidence="low"),
        ],
        evidence=[
            EvidenceFlat(candidate_id="c1", **_evidence(cluster_id=1)),
            EvidenceFlat(candidate_id="c1", **_evidence(cluster_id=2)),
            EvidenceFlat(candidate_id="c2", **_evidence(cluster_id=3)),
        ],
    )
    nested = rehydrate(flat)
    assert isinstance(nested, Attribution)
    assert [c.catalyst for c in nested.candidates] == ["guidance cut", "downgrade"]
    assert [e.cluster_id for e in nested.candidates[0].evidence] == [1, 2]
    assert [e.cluster_id for e in nested.candidates[1].evidence] == [3]


def test_rehydrate_handles_a_candidate_with_no_evidence():
    # enforce_abstention will strip these, but the join must not crash first.
    flat = AttributionFlat(
        **SCALARS, verdict="explained",
        candidates=[CandidateFlat(candidate_id="c1", catalyst="x", event_type="other",
                                  direction_consistent=True, magnitude_plausible=True,
                                  confidence="low")],
        evidence=[],
    )
    assert rehydrate(flat).candidates[0].evidence == []
