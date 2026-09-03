"""Code-level abstention and citation validation.

These are the honesty guarantees. Prompts are suggestions; these are enforced
after the model returns, so they are tested exhaustively.
"""
from __future__ import annotations

import pytest

from swing.agent.abstention import enforce_abstention, unexplained_note
from swing.agent.schema import Attribution, Candidate, Evidence
from swing.agent.validate import validate_citations

SCALARS = {
    "ticker": "MRVL", "date": "2026-08-28", "total_return": -0.06,
    "market_component": -0.004, "sector_component": -0.011, "residual": -0.045,
    "residual_z": -3.2, "swing_type": "gap", "onset_ts": "2026-08-28T13:30:00Z",
    "volume_z": 2.8, "earnings_mode": True,
}


def _ev(cluster_id=1, timing="pre_move", tier=1):
    return Evidence(cluster_id=cluster_id, headline="h", source="sec-edgar",
                    source_tier=tier, published_at="2026-08-27T20:05:00Z",
                    timing=timing, distinct_sources=1, relevance_note="r")


def _cand(evidence=None, direction=True, confidence="high"):
    return Candidate(catalyst="c", event_type="guidance", direction_consistent=direction,
                     magnitude_plausible=True,
                     evidence=evidence if evidence is not None else [_ev()],
                     confidence=confidence)


def _attr(candidates, verdict="explained", **over):
    return Attribution(**{**SCALARS, **over}, verdict=verdict, candidates=candidates)


class TestAbstention:
    def test_post_move_only_evidence_is_rejected(self):
        # The reactive-journalism trap: an article written BECAUSE of the move
        # is not evidence of its cause.
        a = enforce_abstention(_attr([_cand([_ev(timing="post_move")])]))
        assert a.verdict == "unexplained" and a.candidates == []

    def test_direction_inconsistent_candidate_is_rejected(self):
        a = enforce_abstention(_attr([_cand(direction=False)]))
        assert a.verdict == "unexplained"

    def test_tier_4_evidence_is_rejected(self):
        a = enforce_abstention(_attr([_cand([_ev(tier=4)])]))
        assert a.verdict == "unexplained"

    def test_valid_candidate_survives(self):
        a = enforce_abstention(_attr([_cand()]))
        assert a.verdict == "explained" and len(a.candidates) == 1

    def test_mixed_candidates_keeps_only_the_valid_one(self):
        good, bad = _cand(), _cand([_ev(timing="post_move")])
        a = enforce_abstention(_attr([bad, good]))
        assert a.candidates == [good]

    def test_candidate_with_one_valid_evidence_among_many_survives(self):
        a = enforce_abstention(_attr([_cand([_ev(timing="post_move"), _ev()])]))
        assert a.verdict == "explained"

    def test_empty_candidates_gets_a_volume_aware_note(self):
        a = enforce_abstention(_attr([]))
        assert a.verdict == "unexplained"
        assert "flow event" in (a.unexplained_note or "")   # volume_z = 2.8

    def test_big_move_on_only_low_confidence_is_downgraded(self):
        a = enforce_abstention(_attr([_cand(confidence="low")], residual_z=-4.0))
        assert a.verdict == "partially_explained"

    def test_negative_z_triggers_the_downgrade(self):
        """agent-plan.md 3.2 tests `residual_z > 3.0`, which is SIGNED — a -4
        sigma crash would skip the downgrade entirely. We test abs()."""
        a = enforce_abstention(_attr([_cand(confidence="low")], residual_z=-4.5))
        assert a.verdict == "partially_explained"

    def test_model_abstention_is_never_promoted(self):
        # Guards only ever move the verdict toward abstention, never away.
        a = enforce_abstention(_attr([_cand()], verdict="unexplained"))
        assert a.verdict == "unexplained" and a.candidates == []


class TestUnexplainedNote:
    def test_high_volume_reads_as_a_flow_event(self):
        assert "flow event" in unexplained_note(2.5)

    def test_low_volume_reads_as_thin_liquidity(self):
        assert "thin liquidity" in unexplained_note(-1.5)

    def test_normal_volume_is_a_coverage_gap(self):
        n = unexplained_note(0.2)
        assert "unremarkable" in n and "flow event" not in n

    def test_missing_volume_is_stated_not_guessed(self):
        assert "unavailable" in unexplained_note(None)


class TestCitationValidation:
    def test_invented_cluster_id_is_dropped(self):
        attr, res = validate_citations(_attr([_cand([_ev(cluster_id=999)])]), {1, 2})
        assert not res.ok and res.invalid_ids == [999]
        assert attr.candidates == []          # no evidence left -> candidate dropped

    def test_valid_ids_pass_through(self):
        attr, res = validate_citations(_attr([_cand([_ev(cluster_id=2)])]), {1, 2})
        assert res.ok and len(attr.candidates) == 1

    def test_partial_invalidity_keeps_the_valid_evidence(self):
        attr, res = validate_citations(
            _attr([_cand([_ev(cluster_id=1), _ev(cluster_id=999)])]), {1})
        assert not res.ok
        assert [e.cluster_id for e in attr.candidates[0].evidence] == [1]

    def test_hallucinated_citation_then_abstention_yields_unexplained(self):
        # Order matters: validation must run BEFORE abstention, or abstention
        # passes a candidate on evidence that does not exist.
        attr, _ = validate_citations(_attr([_cand([_ev(cluster_id=999)])]), {1})
        attr = enforce_abstention(attr)
        assert attr.verdict == "unexplained"


@pytest.mark.parametrize("timing,tier,ok", [
    ("pre_move", 1, True), ("pre_move", 3, True),
    ("pre_move", 4, False), ("post_move", 1, False),
])
def test_evidence_admissibility_matrix(timing, tier, ok):
    a = enforce_abstention(_attr([_cand([_ev(timing=timing, tier=tier)])]))
    assert (a.verdict == "explained") is ok
