"""Deterministic citation checking. agent-plan.md 3.3: NOT optional.

Models occasionally invent plausible-looking cluster ids. Every cited id must
have appeared in the set we actually passed the model, and the check is done in
code because a prompt cannot guarantee it.
"""
from __future__ import annotations

from dataclasses import dataclass

from swing.agent.schema import Attribution


@dataclass(slots=True)
class ValidationResult:
    ok: bool
    invalid_ids: list[int]
    dropped_candidates: int


def validate_citations(attr: Attribution, allowed_ids: set[int],
                       drop_invalid: bool = True) -> tuple[Attribution, ValidationResult]:
    """Remove evidence citing ids we never supplied, then any candidate left bare.

    Dropping rather than merely flagging matters: a candidate whose only support
    was a hallucinated id has no support at all, and leaving it in would let an
    invented citation reach the user.
    """
    invalid: list[int] = []
    kept_candidates = []
    dropped = 0

    for cand in attr.candidates:
        good_evidence = []
        for ev in cand.evidence:
            if ev.cluster_id in allowed_ids:
                good_evidence.append(ev)
            else:
                invalid.append(ev.cluster_id)
        if not drop_invalid:
            kept_candidates.append(cand)
            continue
        if good_evidence:
            cand.evidence = good_evidence
            kept_candidates.append(cand)
        else:
            dropped += 1

    if drop_invalid:
        attr.candidates = kept_candidates
    return attr, ValidationResult(not invalid, sorted(set(invalid)), dropped)
