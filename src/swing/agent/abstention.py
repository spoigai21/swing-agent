"""Code-level abstention. THE most important function in the agent.

agent-plan.md 3.2: "Prompts are suggestions; code is a guarantee." An LLM handed
a price move and a pile of articles will always produce a story unless you
engineer against it, so this runs AFTER the model returns and overrides it.

A candidate survives only if it is backed by at least one PRE-MOVE piece of
evidence from a credible tier AND the direction is consistent. Everything else
becomes `unexplained` — which is a correct and expected output, not a failure.
"""
from __future__ import annotations

from swing.agent.schema import Attribution, Candidate
from swing.ingest.config import thresholds

# Opening of every unexplained note; the answer screen strips it so the reply
# does not repeat itself.
NOTE_BASE = "No pre-move catalyst identified from credible sources. "


def _is_valid(c: Candidate) -> bool:
    return (
        any(e.timing == "pre_move" and e.source_tier <= 3 for e in c.evidence)
        and c.direction_consistent
    )


def enforce_abstention(attr: Attribution) -> Attribution:
    """Strip unsupported candidates and downgrade the verdict accordingly.

    ⚠️ Deviation from agent-plan.md 3.2, which tests `attr.residual_z > 3.0`.
    That is signed, so a -4 sigma crash — exactly the case where a weak
    explanation is most dangerous — would skip the downgrade entirely. We test
    `abs(residual_z)`.
    """
    valid = [c for c in attr.candidates if _is_valid(c)]

    if not valid:
        attr.verdict = "unexplained"
        attr.candidates = []
        attr.unexplained_note = unexplained_note(attr.volume_z)
        return attr

    attr.candidates = valid
    if abs(attr.residual_z) > 3.0 and all(c.confidence == "low" for c in valid):
        # A big move explained only by weak evidence is not "explained".
        attr.verdict = "partially_explained"
    elif attr.verdict == "unexplained":
        # The model abstained but supplied valid candidates; trust the model's
        # abstention rather than promoting it. Abstention is never overridden
        # upward, only downward.
        attr.candidates = []
        attr.unexplained_note = attr.unexplained_note or unexplained_note(attr.volume_z)
    return attr


def unexplained_note(volume_z: float | None) -> str:
    """Volume turns a dead-end verdict into a usable signal. agent-plan.md 3.2.

    Without this branch a flow event and a genuine coverage gap produce
    identical output; with it, the first is a real answer and the second is a
    prompt to add sources.
    """
    base = NOTE_BASE
    cfg = thresholds()["swings"]
    if volume_z is None:
        return base + "Volume context unavailable."
    if volume_z >= float(cfg["volume_flow_z"]):
        return base + (
            f"Volume was {volume_z:.1f}σ above normal, consistent with a flow event "
            "— index activity, a block trade, or a position unwind — rather than news."
        )
    if volume_z <= float(cfg["volume_thin_z"]):
        return base + (
            "Volume was below normal, so the move may reflect thin liquidity "
            "rather than a catalyst."
        )
    return base + (
        "Volume was unremarkable. The move may reflect positioning or "
        "information not captured in the monitored sources."
    )
