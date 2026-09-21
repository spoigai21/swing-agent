"""Forecast refusal. Runs BEFORE any model call.

`swing-cli.md` Part 8 requires that `swing ask "is NVDA a buy?"` refuses without
spending an LLM call. `agent-plan.md` states the system is not a price predictor
and must not be tuned or evaluated as one, but specifies no mechanism — so this
defines one.

Why refuse rather than answer carefully: every number this system produces is
backward-looking and evaluated backward-looking. An answer about the future
would carry the same confident, cited presentation as an attribution while
having none of the evidence behind it, and a reader cannot tell the difference.
That is the confabulation failure in a different costume.

⚠️ Deliberately conservative. A false refusal costs the user one rephrase; a
false accept produces investment advice from a system with no forecasting
capability. Refusals name the nearest thing the system CAN answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Forward-looking intent. Matched case-insensitively on word boundaries.
_FORECAST_PATTERNS = [
    r"\b(should|shall)\s+i\s+(buy|sell|short|hold|invest|purchase)\b",
    r"\bis\s+\w+\s+a\s+(buy|sell|short|good\s+buy|good\s+investment)\b",
    r"\b(will|would|could|might|going\s+to)\s+\w+\s+(go|rise|fall|drop|rally|crash|moon|tank|climb|decline|increase|decrease|hit|reach|beat)\b",
    r"\b(price\s+target|forecast|prediction|predict|projection)\b",
    r"\bwhat.{0,20}\b(next\s+(week|month|quarter|year)|tomorrow|future)\b",
    r"\b(buy|sell|short)\s+(signal|recommendation|rating)\b",
    r"\bworth\s+(buying|investing|holding)\b",
    r"\b(outlook|upside|downside)\s+for\b",
    r"\bshould\s+\w+\s+be\s+(bought|sold|held)\b",
    r"\bgood\s+(time|entry|moment)\s+to\s+(buy|sell|short)\b",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _FORECAST_PATTERNS]

REFUSAL = (
    "This system explains price moves that have already happened. It is not a "
    "price predictor and is not evaluated as one, so it will not answer "
    "forward-looking or advice questions.\n\n"
    "It can answer things like:\n"
    "  • why did NVDA move on 2026-08-27?\n"
    "  • show me every unexplained swing this month\n"
    "  • which tickers have the highest idiosyncratic share?\n"
    "  • what drove the semiconductor sector last week?"
)


@dataclass(slots=True)
class GuardResult:
    allowed: bool
    reason: str | None = None
    matched: str | None = None


def check(question: str) -> GuardResult:
    """Screen a question before it reaches the model."""
    q = (question or "").strip()
    if not q:
        return GuardResult(False, "empty question")
    for pat in _COMPILED:
        m = pat.search(q)
        if m:
            return GuardResult(False, REFUSAL, m.group(0))
    return GuardResult(True)
