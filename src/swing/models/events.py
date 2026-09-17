"""Phase 5.2 — event classification, and its free labels.

agent-plan.md Step 5.2 wants clusters classified into the `EventType` taxonomy,
with a stated payoff: empirical base rates turn `magnitude_plausible` from an
LLM guess into a data-backed check.

The labels are free. An 8-K carries the SEC Item numbers describing what it
reports, and `normalize.event_hint` already stores them, so a few thousand
filings are self-labelling — no annotation effort at all.

⚠️ But only a few Items map cleanly onto the ten-value taxonomy. 2.02 really is
earnings and 5.02 really is a management change; 7.01 (Reg FD) and 8.01 (Other
Events) can be literally anything, so they can only become `other`. That caps
what any classifier trained here can learn, and the cap is the point of running
the baseline first: agent-plan.md says if the transformer cannot beat LightGBM
on TF-IDF, "you do not have enough labels yet — go label more instead of tuning
hyperparameters."
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from swing.store.session import connect

# 9.01 is "Financial Statements and Exhibits" — it accompanies other items and
# is never itself the event, so it is dropped before deciding.
NON_EVENT_ITEMS = {"9.01"}

# Some SOURCES are a label on their own. Every analyst-ratings row is, by
# construction, an analyst action ("Goldman Sachs reiterates Neutral on Tesla
# (TSLA); price target $360"), which buys a fifth taxonomy class that 8-K Item
# numbers cannot express at all.
#
# ⚠️ This is not the leakage that 8-K Items were. There the label was DERIVED
# from a string printed in the text; here the label comes from provenance and
# the text independently describes the event. But the phrasing is templated, so
# the class is easy and will lift macro-F1 for every model equally — read the
# per-class F1, not the headline number.
SOURCE_LABELS = {"analyst-ratings": "analyst_action"}

# Highest precedence first: an 8-K filed under 2.02 AND 7.01 is an earnings
# release with a Reg FD courtesy copy, not a Reg FD disclosure.
ITEM_TO_EVENT: list[tuple[frozenset[str], str]] = [
    (frozenset({"2.02"}), "earnings"),                      # Results of Operations
    (frozenset({"2.01", "1.01", "1.02"}), "m_and_a"),       # Completion / material agreement
    (frozenset({"5.02", "5.07"}), "management"),            # Officers, directors, votes
    (frozenset({"1.03", "3.01"}), "regulatory"),            # Bankruptcy, listing failure
    (frozenset({"8.01", "7.01", "2.03", "5.03", "3.02",
                "4.01", "5.01", "2.05", "2.06"}), "other"),
]


def event_type_from_items(hint: str | None) -> str | None:
    """Map an 8-K Item string ("2.02,9.01") to an EventType, or None."""
    if not hint:
        return None
    items = {i.strip() for i in hint.split(",") if i.strip()} - NON_EVENT_ITEMS
    if not items:
        return None
    for group, label in ITEM_TO_EVENT:
        if items & group:
            return label
    return "other"


# ⚠️ The labels are DERIVED from the Item numbers, and EDGAR headlines contain
# them verbatim ("NVDA 8-K — Item 2.02,9.01 — ..."). Training on that text lets
# the model read the answer off the input: it scored 0.900 accuracy leaky versus
# 0.835 with the items stripped. Anything that identifies the filing rather than
# describing the event has to go, so the dataset cannot leak by construction.
_ITEMS = re.compile(r"item[s]?\s*[\d.,\s]+", re.IGNORECASE)
_FORM = re.compile(r"\b[86]-K\b|\bFORM\b|\bReport date\b|\d{4}-\d{2}-\d{2}", re.IGNORECASE)
_LEAD_TICKER = re.compile(r"^\s*[A-Z]{1,5}\b")


def strip_label_leakage(text: str) -> str:
    """Remove Item numbers, form names, report dates and the leading ticker."""
    out = _LEAD_TICKER.sub(" ", _ITEMS.sub(" ", text or ""))
    return re.sub(r"\s+", " ", _FORM.sub(" ", out)).strip(" -—·,")


@dataclass(frozen=True, slots=True)
class LabelledEvent:
    article_id: int
    published_at: datetime
    ticker: str | None
    text: str
    label: str


def dataset(min_words: int = 0) -> list[LabelledEvent]:
    """Every filing whose Item numbers imply an event type, oldest first.

    `min_words` keeps only rows with that many real words left after the leakage
    strip. It matters: 59% of these filings are pure boilerplate whose entire
    text was the form name and the Item numbers, so they carry a label and no
    evidence. On the 759 substantive rows a TF-IDF baseline scores 0.855
    accuracy against a 0.296 majority floor; including the empty ones measures
    how predictable each company's filing calendar is, not event classification.
    """
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, published_at, tickers, headline, coalesce(summary,'') AS summary, "
            "       event_hint, source "
            "FROM articles WHERE event_hint IS NOT NULL OR source = ANY(%s) "
            "ORDER BY published_at",
            (list(SOURCE_LABELS),)
        ).fetchall()

    out: list[LabelledEvent] = []
    for r in rows:
        label = event_type_from_items(r["event_hint"]) or SOURCE_LABELS.get(r["source"])
        if label is None:
            continue
        out.append(LabelledEvent(
            article_id=r["id"],
            published_at=r["published_at"],
            ticker=(r["tickers"] or [None])[0],
            # Headline plus summary is what the embedder sees too, so the
            # classifier is judged on the same signal retrieval actually has.
            text=strip_label_leakage(f"{r['headline']} {r['summary']}"),
            label=label,
        ))
    if min_words:
        out = [r for r in out if len(r.text.split()) >= min_words]
    return out


def distribution(rows: list[LabelledEvent]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.label] = counts.get(r.label, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
