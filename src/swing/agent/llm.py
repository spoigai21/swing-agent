"""Provider abstraction for the attribution model.

⚠️ FREE-TIER DAILY CAPS ARE PER MODEL AND SMALL.

Probed 2026-09-03 with a live key:
  gemini-2.5-flash, gemini-2.5-flash-lite   ModelNotFound (retired)
  gemini-3.7-flash                          429, daily cap = 20 requests
  gemini-3.5-flash, gemini-flash-latest,
  gemini-flash-lite-latest                  available

agent-plan.md 1.1 warned that Pro free tiers ran as low as 50/day; a
Flash-class model at 20/day is tighter still. Production is 1-3 attributions a
day and fits easily, but a Phase 4 placebo sweep is 200 requests and does NOT.
Iterate against the 30-case smoke set and budget full sweeps across days.

Swapping providers must be one config line (agent-plan.md 1.1), so nothing
outside this module names a model or a vendor.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from swing.common import logging as log
from swing.common.http import _RateLimiter
from swing.common.settings import REPO_ROOT, get_settings

# ⚠️ The free tier's per-minute cap is 5 RPM, not just 20/day. _RateLimiter takes
# SECONDS BETWEEN CALLS, so 4.0 allowed 15/min — three times over — and every
# batch tripped "GenerateRequestsPerMinutePerProjectPerModel-FreeTier" after a
# handful of calls. That is why runs kept stopping at 7-10 cases and being
# misread as the daily quota running out. 13s = 4.6/min, just under the ceiling.
LLM_MIN_INTERVAL = 13.0
_llm_limiter = _RateLimiter(LLM_MIN_INTERVAL)


logger = log.get("agent.llm")


# Google names the quota it refused on. The per-MINUTE one clears in seconds and
# is worth waiting for; the per-DAY one does not return until midnight Pacific,
# so retrying it burns two more attempts and up to 45s of backoff for nothing.
DAILY_QUOTA_MARKER = "GenerateRequestsPerDayPerProjectPerModel"


def is_daily_cap(exc: BaseException) -> bool:
    return DAILY_QUOTA_MARKER in str(exc)


def _is_transient(exc: BaseException) -> bool:
    """Rate limits (429) and overload (503). Both usually clear quickly.

    ⚠️ Except a DAILY cap, which clears at midnight PT. On 2026-09-19 a fresh
    20-request day produced only 7 stored cases; every failure was the daily
    quota, and each was retried three times because this could not tell the two
    429s apart.
    """
    if is_daily_cap(exc):
        return False
    text = str(exc)
    return any(s in text for s in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"))


# ---------------------------------------------------------------- batch mode
# ⚠️ Under a hard 20/day cap a retry is NOT free: every attempt Google serves —
# or refuses with 503 "high demand" — spends one of the twenty. On 2026-09-20
# the day's last 10 requests bought only 2 scored cases, because three of them
# were retried twice each: 5 extra requests to rescue 1 case, a losing trade.
#
# When the failures are server-side they are independent of WHICH case is being
# asked, so moving to the next case is exactly as likely to succeed as asking
# the same one again, and it adds a case instead of repeating one. Interactive
# use is the opposite — someone is waiting for THAT swing — so this is a mode,
# not a new default.
_BATCH = ContextVar("swing_llm_batch", default=False)


@contextmanager
def batch_mode():
    """Spend each request on a new case instead of retrying the last one."""
    token = _BATCH.set(True)
    try:
        yield
    finally:
        _BATCH.reset(token)


def _should_retry(exc: BaseException) -> bool:
    return not _BATCH.get() and _is_transient(exc)


def _log_attempt(state) -> None:
    """Make the request:case ratio visible.

    A 20-request day yielded 7 stored cases and the gap could not be accounted
    for, because nothing recorded how many requests one case actually costs.
    """
    logger.info("llm retry %d after %s", state.attempt_number,
                type(state.outcome.exception()).__name__ if state.outcome else "?")


# ---------------------------------------------------------------- quota ledger
# The attributions table only stores SUCCESSFUL calls, so it undercounts what the
# day actually cost: a failed call plus its retries spends real quota and leaves
# no row. On 2026-09-16 that made "10 calls left" read from the DB when the true
# figure was 0, and the run walked into 429 RESOURCE_EXHAUSTED. Count attempts
# here instead, where every request passes through.
QUOTA_PATH = REPO_ROOT / "data" / "gemini_usage.json"
DAILY_QUOTA = 20                     # free tier, per model per day
PT = ZoneInfo("America/Los_Angeles")  # Google's free-tier day resets at midnight PT


def _today() -> str:
    return datetime.now(PT).strftime("%Y-%m-%d")


def spent_today() -> int:
    try:
        return int(json.loads(QUOTA_PATH.read_text()).get(_today(), 0))
    except (OSError, ValueError, TypeError):
        return 0


def remaining_today() -> int:
    return max(0, DAILY_QUOTA - spent_today())


def is_quota_rejection(exc: BaseException) -> bool:
    """A 429 RESOURCE_EXHAUSTED: the request was refused, not served."""
    text = f"{type(exc).__name__} {exc}"
    return "RESOURCE_EXHAUSTED" in text or "429" in text


def _write_quota(day: str, value: int) -> int:
    try:
        data = json.loads(QUOTA_PATH.read_text())
    except (OSError, ValueError):
        data = {}
    data[day] = max(0, value)
    # Keep the file small; a fortnight is plenty to debug a bad night.
    for old in sorted(data)[:-14]:
        data.pop(old, None)
    QUOTA_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUOTA_PATH.write_text(json.dumps(data, indent=2, sort_keys=True))
    return data[day]


_daily_cap_day: str | None = None


def daily_cap_reached() -> bool:
    """True once the API itself has refused for the day. Clears at midnight PT."""
    return _daily_cap_day == _today()


def note_daily_cap() -> None:
    """Believe Google over the ledger.

    The ledger counts attempts from this process, so it drifts from Google's
    count in both directions — a hand-edit, a call from another checkout, a
    refusal it could not classify. A daily-cap refusal is the one moment the
    true answer is known, so pin the day to its full quota and let every caller
    that sizes work from remaining_today() see zero.
    """
    global _daily_cap_day
    _daily_cap_day = _today()
    _write_quota(_today(), DAILY_QUOTA)


def record_call(n: int = 1) -> int:
    """Count one request that actually reached the model.

    ⚠️ Attempts that Google REJECTS for quota are not charged against the daily
    allowance, so they must not be counted. Counting them read 36 attempts on a
    20-request day, and since schedule.placebo_if_due sizes the nightly batch
    from remaining_today(), that would skip capacity Gate 4 actually had — the
    opposite of the undercount this ledger was built to fix.
    """
    return _write_quota(_today(), spent_today() + n)


@retry(
    retry=retry_if_exception(_should_retry),
    # A person is waiting at the prompt: three tries over ~30 s, not minutes.
    # On 2026-09-14 every Flash model returned intermittent 503 "high demand";
    # only 429 was retried, so one spike failed the whole question.
    # Google asks for a 34s retry delay on an RPM refusal; a 20s ceiling
    # guaranteed the retry failed too, so two refusals ended the batch.
    wait=wait_exponential(multiplier=2, min=5, max=45),
    stop=stop_after_attempt(3),
    before_sleep=_log_attempt,
    reraise=True,
)
def invoke_with_retry(runnable, prompt: str):
    """Paced, 429-aware invoke. agent-plan.md 3.4 requires this.

    Free-tier quotas throttle by requests-per-minute as well as per-day, and a
    batch firing attributions back to back will trip RPM long before the daily
    cap. The batch has no latency requirement, so pacing costs nothing.
    """
    _llm_limiter.wait()
    record_call()          # a served call costs quota even when it then fails
    try:
        return runnable.invoke(prompt)
    except Exception as exc:
        # ⚠️ This must un-count on EVERY attempt, not only the one that escapes.
        # tenacity swallows the first two, so a batch that served 7 calls logged
        # 23 — and schedule.placebo_if_due sizes the night from remaining_today(),
        # so an inflated ledger makes Gate 4 skip capacity it actually has.
        if is_daily_cap(exc):
            note_daily_cap()    # the day is over; stop guessing at what is left
        elif is_quota_rejection(exc):
            record_call(-1)     # refused, never served, never charged
        raise


def get_llm(**kwargs: Any):
    """The raw chat model.

    ⚠️ This used to pass temperature=0 and claim determinism. Gemini 3.x Flash
    IGNORES temperature, top_p and top_k — silently, apart from a UserWarning —
    and Google has said later models will reject them with HTTP 400. So the
    parameter is gone, and the guarantee it implied never existed here:

    * repeated questions can return DIFFERENT wording, and occasionally a
      different verdict. `explain._reusable_attribution` is a quota cache, not a
      determinism shortcut.
    * the Gate 4 confabulation rate carries sampling noise. A small move between
      runs is not necessarily a real change, which is exactly what agent-plan.md
      3.4 wanted temperature=0 to rule out.

    Output shape is still pinned by constrained decoding (`get_attributor`), and
    the documented lever for steadier output is now `thinking_level` plus the
    response schema, not sampling parameters.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    settings = get_settings()
    settings.require("gemini_api_key")
    model = settings.attribution_model.split(":", 1)[-1]  # tolerate a 'google_genai:' prefix

    # langchain-google-genai reads GOOGLE_API_KEY from the environment, but our
    # key is loaded into Settings from .env and never exported. Pass it
    # explicitly or construction fails with a misleading "API key required".
    return ChatGoogleGenerativeAI(
        model=model,
        # No temperature/top_p/top_k: ignored by Gemini 3.x, HTTP 400 later.
        google_api_key=settings.gemini_api_key,
        # ⚠️ The client defaults to NO timeout and 6 retries. On 2026-09-14 the
        # pinned model stopped answering and a question hung indefinitely
        # instead of reporting "couldn't reach Gemini". Bounded: at most two
        # 60 s attempts, then llm_attribute records llm_error.
        timeout=60,
        max_retries=1,
        # Thinking/reasoning is left at the provider default and should be
        # minimised on this node: longer reasoning chains measurably degrade
        # abstention recall, and abstention is what this node exists to get
        # right. agent-plan.md 1.1 / 3.4.
        **kwargs,
    )


def get_attributor():
    """The model bound to the Attribution schema via constrained decoding.

    Nested schema verified working 2026-08-31 (see agent/schema.py); the flat
    fallback is not used. Constrained decoding pins the SHAPE of the answer; it
    does not make the content reproducible — see get_llm on sampling.
    """
    from swing.agent.schema import Attribution

    return get_llm().with_structured_output(Attribution)
