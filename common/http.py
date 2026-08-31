"""HTTP clients with per-provider rate limiting and retry.

Every free tier here throttles differently, and a silent failure that leaves a
gap in the record is unrecoverable for news. Each provider gets its own client
with its own minimum inter-request interval.

SEC: 10 req/s per IP; exceeding it earns a temporary block, not just a 429.
     data-sources.md D.2 rule 2. We use a 120ms floor for headroom.
"""
from __future__ import annotations

import threading
import time

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from common.settings import get_settings

SEC_MIN_INTERVAL = 0.12  # seconds between SEC requests (limit is 10/s)
RSS_MIN_INTERVAL = 0.25


class _RateLimiter:
    """Process-wide minimum interval between calls. Thread-safe."""

    def __init__(self, min_interval: float) -> None:
        self._min = min_interval
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            delta = time.monotonic() - self._last
            if delta < self._min:
                time.sleep(self._min - delta)
            self._last = time.monotonic()


_sec_limiter = _RateLimiter(SEC_MIN_INTERVAL)
_rss_limiter = _RateLimiter(RSS_MIN_INTERVAL)

RETRYABLE = (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError)


def sec_headers() -> dict[str, str]:
    return {
        "User-Agent": get_settings().sec_user_agent,
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
    }


def _raise_for_retryable(resp: httpx.Response) -> httpx.Response:
    # 429 and 5xx are worth retrying; 403/404 are not (they mean a bad header
    # or a bad CIK, and retrying a 403 extends an SEC block).
    if resp.status_code == 429 or resp.status_code >= 500:
        resp.raise_for_status()
    return resp


@retry(
    retry=retry_if_exception_type(RETRYABLE),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def sec_get(url: str, timeout: float = 30.0) -> httpx.Response:
    """GET against sec.gov / data.sec.gov with the mandatory UA and rate limit."""
    _sec_limiter.wait()
    headers = sec_headers()
    # The Host header must match the actual host, and www.sec.gov != data.sec.gov.
    headers["Host"] = httpx.URL(url).host
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.get(url, headers=headers)
    return _raise_for_retryable(resp)


@retry(
    retry=retry_if_exception_type(RETRYABLE),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def feed_get(url: str, timeout: float = 20.0) -> httpx.Response:
    """GET an RSS/Atom feed. Some publishers 403 a default python UA."""
    _rss_limiter.wait()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) SwingAgent/0.1 research"
        ),
        "Accept": "application/rss+xml, application/atom+xml, application/xml;q=0.9, */*;q=0.8",
    }
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.get(url, headers=headers)
    return _raise_for_retryable(resp)
