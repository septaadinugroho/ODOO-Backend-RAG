"""Authentication, transport enforcement, and rate limiting (NFR-C, NFR-D).

Three requirements, three small pieces:

* **API key per consumer.** "API key berbeda untuk setiap konsumen; Odoo dan
  frontend memakai key terpisah." The key identifies *who* is calling, which is
  what makes the rate limit and the logs meaningful — and what lets one consumer's
  key be rotated without taking the other down. Comparison is constant-time, and
  the key is never logged; a short fingerprint is logged instead.
* **HTTPS only.** "HTTPS wajib; HTTP ditolak."
* **60 requests per minute per API key**, 429 beyond that.

The dependency exposed here is `require_consumer`. It runs as a FastAPI
dependency rather than middleware for one practical reason: as a dependency it
appears in the OpenAPI document as a security scheme, so `/docs` grows an
Authorize button and the demo works from the browser. NFR-F asks for the OpenAPI
spec to be ready before integration, and a spec that omits the auth header is not.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional

from fastapi import Depends, Request, Security
from fastapi.security import APIKeyHeader

from app.config import Settings, get_settings
from app.api.errors import (
    AuthenticationError,
    ConfigurationError,
    InsecureTransportError,
    RateLimitError,
)

logger = logging.getLogger(__name__)

API_KEY_HEADER = "X-API-Key"

api_key_scheme = APIKeyHeader(
    name=API_KEY_HEADER,
    auto_error=False,  # we raise our own FR-4-shaped error, not {"detail": ...}
    description=(
        "Kunci API konsumen. Odoo dan frontend memakai kunci berbeda (NFR-D). "
        "Untuk demo lokal, nilai lalai ada dalam .env.example."
    ),
)

# Shipped in .env.example so a fresh clone runs. Refused in production, because a
# default credential in a public repo is the same as no credential.
DEV_KEY_MARKERS = ("change-me", "dev-key", "changeme", "placeholder")


@dataclass(frozen=True)
class Consumer:
    """The authenticated caller. `fingerprint` is safe to log; the key is not."""

    name: str
    fingerprint: str

    @property
    def is_anonymous(self) -> bool:
        return self.name == "anonymous"


ANONYMOUS = Consumer(name="anonymous", fingerprint="-")


def fingerprint(api_key: str) -> str:
    """First 8 hex of SHA-256. Enough to correlate log lines to one key.

    Truncated deliberately: a full hash of a low-entropy key is brute-forceable,
    and correlation is all the logs need.
    """
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:8]


def authenticate(api_key: Optional[str], settings: Optional[Settings] = None) -> Consumer:
    """Resolve a key to a consumer, or raise.

    The loop compares against every configured key with `compare_digest` instead
    of doing a dict lookup. A dict lookup leaks timing through hashing and early
    exit; the corpus of keys here is two or three entries, so the cost of doing it
    properly is nil.
    """
    settings = settings or get_settings()
    if not settings.require_api_key:
        return ANONYMOUS
    if not api_key:
        raise AuthenticationError(
            f"Missing {API_KEY_HEADER} header.", detail="no api key presented"
        )

    matched: Optional[str] = None
    for known_key, consumer in settings.api_key_map.items():
        if secrets.compare_digest(api_key, known_key):
            matched = consumer
    if matched is None:
        # No hint about which part was wrong, and the presented key is not logged.
        raise AuthenticationError(
            "Invalid API key.", detail=f"unknown key fp={fingerprint(api_key)}"
        )
    return Consumer(name=matched, fingerprint=fingerprint(api_key))


def assert_keys_are_production_ready(settings: Optional[Settings] = None) -> None:
    """Refuse to boot in production with sample keys or a single shared key.

    Called from the app lifespan. Failing at startup is the point: an
    authentication weakness that only shows up under audit has already been
    exploited for however long it was live.
    """
    settings = settings or get_settings()
    if settings.app_env.lower() not in ("production", "prod"):
        return
    if not settings.require_api_key:
        raise ConfigurationError(
            "REQUIRE_API_KEY=false is not allowed in production.",
            detail="auth disabled in production",
        )
    keys = list(settings.api_key_map.keys())
    weak = [k for k in keys if any(m in k.lower() for m in DEV_KEY_MARKERS)]
    if weak:
        raise ConfigurationError(
            "Sample API keys are still configured. Set API_KEYS to real secrets.",
            detail=f"{len(weak)} default key(s) present",
        )
    if len(set(settings.api_key_map.values())) < 2:
        logger.warning(
            "only one consumer is configured; NFR-D expects Odoo and the front end "
            "to hold separate keys",
            extra={"consumers": settings.consumers},
        )
    short = [k for k in keys if len(k) < 24]
    if short:
        raise ConfigurationError(
            "At least one API key is shorter than 24 characters.",
            detail=f"{len(short)} short key(s)",
        )
    if not settings.require_https:
        raise ConfigurationError(
            "REQUIRE_HTTPS=false is not allowed in production (NFR-D).",
            detail="https not enforced in production",
        )


# ─── Transport (NFR-D) ──────────────────────────────────────────────────────


def enforce_https(request: Request, settings: Optional[Settings] = None) -> None:
    """Reject plain HTTP when enforcement is on.

    Behind a TLS-terminating proxy the ASGI scheme is `http` even though the
    client spoke HTTPS, so `X-Forwarded-Proto` decides — but only when
    `TRUST_PROXY_HEADERS` is set, since any client can send that header. If the
    app is exposed directly, turn the trust off and the ASGI scheme is used.
    """
    settings = settings or get_settings()
    if not settings.require_https:
        return

    scheme = request.url.scheme
    if settings.trust_proxy_headers:
        forwarded = request.headers.get("x-forwarded-proto")
        if forwarded:
            scheme = forwarded.split(",")[0].strip().lower()

    if scheme != "https":
        raise InsecureTransportError(
            "This service accepts HTTPS only. Retry over https://.",
            detail=f"scheme={scheme}",
        )


# ─── Rate limit (NFR-C) ─────────────────────────────────────────────────────


class RateLimiter:
    """Sliding-window counter, 60 requests per minute per key by default.

    Sliding rather than fixed-window: a fixed window lets a caller send 60 at
    59.9s and 60 more at 60.1s, which is 120 in a fifth of a second and defeats
    the point. One deque of timestamps per consumer, trimmed on read — bounded by
    the limit itself, so memory is a few hundred floats.
    """

    def __init__(self, limit: Optional[int] = None, window_seconds: float = 60.0) -> None:
        settings = get_settings()
        self.limit = settings.rate_limit_per_minute if limit is None else limit
        self.window = window_seconds
        self._hits: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        """Record a hit, or raise `RateLimitError` with a `Retry-After` header."""
        if self.limit <= 0:  # 0 disables the limit; used by the eval CLI
            return
        now = time.monotonic()
        with self._lock:
            window = self._hits.setdefault(key, deque())
            cutoff = now - self.window
            while window and window[0] < cutoff:
                window.popleft()
            if len(window) >= self.limit:
                retry_after = max(1, int(self.window - (now - window[0])) + 1)
                raise RateLimitError(
                    f"Rate limit of {self.limit} requests per minute exceeded. "
                    f"Retry in {retry_after}s.",
                    detail=f"key={key} hits={len(window)}",
                    headers={
                        "Retry-After": str(retry_after),
                        "X-RateLimit-Limit": str(self.limit),
                        "X-RateLimit-Remaining": "0",
                    },
                )
            window.append(now)

    def remaining(self, key: str) -> int:
        now = time.monotonic()
        with self._lock:
            window = self._hits.get(key, deque())
            cutoff = now - self.window
            live = sum(1 for t in window if t >= cutoff)
            return max(0, self.limit - live)

    def reset(self, key: Optional[str] = None) -> None:
        with self._lock:
            if key is None:
                self._hits.clear()
            else:
                self._hits.pop(key, None)


_limiter: Optional[RateLimiter] = None
_limiter_lock = threading.Lock()


def get_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        with _limiter_lock:
            if _limiter is None:
                _limiter = RateLimiter()
    return _limiter


def reset_limiter() -> None:
    global _limiter
    with _limiter_lock:
        _limiter = None


# ─── The dependency ─────────────────────────────────────────────────────────


def require_consumer(
    request: Request,
    api_key: Optional[str] = Security(api_key_scheme),
) -> Consumer:
    """HTTPS → authenticate → rate limit, in that order.

    Order matters. An unauthenticated caller must not be able to consume another
    consumer's rate budget, so the limit is keyed by the *authenticated* identity
    and applied last. And HTTPS is checked first because a credential sent over
    plain HTTP is already compromised — rejecting the request without inspecting
    it is the least bad response.
    """
    settings = get_settings()
    enforce_https(request, settings)
    consumer = authenticate(api_key, settings)
    limiter = get_limiter()
    limiter.check(consumer.fingerprint if not consumer.is_anonymous else "anonymous")

    request.state.consumer = consumer.name
    request.state.consumer_fp = consumer.fingerprint
    return consumer


ConsumerDep = Depends(require_consumer)


__all__ = [
    "ANONYMOUS",
    "API_KEY_HEADER",
    "Consumer",
    "ConsumerDep",
    "RateLimiter",
    "api_key_scheme",
    "assert_keys_are_production_ready",
    "authenticate",
    "enforce_https",
    "fingerprint",
    "get_limiter",
    "require_consumer",
    "reset_limiter",
]
