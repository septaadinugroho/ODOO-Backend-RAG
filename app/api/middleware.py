"""ASGI middleware: trace context, the per-request log line, and hardening.

Split from `security.py` because these run for *every* request including `/health`
and `/docs`, whereas authentication is a dependency on the endpoints that need it.

The one request log line lives here rather than in the router, so it is emitted
even when the request never reaches the router — a 401, a 429, a 422, or an
unhandled crash. Those are exactly the requests you want in the log, and a router
that logs its own successes would miss all of them.
"""

from __future__ import annotations

import logging
import time
from typing import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.config import get_settings
from app.api.errors import PayloadTooLargeError, error_body
from app.infrastructure.observability import TRACE_HEADERS, get_metrics, resolve_trace_id

logger = logging.getLogger("app.request")

# A `/v1/chat` body is a message of at most 2,000 characters plus a session id.
# 32 KB is generous for that and small enough that a hostile body is rejected on
# the header rather than buffered.
MAX_REQUEST_BYTES = 32_768

# Paths that are noise in the request log: liveness probes fire every few seconds
# and would bury real traffic. Errors on them are still logged.
_QUIET_PATHS = frozenset({"/health", "/metrics", "/favicon.ico"})


class TraceContextMiddleware(BaseHTTPMiddleware):
    """Trace id in, timing and one structured log line out (NFR-G).

    The trace id is taken from the client's header when present so a request can
    be followed from Odoo through this service, and echoed back on the response
    so the caller can log it too.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        incoming = None
        for header in TRACE_HEADERS:
            incoming = request.headers.get(header)
            if incoming:
                break
        trace_id = resolve_trace_id(incoming)
        request.state.trace_id = trace_id
        request.state.session_id = None
        request.state.response_state = None
        request.state.degraded = False

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # Reaching here means an exception escaped the exception-handler
            # middleware (rare — usually a bug in a handler itself). Log it with
            # the trace id and re-raise; the server turns it into a 500.
            latency_ms = int((time.perf_counter() - started) * 1000)
            logger.exception(
                "request crashed",
                extra={
                    "trace_id": trace_id,
                    "session_id": getattr(request.state, "session_id", None),
                    "state": None,
                    "latency_ms": latency_ms,
                    "method": request.method,
                    "path": request.url.path,
                },
            )
            raise

        latency_ms = int((time.perf_counter() - started) * 1000)
        response.headers["X-Trace-Id"] = trace_id
        response.headers["X-Response-Time-Ms"] = str(latency_ms)

        path = request.url.path
        if path in _QUIET_PATHS and response.status_code < 400:
            return response

        level = logging.INFO if response.status_code < 500 else logging.ERROR
        logger.log(
            level,
            "%s %s -> %s",
            request.method,
            path,
            response.status_code,
            extra={
                "trace_id": trace_id,
                # Random and personal-data-free by construction (FR-26, NFR-D),
                # which is what makes it loggable at all.
                "session_id": getattr(request.state, "session_id", None),
                "state": getattr(request.state, "response_state", None),
                "latency_ms": latency_ms,
                "method": request.method,
                "path": path,
                "status_code": response.status_code,
                "consumer": getattr(request.state, "consumer", None),
                "consumer_fp": getattr(request.state, "consumer_fp", None),
                "degraded": getattr(request.state, "degraded", False),
            },
        )
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Feed `/metrics` and evaluate NFR-G's alert thresholds.

    Only `/v1/chat` is counted. The degraded ratio and error rate in NFR-G are
    about answering behaviour; folding health probes into the denominator would
    make a 3% degraded rate look like 0.1% simply because Kubernetes is polling.
    """

    COUNTED_SUFFIX = "/chat"

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        started = time.perf_counter()
        response = await call_next(request)
        if not request.url.path.endswith(self.COUNTED_SUFFIX):
            return response

        latency_ms = int((time.perf_counter() - started) * 1000)
        metrics = get_metrics()
        state = getattr(request.state, "response_state", None) or f"http_{response.status_code}"
        metrics.observe(
            state=state,
            latency_ms=latency_ms,
            degraded=bool(getattr(request.state, "degraded", False)),
            error=response.status_code >= 500,
            tokens=int(getattr(request.state, "llm_tokens", 0) or 0),
            cost_rm=float(getattr(request.state, "llm_cost_rm", 0.0) or 0.0),
            error_code=getattr(request.state, "error_code", None),
        )
        metrics.emit_alerts()
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Response hardening.

    `no-store` because a response embeds the caller's search; a shared cache
    holding it would leak one user's query to another. HSTS is only sent when the
    request actually arrived over TLS — sending it over plain HTTP is meaningless
    and, on a dev machine, pins localhost to https in the developer's browser for
    a year.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        headers = response.headers
        headers.setdefault("X-Content-Type-Options", "nosniff")
        headers.setdefault("Referrer-Policy", "no-referrer")
        headers.setdefault("X-Frame-Options", "DENY")
        if request.url.path.startswith("/v1"):
            headers.setdefault("Cache-Control", "no-store")

        settings = get_settings()
        forwarded = (request.headers.get("x-forwarded-proto") or "").split(",")[0].strip()
        over_tls = request.url.scheme == "https" or (
            settings.trust_proxy_headers and forwarded == "https"
        )
        if settings.require_https and over_tls:
            headers.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
            )
        return response


class BodyLimitMiddleware(BaseHTTPMiddleware):
    """Reject oversized bodies on the header, before buffering.

    Pydantic already caps `message` at 2,000 characters, but that check happens
    after the body has been read and parsed. This one is about not reading a
    100 MB body in the first place.

    A chunked request without `Content-Length` slips past this check; Pydantic's
    limits still bound what is accepted, and a proxy in front of the app should
    carry a body limit of its own.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > MAX_REQUEST_BYTES:
            exc = PayloadTooLargeError(
                f"Request body exceeds {MAX_REQUEST_BYTES} bytes."
            )
            logger.warning(
                "request body too large",
                extra={
                    "trace_id": getattr(request.state, "trace_id", None),
                    "content_length": int(declared),
                    "path": request.url.path,
                },
            )
            from starlette.responses import JSONResponse

            return JSONResponse(
                status_code=exc.status_code,
                content=error_body(
                    exc.code, exc.message, getattr(request.state, "trace_id", None)
                ),
            )
        return await call_next(request)


__all__ = [
    "BodyLimitMiddleware",
    "MAX_REQUEST_BYTES",
    "MetricsMiddleware",
    "SecurityHeadersMiddleware",
    "TraceContextMiddleware",
]
