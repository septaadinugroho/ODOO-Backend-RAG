"""Every failure the client sees, in one shape (FR-4).

FR-4's acceptance criterion is about what a client must *never* receive: "bukan
HTML, body kosong, atau format yang tidak terduga". FastAPI's defaults already
violate that in two places — `HTTPException` renders `{"detail": ...}` and an
unhandled exception renders `Internal Server Error` as plain text — so this module
defines one exception family and the handlers that turn all of it, including
crashes and validation failures, into `ErrorResponse`.

Two decisions worth stating.

Timeouts are 503, not 500. NFR-A puts an 8-second deadline on the LLM and FR-4
names 503 explicitly; the distinction matters because 503 tells a client "retry
later" while 500 tells it "this request will never work".

Error messages never quote the user's input back, and never include a stack trace
or a database error string. Those leak schema and file paths to whoever is probing
the endpoint. The detail goes to the log, keyed by `trace_id`, and the client gets
the id — which is also what makes a support conversation possible without the user
having to describe symptoms.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import SCHEMA_VERSION
from app.core.schemas import ErrorDetail, ErrorResponse

logger = logging.getLogger(__name__)


class ServiceError(Exception):
    """A failure with a client-facing code and an HTTP status.

    `detail` is for the log only — it may contain exception text, so it must not
    reach the response body.
    """

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "internal_error"
    message: str = "The service could not complete this request."

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        code: Optional[str] = None,
        status_code: Optional[int] = None,
        detail: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.message = message or self.message
        self.code = code or self.code
        self.status_code = status_code or self.status_code
        self.detail = detail
        self.headers = headers or {}
        super().__init__(self.message)


class AuthenticationError(ServiceError):
    status_code = status.HTTP_401_UNAUTHORIZED
    code = "unauthorized"
    message = "A valid X-API-Key header is required."


class InsecureTransportError(ServiceError):
    # 400, not 403: the request is malformed for this service's contract rather
    # than forbidden to this caller. NFR-D: "HTTPS wajib; HTTP ditolak".
    status_code = status.HTTP_400_BAD_REQUEST
    code = "https_required"
    message = "This service accepts HTTPS only."


class RateLimitError(ServiceError):
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    code = "rate_limited"
    message = "Rate limit exceeded. Try again shortly."


class PayloadTooLargeError(ServiceError):
    status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    code = "payload_too_large"
    message = "The request body is larger than this endpoint accepts."


class LLMTimeoutError(ServiceError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "llm_timeout"
    message = "The model provider did not respond within the time limit."


class DatabaseUnavailableError(ServiceError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "database_unavailable"
    message = "The class database is not reachable right now."


class ConfigurationError(ServiceError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    code = "misconfigured"
    message = "The service is not configured correctly."


def error_body(code: str, message: str, trace_id: Optional[str] = None) -> Dict[str, Any]:
    return ErrorResponse(
        schema_version=SCHEMA_VERSION,
        error=ErrorDetail(code=code, message=message, trace_id=trace_id),
    ).model_dump(mode="json")


def error_response(
    code: str,
    message: str,
    status_code: int,
    trace_id: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=error_body(code, message, trace_id),
        headers=headers or None,
    )


def _trace_id(request: Request) -> Optional[str]:
    return getattr(request.state, "trace_id", None)


async def service_error_handler(request: Request, exc: ServiceError) -> JSONResponse:
    trace_id = _trace_id(request)
    log = logger.warning if exc.status_code < 500 else logger.error
    log(
        "request failed: %s",
        exc.code,
        extra={
            "trace_id": trace_id,
            "error_code": exc.code,
            "status_code": exc.status_code,
            "detail": exc.detail,
            "path": request.url.path,
        },
    )
    return error_response(
        exc.code, exc.message, exc.status_code, trace_id, exc.headers
    )


async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """422 in FR-4's shape.

    The field names and error types are safe to return — they describe the
    caller's own request and are what makes a 422 fixable — but the submitted
    values are not echoed. A rejected body can contain anything.
    """
    problems = []
    for err in exc.errors()[:10]:
        location = ".".join(str(p) for p in err.get("loc", ()) if p != "body")
        problems.append(f"{location or 'body'}: {err.get('msg', 'invalid')}")
    message = "Request body is invalid. " + "; ".join(problems)
    trace_id = _trace_id(request)
    logger.info(
        "request rejected by schema validation",
        extra={"trace_id": trace_id, "problems": problems, "path": request.url.path},
    )
    return error_response(
        "validation_error",
        message[:1_000],
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        trace_id,
    )


async def http_exception_handler(request: Request, exc: Any) -> JSONResponse:
    """Re-shape `HTTPException`, including 404 and 405 raised by the router."""
    status_code = getattr(exc, "status_code", 500)
    detail = getattr(exc, "detail", None)
    codes = {
        404: "not_found",
        405: "method_not_allowed",
        401: "unauthorized",
        403: "forbidden",
        413: "payload_too_large",
        429: "rate_limited",
    }
    code = codes.get(status_code, "http_error")
    message = detail if isinstance(detail, str) and detail else "Request could not be served."
    return error_response(
        code, message, status_code, _trace_id(request), getattr(exc, "headers", None)
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Last resort. The client gets a trace id; the log gets the traceback."""
    trace_id = _trace_id(request)
    logger.exception(
        "unhandled exception",
        extra={
            "trace_id": trace_id,
            "error_code": "internal_error",
            "path": request.url.path,
        },
    )
    return error_response(
        "internal_error",
        "The service hit an unexpected error. Quote the trace_id when reporting it.",
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        trace_id,
    )


__all__ = [
    "AuthenticationError",
    "ConfigurationError",
    "DatabaseUnavailableError",
    "InsecureTransportError",
    "LLMTimeoutError",
    "PayloadTooLargeError",
    "RateLimitError",
    "ServiceError",
    "error_body",
    "error_response",
    "http_exception_handler",
    "service_error_handler",
    "unhandled_error_handler",
    "validation_error_handler",
]
