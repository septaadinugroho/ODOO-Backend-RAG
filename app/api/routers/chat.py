"""`POST /v1/chat` — the only endpoint that answers questions.

Thin on purpose. The router's job is HTTP: authenticate, hand the validated body
to `app.core.service`, translate infrastructure failures into FR-4-shaped errors, and
publish the few facts the middleware needs for logging and metrics. Every decision
about *what to say* lives in the service and the guardrails.

The path carries the version (`/v1/chat`) per NFR-F, and `response_model` is bound
to `ChatResponse`, so FastAPI validates our own output before it leaves. That is
what makes FR-16 and FR-17 structural rather than aspirational: `fee_rm` and
`schedule` are typed `None`, so a number in either field fails serialisation here,
after every guardrail, as the last line of defence.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Body, Depends, Request, status
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.infrastructure.db import get_db
from app.api.errors import DatabaseUnavailableError, LLMTimeoutError
from app.core.schemas import ChatRequest, ChatResponse, ErrorResponse
from app.api.security import Consumer, require_consumer
from app.core.service import handle_chat

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["chat"])

ERROR_RESPONSES = {
    status.HTTP_401_UNAUTHORIZED: {
        "model": ErrorResponse,
        "description": "Kunci API tiada atau tidak sah (NFR-D).",
    },
    status.HTTP_422_UNPROCESSABLE_ENTITY: {
        "model": ErrorResponse,
        "description": "Badan permintaan tidak menepati skema.",
    },
    status.HTTP_429_TOO_MANY_REQUESTS: {
        "model": ErrorResponse,
        "description": "Melebihi 60 permintaan seminit bagi satu kunci (NFR-C).",
    },
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "model": ErrorResponse,
        "description": (
            "Pangkalan data tidak dapat dihubungi, atau — apabila "
            "LLM_TIMEOUT_RETURNS_503=true — model tamat masa (FR-4)."
        ),
    },
}


@router.post(
    "/chat",
    response_model=ChatResponse,
    response_model_exclude_none=False,
    status_code=status.HTTP_200_OK,
    summary="Cari kelas tuisyen daripada satu mesej bahasa asli",
    responses=ERROR_RESPONSES,
    description=(
        "Menerima satu mesej dan mengembalikan salah satu daripada enam state "
        "(FR-2). `results[]` sentiasa datang terus daripada pangkalan data (FR-5), "
        "`fee_rm` dan `schedule` sentiasa null (FR-16, FR-17), dan `trace` sentiasa "
        "memuat `retrieved_ids[]` serta `filters_applied` (FR-29).\n\n"
        "Hantar `session_id` daripada respons sebelumnya untuk meneruskan "
        "perbualan; konteks tamat selepas 30 minit tanpa aktiviti (FR-28)."
    ),
)
def chat(
    request: Request,
    payload: ChatRequest = Body(...),
    db: Session = Depends(get_db),
    consumer: Consumer = Depends(require_consumer),
) -> ChatResponse:
    trace_id = getattr(request.state, "trace_id", "t_unknown")
    settings = get_settings()

    try:
        turn = handle_chat(payload, db, trace_id=trace_id, consumer=consumer.name)
    except OperationalError as exc:
        # Postgres unreachable, credentials wrong, or the pool timed out. 503 with
        # a code, never a stack trace or an HTML page (FR-4).
        raise DatabaseUnavailableError(detail=str(exc)[:500]) from exc
    except SQLAlchemyError as exc:
        logger.exception(
            "database error while answering", extra={"trace_id": trace_id}
        )
        raise DatabaseUnavailableError(
            "The class database returned an error.", detail=str(exc)[:500]
        ) from exc

    response = turn.response

    # Published for the middleware: the log line and the metrics both need the
    # state and the cost, and neither can see the response body.
    request.state.session_id = response.session_id
    request.state.response_state = response.state.value
    request.state.degraded = response.degraded
    request.state.llm_tokens = turn.llm_tokens
    request.state.llm_cost_rm = turn.llm_cost_rm
    if turn.validation_failed:
        request.state.error_code = turn.validation_failed[0]

    if (
        settings.llm_timeout_returns_503
        and "llm_timeout" in turn.validation_failed
    ):
        # Literal FR-4. Off by default — see the comment on the setting: NFR-C
        # would rather send the correct class list with degraded=true than discard
        # it because the paragraph is missing.
        raise LLMTimeoutError(detail="llm timeout surfaced as 503 by configuration")

    return response
