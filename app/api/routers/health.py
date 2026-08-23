"""`/health` and `/metrics` — the two endpoints an operator needs (NFR-G).

`/health` is unauthenticated so a load balancer or `docker healthcheck` can use it
without holding a key, and it reports only facts that are already public to anyone
who can reach the port: whether the corpus loaded, which provider is configured,
whether it is ready. No key material, no connection string, no counts of anything
sensitive.

`/metrics` does require a key. The numbers are not secret, but traffic volume and
error rates are the kind of thing you would rather not hand to an unauthenticated
stranger, and the scraper is under the same operator's control as the API.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import SCHEMA_VERSION, get_settings
from app.infrastructure.db import get_db
from app.infrastructure.llm import get_provider
from app.infrastructure.observability import get_metrics
from app.core.retrieval import corpus_size
from app.core.schemas import HealthResponse
from app.api.security import Consumer, require_consumer
from app.core.sessions import get_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ops"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Kesediaan servis, korpus, dan penyedia LLM",
)
def health(db: Session = Depends(get_db)) -> HealthResponse:
    """Reports `degraded` rather than failing when a dependency is down.

    A health endpoint that 500s the moment the LLM is unreachable would make a
    load balancer pull a service that, per NFR-C, is still perfectly able to answer
    with filters. Only an unreachable *database* means this service cannot do its
    job, so that is the one condition reported as unhealthy.
    """
    settings = get_settings()

    try:
        records = corpus_size(db)
        db_ok = True
    except SQLAlchemyError as exc:
        logger.error("health: database unreachable", extra={"detail": str(exc)[:300]})
        records, db_ok = 0, False

    provider = get_provider()
    try:
        llm_ready = bool(provider.ready())
    except Exception as exc:  # pragma: no cover - provider decides
        logger.warning("health: provider readiness check failed: %s", exc)
        llm_ready = False

    if not db_ok:
        status = "unhealthy"
    elif records == 0:
        # Reachable but empty: the seed never ran. Every query would return
        # no_result, which looks like a data problem and is a deployment one.
        status = "degraded"
    elif not llm_ready:
        status = "degraded"
    else:
        status = "ok"

    return HealthResponse(
        status=status,
        schema_version=SCHEMA_VERSION,
        corpus_records=records,
        llm_provider=settings.llm_provider,
        llm_ready=llm_ready,
        active_sessions=get_store().count(),
    )


@router.get(
    "/metrics",
    summary="Latency, request per state, rasio degraded, error rate, kos/token",
    description=(
        "Metrik dalam tetingkap satu jam (NFR-G). Nilainya per-proses dan hilang "
        "apabila proses dimulakan semula — retensi 90 hari adalah tanggungjawab "
        "sistem yang mengumpul metrik ini."
    ),
)
def metrics(consumer: Consumer = Depends(require_consumer)) -> Dict[str, Any]:
    snapshot = get_metrics().snapshot()
    settings = get_settings()
    snapshot["service"] = {
        "schema_version": SCHEMA_VERSION,
        "env": settings.app_env,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.vertex_model,
        "active_sessions": get_store().count(),
    }
    return snapshot
