"""FastAPI application factory.

Assembles the app and nothing else: middleware order, exception handlers, routers,
and the startup checks that turn a misconfiguration into a loud failure instead of
a subtle one.

**Middleware order matters here.** Starlette runs the last-added middleware
outermost, so the list below is written bottom-up. The outermost layer is the trace
context, because every other layer's log line needs the trace id — including the
403 from a body-size rejection. The metrics layer is innermost, so it measures the
handler rather than the middleware around it.

**Startup fails loudly.** Production with sample API keys, or with HTTPS
enforcement off, raises before the first request (NFR-D). An unreachable database
or a cold LLM only warns: NFR-C says filter search must keep working without the
model, and a service that refuses to boot because Vertex is unreachable has turned
a degraded mode into an outage.

Run it with:

    uvicorn app.main:app --reload --port 8000

Then open http://127.0.0.1:8000/docs, press Authorize, paste the key from `.env`,
and try `POST /v1/chat`.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import SCHEMA_VERSION, get_settings
from app.infrastructure.db import ping, session_scope
from app.api.errors import (
    ServiceError,
    http_exception_handler,
    service_error_handler,
    unhandled_error_handler,
    validation_error_handler,
)
from app.infrastructure.llm import get_provider
from app.api.middleware import (
    BodyLimitMiddleware,
    MetricsMiddleware,
    SecurityHeadersMiddleware,
    TraceContextMiddleware,
)
from app.infrastructure.observability import configure_logging
from app.core.retrieval import corpus_size
from app.api.routers import chat_router, health_router
from app.api.security import API_KEY_HEADER, assert_keys_are_production_ready
from app.core.sessions import get_store

logger = logging.getLogger(__name__)

DESCRIPTION = """\
**Class Bot** — carian kelas tuisyen Malaysia daripada korpus 200 rekod yang telah
disemak. Satu endpoint, `POST /v1/chat`, dan enam state respons yang tertutup.

Tiga perkara yang perlu diketahui sebelum integrasi:

* **Perkhidmatan ini tidak menyimpan harga atau jadual.** `fee_rm` dan `schedule`
  sentiasa `null`, dan model dilarang menyebut sebarang jumlah atau waktu. Soalan
  tentang kedua-duanya dijawab dengan `state: unavailable_field` beserta pautan
  penyedia (FR-16, FR-17, FR-20).
* **Tiada penarafan.** Korpus tiada data kualiti, jadi perkhidmatan ini tidak
  menyusun kedudukan penyedia dan tidak menggunakan kata seperti "terbaik" (FR-22).
* **Setiap respons boleh dijejak.** `trace.retrieved_ids[]` dan
  `trace.filters_applied` sentiasa hadir (FR-29). Jika naratif gagal pengesahan,
  `state` menjadi `degraded`, ayat naratif diganti templat selamat, tetapi
  `results[]` kekal lengkap (FR-18, FR-19).

Pengesahan: hantar kunci dalam header `X-API-Key`. Odoo dan frontend memakai kunci
berasingan (NFR-D).
"""

TAGS_METADATA = [
    {"name": "chat", "description": "Carian kelas dan perbualan berbilang giliran."},
    {"name": "ops", "description": "Kesediaan servis dan metrik operasi."},
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging()

    # Raises on a production misconfiguration. Deliberately before anything else:
    # a service with sample credentials should never reach the accept() call.
    assert_keys_are_production_ready(settings)

    logger.info(
        "starting Class Bot",
        extra={
            "env": settings.app_env,
            "schema_version": SCHEMA_VERSION,
            "llm_provider": settings.llm_provider,
            "llm_model": settings.vertex_model,
            "require_api_key": settings.require_api_key,
            "require_https": settings.require_https,
            "consumers": settings.consumers,
            "rate_limit_per_minute": settings.rate_limit_per_minute,
            "session_ttl_seconds": settings.session_ttl_seconds,
            # Never the full URL: it carries the password.
            "database_host": settings.database_url.rsplit("@", 1)[-1],
        },
    )

    if ping():
        try:
            with session_scope() as db:
                records = corpus_size(db)
            if records == 0:
                logger.warning(
                    "database is reachable but the corpus is empty; every query "
                    "will return no_result. Run scripts/init_db.py or "
                    "`docker compose up` so db/init/*.sql is applied."
                )
            else:
                logger.info("corpus loaded", extra={"corpus_records": records})
        except Exception as exc:  # pragma: no cover - schema not yet created
            logger.warning("could not count corpus records: %s", exc)
    else:
        # A warning, not a failure: the app should be inspectable at /docs and
        # able to report its own unhealthiness at /health.
        logger.error(
            "database is not reachable at startup; /v1/chat will return 503 until "
            "it is. Check DATABASE_URL and that Postgres is listening.",
            extra={"database_host": settings.database_url.rsplit("@", 1)[-1]},
        )

    try:
        if get_provider().ready():
            logger.info("llm provider ready", extra={"provider": settings.llm_provider})
        else:
            logger.warning(
                "llm provider is not ready; answers will be template-only with "
                "degraded=true (NFR-C). For Vertex, set GOOGLE_CLOUD_PROJECT and "
                "run `gcloud auth application-default login`.",
                extra={"provider": settings.llm_provider},
            )
    except Exception as exc:  # pragma: no cover
        logger.warning("llm provider check failed: %s", exc)

    try:
        yield
    finally:
        get_store().clear()
        logger.info("shutting down Class Bot")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Class Bot API",
        description=DESCRIPTION,
        version="1.0.0",
        openapi_tags=TAGS_METADATA,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        swagger_ui_parameters={"displayRequestDuration": True, "tryItOutEnabled": True},
    )

    # ── Handlers: everything the client sees has FR-4's shape ────────────────
    app.add_exception_handler(ServiceError, service_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)

    # ── Middleware (added inner-first; see the module docstring) ─────────────
    app.add_middleware(MetricsMiddleware)
    origins = settings.cors_origin_list
    if origins:
        if "*" in origins and settings.require_api_key:
            # A wildcard origin with a credentialed header means any page on the
            # internet can spend this key from a visitor's browser.
            raise RuntimeError(
                "CORS_ORIGINS='*' is not allowed while REQUIRE_API_KEY is true; "
                "list the front-end origins explicitly."
            )
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=[API_KEY_HEADER, "Content-Type", "X-Trace-Id"],
            expose_headers=["X-Trace-Id", "X-Response-Time-Ms"],
            max_age=600,
        )
    app.add_middleware(BodyLimitMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(TraceContextMiddleware)  # outermost

    # ── Routes ──────────────────────────────────────────────────────────────
    app.include_router(chat_router)
    app.include_router(health_router)

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    return app


app = create_app()
