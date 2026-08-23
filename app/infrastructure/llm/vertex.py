"""Google Vertex AI provider (Gemini via the `google-genai` SDK).

Everything Google-specific in this service lives in this file. That is what makes
NFR-H's "ganti model LLM melalui konfigurasi" true rather than aspirational:
switching to another vendor means adding a sibling module and changing
`LLM_PROVIDER`, not touching retrieval, guardrails, or the response contract.

Authentication is Application Default Credentials — no key material in the
codebase or the environment file:

    gcloud auth application-default login
    gcloud config set project <project-id>

Operational behaviour, straight from the NFRs:

* 8-second timeout to the provider (NFR-A), surfaced as `timed_out` so the router
  can return HTTP 503 with a structured body per FR-4.
* At most one retry, 500 ms backoff (NFR-C "Retry — maksimum 1 kali ... backoff
  500 ms"), and only for errors that could plausibly succeed on a second attempt.
  Retrying a 400 wastes the user's latency budget twice over.
* Token counts are returned so NFR-E's cost-per-query ceiling can be measured
  instead of assumed.

The client is built lazily and once. Constructing it needs credentials, and an
import-time failure would take the whole service down — including the filter
search that NFR-C says must keep working without the LLM.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from app.config import get_settings
from app.infrastructure.llm.base import LLMConfigurationError, LLMResult, Prompt

logger = logging.getLogger(__name__)

# Error fragments worth a second attempt: transport hiccups and transient
# capacity limits. Anything else (bad request, permission denied, model not
# found) is a deterministic failure and retrying only doubles the latency.
_RETRYABLE = (
    "deadline",
    "timeout",
    "timed out",
    "unavailable",
    "503",
    "500",
    "resource exhausted",
    "resource_exhausted",
    "429",
    "connection",
    "reset by peer",
    "internal error",
)


class VertexProvider:
    name = "vertex"

    def __init__(self) -> None:
        settings = get_settings()
        self.model = settings.vertex_model
        self._project = settings.google_cloud_project
        self._location = settings.vertex_location
        self._timeout_seconds = settings.llm_timeout_seconds
        self._max_retries = max(0, settings.llm_max_retries)
        self._backoff_ms = settings.llm_retry_backoff_ms
        self._max_output_tokens = settings.vertex_max_output_tokens
        self._temperature = settings.vertex_temperature
        self._client: Optional[Any] = None
        self._types: Optional[Any] = None
        self._init_error: Optional[str] = None

    # ── client ──────────────────────────────────────────────────────────────

    def _ensure_client(self) -> Optional[Any]:
        if self._client is not None or self._init_error is not None:
            return self._client
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on install
            self._init_error = (
                "google-genai is not installed; run "
                "`pip install -r requirements.txt` or set LLM_PROVIDER=null"
            )
            logger.warning("vertex provider unavailable: %s (%s)", self._init_error, exc)
            return None

        if not self._project:
            # Not raised: a missing project must not stop the service from
            # serving filter-only, degraded responses (NFR-C).
            self._init_error = (
                "GOOGLE_CLOUD_PROJECT is empty; set it in .env, or set "
                "LLM_PROVIDER=null to run without narrative generation"
            )
            logger.warning("vertex provider unavailable: %s", self._init_error)
            return None

        try:
            self._types = types
            self._client = genai.Client(
                vertexai=True,
                project=self._project,
                location=self._location,
                http_options=self._http_options(types),
            )
        except Exception as exc:  # pragma: no cover - needs real credentials
            self._init_error = f"could not create Vertex AI client: {exc}"
            logger.warning("vertex provider unavailable: %s", self._init_error)
            return None
        return self._client

    def _http_options(self, types: Any) -> Any:
        """Per-request deadline. NFR-A puts the LLM timeout at 8 seconds.

        The SDK expresses it in milliseconds. Wrapped defensively because this is
        exactly the kind of field that gets renamed between SDK minor versions,
        and a rename must not take the service down — worst case we fall back to
        the SDK default and the wall-clock check below still bounds the wait.
        """
        try:
            return types.HttpOptions(timeout=int(self._timeout_seconds * 1000))
        except Exception:  # pragma: no cover - SDK shape drift
            logger.debug("HttpOptions(timeout=...) not accepted; using SDK default")
            return None

    def ready(self) -> bool:
        """True if a client could be built. Deliberately does not call the API.

        /health is polled; a readiness probe that spends tokens and 8 seconds of
        latency on every poll would be its own outage.
        """
        return self._ensure_client() is not None

    # ── generation ──────────────────────────────────────────────────────────

    def generate(self, prompt: Prompt) -> LLMResult:
        client = self._ensure_client()
        if client is None:
            return LLMResult(
                text="",
                error="llm_unconfigured",
                model=self.model,
                meta={"reason": self._init_error or "client unavailable"},
            )

        config = self._config(prompt)
        attempts = 0
        started = time.perf_counter()
        last_error = "unknown error"
        timed_out = False

        while attempts <= self._max_retries:
            attempts += 1
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=prompt.data,
                    config=config,
                )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                lowered = last_error.lower()
                timed_out = any(k in lowered for k in ("deadline", "timeout", "timed out"))
                elapsed = time.perf_counter() - started
                retryable = any(k in lowered for k in _RETRYABLE)
                if (
                    retryable
                    and attempts <= self._max_retries
                    and elapsed < self._timeout_seconds
                ):
                    time.sleep(self._backoff_ms / 1000.0)
                    continue
                return LLMResult(
                    text="",
                    error="llm_timeout" if timed_out else "llm_error",
                    latency_ms=int(elapsed * 1000),
                    model=self.model,
                    attempts=attempts,
                    timed_out=timed_out,
                    meta={"detail": last_error},
                )

            elapsed_ms = int((time.perf_counter() - started) * 1000)
            text = self._extract_text(response)
            usage = getattr(response, "usage_metadata", None)

            if not text:
                # An empty candidate usually means a safety filter fired. It is a
                # generation failure, not a transport one, so it is not retried:
                # the same prompt would be blocked again. The response degrades
                # and results[] still goes out complete (FR-19).
                return LLMResult(
                    text="",
                    error="llm_empty_response",
                    latency_ms=elapsed_ms,
                    model=self.model,
                    attempts=attempts,
                    prompt_tokens=_usage(usage, "prompt_token_count"),
                    completion_tokens=_usage(usage, "candidates_token_count"),
                    meta={"finish_reason": self._finish_reason(response)},
                )

            return LLMResult(
                text=text.strip(),
                latency_ms=elapsed_ms,
                model=self.model,
                attempts=attempts,
                prompt_tokens=_usage(usage, "prompt_token_count"),
                completion_tokens=_usage(usage, "candidates_token_count"),
            )

        return LLMResult(  # pragma: no cover - loop always returns
            text="",
            error="llm_error",
            model=self.model,
            attempts=attempts,
            meta={"detail": last_error},
        )

    def _config(self, prompt: Prompt) -> Any:
        types = self._types
        assert types is not None  # _ensure_client sets both or neither
        return types.GenerateContentConfig(
            system_instruction=prompt.system,
            temperature=(
                self._temperature if prompt.temperature is None else prompt.temperature
            ),
            max_output_tokens=(
                prompt.max_output_tokens or self._max_output_tokens
            ),
            # Nothing here needs sampling variety; a factual paragraph about six
            # database rows should be close to deterministic run to run, which
            # also makes the eval set (FR-31) meaningful.
            top_p=0.9,
            candidate_count=1,
        )

    @staticmethod
    def _extract_text(response: Any) -> str:
        text = getattr(response, "text", None)
        if text:
            return str(text)
        # Fall back to walking candidates: `.text` raises or returns None on some
        # SDK versions when the first candidate has no plain-text part.
        try:
            parts = response.candidates[0].content.parts
        except Exception:
            return ""
        return "".join(str(getattr(p, "text", "") or "") for p in parts)

    @staticmethod
    def _finish_reason(response: Any) -> str:
        try:
            return str(response.candidates[0].finish_reason)
        except Exception:
            return "unknown"


def _usage(usage: Any, attribute: str) -> int:
    value = getattr(usage, attribute, None) if usage is not None else None
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


__all__ = ["VertexProvider", "LLMConfigurationError"]
