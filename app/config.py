"""Application configuration.

Every tunable lives here and is driven by environment variables, so that
NFR-H ("Ganti model LLM melalui konfigurasi, tanpa mengubah logic utama")
holds: swapping the LLM or model never requires touching business logic.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

SCHEMA_VERSION = "v1"
"""Response-contract version. FR-3: breaking changes ship as /v2, never in place."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Service ─────────────────────────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "INFO"

    # ── Database ────────────────────────────────────────────────────────────
    database_url: str = (
        "postgresql+psycopg2://classbot:classbot@127.0.0.1:5433/classbot"
    )
    db_pool_size: int = 10
    db_max_overflow: int = 10

    # ── Auth (NFR-D) ────────────────────────────────────────────────────────
    # Raw form: "odoo:key1,frontend:key2". Kept as a string so pydantic-settings
    # does not try to JSON-decode it; parsed by `api_key_map`.
    api_keys: str = "odoo:odoo-dev-key-change-me,frontend:frontend-dev-key-change-me"
    require_api_key: bool = True

    # NFR-D: "HTTPS wajib; HTTP ditolak". Defaults to false so `uvicorn` on
    # localhost works out of the box; deployment must set REQUIRE_HTTPS=true.
    # The app almost certainly sits behind a TLS-terminating proxy, so the check
    # trusts X-Forwarded-Proto — which is only safe when nothing but that proxy
    # can reach the port, hence the separate switch below.
    require_https: bool = False
    trust_proxy_headers: bool = True

    # Comma-separated origins allowed to call the API from a browser. Empty means
    # no browser origin is allowed, which is the right default for a service whose
    # first consumer is server-side Odoo. "*" is refused when auth is on: a
    # wildcard plus a credentialed header is how a key leaks to any page.
    cors_origins: str = ""

    # ── Rate limit (NFR-C) ──────────────────────────────────────────────────
    rate_limit_per_minute: int = 60

    # ── Sessions (FR-28) ────────────────────────────────────────────────────
    session_ttl_seconds: int = 1800  # 30 minutes
    max_concurrent_sessions: int = 200

    # ── Retrieval (NFR-B) ───────────────────────────────────────────────────
    max_results: int = 20
    clarify_threshold: int = 20
    max_response_bytes: int = 102_400  # 100 KB

    # DD open question D3: should "Malaysia" / "Multi-state" providers surface
    # when the user names a specific state? Default yes — a Selangor student can
    # legitimately join a nationwide online class, and the widening is always
    # visible in trace.filters_applied.
    include_cross_region_states: bool = True

    # ── Query understanding (FR-8) ──────────────────────────────────────────
    # Two thresholds because short tokens are cheap to mangle: at 4 characters
    # "sains" and "seni" already score ~67, so short input must clear a higher
    # bar before we accept a fuzzy hit. Below the bar we ask (FR-9) instead of
    # guessing.
    fuzzy_min_score: int = 86
    fuzzy_min_score_short: int = 92
    fuzzy_short_term_chars: int = 5
    # Minimum gap between best and runner-up before a fuzzy hit is trusted.
    fuzzy_min_margin: int = 4
    # How often to stat() the synonym file for hot-reload (NFR-H).
    synonyms_reload_seconds: float = 2.0

    # ── LLM ─────────────────────────────────────────────────────────────────
    llm_provider: Literal["vertex", "null"] = "vertex"
    llm_timeout_seconds: float = 8.0
    llm_max_retries: int = 1
    llm_retry_backoff_ms: int = 500

    # The PRD says two different things about a model timeout. FR-4: "Jika terjadi
    # timeout pada model, service mengembalikan HTTP 503." NFR-C: "Saat LLM down —
    # pencarian berbasis filter tetap berjalan; respons memakai degraded tanpa
    # narasi generatif." Both cannot hold for one request.
    #
    # Default false, i.e. NFR-C wins: the user gets HTTP 200, the real class list,
    # and degraded=true, because throwing away correct results over a missing
    # paragraph serves nobody. Set true for the literal FR-4 reading — the trace
    # and the error code are identical either way, so this is a one-line switch
    # rather than a fork in the logic. Documented in README §Inconsistencies.
    llm_timeout_returns_503: bool = False

    # ── Vertex AI ───────────────────────────────────────────────────────────
    google_cloud_project: str = ""
    vertex_location: str = "asia-southeast1"
    vertex_model: str = "gemini-2.0-flash"
    vertex_max_output_tokens: int = 800
    vertex_temperature: float = 0.2

    # ── Cost (NFR-E) ────────────────────────────────────────────────────────
    cost_per_query_limit_rm: float = 0.05
    monthly_budget_rm: float = 500.0
    cost_alert_threshold: float = 0.8
    # Token prices in RM per 1,000 tokens, used to turn the token counts the
    # provider returns into the RM figures NFR-E is written in. Defaults are
    # Gemini 2.0 Flash list price (USD 0.10 / 0.40 per 1M tokens) at ~RM4.70 to
    # the dollar. Vendor pricing and FX both move — set these from the current
    # rate card rather than trusting the defaults for budget decisions.
    llm_input_price_per_1k_rm: float = 0.00047
    llm_output_price_per_1k_rm: float = 0.00188

    # ── Observability (NFR-G) ───────────────────────────────────────────────
    # Alert windows and thresholds, straight from the NFR-G table: degraded ratio
    # above 2% in an hour, error rate above 5%, p95 latency above 5 s.
    metrics_window_seconds: int = 3600
    alert_degraded_ratio: float = 0.02
    alert_error_rate: float = 0.05
    alert_latency_p95_ms: int = 5_000
    # Below this many requests in the window, ratios are too noisy to alert on:
    # one degraded response out of three is 33% and means nothing.
    alert_min_samples: int = 20
    # Seconds of silence after firing an alert, so a sustained problem produces a
    # readable log rather than one line per request.
    alert_cooldown_seconds: int = 300
    log_json: bool = True

    # ── Data files ──────────────────────────────────────────────────────────
    corpus_csv: str = "data/malaysia_tuition_classes_rag_200.csv"
    synonyms_path: str = "data/synonyms.json"

    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    @property
    def api_key_map(self) -> Dict[str, str]:
        """Map api-key -> consumer name.

        NFR-D: "API key berbeda untuk setiap konsumen; Odoo dan frontend
        memakai key terpisah." Storing it keyed by the secret lets auth be an
        O(1) lookup while still telling us *who* called, for logging.
        """
        mapping: Dict[str, str] = {}
        for chunk in self.api_keys.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if ":" in chunk:
                consumer, key = chunk.split(":", 1)
                mapping[key.strip()] = consumer.strip()
            else:
                mapping[chunk] = "unknown"
        return mapping

    @property
    def consumers(self) -> List[str]:
        return sorted(set(self.api_key_map.values()))

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor (import-safe, testable via cache_clear())."""
    return Settings()
