"""Request/response contract for POST /v1/chat.

This module *is* the API contract (FR-1, FR-2, FR-3, FR-12). Two rules govern
changes here:

* FR-3 — additions to /v1 must be additive. Removing a field or changing a
  type is a breaking change and ships as /v2.
* FR-2 — `state` is a closed enum of exactly six values. Clients branch on it
  instead of parsing prose out of `answer`.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from app.config import SCHEMA_VERSION


class ChatState(str, Enum):
    """FR-2: the only six legal values. Anything else is invalid.

    Each maps to a scenario in the PRD:
      answered             — results found and narrated          (skenario 1)
      needs_clarification  — too broad, or an out-of-vocab value (skenario 2)
      no_result            — filters valid but nothing matched   (skenario 10)
      unavailable_field    — asked about fee_rm/schedule         (skenario 6)
      out_of_scope         — not a class-search question         (skenario 11)
      degraded             — narrative failed validation         (skenario 12)
    """

    ANSWERED = "answered"
    NEEDS_CLARIFICATION = "needs_clarification"
    NO_RESULT = "no_result"
    UNAVAILABLE_FIELD = "unavailable_field"
    OUT_OF_SCOPE = "out_of_scope"
    DEGRADED = "degraded"


# ─── Request ────────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    """FR-1: accepts `message` plus an optional `session_id`."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "message": "kelas Add Maths SPM online di Selangor bawah RM200",
                    "session_id": "s_demo_001",
                }
            ]
        },
    )

    message: str = Field(
        min_length=1,
        max_length=2_000,
        description=(
            "Pertanyaan user dalam bahasa alami. Bahasa Melayu, English, atau "
            "campuran keduanya (NFR-F). Diperlakukan sebagai data, bukan "
            "instruksi (NFR-D, prompt injection)."
        ),
        examples=["kelas Add Maths SPM online di Selangor bawah RM200"],
    )
    session_id: Optional[str] = Field(
        default=None,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
        description=(
            "Opsional. Kalau kosong, service membuat ID acak. NFR-D: tidak "
            "boleh diturunkan dari nama, email, telepon, atau data pribadi."
        ),
        examples=["s_demo_001"],
    )


# ─── Result item ────────────────────────────────────────────────────────────


class ClassResult(BaseModel):
    """One class, copied verbatim from the database row.

    FR-5: "Data kelas dalam results[] harus berasal langsung dari record
    database, bukan ditulis ulang oleh model." Nothing on this object is ever
    produced by the LLM except `match_reason`, which is validated against the
    record's own facts (FR-13).

    FR-12: the eleven fields below `record_id` are mandatory and non-empty.
    """

    model_config = ConfigDict(extra="forbid")

    # ── FR-12 mandatory eleven ──────────────────────────────────────────────
    record_id: str = Field(pattern=r"^MYTC-\d{3}$", examples=["MYTC-081"])
    class_title: str = Field(min_length=1)
    provider_name: str = Field(min_length=1)
    level: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    delivery_mode: str = Field(min_length=1)
    instruction_language: str = Field(min_length=1)
    state: str = Field(min_length=1)
    city_area: str = Field(min_length=1)
    provider_source_url: HttpUrl
    verified_on: date = Field(
        description=(
            "FR-14: tanggal dari record, bukan tanggal hari ini. Seluruh "
            "korpus v1 bernilai 2026-08-18."
        ),
        examples=["2026-08-18"],
    )

    # ── Display-only extras (DD "Peran" = T) ────────────────────────────────
    curriculum: Optional[str] = None
    education_stage: Optional[str] = None
    school_category: Optional[str] = None

    # ── Always-null fields ──────────────────────────────────────────────────
    # DD: "Keduanya tetap ada di skema respons (bernilai null) supaya klien
    # bisa membedakan 'tidak tersedia' dari 'tidak ditanyakan'." The annotation
    # is literally `None`, so any number here fails validation — which is the
    # contract the DD asks for: "angka apa pun melanggar kontrak" (FR-16/17).
    fee_rm: None = Field(
        default=None,
        description="Selalu null di korpus v1 (200/200). FR-16.",
    )
    schedule: None = Field(
        default=None,
        description="Selalu null di korpus v1 (200/200). FR-17.",
    )

    # ── FR-13: reason built only from mode/language/location/curriculum ─────
    match_reason: Optional[str] = Field(
        default=None,
        max_length=300,
        description=(
            "FR-13: hanya boleh merujuk mode pengajaran, bahasa, lokasi, dan "
            "kurikulum. Harga, jadwal, dan klaim kualitas dilarang."
        ),
    )


# ─── Supporting structures ──────────────────────────────────────────────────


class RelaxationOption(BaseModel):
    """FR-15: a concrete way to widen a search that returned nothing.

    Every option carries `field`, `current_value`, and `result_count` so the
    client can render "melonggarkan delivery_mode → 2 hasil" without guessing.
    """

    model_config = ConfigDict(extra="forbid")

    field: str = Field(examples=["delivery_mode"])
    current_value: str = Field(examples=["Online live"])
    result_count: int = Field(ge=0, examples=[2])
    suggested_values: List[str] = Field(default_factory=list)


class Trace(BaseModel):
    """FR-29: every response is traceable back to its data and filters."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    retrieved_ids: List[str] = Field(
        default_factory=list,
        description="FR-29 + FR-18: the only records `answer` may talk about.",
    )
    filters_applied: Dict[str, object] = Field(default_factory=dict)
    candidate_count: int = 0
    llm_used: bool = False
    latency_ms: Optional[int] = None
    validation_failed: List[str] = Field(
        default_factory=list,
        description="FR-30: which honesty checks failed, when degraded is true.",
    )


# ─── Response ───────────────────────────────────────────────────────────────


class ChatResponse(BaseModel):
    """The single response shape for all six states.

    Fields are always present (never conditionally omitted) so integrators can
    bind to a stable structure — scenario 15/16. Lists are empty rather than
    null when they do not apply.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "schema_version": "v1",
                    "state": "answered",
                    "session_id": "s_demo_001",
                    "answer": (
                        "Saya jumpa 2 kelas Additional Mathematics Form 4 di "
                        "Selangor. Kedua-duanya diajar secara Online live "
                        "dalam Malay-English bilingual."
                    ),
                    "results": [
                        {
                            "record_id": "MYTC-081",
                            "class_title": "Additional Mathematics Intensive",
                            "provider_name": "Tuition Hero Malaysia",
                            "level": "Form 4",
                            "subject": "Additional Mathematics",
                            "delivery_mode": "Online live",
                            "instruction_language": "Malay-English bilingual",
                            "state": "Selangor",
                            "city_area": "Petaling Jaya / Kota Kemuning",
                            "provider_source_url": "https://example.com/tuition-hero",
                            "verified_on": "2026-08-18",
                            "curriculum": "KSSM / SPM",
                            "education_stage": "Secondary",
                            "school_category": "National / private secondary (SPM track)",
                            "fee_rm": None,
                            "schedule": None,
                            "match_reason": (
                                "Online live, pengantar Malay-English "
                                "bilingual, di Selangor, kurikulum KSSM / SPM."
                            ),
                        }
                    ],
                    "notice": (
                        "Data ini contoh berasaskan rujukan web, bukan senarai "
                        "kelas langsung daripada penyedia. Sila semak laman "
                        "penyedia untuk maklumat terkini."
                    ),
                    "unavailable_fields": [],
                    "relaxation_options": [],
                    "missing_field": None,
                    "available_values": [],
                    "suggested_questions": [],
                    "degraded": False,
                    "trace": {
                        "trace_id": "tr_9f2c1a",
                        "retrieved_ids": ["MYTC-081", "MYTC-107"],
                        "filters_applied": {
                            "subject": "Additional Mathematics",
                            "level": "Form 4",
                            "state": "Selangor",
                        },
                        "candidate_count": 2,
                        "llm_used": True,
                        "latency_ms": 812,
                        "validation_failed": [],
                    },
                }
            ]
        },
    )

    schema_version: str = Field(
        default=SCHEMA_VERSION,
        description="FR-3: bumped only via a new path, e.g. /v2/chat.",
    )
    state: ChatState
    session_id: str

    answer: str = Field(
        description=(
            "Narasi untuk user. FR-18: setiap fakta di sini harus dapat "
            "dibuktikan dari trace.retrieved_ids."
        )
    )
    results: List[ClassResult] = Field(
        default_factory=list,
        max_length=20,
        description="NFR-B: maksimum 20 item.",
    )

    notice: str = Field(
        min_length=1,
        description=(
            "FR-21: tidak boleh kosong. Menjelaskan bahwa data adalah contoh "
            "berasaskan rujukan web, bukan senarai langsung dari penyedia."
        ),
    )

    unavailable_fields: List[str] = Field(
        default_factory=list,
        description='FR-20: mis. ["fee_rm"].',
        examples=[["fee_rm"]],
    )
    relaxation_options: List[RelaxationOption] = Field(
        default_factory=list,
        description="FR-15: hanya terisi saat state=no_result.",
    )

    missing_field: Optional[str] = Field(
        default=None,
        description="FR-10: maksimum satu field per respons.",
        examples=["state"],
    )
    available_values: List[str] = Field(
        default_factory=list,
        description="FR-10: pilihan nilai untuk missing_field, buat dropdown.",
    )

    suggested_questions: List[str] = Field(
        default_factory=list,
        description="FR-23: contoh pertanyaan yang memang bisa dijawab sistem.",
    )

    degraded: bool = Field(
        default=False,
        description="FR-19: true berarti answer diganti template aman.",
    )

    trace: Trace


# ─── Errors ─────────────────────────────────────────────────────────────────


class ErrorDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(examples=["llm_timeout"])
    message: str = Field(examples=["Model provider did not respond in 8s."])
    trace_id: Optional[str] = None


class ErrorResponse(BaseModel):
    """FR-4: one consistent error envelope. Never HTML, never an empty body."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "schema_version": "v1",
                    "error": {
                        "code": "llm_timeout",
                        "message": "Model provider did not respond in 8s.",
                        "trace_id": "tr_9f2c1a",
                    },
                }
            ]
        },
    )

    schema_version: str = SCHEMA_VERSION
    error: ErrorDetail


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    schema_version: str = SCHEMA_VERSION
    corpus_records: int
    llm_provider: str
    llm_ready: bool
    active_sessions: int
