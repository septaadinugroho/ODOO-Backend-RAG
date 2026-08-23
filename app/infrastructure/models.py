"""SQLAlchemy models.

Structure follows the ER diagram in the Data Dictionary: the facets that the DD
declares as closed enums become dimension tables, and `class_records` references
them by foreign key. That keeps V2 ("seluruh field enum berisi nilai dari daftar
tertutup") enforceable by the database itself rather than by convention.

Deliberately *not* modelled here:

* `sessions` — PRD NFR-B says "Retensi sesi: 30 menit, in-memory, tidak
  persisten", and NG2 rules out cross-session history. The DD's SESSION entity
  is therefore implemented in `app/sessions.py` as an in-memory store, not a
  table. See README "Catatan PRD vs DD".
* pgvector — NFR-B: "belum menggunakan vector DB di v1".
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional

from sqlalchemy import (
    DDL,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    event,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ─── Dimension tables ───────────────────────────────────────────────────────


class Provider(Base):
    """DD: 13 values, each bound 1:1 to url + state + city_area."""

    __tablename__ = "providers"

    provider_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider_name: Mapped[str] = mapped_column(String(200), unique=True, nullable=False)
    provider_source_url: Mapped[str] = mapped_column(String(500), nullable=False)
    state: Mapped[str] = mapped_column(String(60), nullable=False)
    city_area: Mapped[str] = mapped_column(String(120), nullable=False)

    __table_args__ = (
        # V3: https only.
        CheckConstraint(
            "provider_source_url LIKE 'https://%'",
            name="ck_provider_url_https",
        ),
    )


class Subject(Base):
    __tablename__ = "subjects"

    subject_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)


class Level(Base):
    __tablename__ = "levels"

    level_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(60), unique=True, nullable=False)
    stage: Mapped[str] = mapped_column(String(20), nullable=False)
    # DD: "Year 1 sebelum Form 1"
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class DeliveryMode(Base):
    __tablename__ = "delivery_modes"

    delivery_mode_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    # DD: online | physical | hybrid | home
    mode_group: Mapped[str] = mapped_column(String(20), nullable=False)


class InstructionLanguage(Base):
    __tablename__ = "instruction_languages"

    instruction_lang_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)


class Curriculum(Base):
    __tablename__ = "curricula"

    curriculum_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(60), unique=True, nullable=False)
    # DD: "rujukan kurikulum, BUKAN sumber kelas" — internal, never displayed
    # as a class source because doing so would mislead.
    curriculum_source_url: Mapped[Optional[str]] = mapped_column(String(500))


# ─── Fact table ─────────────────────────────────────────────────────────────


class ClassRecord(Base):
    """One curated class. 23 CSV columns land here plus the dimension tables."""

    __tablename__ = "class_records"

    # V1: PK matching ^MYTC-\d{3}$
    record_id: Mapped[str] = mapped_column(String(12), primary_key=True)
    # DD: 182 unique of 200 — display only, explicitly NOT a natural key.
    class_title: Mapped[str] = mapped_column(String(300), nullable=False)

    provider_id: Mapped[int] = mapped_column(
        ForeignKey("providers.provider_id"), nullable=False
    )
    curriculum_id: Mapped[int] = mapped_column(
        ForeignKey("curricula.curriculum_id"), nullable=False
    )
    subject_id: Mapped[int] = mapped_column(
        ForeignKey("subjects.subject_id"), nullable=False
    )
    level_id: Mapped[int] = mapped_column(ForeignKey("levels.level_id"), nullable=False)
    delivery_mode_id: Mapped[int] = mapped_column(
        ForeignKey("delivery_modes.delivery_mode_id"), nullable=False
    )
    instruction_lang_id: Mapped[int] = mapped_column(
        ForeignKey("instruction_languages.instruction_lang_id"), nullable=False
    )

    school_category: Mapped[str] = mapped_column(String(120), nullable=False)
    education_stage: Mapped[str] = mapped_column(String(20), nullable=False)

    # ── Always NULL in v1 (DD "Field kosong — perlakuan khusus") ────────────
    # Nullable columns kept so the shape survives, with a CHECK that makes the
    # v1 assumption explicit at the storage layer. DD V8 deliberately treats a
    # *present* value as the alarm: it means the v1 contract needs revisiting.
    fee_rm: Mapped[Optional[float]] = mapped_column(Numeric(10, 2), nullable=True)
    schedule: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)

    class_status: Mapped[str] = mapped_column(
        String(60), nullable=False, default="example_not_live_listing"
    )
    data_label: Mapped[str] = mapped_column(
        String(60), nullable=False, default="web-grounded synthetic example"
    )

    # FR-14: shown as-is, never "today".
    verified_on: Mapped[date] = mapped_column(Date, nullable=False)

    # ── Internal-only fields (DD "Peran" = I) ───────────────────────────────
    retrieval_text: Mapped[Optional[str]] = mapped_column(Text)
    keywords: Mapped[Optional[str]] = mapped_column(Text)
    source_basis: Mapped[Optional[str]] = mapped_column(Text)

    loaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        # server_default as well, so rows inserted by raw SQL (db/init/*.sql,
        # loaded by the Postgres container on first boot) get a timestamp too.
        server_default=func.now(),
        nullable=False,
    )

    provider: Mapped[Provider] = relationship(lazy="joined")
    subject: Mapped[Subject] = relationship(lazy="joined")
    level: Mapped[Level] = relationship(lazy="joined")
    delivery_mode: Mapped[DeliveryMode] = relationship(lazy="joined")
    instruction_language: Mapped[InstructionLanguage] = relationship(lazy="joined")
    curriculum: Mapped[Curriculum] = relationship(lazy="joined")

    __table_args__ = (
        # V1: record_id must match ^MYTC-\d{3}$. That check uses PostgreSQL's `~`
        # regex operator, which SQLite cannot parse — it has no binary `~`, so a
        # CREATE TABLE carrying it fails outright and would break the SQLite-backed
        # test suite. It is therefore attached as a Postgres-only DDL event just
        # below this class instead of here. Docker's hand-written schema
        # (db/init/001_schema.sql) carries the same named constraint, so Postgres
        # enforces the pattern either way.
        CheckConstraint(
            "education_stage IN ('Primary', 'Secondary')",
            name="ck_education_stage",
        ),
        # Composite index covering the common filter combination from FR-11.
        Index(
            "ix_class_records_facets",
            "subject_id",
            "level_id",
            "delivery_mode_id",
            "provider_id",
        ),
        Index("ix_class_records_verified_on", "verified_on"),
    )


# V1 (Postgres only): enforce the MYTC-### pattern with a regex CHECK, attached
# through a DDL event guarded by execute_if(dialect="postgresql"). SQLite — used
# by the test suite — has no binary `~` operator and would reject the CREATE
# TABLE, so the constraint must not be emitted for it. On Postgres created via
# the ORM (scripts/init_db.py) it is added right after the table, matching the
# hand-written db/init/001_schema.sql that Docker applies.
event.listen(
    ClassRecord.__table__,
    "after_create",
    DDL(
        "ALTER TABLE class_records "
        "ADD CONSTRAINT ck_record_id_pattern "
        r"CHECK (record_id ~ '^MYTC-[0-9]{3}$')"
    ).execute_if(dialect="postgresql"),
)


# ─── Synonyms ───────────────────────────────────────────────────────────────


class Synonym(Base):
    """Mirror of the editable synonym dictionary.

    NFR-H requires the dictionary to live outside the code and be updatable
    without a redeploy, so `data/synonyms.json` is the editable source of truth
    and is hot-reloaded on mtime change (see app/synonyms.py). This table is the
    persisted mirror the DD models, refreshed by scripts/load_data.py, so the
    dictionary is also queryable in SQL for audits.
    """

    __tablename__ = "synonyms"

    synonym_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    term: Mapped[str] = mapped_column(String(120), nullable=False)
    facet: Mapped[str] = mapped_column(String(40), nullable=False)
    canonical_value: Mapped[str] = mapped_column(String(120), nullable=False)
    lang: Mapped[str] = mapped_column(String(10), nullable=False, default="mixed")

    __table_args__ = (
        UniqueConstraint("term", "facet", name="uq_synonym_term_facet"),
        Index("ix_synonym_term", "term"),
    )


# ─── Query log ──────────────────────────────────────────────────────────────


class QueryLog(Base):
    """FR-30 + NFR-D: per-request trace, no user identity, 30-day retention.

    Structured JSON logs (app/observability.py) remain the primary observability
    surface per NFR-G; this table exists so degraded responses can be analysed
    with SQL, which FR-30 asks for. Writes never block or fail a response.
    """

    __tablename__ = "query_logs"

    trace_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    consumer: Mapped[Optional[str]] = mapped_column(String(40))
    # NFR-D: "Query dicatat tanpa identitas user."
    user_message: Mapped[str] = mapped_column(Text, nullable=False)
    filters_applied: Mapped[Optional[str]] = mapped_column(Text)  # JSON string
    retrieved_ids: Mapped[Optional[str]] = mapped_column(Text)  # JSON string
    state: Mapped[str] = mapped_column(String(30), nullable=False)
    degraded: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    validation_failed: Mapped[Optional[str]] = mapped_column(Text)  # JSON string
    llm_used: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer)
    token_cost: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_query_logs_created_at", "created_at"),
        Index("ix_query_logs_state", "state"),
        Index("ix_query_logs_degraded", "degraded"),
        Index("ix_query_logs_session", "session_id"),
    )
