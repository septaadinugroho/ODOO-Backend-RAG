"""Shared pytest configuration and fixtures.

Two decisions worth knowing about.

**The test database is SQLite, seeded from `data/sample_corpus.csv`.** Every
model in `app/models.py` uses portable column types, so the schema builds on
SQLite unchanged, and the tests then run against the same twelve rows the
Postgres fixture seeds. That keeps the suite runnable with no container, no
network, and no `docker compose up` — a test suite you have to set up is a test
suite that stops being run. `tests/test_schema_matches_orm.py` is the counterpart
that checks the real Postgres schema, and it skips when Postgres is absent.

**Settings are frozen before `app.config` is imported.** `get_settings()` is
`lru_cache`d, so the environment has to be right at import time; setting it later
would silently have no effect.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Set before anything imports app.config, whose get_settings() is lru_cached.
os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")
os.environ.setdefault("LLM_PROVIDER", "null")
os.environ.setdefault("API_KEYS", "pytest:test-key-0123456789abcdef")
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("REQUIRE_HTTPS", "false")
os.environ.setdefault("LOG_JSON", "false")
os.environ.setdefault("LOG_LEVEL", "CRITICAL")
# The suite fires hundreds of requests on one key in a few seconds; the
# production 60/min limit has its own dedicated test instead.
os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "1000000")

import pytest  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import Session, sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

CORPUS_CSV = ROOT / "data" / "sample_corpus.csv"

TEST_API_KEY = "test-key-0123456789abcdef"


def _seed(session: Session) -> int:
    """Load the CSV fixture into the normalised schema.

    Mirrors `db/init/002_seed_corpus.sql` through the ORM rather than repeating
    the row values here, so the two never drift apart.
    """
    from app.infrastructure.models import (
        ClassRecord,
        Curriculum,
        DeliveryMode,
        InstructionLanguage,
        Level,
        Provider,
        Subject,
    )
    from app.core.vocabulary import LEVEL_SORT_ORDER, LEVEL_STAGE, MODE_TO_GROUP

    from datetime import date

    providers: dict[str, Provider] = {}
    subjects: dict[str, Subject] = {}
    levels: dict[str, Level] = {}
    modes: dict[str, DeliveryMode] = {}
    languages: dict[str, InstructionLanguage] = {}
    curricula: dict[str, Curriculum] = {}
    count = 0

    with CORPUS_CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = row["provider_name"]
            if name not in providers:
                providers[name] = Provider(
                    provider_name=name,
                    provider_source_url=row["provider_source_url"],
                    state=row["state"],
                    city_area=row["city_area"],
                )
                session.add(providers[name])

            if row["subject"] not in subjects:
                subjects[row["subject"]] = Subject(name=row["subject"])
                session.add(subjects[row["subject"]])

            if row["level"] not in levels:
                levels[row["level"]] = Level(
                    name=row["level"],
                    stage=LEVEL_STAGE.get(row["level"], row["education_stage"]),
                    sort_order=LEVEL_SORT_ORDER.get(row["level"], 0),
                )
                session.add(levels[row["level"]])

            if row["delivery_mode"] not in modes:
                modes[row["delivery_mode"]] = DeliveryMode(
                    name=row["delivery_mode"],
                    mode_group=MODE_TO_GROUP.get(row["delivery_mode"], "unspecified"),
                )
                session.add(modes[row["delivery_mode"]])

            if row["instruction_language"] not in languages:
                languages[row["instruction_language"]] = InstructionLanguage(
                    name=row["instruction_language"]
                )
                session.add(languages[row["instruction_language"]])

            if row["curriculum"] not in curricula:
                curricula[row["curriculum"]] = Curriculum(
                    name=row["curriculum"],
                    curriculum_source_url=row.get("curriculum_source_url") or None,
                )
                session.add(curricula[row["curriculum"]])

            session.flush()

            session.add(
                ClassRecord(
                    record_id=row["record_id"],
                    class_title=row["class_title"],
                    provider_id=providers[name].provider_id,
                    curriculum_id=curricula[row["curriculum"]].curriculum_id,
                    subject_id=subjects[row["subject"]].subject_id,
                    level_id=levels[row["level"]].level_id,
                    delivery_mode_id=modes[row["delivery_mode"]].delivery_mode_id,
                    instruction_lang_id=languages[
                        row["instruction_language"]
                    ].instruction_lang_id,
                    school_category=row["school_category"],
                    education_stage=row["education_stage"],
                    # FR-16 / FR-17: empty in the corpus, and the suite proves the
                    # service keeps them empty all the way to the wire.
                    fee_rm=None,
                    schedule=None,
                    class_status=row["class_status"],
                    data_label=row["data_label"],
                    verified_on=date.fromisoformat(row["verified_on"]),
                    retrieval_text=row.get("retrieval_text"),
                    keywords=row.get("keywords"),
                    source_basis=row.get("source_basis"),
                )
            )
            count += 1

    session.commit()
    return count


@pytest.fixture(scope="session")
def engine():
    from app.infrastructure.models import Base

    eng = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(eng)
    factory = sessionmaker(bind=eng, expire_on_commit=False)
    with factory() as session:
        _seed(session)
    return eng


@pytest.fixture()
def db(engine) -> Session:
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    session = factory()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture(autouse=True)
def _clean_process_state():
    """Sessions, metrics and the rate-limit window are process-global by design.

    Resetting them between tests keeps one test's traffic from showing up in
    another's assertions.
    """
    from app.infrastructure.llm import reset_provider
    from app.infrastructure.observability import reset_metrics
    from app.api.security import reset_limiter
    from app.core.sessions import reset_store

    reset_store()
    reset_metrics()
    reset_limiter()
    reset_provider()
    yield
    reset_store()
    reset_metrics()
    reset_limiter()
    reset_provider()


@pytest.fixture()
def client(db):
    """The real app, with only the database dependency swapped for SQLite."""
    from fastapi.testclient import TestClient

    from app.infrastructure.db import get_db
    from app.main import create_app

    app = create_app()
    app.dependency_overrides[get_db] = lambda: db
    # No `with` block: the lifespan's startup probes talk to the configured
    # engine rather than this fixture, and test_app.py exercises it separately.
    test_client = TestClient(app)
    test_client.headers.update({"X-API-Key": TEST_API_KEY})
    return test_client


@pytest.fixture()
def rows(db):
    """Every fixture row as `ClassRow` objects, for the guardrail tests."""
    from app.core.retrieval import search

    return list(search(db, {}, limit=50).rows)
