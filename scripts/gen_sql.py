#!/usr/bin/env python3
"""Generate the SQL that PostgreSQL loads on its very first boot.

Why this script exists
----------------------
`docker-entrypoint-initdb.d/` is run by the postgres image exactly once, when the
data directory is empty. Dropping SQL there means `docker compose up -d` yields a
database that already contains the schema, the closed vocabulary, the synonym
mirror and the class corpus — no `pip install`, no Python, no extra step.

That constraint is why this file uses **nothing but the standard library**: it has
to be runnable before the project's dependencies exist. It imports
`app.core.vocabulary`, `app.core.ingest` and `app.core.textnorm`, all of which are dependency
free on purpose, so the SQL is generated from the same source of truth the
running application uses. Nothing here is hand-written twice.

No SQL injection surface, by construction
-----------------------------------------
None of the corpus data is ever interpolated into a SQL string. Every value
travels through `COPY ... FROM stdin WITH (FORMAT csv)`, where the payload is a
CSV *data stream* that PostgreSQL never parses as SQL — the same mechanism
`pg_dump` uses. A cell containing `'); DROP TABLE class_records; --` is therefore
just a 34-character string, not a statement, and it lands in the column intact.

The only SQL text this script composes is the fixed DDL and the fixed column
lists, both of which come from the code, never from a data file. `_sql_ident()`
enforces that with an allow-list, so a stray column name cannot smuggle anything
in either.

The application's own query path is held to the same standard a different way:
see `db/README.md` and `tests/test_sql_injection.py` — every runtime query goes
through SQLAlchemy bound parameters, and user text additionally has to survive
the FR-9 closed-vocabulary gate before it can influence a query at all.

Usage
-----
    python scripts/gen_sql.py                            # uses data/sample_corpus.csv
    python scripts/gen_sql.py data/real_corpus_200.csv   # the real corpus
    python scripts/gen_sql.py --check                    # fail if files are stale

Exit codes:
    0  written (or, with --check, up to date)
    1  the corpus failed validation, or --check found drift
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.ingest import (  # noqa: E402
    HARD_WARNING,
    IngestReport,
    read_rows,
    validate,
)
from app.core.textnorm import normalize  # noqa: E402
from app.core.vocabulary import (  # noqa: E402
    CURRICULA,
    DELIVERY_MODES,
    INSTRUCTION_LANGUAGES,
    LEVEL_SORT_ORDER,
    LEVEL_STAGE,
    LEVELS,
    MODE_TO_GROUP,
    SUBJECTS,
)

OUT_DIR = ROOT / "db" / "init"
SCHEMA_FILE = "001_schema.sql"
SEED_FILE = "002_seed_corpus.sql"

_IDENT = re.compile(r"^[a-z_][a-z0-9_]*$")


def _sql_ident(name: str) -> str:
    """Allow-list guard for every identifier this script writes into SQL.

    Column and table names come from code, not from data, so this can never fire
    in normal operation. It exists so that if someone later feeds a CSV header
    into a column list, the script stops instead of composing SQL out of it.
    """
    if not _IDENT.match(name):
        raise ValueError(
            f"refusing to build SQL with identifier {name!r}: "
            "identifiers must match ^[a-z_][a-z0-9_]*$"
        )
    return name


def _copy_block(table: str, columns: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    """A `COPY ... FROM stdin` statement with its CSV payload.

    The rows are written by `csv.writer`, so quoting and escaping are handled by
    the standard library rather than by hand. In COPY's CSV mode an unquoted
    empty field is NULL and a quoted empty field (`""`) is the empty string,
    which is exactly the distinction we want: `None` becomes NULL, `""` stays
    empty. Every column is staged as text and converted in SQL afterwards.
    """
    cols = ", ".join(_sql_ident(c) for c in columns)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    count = 0
    for row in rows:
        if len(row) != len(columns):
            raise ValueError(
                f"{table}: row has {len(row)} values but {len(columns)} columns"
            )
        writer.writerow(["" if v is None else v for v in row])
        count += 1
    payload = buffer.getvalue()
    if "\\." in payload:
        # psql ends the data stream at a line containing only "\.", so a value
        # holding that sequence could truncate the load. Nothing in the corpus
        # does, but a silent short load is the kind of bug that hides for weeks.
        raise ValueError(f"{table}: data contains a backslash-dot sequence")
    return (
        f"COPY {_sql_ident(table)} ({cols}) FROM stdin WITH (FORMAT csv);\n"
        f"{payload}\\.\n"
    ), count


# ─── 001_schema.sql ─────────────────────────────────────────────────────────

SCHEMA_SQL = """\
-- ===========================================================================
--  Class Bot v1 — schema
--
--  GENERATED by scripts/gen_sql.py. Do not edit by hand; edit app/models.py
--  and re-run the generator.
--
--  Loaded automatically by the postgres image from
--  /docker-entrypoint-initdb.d/ the first time the data volume is created.
--  It is a mirror of app/models.py; tests/test_schema_matches_orm.py compares
--  the live database against SQLAlchemy's metadata, so the two cannot drift
--  apart unnoticed.
--
--  Design notes worth keeping in view:
--
--   * No pgvector. PRD NFR-B: "belum menggunakan vector DB di v1". Retrieval is
--     deterministic SQL filtering over closed-vocabulary facets.
--   * The enum facets from the Data Dictionary are dimension *tables*, not
--     CHECK constraints on text, so DD rule V2 ("seluruh field enum berisi
--     nilai dari daftar tertutup") is enforced by foreign keys. An unknown
--     subject cannot enter even through a direct INSERT.
--   * There is no `sessions` table. PRD NFR-B caps session retention at 30
--     minutes in memory and NG2 rules out cross-session history, so the DD's
--     SESSION entity is implemented in app/sessions.py instead. See the README
--     section "Catatan PRD vs DD".
-- ===========================================================================

SET client_min_messages = warning;

-- ─── Dimension tables (closed vocabulary) ──────────────────────────────────

CREATE TABLE IF NOT EXISTS subjects (
    subject_id  SERIAL       PRIMARY KEY,
    name        VARCHAR(120) NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS levels (
    level_id    SERIAL      PRIMARY KEY,
    name        VARCHAR(60) NOT NULL UNIQUE,
    stage       VARCHAR(20) NOT NULL,
    -- DD ER diagram: "Year 1 sebelum Form 1", so ordering is explicit data
    -- rather than something inferred from the label at query time.
    sort_order  INTEGER     NOT NULL
);

CREATE TABLE IF NOT EXISTS delivery_modes (
    delivery_mode_id  SERIAL      PRIMARY KEY,
    name              VARCHAR(80) NOT NULL UNIQUE,
    -- online | physical | hybrid | home | unspecified.
    -- "unspecified" covers the three modes whose names carry no channel signal
    -- (Small group, Private one-to-one, Personalised tuition). Forcing them
    -- into a channel would be the guess FR-9 forbids.
    mode_group        VARCHAR(20) NOT NULL
);

CREATE TABLE IF NOT EXISTS instruction_languages (
    instruction_lang_id  SERIAL      PRIMARY KEY,
    name                 VARCHAR(80) NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS curricula (
    curriculum_id          SERIAL      PRIMARY KEY,
    name                   VARCHAR(60) NOT NULL UNIQUE,
    -- DD: "rujukan kurikulum, BUKAN sumber kelas". Kept on the curriculum row
    -- and never returned as a class source, because showing it as one would
    -- mislead the user about where the class information came from.
    curriculum_source_url  VARCHAR(500)
);

-- ─── Providers ─────────────────────────────────────────────────────────────
-- DD: provider_name is bound 1:1 to url + state + city_area, so those three
-- live here instead of being repeated on every class row.

CREATE TABLE IF NOT EXISTS providers (
    provider_id          SERIAL       PRIMARY KEY,
    provider_name        VARCHAR(200) NOT NULL UNIQUE,
    provider_source_url  VARCHAR(500) NOT NULL,
    state                VARCHAR(60)  NOT NULL,
    city_area            VARCHAR(120) NOT NULL,
    -- V3: sumber wajib https.
    CONSTRAINT ck_provider_url_https
        CHECK (provider_source_url LIKE 'https://%')
);

-- ─── Class records ─────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS class_records (
    -- V1: PK matching ^MYTC-\\d{3}$.
    record_id            VARCHAR(12)  PRIMARY KEY,
    -- DD: 182 unique titles out of 200 rows — display only, explicitly NOT a
    -- natural key.
    class_title          VARCHAR(300) NOT NULL,

    provider_id          INTEGER      NOT NULL REFERENCES providers(provider_id),
    curriculum_id        INTEGER      NOT NULL REFERENCES curricula(curriculum_id),
    subject_id           INTEGER      NOT NULL REFERENCES subjects(subject_id),
    level_id             INTEGER      NOT NULL REFERENCES levels(level_id),
    delivery_mode_id     INTEGER      NOT NULL REFERENCES delivery_modes(delivery_mode_id),
    instruction_lang_id  INTEGER      NOT NULL REFERENCES instruction_languages(instruction_lang_id),

    school_category      VARCHAR(120) NOT NULL,
    education_stage      VARCHAR(20)  NOT NULL,

    -- Empty for all 200 records in v1 (DD "Field kosong — perlakuan khusus").
    -- The columns are kept so the shape survives, but the API contract is that
    -- they are never returned as a value: FR-16 (fee) and FR-17 (schedule).
    -- DD rule V8 is inverted for exactly this reason — a *populated* value here
    -- is the alarm, because it means the v1 contract needs a conscious review.
    fee_rm               NUMERIC(10, 2),
    schedule             VARCHAR(300),

    class_status         VARCHAR(60)  NOT NULL,
    data_label           VARCHAR(60)  NOT NULL,

    -- FR-14: shown to the user as-is. Never rendered as "today" or "current".
    verified_on          DATE         NOT NULL,

    -- Internal only (DD "Peran" = I). Never serialised into a response.
    retrieval_text       TEXT,
    keywords             TEXT,
    source_basis         TEXT,

    -- server_default so rows inserted by this file get a timestamp without the
    -- Python layer being involved.
    loaded_at            TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),

    CONSTRAINT ck_record_id_pattern
        CHECK (record_id ~ '^MYTC-[0-9]{3}$'),
    CONSTRAINT ck_education_stage
        CHECK (education_stage IN ('Primary', 'Secondary'))
);

-- Covers the usual FR-11 filter combination.
CREATE INDEX IF NOT EXISTS ix_class_records_facets
    ON class_records (subject_id, level_id, delivery_mode_id, provider_id);
CREATE INDEX IF NOT EXISTS ix_class_records_verified_on
    ON class_records (verified_on);

-- ─── Synonym mirror ────────────────────────────────────────────────────────
-- NFR-H requires the dictionary to be editable without a redeploy, so
-- data/synonyms.json is the source of truth and is hot-reloaded on mtime
-- change. This table is a read-only mirror so the dictionary is auditable in
-- SQL. `term` is stored already normalised (app/textnorm.normalize).

CREATE TABLE IF NOT EXISTS synonyms (
    synonym_id       SERIAL       PRIMARY KEY,
    term             VARCHAR(120) NOT NULL,
    facet            VARCHAR(40)  NOT NULL,
    canonical_value  VARCHAR(120) NOT NULL,
    lang             VARCHAR(10)  NOT NULL,
    CONSTRAINT uq_synonym_term_facet UNIQUE (term, facet)
);

CREATE INDEX IF NOT EXISTS ix_synonym_term ON synonyms (term);

-- ─── Query log ─────────────────────────────────────────────────────────────
-- FR-30 + NFR-D: one row per request, no user identity, 30-day retention.
-- Structured JSON logs remain the primary observability surface (NFR-G); this
-- table exists so degraded answers can be analysed with SQL.

CREATE TABLE IF NOT EXISTS query_logs (
    trace_id           VARCHAR(40) PRIMARY KEY,
    session_id         VARCHAR(64) NOT NULL,
    consumer           VARCHAR(40),
    -- NFR-D: "Query dicatat tanpa identitas user." The message text is kept,
    -- never a name, e-mail or phone number.
    user_message       TEXT        NOT NULL,
    filters_applied    TEXT,
    retrieved_ids      TEXT,
    state              VARCHAR(30) NOT NULL,
    degraded           BOOLEAN     NOT NULL,
    validation_failed  TEXT,
    llm_used           BOOLEAN     NOT NULL,
    latency_ms         INTEGER,
    token_cost         INTEGER,
    created_at         TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_query_logs_created_at ON query_logs (created_at);
CREATE INDEX IF NOT EXISTS ix_query_logs_state      ON query_logs (state);
CREATE INDEX IF NOT EXISTS ix_query_logs_degraded   ON query_logs (degraded);
CREATE INDEX IF NOT EXISTS ix_query_logs_session    ON query_logs (session_id);
"""


# ─── 002_seed_corpus.sql ────────────────────────────────────────────────────

STAGE_COLUMNS: Tuple[str, ...] = (
    "record_id",
    "class_title",
    "provider_name",
    "curriculum",
    "subject",
    "level",
    "delivery_mode",
    "instruction_language",
    "school_category",
    "education_stage",
    "class_status",
    "data_label",
    "verified_on",
    "retrieval_text",
    "keywords",
    "source_basis",
)


def _providers_from(rows: Sequence[Dict[str, str]]) -> List[Tuple[str, str, str, str]]:
    """Derive the provider dimension from the corpus, first occurrence winning.

    Same rule as app/ingest.load_rows, so the SQL path and the Python path
    produce identical provider rows. V7 has already warned if a name maps to
    more than one URL.
    """
    seen: Dict[str, Tuple[str, str, str, str]] = {}
    for row in rows:
        name = row["provider_name"]
        if name in seen:
            continue
        seen[name] = (
            name,
            row["provider_source_url"],
            row["state"],
            row["city_area"],
        )
    return [seen[k] for k in sorted(seen)]


def _curriculum_urls(rows: Sequence[Dict[str, str]]) -> Dict[str, str]:
    """curriculum name -> first non-empty curriculum_source_url."""
    urls: Dict[str, str] = {}
    for row in rows:
        name = (row.get("curriculum") or "").strip()
        url = (row.get("curriculum_source_url") or "").strip()
        if name and url and name not in urls:
            urls[name] = url
    return urls


def _synonym_entries(path: Path) -> List[Tuple[str, str, str, str]]:
    """Flatten data/synonyms.json exactly as SynonymDictionary.all_entries does.

    The dictionary keys on (facet, normalised term), so a repeated pair collapses
    to one row — which is also what the uq_synonym_term_facet constraint
    requires. Deduplicating here rather than letting PostgreSQL raise means a
    sloppy edit to the JSON does not break `docker compose up`.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    by_facet: Dict[Tuple[str, str], str] = {}
    for entry in raw.get("entries", []):
        term = normalize(entry["term"])
        if not term:
            continue
        by_facet[(entry["facet"], term)] = entry["canonical_value"]
    return [
        (term, facet, value, "mixed")
        for (facet, term), value in sorted(by_facet.items())
    ]


def build_seed_sql(rows: Sequence[Dict[str, str]], source: Path, synonyms: Path) -> str:
    providers = _providers_from(rows)
    curriculum_urls = _curriculum_urls(rows)
    synonym_rows = _synonym_entries(synonyms)

    parts: List[str] = []
    counts: Dict[str, int] = {}

    parts.append(
        f"""\
-- ===========================================================================
--  Class Bot v1 — seed data
--
--  GENERATED by scripts/gen_sql.py from
--    corpus   : {source.name}
--    synonyms : {synonyms.name}
--  Do not edit by hand; change the source files and re-run the generator.
--
--  Every value below arrives through COPY ... FROM stdin, which PostgreSQL
--  reads as a CSV data stream and never as SQL. No corpus value is
--  interpolated into a statement, so there is no injection surface here even
--  if the corpus file is untrusted.
--
--  The corpus was validated against Data Dictionary rules V1-V8 before this
--  file was written; only rows that passed are present. The DO block at the
--  end re-checks the resulting row counts and raises, which aborts the
--  container's initialisation rather than leaving a half-loaded database.
-- ===========================================================================

SET client_min_messages = warning;

BEGIN;

-- ─── Closed vocabulary ─────────────────────────────────────────────────────
-- Seeded from app/vocabulary.py, deliberately NOT from the corpus file. That
-- is what makes DD rule V2 enforceable by the database: a foreign key can only
-- point at a value the Data Dictionary sanctions, so an unknown enum cannot
-- enter even by hand-written SQL.

CREATE TEMP TABLE _stage_subjects (name text) ON COMMIT DROP;
"""
    )

    block, n = _copy_block("_stage_subjects", ("name",), ((s,) for s in SUBJECTS))
    parts.append(block)
    counts["subjects"] = n
    parts.append(
        "\nINSERT INTO subjects (name)\n"
        "SELECT name FROM _stage_subjects\n"
        "ON CONFLICT (name) DO NOTHING;\n\n"
    )

    parts.append(
        "CREATE TEMP TABLE _stage_levels (name text, stage text, sort_order text)\n"
        "    ON COMMIT DROP;\n"
    )
    block, n = _copy_block(
        "_stage_levels",
        ("name", "stage", "sort_order"),
        ((name, LEVEL_STAGE[name], LEVEL_SORT_ORDER[name]) for name in LEVELS),
    )
    parts.append(block)
    counts["levels"] = n
    parts.append(
        "\nINSERT INTO levels (name, stage, sort_order)\n"
        "SELECT name, stage, sort_order::int FROM _stage_levels\n"
        "ON CONFLICT (name) DO NOTHING;\n\n"
    )

    parts.append(
        "CREATE TEMP TABLE _stage_delivery_modes (name text, mode_group text)\n"
        "    ON COMMIT DROP;\n"
    )
    block, n = _copy_block(
        "_stage_delivery_modes",
        ("name", "mode_group"),
        ((name, MODE_TO_GROUP[name]) for name in DELIVERY_MODES),
    )
    parts.append(block)
    counts["delivery_modes"] = n
    parts.append(
        "\nINSERT INTO delivery_modes (name, mode_group)\n"
        "SELECT name, mode_group FROM _stage_delivery_modes\n"
        "ON CONFLICT (name) DO NOTHING;\n\n"
    )

    parts.append(
        "CREATE TEMP TABLE _stage_instruction_languages (name text) ON COMMIT DROP;\n"
    )
    block, n = _copy_block(
        "_stage_instruction_languages", ("name",), ((s,) for s in INSTRUCTION_LANGUAGES)
    )
    parts.append(block)
    counts["instruction_languages"] = n
    parts.append(
        "\nINSERT INTO instruction_languages (name)\n"
        "SELECT name FROM _stage_instruction_languages\n"
        "ON CONFLICT (name) DO NOTHING;\n\n"
    )

    parts.append(
        "CREATE TEMP TABLE _stage_curricula (name text, curriculum_source_url text)\n"
        "    ON COMMIT DROP;\n"
    )
    block, n = _copy_block(
        "_stage_curricula",
        ("name", "curriculum_source_url"),
        ((name, curriculum_urls.get(name)) for name in CURRICULA),
    )
    parts.append(block)
    counts["curricula"] = n
    parts.append(
        "\nINSERT INTO curricula (name, curriculum_source_url)\n"
        "SELECT name, curriculum_source_url FROM _stage_curricula\n"
        "ON CONFLICT (name) DO NOTHING;\n\n"
    )

    # ── Providers ──
    parts.append(
        "-- ─── Providers ─────────────────────────────────────────────────────────────\n"
        "-- Derived from the corpus: the DD binds provider_name 1:1 to url, state and\n"
        "-- city_area, and V7 has already flagged any name that maps to two URLs.\n\n"
        "CREATE TEMP TABLE _stage_providers (\n"
        "    provider_name text, provider_source_url text, state text, city_area text\n"
        ") ON COMMIT DROP;\n"
    )
    block, n = _copy_block(
        "_stage_providers",
        ("provider_name", "provider_source_url", "state", "city_area"),
        providers,
    )
    parts.append(block)
    counts["providers"] = n
    parts.append(
        "\nINSERT INTO providers (provider_name, provider_source_url, state, city_area)\n"
        "SELECT provider_name, provider_source_url, state, city_area\n"
        "FROM _stage_providers\n"
        "ON CONFLICT (provider_name) DO NOTHING;\n\n"
    )

    # ── Class records ──
    parts.append(
        "-- ─── Class records ─────────────────────────────────────────────────────────\n"
        "-- Facet values are staged as text and resolved to dimension ids by name. The\n"
        "-- joins are inner joins on purpose: a value outside the closed vocabulary\n"
        "-- would drop its row rather than land in the table, and the count assertion\n"
        "-- at the end of this file then fails, which aborts initialisation. Silent\n"
        "-- partial loads are the failure mode this design refuses to allow.\n\n"
        "CREATE TEMP TABLE _stage_classes (\n"
        + ",\n".join(f"    {_sql_ident(c)} text" for c in STAGE_COLUMNS)
        + "\n) ON COMMIT DROP;\n"
    )
    block, n = _copy_block(
        "_stage_classes",
        STAGE_COLUMNS,
        (
            tuple(row.get(column, "") for column in STAGE_COLUMNS)
            for row in rows
        ),
    )
    parts.append(block)
    counts["class_records"] = n
    parts.append(
        """
INSERT INTO class_records (
    record_id, class_title,
    provider_id, curriculum_id, subject_id, level_id,
    delivery_mode_id, instruction_lang_id,
    school_category, education_stage,
    fee_rm, schedule,
    class_status, data_label, verified_on,
    retrieval_text, keywords, source_basis
)
SELECT
    v.record_id,
    v.class_title,
    p.provider_id,
    c.curriculum_id,
    s.subject_id,
    l.level_id,
    d.delivery_mode_id,
    i.instruction_lang_id,
    v.school_category,
    v.education_stage,
    -- Forced NULL regardless of what the source file contained. Rule V8 has
    -- already raised a hard warning if a value was present; the v1 API
    -- contract (FR-16, FR-17) wins until someone decides otherwise.
    NULL,
    NULL,
    COALESCE(NULLIF(v.class_status, ''), 'example_not_live_listing'),
    COALESCE(NULLIF(v.data_label,   ''), 'web-grounded synthetic example'),
    v.verified_on::date,
    NULLIF(v.retrieval_text, ''),
    NULLIF(v.keywords,       ''),
    NULLIF(v.source_basis,   '')
FROM _stage_classes           v
JOIN providers                p ON p.provider_name = v.provider_name
JOIN curricula                c ON c.name          = v.curriculum
JOIN subjects                 s ON s.name          = v.subject
JOIN levels                   l ON l.name          = v.level
JOIN delivery_modes           d ON d.name          = v.delivery_mode
JOIN instruction_languages    i ON i.name          = v.instruction_language
ON CONFLICT (record_id) DO NOTHING;

"""
    )

    # ── Synonym mirror ──
    parts.append(
        "-- ─── Synonym mirror ────────────────────────────────────────────────────────\n"
        "-- data/synonyms.json stays the editable source of truth (NFR-H). This is a\n"
        "-- rebuilt-from-scratch mirror, matching app.core.ingest.mirror_synonyms, so the\n"
        "-- table can never accumulate entries the JSON has since dropped.\n\n"
        "DELETE FROM synonyms;\n\n"
        "CREATE TEMP TABLE _stage_synonyms (\n"
        "    term text, facet text, canonical_value text, lang text\n"
        ") ON COMMIT DROP;\n"
    )
    block, n = _copy_block(
        "_stage_synonyms",
        ("term", "facet", "canonical_value", "lang"),
        synonym_rows,
    )
    parts.append(block)
    counts["synonyms"] = n
    parts.append(
        "\nINSERT INTO synonyms (term, facet, canonical_value, lang)\n"
        "SELECT term, facet, canonical_value, lang FROM _stage_synonyms\n"
        "ON CONFLICT (term, facet) DO NOTHING;\n\n"
    )

    # ── Self-check ──
    parts.append(_assertion_block(counts))
    parts.append("\nCOMMIT;\n")
    return "".join(parts)


def _assertion_block(counts: Dict[str, int]) -> str:
    """A DO block that raises if the load did not produce what we expected.

    The counts are baked in at generation time. Inside a container's
    initialisation an exception here aborts the whole startup, which is the
    behaviour we want: an obviously-broken database is far better than a
    quietly half-loaded one that answers questions with missing classes.
    """
    checks = "\n".join(
        f"""    SELECT count(*) INTO n FROM {_sql_ident(table)};
    IF n <> {expected} THEN
        RAISE EXCEPTION 'seed check failed: {table} has % row(s), expected {expected}', n;
    END IF;
"""
        for table, expected in sorted(counts.items())
    )
    return f"""\
-- ─── Self-check ────────────────────────────────────────────────────────────
-- Counts are baked in by the generator. If any of them is wrong the exception
-- aborts this transaction, and with it the container's initialisation, instead
-- of leaving a partially loaded corpus behind.

DO $$
DECLARE
    n bigint;
BEGIN
{checks}
    -- FR-16 / FR-17: these two columns must be empty for every row. The API
    -- contract is built on it, so it is asserted at load time rather than
    -- trusted.
    SELECT count(*) INTO n
    FROM class_records
    WHERE fee_rm IS NOT NULL OR schedule IS NOT NULL;
    IF n <> 0 THEN
        RAISE EXCEPTION
            'seed check failed: % row(s) have fee_rm or schedule set; FR-16/FR-17 require both to be NULL', n;
    END IF;

    -- Every class must resolve to a full set of dimension rows. A NULL here
    -- would mean a join silently dropped something.
    SELECT count(*) INTO n
    FROM class_records
    WHERE provider_id IS NULL OR curriculum_id IS NULL OR subject_id IS NULL
       OR level_id IS NULL OR delivery_mode_id IS NULL
       OR instruction_lang_id IS NULL;
    IF n <> 0 THEN
        RAISE EXCEPTION 'seed check failed: % class row(s) have an unresolved facet', n;
    END IF;

    -- education_stage is redundant with level; a disagreement would make
    -- stage filters silently wrong.
    SELECT count(*) INTO n
    FROM class_records cr
    JOIN levels l ON l.level_id = cr.level_id
    WHERE l.stage <> cr.education_stage;
    IF n <> 0 THEN
        RAISE EXCEPTION
            'seed check failed: % row(s) where education_stage disagrees with the level stage', n;
    END IF;

    RAISE NOTICE 'Class Bot seed loaded and verified.';
END $$;
"""


# ─── Entry point ────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        default="data/sample_corpus.csv",
        help="corpus CSV or XLSX to turn into SQL (default: data/sample_corpus.csv)",
    )
    parser.add_argument(
        "--synonyms",
        default="data/synonyms.json",
        help="synonym dictionary to mirror (default: data/synonyms.json)",
    )
    parser.add_argument(
        "--out",
        default=str(OUT_DIR),
        help=f"output directory (default: {OUT_DIR})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 1 if the files on disk differ from what would be generated",
    )
    args = parser.parse_args()

    source = Path(args.path)
    if not source.is_absolute():
        source = ROOT / source
    synonyms = Path(args.synonyms)
    if not synonyms.is_absolute():
        synonyms = ROOT / synonyms

    for label, path in (("corpus", source), ("synonyms", synonyms)):
        if not path.exists():
            print(f"{label} file not found: {path}", file=sys.stderr)
            return 1

    print(f"Reading {source} ...")
    rows, header = read_rows(source)

    report = IngestReport(source=str(source))
    report.rows_read = len(rows)
    accepted = validate(rows, header, report)

    if report.batch_rejected:
        print("\n=== BATCH REJECTED — no SQL written ===", file=sys.stderr)
        print(report.summary(), file=sys.stderr)
        return 1
    if not accepted:
        print("\nNo rows survived validation; no SQL written.", file=sys.stderr)
        print(report.summary(), file=sys.stderr)
        return 1

    report.rows_loaded = len(accepted)
    print(report.summary())
    if report.by_severity(HARD_WARNING):
        print(
            "\nV8 fired: fee_rm and/or schedule arrived populated. They are "
            "written as NULL to keep the FR-16/FR-17 contract intact.",
            file=sys.stderr,
        )

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    generated = {
        SCHEMA_FILE: SCHEMA_SQL,
        SEED_FILE: build_seed_sql(accepted, source, synonyms),
    }

    if args.check:
        stale = []
        for name, content in generated.items():
            target = out_dir / name
            if not target.exists() or target.read_text(encoding="utf-8") != content:
                stale.append(name)
        if stale:
            print(
                f"\nOut of date: {', '.join(stale)}\nRun: python scripts/gen_sql.py",
                file=sys.stderr,
            )
            return 1
        print("\nSQL files are up to date.")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, content in generated.items():
        (out_dir / name).write_text(content, encoding="utf-8")
        size = len(content.encode("utf-8"))
        print(f"  wrote {out_dir / name}  ({size:,} bytes)")

    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    print(
        f"\nGenerated at {stamp}. PostgreSQL loads these on first boot:\n"
        "  docker compose up -d\n"
        "To rebuild an already-initialised database from scratch:\n"
        "  docker compose down -v && docker compose up -d"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
