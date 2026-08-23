# Class Bot — Malaysia Tuition-Class Search API

Class Bot answers natural-language questions about Malaysian tuition classes from a
curated corpus of web-grounded example records. It exposes a single endpoint,
`POST /v1/chat`, that turns one message ("kelas Add Maths Form 4 online di
Selangor") into exactly one of six closed response states, with the matching
classes returned verbatim from the database and a narrative paragraph written by an
LLM only when the guardrails can verify every claim in it.

The design goal that shapes almost every file here is *honesty over helpfulness
under uncertainty*: the service would rather say "we don't hold prices" or return a
plain list with `degraded: true` than invent a fact. Fees and schedules are absent
from the corpus, so they are absent from every response, and that absence is
enforced structurally — a number in either field fails serialization at the API
boundary, after every other check.

This README explains how the code is organized, how to run and test it, and — in
the traceability matrix near the end — where each PRD requirement lives. For a
step-by-step run guide in Bahasa Indonesia, see **[RUNNING.md](RUNNING.md)**.


## Quick start

```bash
./setup.sh                          # venv + dependencies + .env, one command
source .venv/bin/activate
docker compose up -d                # PostgreSQL 16 on port 5433, schema + corpus preloaded
uvicorn app.main:app --reload --port 8000
```

Then open http://127.0.0.1:8000/docs, click **Authorize**, paste a key from your
`.env`, and try `POST /v1/chat`. To demo without a Google Cloud project, set
`LLM_PROVIDER=null` in `.env`: search still works, responses just come back as
`state: degraded` with no generated paragraph.


## Architecture

The code is organized into three layers, and the dependencies only ever point one
way: `api → core → infrastructure`. Nothing in `core` imports from `api`, and
nothing imports a router except the app factory. That rule is what keeps the
business logic — the part that decides what is honest to say — testable without a
web server or a database.

```
app/
  config.py                 cross-cutting settings (pydantic-settings, @lru_cache)
  main.py                   application factory: middleware order, handlers, routers
  api/                      the HTTP edge — nothing about classes lives here
    routers/chat.py         POST /v1/chat (thin: auth, delegate, shape errors)
    routers/health.py       GET /health (public), GET /metrics (auth)
    security.py             API-key auth, HTTPS enforcement, rate limiting
    errors.py               one error envelope for every failure (FR-4)
    middleware.py           trace id, security headers, body limit, metrics
  core/                     the decisions — pure of HTTP and, where possible, of I/O
    service.py              the six-state machine (the only place the state is chosen)
    query_understanding.py  message → filters, with the FR-9 closed-vocab gate
    synonyms.py             hot-reloadable synonym dictionary + fuzzy matcher (FR-7/8)
    retrieval.py            deterministic conjunctive SQL, bound parameters only (FR-11)
    honesty.py              post-hoc validator: grounding, fee/schedule silence (FR-16/17/18)
    narrative.py            prompt building + safe templates (FR-13/19/23/24)
    sessions.py             in-memory 30-minute session context (FR-25/28)
    schemas.py              the request/response contract itself (FR-1/2/3/12)
    vocabulary.py           the closed enums every filter is checked against
    ingest.py               CSV → validated rows → database
  infrastructure/           the outside world
    db.py, models.py        SQLAlchemy engine + ORM (relational only; no pgvector)
    observability.py        structured logs, metrics, cost + redaction (NFR-D/E/G)
    llm/                    provider interface + vertex.py and null.py implementations
scripts/                    init_db.py, load_data.py, gen_sql.py, run_eval.py
db/init/                    001_schema.sql + 002_seed_corpus.sql (generated; run by Docker)
tests/                      SQLite-backed contract suite + a Postgres schema-drift test
eval/eval_set.yaml          the FR-31 evaluation set
```


## The response contract

Every `POST /v1/chat` reply is the same JSON shape (`ChatResponse`), and clients
branch on `state` rather than parsing prose out of `answer`:

| state | when |
|---|---|
| `answered` | matches found, narrative passed validation |
| `needs_clarification` | too broad, or an out-of-vocabulary value was named |
| `no_result` | filters valid but nothing matched — carries relaxation options |
| `unavailable_field` | user asked for `fee_rm`/`schedule` — returned *with* the matching classes |
| `out_of_scope` | not a class-search question (includes distress, handled with care) |
| `degraded` | narrative failed validation or the LLM was unavailable — list intact |

`results[]`, `notice`, and `trace` (with `retrieved_ids[]` and `filters_applied`)
are present in every state. `fee_rm` and `schedule` are `null` on every result, in
every state.


## Testing

The suite runs on an in-memory SQLite database seeded from
`data/sample_corpus.csv`, with `LLM_PROVIDER=null`, so it needs no container, no
network, and no credentials:

```bash
pytest                              # whole suite
pytest tests/test_app.py           # endpoint contract tests (six states, auth, errors)
pytest tests/test_honesty.py       # the guardrails (FR-16/17/18)
pytest tests/test_sql_injection.py # bound-parameter / injection-resistance checks
```

`tests/test_schema_matches_orm.py` is the one test that needs PostgreSQL; it
reflects the live schema and compares it to the ORM, and **skips cleanly** when
Postgres is unreachable. Run it after `docker compose up -d`.

The 100% requirement on FR-16/17/18 is checked by measuring coverage of the
guardrail module specifically:

```bash
pytest --cov=app.core.honesty --cov-report=term-missing --cov-fail-under=100 \
       tests/test_honesty.py tests/test_app.py
```


## Evaluation (FR-31)

`scripts/run_eval.py` drives the real app in-process through `TestClient` and prints
the M1–M9 table. Field accuracy and filter precision are checked by re-reading each
returned record straight from the database with a parameterized query, so the
response is graded against the data, not against itself.

```bash
python scripts/run_eval.py                 # against the configured LLM
python scripts/run_eval.py --llm null      # retrieval metrics only, no tokens spent
python scripts/run_eval.py --json out.json --md out.md
```

| metric | what it measures | target |
|---|---|---|
| M1 | Groundedness — output values ⊆ retrieved record (incl. FR-16/17 silence, FR-22 no ranking) | 100% |
| M2 | Field accuracy vs database | ≥ 0.90 |
| M3 | Recall@5 | ≥ 0.95 |
| M4 | Filter precision on trap queries (one leaked row fails the case) | 100% |
| M5 | Unanswerable-field handling | 100% |
| M6 | Disclaimer (`notice`) presence | 100% |
| M7 | Schema validity | ≥ 0.90 |
| M8 | Multi-turn constraint retention (via `trace.filters_applied`) | 100% |
| M9 | Latency p95, client-side in-process | ≤ 3000 ms |

Cost per query is reported alongside as NFR-E (target ≤ RM0.05) whenever the LLM
runs.


## Requirement traceability (FR-1 … FR-31)

Every requirement is annotated in the source with its `FR-n` tag; grep for the tag
to see the acceptance-criterion notes inline. Primary implementation sites:

| FR | Requirement | Primary location |
|---|---|---|
| FR-1 | Accept `message` + optional `session_id` | `core/schemas.py` |
| FR-2 | Closed set of six response states | `core/service.py`, `core/schemas.py` |
| FR-3 | `/v1` additive; breaking change ⇒ `/v2` | `core/schemas.py`, `config.py` |
| FR-4 | One error envelope; never HTML/empty; 503 on timeout | `api/errors.py`, `main.py` |
| FR-5 | `results[]` come verbatim from DB records | `core/service.py`, `api/routers/chat.py` |
| FR-6 | Parse facets from natural language | `core/query_understanding.py` |
| FR-7 | Synonym / variant resolution | `core/synonyms.py` |
| FR-8 | Typo tolerance, stated not silently applied | `core/synonyms.py`, `core/query_understanding.py` |
| FR-9 | Never guess out-of-vocabulary values (closed-vocab gate) | `core/query_understanding.py`, `core/vocabulary.py` |
| FR-10 | Too broad ⇒ one clarifying question with real values | `core/service.py`, `core/retrieval.py` |
| FR-11 | Deterministic conjunctive SQL retrieval | `core/retrieval.py`, `infrastructure/models.py` |
| FR-12 | Eleven mandatory result fields, non-empty | `core/schemas.py` |
| FR-13 | `match_reason` from record facts only | `core/narrative.py`, `core/honesty.py` |
| FR-14 | `verified_on` from the record, not "today" | `core/schemas.py`, `core/retrieval.py` |
| FR-15 | `no_result` ⇒ relaxation options with real counts | `core/retrieval.py`, `core/service.py` |
| FR-16 | `fee_rm` always `null` (structural) | `core/schemas.py`, `core/honesty.py` |
| FR-17 | `schedule` always `null` (structural) | `core/schemas.py`, `core/honesty.py` |
| FR-18 | Every claim in `answer` grounded in `retrieved_ids` | `core/honesty.py`, `core/service.py` |
| FR-19 | Failed validation ⇒ `degraded` + safe template | `core/honesty.py`, `core/service.py` |
| FR-20 | Asked-for missing field ⇒ `unavailable_field` with results | `core/service.py`, `core/narrative.py` |
| FR-21 | Non-empty `notice` on every response | `core/honesty.py`, `core/schemas.py` |
| FR-22 | No ranking or quality claims ("best") | `core/honesty.py`, `core/narrative.py` |
| FR-23 | Out of scope ⇒ polite + `suggested_questions` | `core/service.py`, `core/narrative.py` |
| FR-24 | Distress ⇒ empathy + trusted adult, no counseling | `core/narrative.py`, `core/service.py` |
| FR-25 | Multi-turn context merge | `core/sessions.py`, `core/service.py` |
| FR-26 | Session lifecycle | `core/sessions.py`, `api/middleware.py` |
| FR-27 | Subject change resets carried filters | `core/sessions.py`, `core/service.py` |
| FR-28 | 30-minute in-memory session via `session_id` | `core/sessions.py`, `config.py` |
| FR-29 | `trace` always has `retrieved_ids` + `filters_applied` | `core/service.py`, `core/schemas.py` |
| FR-30 | Query log: query, model output, failed checks; no identity | `infrastructure/observability.py`, `infrastructure/models.py` |
| FR-31 | Eval CLI reporting M1–M9 | `scripts/run_eval.py` |

Non-functional requirements:

| NFR | Requirement | Primary location |
|---|---|---|
| NFR-A | Latency / timeout budgets (LLM 8 s; p95/p99) | `api/errors.py`, `config.py`, `scripts/run_eval.py` |
| NFR-B | Relational storage only; ≤ 20 results; 30-min sessions | `infrastructure/models.py`, `config.py` |
| NFR-C | Degrade without the LLM; filter search survives; rate limit | `infrastructure/llm/null.py`, `core/service.py`, `api/security.py` |
| NFR-D | Per-consumer keys, HTTPS, no identity in logs, prompt-injection defense | `api/security.py`, `infrastructure/observability.py` |
| NFR-E | Cost per query tracked, ≤ RM0.05 | `infrastructure/observability.py` |
| NFR-F | Versioned path + OpenAPI ready for integration | `main.py`, `api/security.py` |
| NFR-G | Structured logging + metrics | `infrastructure/observability.py` |
| NFR-H | Model and synonyms swappable via config, no redeploy | `infrastructure/llm/__init__.py`, `core/synonyms.py`, `config.py` |


## Security notes

Every runtime query goes through SQLAlchemy bound parameters — user text never
becomes SQL. It also has to survive the FR-9 closed-vocabulary gate before it can
influence a query at all, and `tests/test_sql_injection.py` asserts both. The
one-time seed SQL is generated by `scripts/gen_sql.py`, which streams corpus values
through `COPY ... FROM stdin` (a data stream Postgres never parses as SQL) rather
than interpolating them.

Odoo and the front end must hold **different** API keys (NFR-D); the service refuses
to boot in production with sample keys, a single shared key, or HTTPS disabled.
Query logs carry the message and model output for diagnosis but no user identity,
and `session_id` is a random token that must never be derived from personal data.
User input is wrapped as data — never instructions — when passed to the model, and
provider source URLs are constrained to `https://` at the database layer.


## Known PRD-vs-DD inconsistencies and how they are resolved

A few places in the PRD and Data Dictionary pull in slightly different directions.
Each is resolved deliberately, and noted here so the choice is visible:

The **`sessions` entity** in the DD is modeled as a table, but PRD NFR-B specifies
"30 minutes, in-memory, not persisted." The in-memory store in `core/sessions.py`
follows the PRD; no session table is created.

**LLM timeout status** — FR-4 names `503` for a timeout, while NFR-C would rather
return the correct class list with `degraded: true` than discard it because the
paragraph is missing. The default follows NFR-C (degrade, keep the list);
`LLM_TIMEOUT_RETURNS_503=true` switches to the literal FR-4 behavior for callers who
prefer it.

The **real 200-record corpus** (`data/malaysia_tuition_classes_rag_200.csv`) is not
in the repository — only the 12-row `data/sample_corpus.csv`, which has the same 23
columns and covers every facet group. `CORPUS_CSV` in `.env.example` points at the
200-row file by default; until it is supplied, point it at the sample, or leave
Docker's preloaded sample in place.
