# `db/` — schema and seed data

`db/init/` is mounted into the PostgreSQL container at
`/docker-entrypoint-initdb.d/`. The postgres image runs everything in that
directory, in filename order, **once** — the first time the data volume is
created. So this is all it takes to get a working database:

```bash
docker compose up -d
```

No `pip install`, no Python, no loader script. The container comes up with the
schema, the closed vocabulary, the synonym mirror and the class corpus already
in place.

| File | What it does |
| --- | --- |
| `init/001_schema.sql` | Tables, foreign keys, check constraints, indexes. Mirrors `app/models.py`. |
| `init/002_seed_corpus.sql` | Closed vocabulary from `app/vocabulary.py`, providers and classes from the corpus CSV, synonyms from `data/synonyms.json`. |

Both files are **generated**. Do not edit them:

```bash
python scripts/gen_sql.py                          # regenerate from data/sample_corpus.csv
python scripts/gen_sql.py data/real_corpus.csv     # or from the real 200-record corpus
python scripts/gen_sql.py --check                  # CI: fail if the files are stale
```

The generator deliberately depends on nothing outside the standard library,
because it has to be runnable before the project's dependencies are installed.

To rebuild an already-initialised database:

```bash
docker compose down -v && docker compose up -d
```

`down -v` is the important part — without it the volume survives and the init
scripts are skipped. To load a corpus into a *running* database instead, use
`python scripts/load_data.py path/to/corpus.csv`, which goes through the same
V1–V8 validation via the ORM.

## The seed refuses to load half a corpus

`002_seed_corpus.sql` runs inside a single transaction and ends with a `DO`
block that re-counts every table and compares against numbers baked in at
generation time. It also asserts the invariants the API contract depends on:

- `fee_rm` and `schedule` are NULL for every row (FR-16, FR-17)
- no class row has an unresolved facet — i.e. no join silently dropped a record
- `education_stage` agrees with the stage on the row's level

If any check fails, the exception aborts the transaction and with it the
container's initialisation. You get an obviously broken database rather than a
quietly incomplete one that answers questions with classes missing. That
tradeoff is deliberate: a wrong answer delivered confidently is the failure mode
this whole product is built to avoid, and it starts here.

## Why there is no `sessions` table

The Data Dictionary models a `SESSION` entity, but PRD NFR-B caps session
retention at "30 menit, in-memory, tidak persisten" and NG2 rules out
cross-session history. The two documents disagree; the PRD wins, so session
context lives in `app/sessions.py` as an in-memory store with a TTL. See the
README section "Catatan PRD vs DD".

There is also no `vector` extension. NFR-B: "belum menggunakan vector DB di v1".
Retrieval in v1 is deterministic SQL filtering over closed-vocabulary facets.

## SQL injection: what actually protects this service

Three separate mechanisms, and the third is the one that matters most.

**1. The application never builds SQL from strings.** Every runtime query goes
through SQLAlchemy's expression language, which sends values as bound
parameters — the driver transmits them out-of-band from the statement text, so
there is no parsing step for a payload to escape from. The single use of
`text()` in the codebase is `text("SELECT 1")`, a constant, in the health check.
`tests/test_sql_injection.py` enforces this with an AST scan over `app/`, so a
future f-string in a query fails the build rather than shipping.

**2. The generated seed carries data in `COPY`, not in `INSERT` literals.** No
corpus value is interpolated into a statement. Everything travels through
`COPY … FROM stdin WITH (FORMAT csv)`, which PostgreSQL reads as a data stream
and never parses as SQL — the same mechanism `pg_dump` uses. A cell containing
`'); DROP TABLE class_records; --` arrives in the column as a 34-character
string. The test suite proves this by generating SQL from a deliberately hostile
corpus and asserting the payload never appears outside a `COPY` block.

**3. User text cannot reach a query at all unless it is already vocabulary.**
This is the real guarantee, and it is stronger than escaping. FR-9 requires
every value parsed out of a message to be checked against the closed lists in
`app/vocabulary.py` *before* it reaches the query layer. A filter can therefore
only ever hold one of 22 subjects, 17 levels, 7 states, 11 delivery modes, 6
instruction languages, 5 curricula or 2 education stages — values that come from
our own code, not from the request. `'; DROP TABLE …` is not a subject, so it
does not become a filter that needs escaping; it becomes
`state: needs_clarification` and the user is asked what they meant.

Escaping protects you when hostile input reaches the SQL layer. Design point 3
means it does not get there.

Two related habits, for the same reason: user text is passed to the LLM as data
and never as instructions (`sanitize_user_text` strips control characters and
prompt-delimiter sequences), and model output is never executed — it is
validated against the retrieved records and discarded if it does not ground.
