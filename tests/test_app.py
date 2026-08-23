"""End-to-end contract tests for `POST /v1/chat` and the ops endpoints.

These exercise the real application through `TestClient` — every middleware,
the auth dependency, the exception handlers, and the full service state machine —
with only the database swapped for the seeded SQLite fixture (see `conftest.py`).
The LLM provider is `null`, so answered searches surface as `degraded`: that is
NFR-C's contract, "filter search keeps working without the model", and asserting
it here is the point rather than a limitation.

What each requirement pins down:

* FR-2  — every response carries one of the six `state` values, in one envelope.
* FR-4  — failures (401/404/405/422/429) come back as `ErrorResponse`, never as
          HTML or `{"detail": ...}`.
* FR-5  — `results[]` come from the database; ids echo into `trace.retrieved_ids`.
* FR-12 — the eleven mandatory fields are present and non-empty on every result.
* FR-16/17 — `fee_rm` and `schedule` are `null` on every result, in every state.
* FR-20 — a fee/schedule question yields `unavailable_field` *with* the classes.
* FR-21 — `notice` is always non-empty.
* FR-23 — a non-class message yields `out_of_scope` with `suggested_questions`.
* FR-29 — `trace.retrieved_ids` and `trace.filters_applied` are always present.
* NFR-C — no model → `degraded`, list intact.
* NFR-D — missing/invalid key → 401; 60/min rate limit → 429.
"""

from __future__ import annotations

import re

import pytest

from conftest import TEST_API_KEY

from app.core.vocabulary import CROSS_REGION_STATES

RECORD_ID = re.compile(r"^MYTC-\d{3}$")

# One class that certainly exists in the fixture (MYTC-001).
ANSWERABLE = "Additional Mathematics Form 4 Selangor"


# ─── Envelope invariants (FR-2, FR-21, FR-29) ────────────────────────────────


def _assert_envelope(body: dict) -> None:
    """Every /v1/chat body — whatever the state — has this exact shape."""
    for key in (
        "schema_version",
        "state",
        "session_id",
        "answer",
        "results",
        "notice",
        "unavailable_fields",
        "relaxation_options",
        "missing_field",
        "available_values",
        "suggested_questions",
        "degraded",
        "trace",
    ):
        assert key in body, f"missing top-level field: {key}"

    assert body["schema_version"] == "v1"
    assert body["state"] in {
        "answered",
        "needs_clarification",
        "no_result",
        "unavailable_field",
        "out_of_scope",
        "degraded",
    }
    # FR-21: the disclaimer is never empty.
    assert isinstance(body["notice"], str) and body["notice"].strip()
    # FR-29: the trace is always traceable back to data and filters.
    trace = body["trace"]
    assert "retrieved_ids" in trace and "filters_applied" in trace
    assert trace["trace_id"]


def _assert_result_shape(result: dict) -> None:
    """FR-12 mandatory eleven present + FR-16/17 always-null fields."""
    for key in (
        "record_id",
        "class_title",
        "provider_name",
        "level",
        "subject",
        "delivery_mode",
        "instruction_language",
        "state",
        "city_area",
        "provider_source_url",
        "verified_on",
    ):
        assert result.get(key), f"mandatory field empty or missing: {key}"

    assert RECORD_ID.match(result["record_id"])
    # V3 / FR-16 / FR-17: provider link is https, fee and schedule are null.
    assert result["provider_source_url"].startswith("https://")
    assert result["fee_rm"] is None
    assert result["schedule"] is None


# ─── /health ─────────────────────────────────────────────────────────────────


def test_health_is_public_and_reports_corpus(client):
    """/health needs no key and reports the seeded corpus + provider state."""
    client.headers.pop("X-API-Key", None)
    resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["corpus_records"] == 12
    assert body["llm_provider"] == "null"
    assert body["llm_ready"] is False
    # 12 rows are present but the null provider is not ready → degraded, not ok.
    assert body["status"] == "degraded"
    assert body["schema_version"] == "v1"


# ─── Authentication (NFR-D, FR-4) ────────────────────────────────────────────


def test_missing_api_key_is_json_401(client):
    client.headers.pop("X-API-Key", None)
    resp = client.post("/v1/chat", json={"message": ANSWERABLE})

    assert resp.status_code == 401
    assert resp.headers["content-type"].startswith("application/json")
    body = resp.json()
    # FR-4: our envelope, never FastAPI's {"detail": ...}.
    assert "detail" not in body
    assert body["schema_version"] == "v1"
    assert body["error"]["code"] == "unauthorized"
    assert body["error"]["message"]


def test_invalid_api_key_is_401(client):
    client.headers["X-API-Key"] = "not-a-real-key"
    resp = client.post("/v1/chat", json={"message": ANSWERABLE})

    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


def test_valid_api_key_is_accepted(client):
    # The fixture presets the header; a plain call must authenticate.
    resp = client.post("/v1/chat", json={"message": ANSWERABLE})
    assert resp.status_code == 200


# ─── answered → degraded, with a full, honest list (NFR-C, FR-5, FR-16/17) ───


def test_answerable_query_degrades_but_keeps_results(client):
    resp = client.post("/v1/chat", json={"message": ANSWERABLE})

    assert resp.status_code == 200
    body = resp.json()
    _assert_envelope(body)

    # NFR-C: the model is off, so the narrative is templated and degraded is true,
    # but the retrieval result is the product and it survives intact.
    assert body["state"] == "degraded"
    assert body["degraded"] is True
    assert "llm_disabled" in body["trace"]["validation_failed"]

    assert body["results"], "a matching query must return classes"
    ids = [r["record_id"] for r in body["results"]]
    assert "MYTC-001" in ids
    for result in body["results"]:
        _assert_result_shape(result)

    # FR-5 + FR-29: the ids in the list are exactly the ids in the trace.
    assert body["trace"]["retrieved_ids"] == ids
    # The parsed filters made it into the trace (FR-29).
    applied = body["trace"]["filters_applied"]
    assert applied.get("subject") == "Additional Mathematics"
    assert applied.get("level") == "Form 4"
    # DD open-question D3: a specific-state search widens to also include the
    # cross-region providers ("Malaysia" / "Multi-state") when
    # INCLUDE_CROSS_REGION_STATES is on (the default). The trace deliberately
    # shows the widened list rather than hiding it, so filters_applied.state is a
    # list of Selangor *plus* the cross-region states, not a bare "Selangor".
    applied_state = applied.get("state")
    assert isinstance(applied_state, list)
    assert set(applied_state) == {"Selangor", *CROSS_REGION_STATES}


# ─── unavailable_field: the question we cannot answer, with the ones we can ──


def test_fee_question_is_unavailable_field_with_results(client):
    resp = client.post(
        "/v1/chat",
        json={"message": "Berapa harga kelas Additional Mathematics Form 4 Selangor?"},
    )

    assert resp.status_code == 200
    body = resp.json()
    _assert_envelope(body)

    # FR-20: the field is named, the classes are still returned.
    assert body["state"] == "unavailable_field"
    assert "fee_rm" in body["unavailable_fields"]
    assert body["results"], "unavailable_field must still carry the matching classes"
    for result in body["results"]:
        _assert_result_shape(result)


def test_schedule_question_is_unavailable_field(client):
    resp = client.post(
        "/v1/chat",
        json={"message": "Kelas Additional Mathematics Form 4 Selangor jadual pukul berapa?"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "unavailable_field"
    assert "schedule" in body["unavailable_fields"]


# ─── out_of_scope (FR-23) ────────────────────────────────────────────────────


def test_non_class_message_is_out_of_scope(client):
    resp = client.post(
        "/v1/chat", json={"message": "What is the capital of France?"}
    )

    assert resp.status_code == 200
    body = resp.json()
    _assert_envelope(body)
    assert body["state"] == "out_of_scope"
    # FR-23: show what this service *can* answer.
    assert body["suggested_questions"]
    assert body["results"] == []


# ─── needs_clarification (FR-10) ─────────────────────────────────────────────


def test_broad_query_asks_one_question(client):
    resp = client.post("/v1/chat", json={"message": "Saya nak cari kelas tuisyen"})

    assert resp.status_code == 200
    body = resp.json()
    _assert_envelope(body)
    assert body["state"] == "needs_clarification"
    # FR-10: exactly one field, highest priority first (subject), with real values.
    assert body["missing_field"] == "subject"
    assert body["available_values"]


# ─── Request validation (FR-1, FR-4) ─────────────────────────────────────────


def test_empty_message_is_422_envelope(client):
    resp = client.post("/v1/chat", json={"message": ""})

    assert resp.status_code == 422
    body = resp.json()
    assert "detail" not in body
    assert body["schema_version"] == "v1"
    assert body["error"]["code"] == "validation_error"


def test_unknown_field_is_rejected(client):
    # ChatRequest is extra="forbid": a stray field is a client bug worth surfacing.
    resp = client.post(
        "/v1/chat", json={"message": ANSWERABLE, "surprise": "value"}
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_bad_session_id_pattern_is_422(client):
    resp = client.post(
        "/v1/chat", json={"message": ANSWERABLE, "session_id": "has spaces!"}
    )
    assert resp.status_code == 422


# ─── Routing errors come back shaped (FR-4) ──────────────────────────────────


def test_unknown_route_is_json_not_found(client):
    resp = client.get("/v1/does-not-exist")
    assert resp.status_code == 404
    assert resp.headers["content-type"].startswith("application/json")
    body = resp.json()
    assert "detail" not in body
    assert body["error"]["code"] == "not_found"


def test_method_not_allowed_is_shaped(client):
    # /v1/chat is POST-only; a GET must not leak Starlette's default text body.
    resp = client.get("/v1/chat")
    assert resp.status_code == 405
    body = resp.json()
    assert body["error"]["code"] == "method_not_allowed"


# ─── Rate limiting (NFR-C) ───────────────────────────────────────────────────


def test_rate_limit_returns_429_with_retry_after(client):
    """Force the per-key limit down to 1 and prove the second call is refused.

    The suite as a whole runs with the limit effectively disabled (conftest sets
    a very high ceiling); this test owns the one scenario that needs it low.
    """
    from app.api import security

    limiter = security.get_limiter()
    limiter.limit = 1

    first = client.post("/v1/chat", json={"message": ANSWERABLE})
    assert first.status_code == 200

    second = client.post("/v1/chat", json={"message": ANSWERABLE})
    assert second.status_code == 429
    body = second.json()
    assert body["error"]["code"] == "rate_limited"
    assert "Retry-After" in second.headers


# ─── Session continuity (FR-25/28) ───────────────────────────────────────────


def test_session_id_is_echoed(client):
    resp = client.post(
        "/v1/chat", json={"message": ANSWERABLE, "session_id": "s_test_001"}
    )
    assert resp.status_code == 200
    assert resp.json()["session_id"] == "s_test_001"


def test_missing_session_id_is_generated(client):
    resp = client.post("/v1/chat", json={"message": ANSWERABLE})
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]
    assert session_id  # non-empty, server-minted
    # NFR-D: it is a random token, not derived from anything in the request.
    assert " " not in session_id


def test_context_carries_across_turns(client):
    """FR-25/26: a follow-up that only narrows still resolves against the carried
    subject, so a bare 'Form 5' after an Additional Mathematics turn stays on
    topic rather than resetting to a broad search."""
    first = client.post(
        "/v1/chat",
        json={"message": "Additional Mathematics Selangor", "session_id": "s_ctx_1"},
    )
    assert first.status_code == 200

    second = client.post(
        "/v1/chat", json={"message": "Form 5 sahaja", "session_id": "s_ctx_1"}
    )
    assert second.status_code == 200
    body = second.json()
    _assert_envelope(body)
    # The subject carried from turn one is still in force.
    applied = body["trace"]["filters_applied"]
    assert applied.get("subject") == "Additional Mathematics"


# ─── FR-16/17 across every state that returns rows ───────────────────────────


@pytest.mark.parametrize(
    "message",
    [
        ANSWERABLE,
        "Berapa harga kelas Additional Mathematics Form 4 Selangor?",
        "Physics Form 5 Johor",
        "kelas Mathematics Form 4 Penang online",
    ],
)
def test_fee_and_schedule_never_populated(client, message):
    """FR-16/17 are structural: no reachable path puts a value in either field."""
    resp = client.post("/v1/chat", json={"message": message})
    assert resp.status_code == 200
    for result in resp.json()["results"]:
        assert result["fee_rm"] is None
        assert result["schedule"] is None
