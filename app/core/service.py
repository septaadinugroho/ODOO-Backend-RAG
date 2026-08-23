"""The state machine behind `POST /v1/chat`.

Everything above this file is a mechanism — understanding, retrieval, guardrails,
sessions, the model. This is where one message becomes exactly one of the six
states in FR-2, and it is deliberately the only place that decision is made.

Order of decisions, and why each comes where it does:

1. **Distress (FR-24, Must).** Checked before anything else, because it is the one
   case where continuing to be a search engine is the wrong answer. Returns
   `out_of_scope` — the PRD files FR-24 under scenario 11 — with a reply that
   points to a trusted adult and does not counsel.
2. **Out of scope (FR-23).** No class signal, nothing filterable, no carried
   context: answer politely and show what this service *can* answer.
3. **Unresolved vocabulary (FR-9).** An unrecognised value is asked about, never
   guessed at, and never sent to SQL.
4. **Session merge (FR-25/26/27).** Carried filters join this turn's; a changed
   subject wipes the slate.
5. **Too broad (FR-10).** One question, the highest-impact missing field, with the
   values that actually exist under the current filters.
6. **Retrieve (FR-11).** Deterministic conjunctive SQL, bound parameters only.
7. **Nothing matched (FR-15).** `no_result` plus per-filter relaxation offers with
   real counts.
8. **Asked for a field we do not hold (FR-20).** `unavailable_field`, with the
   matching classes still in `results[]` — "we don't know the price, here are the
   four classes and their provider links" is the useful shape of that answer.
9. **Answered.** The model writes the paragraph; `honesty.enforce` decides whether
   the user ever sees it (FR-18/19). `results[]` is built from database rows in
   every branch (FR-5) and never from generated text.

The invariant worth stating plainly: a failure anywhere in step 9 costs the user
the prose and nothing else. The list, the notice, the trace, and the session
context all survive, which is what makes `degraded` a usable state rather than an
apology.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from sqlalchemy.orm import Session

from app.core import narrative, retrieval
from app.config import SCHEMA_VERSION, get_settings
from app.core.honesty import enforce, notice, safe_answer
from app.infrastructure.llm import get_provider
from app.infrastructure.llm.base import LLMResult
from app.infrastructure.observability import estimate_cost_rm, get_metrics, redact
from app.core.query_understanding import MISSING_FIELD_PRIORITY, Understanding, analyze
from app.core.retrieval import ClassRow, Retrieval
from app.core.schemas import (
    ChatRequest,
    ChatResponse,
    ChatState,
    ClassResult,
    RelaxationOption,
    Trace,
)
from app.core.sessions import SessionState, get_store

logger = logging.getLogger(__name__)
degraded_logger = logging.getLogger("app.degraded")

# Provider names change only when the corpus is reloaded, and the list is read on
# every answered turn for the FR-18 check. Cached per process with a short TTL:
# long enough to save a query per request, short enough that a reseed is picked up
# without a restart.
_PROVIDER_CACHE_TTL = 300.0
_provider_names: Tuple[float, Tuple[str, ...]] = (0.0, ())


def _known_providers(session: Session) -> Tuple[str, ...]:
    global _provider_names
    cached_at, names = _provider_names
    now = time.monotonic()
    if names and (now - cached_at) < _PROVIDER_CACHE_TTL:
        return names
    fresh = tuple(retrieval.known_providers(session))
    _provider_names = (now, fresh)
    return fresh


def reset_provider_cache() -> None:
    global _provider_names
    _provider_names = (0.0, ())


@dataclass
class Turn:
    """The response plus the telemetry the router needs for NFR-E and NFR-G."""

    response: ChatResponse
    llm_tokens: int = 0
    llm_cost_rm: float = 0.0
    llm_used: bool = False
    validation_failed: List[str] = field(default_factory=list)


# ─── Entry point ────────────────────────────────────────────────────────────


def handle_chat(
    request: ChatRequest,
    db: Session,
    trace_id: str,
    consumer: str = "unknown",
) -> Turn:
    started = time.perf_counter()
    settings = get_settings()
    store = get_store()

    message = request.message
    understanding = analyze(message)
    session, is_new = store.get_or_create(request.session_id)
    lang = narrative.detect_language(message)

    ctx = _Context(
        db=db,
        trace_id=trace_id,
        consumer=consumer,
        message=message,
        understanding=understanding,
        session=session,
        lang=lang,
        started=started,
        settings=settings,
    )

    logger.debug(
        "turn accepted",
        extra={
            "trace_id": trace_id,
            "session_id": session.session_id,
            "new_session": is_new,
            "turn": session.turns + 1,
            "lang": lang,
            "consumer": consumer,
        },
    )

    # 1. FR-24 — before the search machinery gets a say.
    if understanding.shows_distress:
        return ctx.finish(
            state=ChatState.OUT_OF_SCOPE,
            answer=narrative.distress_answer(lang),
            suggested=narrative.suggested_questions(lang),
        )

    # 2. FR-23 — nothing about classes, and no context to lean on.
    carried_available = bool(session.filters)
    if not understanding.is_searchable and not carried_available:
        if not understanding.has_class_signal and not understanding.asked_unanswerable_question:
            return ctx.finish(
                state=ChatState.OUT_OF_SCOPE,
                answer=narrative.out_of_scope_answer(lang),
                suggested=narrative.suggested_questions(lang),
            )

    # 3. FR-9 — refuse to guess an out-of-vocabulary value.
    if understanding.ambiguities:
        ambiguity = understanding.ambiguities[0]
        return ctx.finish(
            state=ChatState.NEEDS_CLARIFICATION,
            answer=narrative.ambiguity_answer(
                ambiguity.fragment, ambiguity.candidates, lang
            ),
            missing_field=ambiguity.facet,
            available_values=list(ambiguity.candidates),
        )

    # 4. FR-25/26/27.
    effective, reset_fired = session.merge(dict(understanding.filters))
    if reset_fired:
        logger.info(
            "session context reset on subject change (FR-27)",
            extra={"trace_id": trace_id, "session_id": session.session_id},
        )

    # 5. FR-10 — one question at a time, only when the answer set is unusable.
    clarification = _maybe_clarify(ctx, effective)
    if clarification is not None:
        return clarification

    # 6. FR-11.
    result = retrieval.search(db, effective, limit=settings.max_results)
    ctx.retrieval = result

    # 7. FR-15.
    if result.is_empty:
        offers = result.relaxation_options
        return ctx.finish(
            state=ChatState.NO_RESULT,
            answer=narrative.no_result_answer(offers, effective, lang),
            relaxation=offers,
            unavailable=understanding.unavailable_fields,
        )

    rows = result.rows
    results = _to_results(rows, understanding, lang)
    session.remember_results([r.record_id for r in rows], "answered", lang)

    # 8. FR-20 — the question we cannot answer, with the classes we can.
    if understanding.asked_unanswerable_question:
        fields = understanding.unavailable_fields
        return ctx.finish(
            state=ChatState.UNAVAILABLE_FIELD,
            answer=narrative.unavailable_field_answer(fields, rows, lang),
            results=results,
            unavailable=fields,
        )

    # 9. Narrate, then let the guardrails decide what ships.
    return _answer(ctx, rows, results, effective)


# ─── Steps ──────────────────────────────────────────────────────────────────


@dataclass
class _Context:
    """Per-turn state, so the step functions stay short and testable."""

    db: Session
    trace_id: str
    consumer: str
    message: str
    understanding: Understanding
    session: SessionState
    lang: str
    started: float
    settings: object
    retrieval: Optional[Retrieval] = None
    llm: Optional[LLMResult] = None

    @property
    def filters_applied(self) -> Dict[str, object]:
        if self.retrieval is not None:
            return dict(self.retrieval.filters_applied)
        return dict(self.session.filters)

    def latency_ms(self) -> int:
        return int((time.perf_counter() - self.started) * 1000)

    def finish(
        self,
        state: ChatState,
        answer: str,
        results: Optional[List[ClassResult]] = None,
        relaxation: Sequence[object] = (),
        unavailable: Sequence[str] = (),
        missing_field: Optional[str] = None,
        available_values: Sequence[str] = (),
        suggested: Sequence[str] = (),
        degraded: bool = False,
        validation_failed: Sequence[str] = (),
    ) -> Turn:
        """Assemble the response. Every state comes through here.

        Centralised so the invariants hold everywhere without repetition: FR-21's
        notice is always present, FR-29's trace always carries `retrieved_ids` and
        `filters_applied`, and `session_id` is always echoed so a client can keep
        the thread.
        """
        results = results or []
        response = ChatResponse(
            schema_version=SCHEMA_VERSION,
            state=state,
            session_id=self.session.session_id,
            answer=answer,
            results=results,
            notice=notice(self.lang),
            unavailable_fields=list(unavailable),
            relaxation_options=[
                RelaxationOption(
                    field=o.field,
                    current_value=o.current_value,
                    result_count=o.result_count,
                    suggested_values=list(o.suggested_values),
                )
                for o in relaxation
            ],
            missing_field=missing_field,
            available_values=list(available_values),
            suggested_questions=list(suggested),
            degraded=degraded,
            trace=Trace(
                trace_id=self.trace_id,
                retrieved_ids=[r.record_id for r in results],
                filters_applied=self.filters_applied,
                candidate_count=(
                    self.retrieval.candidate_count if self.retrieval else 0
                ),
                llm_used=self.llm is not None,
                latency_ms=self.latency_ms(),
                validation_failed=list(validation_failed),
            ),
        )
        self.session.remember_results(
            [r.record_id for r in results], state.value, self.lang
        )
        tokens = self.llm.total_tokens if self.llm else 0
        cost = (
            estimate_cost_rm(self.llm.prompt_tokens, self.llm.completion_tokens)
            if self.llm
            else 0.0
        )
        return Turn(
            response=response,
            llm_tokens=tokens,
            llm_cost_rm=cost,
            llm_used=self.llm is not None,
            validation_failed=list(validation_failed),
        )


def _maybe_clarify(ctx: _Context, effective: Dict[str, object]) -> Optional[Turn]:
    """FR-10: ask one question when the result set would be unusable.

    Two triggers. Nothing to filter on at all, and a candidate set larger than
    `clarify_threshold`. The threshold is a product judgement, not a technical
    one: twenty classes is a list a parent can scan, two hundred is a wall.

    If nothing is left to ask — every priority field is already set and the count
    is still high — the search proceeds. Asking about a field the user has already
    given would be the worst of both, and `max_results` bounds the payload anyway.
    """
    missing = [f for f in MISSING_FIELD_PRIORITY if f not in effective]
    candidate_count = retrieval.count_candidates(ctx.db, effective)

    if effective and candidate_count <= ctx.settings.clarify_threshold:
        return None
    if not missing:
        return None

    field_name = missing[0]
    values = retrieval.distinct_values(
        ctx.db, field_name, effective, limit=retrieval.MAX_SUGGESTED_VALUES * 2
    )
    if not values:
        # The field cannot narrow anything under these filters — asking about it
        # would be a dead end. Try the next one before giving up on clarifying.
        for candidate in missing[1:]:
            values = retrieval.distinct_values(
                ctx.db, candidate, effective, limit=retrieval.MAX_SUGGESTED_VALUES * 2
            )
            if values:
                field_name = candidate
                break
        else:
            return None

    return ctx.finish(
        state=ChatState.NEEDS_CLARIFICATION,
        answer=narrative.clarification_answer(
            field_name, values, candidate_count, ctx.lang
        ),
        missing_field=field_name,
        available_values=values,
        unavailable=ctx.understanding.unavailable_fields,
    )


def _to_results(
    rows: Sequence[ClassRow],
    understanding: Understanding,
    lang: str,
) -> List[ClassResult]:
    """Database rows to response items (FR-5, FR-12, FR-16, FR-17).

    Field-for-field from the row. `fee_rm` and `schedule` are not passed at all —
    their annotation is literally `None`, so the schema itself refuses a value, and
    leaving them to their default is the one construction that cannot regress.
    `match_reason` is built from the row's own facts (FR-13), never generated.
    """
    out: List[ClassResult] = []
    for row in rows:
        out.append(
            ClassResult(
                record_id=row.record_id,
                class_title=row.class_title,
                provider_name=row.provider_name,
                level=row.level,
                subject=row.subject,
                delivery_mode=row.delivery_mode,
                instruction_language=row.instruction_language,
                state=row.state,
                city_area=row.city_area,
                provider_source_url=row.provider_source_url,
                verified_on=row.verified_on,
                curriculum=row.curriculum,
                education_stage=row.education_stage,
                school_category=row.school_category,
                match_reason=narrative.build_match_reason(row, understanding, lang),
            )
        )
    return out


def _answer(
    ctx: _Context,
    rows: Sequence[ClassRow],
    results: List[ClassResult],
    effective: Dict[str, object],
) -> Turn:
    """Generate the paragraph, validate it, ship whichever version is honest."""
    understanding = ctx.understanding
    lang = ctx.lang
    unavailable = understanding.unavailable_fields
    providers = _known_providers(ctx.db)

    extra: List[str] = []
    if understanding.wants_quality_ranking:
        # FR-22: the corpus has no quality data, so a "which is best" question is
        # answered by comparing facts and saying so.
        extra.append(
            "The user asked which option is best. State plainly that this service "
            "holds no quality, rating, or popularity data and cannot rank "
            "providers, then compare only the facts in RECORDS."
        )
    if understanding.fuzzy_facets:
        assumed = ", ".join(
            f"{facet}={value}" for facet, (value, _) in understanding.fuzzy_facets.items()
        )
        # FR-8: a corrected spelling is stated, not silently applied.
        extra.append(
            f"The service interpreted an approximate spelling as: {assumed}. "
            "Mention this interpretation in one short clause so the user can "
            "correct it."
        )
    if ctx.retrieval is not None and ctx.retrieval.truncated:
        extra.append(
            "The list shown is partial. Do not state any total or count; instead "
            "invite the user to narrow the search further."
        )
    if ctx.retrieval is not None and ctx.retrieval.state_widened:
        extra.append(
            "Some records are nationwide or multi-state providers rather than the "
            "specific state requested. Mention this briefly."
        )

    prompt = narrative.build_prompt(ctx.message, rows, understanding, lang, extra)
    provider = get_provider()
    llm = provider.generate(prompt)
    ctx.llm = llm
    get_metrics().observe_llm(llm.ok)

    if not llm.ok:
        # NFR-C: no model, no problem — the list is the product. `degraded` is
        # true because the user is getting less than the full experience, and the
        # trace says why.
        code = llm.error or "llm_unavailable"
        logger.warning(
            "narrative unavailable; serving template answer",
            extra={
                "trace_id": ctx.trace_id,
                "session_id": ctx.session.session_id,
                "state": ChatState.DEGRADED.value,
                "llm_error": code,
                "llm_attempts": llm.attempts,
                "llm_latency_ms": llm.latency_ms,
            },
        )
        return ctx.finish(
            state=ChatState.DEGRADED,
            answer=safe_answer(rows, lang, unavailable),
            results=results,
            unavailable=unavailable,
            degraded=True,
            validation_failed=[code],
        )

    verdict = enforce(
        llm.text,
        rows,
        lang=lang,
        unavailable_fields=unavailable,
        known_providers=providers,
    )

    if verdict.degraded:
        _log_degraded(ctx, llm, verdict.failures, effective)
        return ctx.finish(
            state=ChatState.DEGRADED,
            answer=verdict.answer,
            results=results,
            unavailable=unavailable,
            degraded=True,
            validation_failed=verdict.failures,
        )

    answer = verdict.answer
    if understanding.wants_quality_ranking:
        # Belt and braces on NG8: even a clean narrative gets the "we do not rank"
        # sentence in front of it, because the question invited a ranking and the
        # answer should refuse it visibly.
        answer = f"{narrative.comparison_note(lang)} {answer}"

    return ctx.finish(
        state=ChatState.ANSWERED,
        answer=answer,
        results=results,
        unavailable=unavailable,
    )


def _log_degraded(
    ctx: _Context,
    llm: LLMResult,
    failures: Sequence[str],
    effective: Dict[str, object],
) -> None:
    """FR-30: enough to diagnose, nothing that identifies a person.

    The acceptance criterion names three things — "query user, output model, dan
    validasi mana yang gagal" — and all three are here. Both texts go through
    `redact()` first: FR-30 wants the query, NFR-D wants no user identity in the
    logs, and a query is the one place a phone number tends to appear. Truncated
    as well, because a 30-day log of full model output is a liability with no
    diagnostic value beyond the first few hundred characters.
    """
    degraded_logger.warning(
        "degraded response: %s",
        ",".join(failures) or "unknown",
        extra={
            "trace_id": ctx.trace_id,
            "session_id": ctx.session.session_id,
            "state": ChatState.DEGRADED.value,
            "latency_ms": ctx.latency_ms(),
            "validation_failed": list(failures),
            "user_query": redact(ctx.message),
            "model_output": redact(llm.text, limit=1_200),
            "model": llm.model,
            "filters_applied": {k: str(v) for k, v in effective.items()},
            "retrieved_ids": (
                list(ctx.retrieval.record_ids) if ctx.retrieval else []
            ),
            "consumer": ctx.consumer,
        },
    )


__all__ = ["Turn", "handle_chat", "reset_provider_cache"]
