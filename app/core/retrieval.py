"""Retrieval over the curated corpus (FR-11, FR-14, FR-15).

There is no embedding model and no similarity score anywhere in this file, and
that is deliberate. NFR-B: "belum menggunakan vector DB di v1". The corpus is
200 rows described entirely by closed enums, so the honest implementation is a
conjunctive SQL filter: every facet the user named must match, and a class either
qualifies or it does not. Nothing is ranked by relevance, because there is no
relevance signal in the data — inventing one would produce exactly the confident
guessing the PRD spends most of its length forbidding.

Three consequences worth stating up front:

* **Ordering is deterministic, not relevance-based.** Level order first (DD:
  "Year 1 sebelum Form 1"), then most recently verified, then record_id as the
  tiebreak. The same query returns the same list in the same order, which is
  what makes FR-31's eval reproducible.

* **`verified_on` comes off the row** (FR-14). It is never "today", never
  `now()`, and never omitted. A user must be able to see how stale the entry is.

* **An empty result is a first-class outcome** (FR-15). When nothing matches, the
  interesting information is *which single filter* is responsible, so the
  response can offer "drop the language requirement and there are 4 classes"
  instead of a dead end.

Every value that reaches a WHERE clause has already passed the FR-9 closed
vocabulary gate in `app/query_understanding.py`, and travels as a bound
parameter through SQLAlchemy's expression language. See `db/README.md` for why
that ordering — gate first, then bind — is the part that actually protects the
service.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from app.config import get_settings
from app.infrastructure.models import (
    ClassRecord,
    Curriculum,
    DeliveryMode,
    InstructionLanguage,
    Level,
    Provider,
    Subject,
)
from app.core.vocabulary import (
    CROSS_REGION_STATES,
    LEVEL_SORT_ORDER,
    MODE_TO_GROUP,
)

# ── Facet -> the column it filters on ───────────────────────────────────────
# Keys match FILTERABLE_FACETS exactly. A facet absent from this map cannot be
# filtered on at all, which is the intended failure mode: adding a filter
# requires adding it here, in the open, rather than assembling a column name
# from a request field.
FACET_COLUMNS: Dict[str, ColumnElement] = {
    "subject": Subject.name,
    "level": Level.name,
    "state": Provider.state,
    "city_area": Provider.city_area,
    "delivery_mode": DeliveryMode.name,
    "instruction_language": InstructionLanguage.name,
    "curriculum": Curriculum.name,
    "education_stage": ClassRecord.education_stage,
}

# How many alternative values to name inside a single relaxation option. Long
# lists stop being a suggestion and start being a data dump.
MAX_SUGGESTED_VALUES = 6
# How many relaxation options to return. FR-10 allows one clarifying question;
# FR-15 is a different thing (offers, not questions) but the same restraint
# applies — four is enough to be useful and still readable in a chat bubble.
MAX_RELAXATION_OPTIONS = 4


# ─── Row snapshot ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ClassRow:
    """A flattened, session-independent snapshot of one class.

    Columns are selected explicitly rather than loading `ClassRecord` entities,
    for two reasons. It keeps the query to a single round trip with no lazy
    loading behind the scenes, and it hands the honesty layer a plain object it
    can validate without an open session — `app/honesty.py` has to be able to
    assert things about `fee_rm` and `schedule` *as they came out of the
    database*, not as the ORM promises they will be.
    """

    record_id: str
    class_title: str
    provider_name: str
    provider_source_url: str
    state: str
    city_area: str
    subject: str
    level: str
    level_sort_order: int
    delivery_mode: str
    mode_group: str
    instruction_language: str
    curriculum: str
    education_stage: str
    school_category: str
    verified_on: date
    class_status: str
    data_label: str
    # Selected on purpose even though both are NULL for every row: the guardrail
    # in app/honesty.py checks the actual value rather than trusting the
    # contract (FR-16, FR-17, DD V8).
    fee_rm: Optional[float]
    schedule: Optional[str]
    # Internal-only (DD "Peran" = I). Never rendered to a user; available to the
    # grounding validator and the eval harness.
    retrieval_text: Optional[str]
    keywords: Optional[str]

    @classmethod
    def from_row(cls, row) -> "ClassRow":
        m = row._mapping
        return cls(
            record_id=m["record_id"],
            class_title=m["class_title"],
            provider_name=m["provider_name"],
            provider_source_url=m["provider_source_url"],
            state=m["state"],
            city_area=m["city_area"],
            subject=m["subject"],
            level=m["level"],
            level_sort_order=m["level_sort_order"],
            delivery_mode=m["delivery_mode"],
            mode_group=m["mode_group"],
            instruction_language=m["instruction_language"],
            curriculum=m["curriculum"],
            education_stage=m["education_stage"],
            school_category=m["school_category"],
            verified_on=m["verified_on"],
            class_status=m["class_status"],
            data_label=m["data_label"],
            fee_rm=m["fee_rm"],
            schedule=m["schedule"],
            retrieval_text=m["retrieval_text"],
            keywords=m["keywords"],
        )

    def facet(self, name: str) -> Optional[str]:
        """Value of a named facet on this row, or None if it is not a facet.

        Used by the grounding validator (FR-19) to check a claim against the
        record it is supposed to be about, and by FR-13's match-reason builder.
        """
        return {
            "subject": self.subject,
            "level": self.level,
            "state": self.state,
            "city_area": self.city_area,
            "delivery_mode": self.delivery_mode,
            "instruction_language": self.instruction_language,
            "curriculum": self.curriculum,
            "education_stage": self.education_stage,
            "school_category": self.school_category,
            "provider_name": self.provider_name,
        }.get(name)


SELECTED_COLUMNS: Tuple[ColumnElement, ...] = (
    ClassRecord.record_id,
    ClassRecord.class_title,
    Provider.provider_name,
    Provider.provider_source_url,
    Provider.state,
    Provider.city_area,
    Subject.name.label("subject"),
    Level.name.label("level"),
    Level.sort_order.label("level_sort_order"),
    DeliveryMode.name.label("delivery_mode"),
    DeliveryMode.mode_group.label("mode_group"),
    InstructionLanguage.name.label("instruction_language"),
    Curriculum.name.label("curriculum"),
    ClassRecord.education_stage,
    ClassRecord.school_category,
    ClassRecord.verified_on,
    ClassRecord.class_status,
    ClassRecord.data_label,
    ClassRecord.fee_rm,
    ClassRecord.schedule,
    ClassRecord.retrieval_text,
    ClassRecord.keywords,
)


# ─── Results ────────────────────────────────────────────────────────────────


@dataclass
class RelaxationOffer:
    """One "drop this and you get results" offer (FR-15)."""

    field: str
    current_value: str
    result_count: int
    suggested_values: List[str] = field(default_factory=list)


@dataclass
class Retrieval:
    rows: List[ClassRow] = field(default_factory=list)
    # Matches before the `max_results` cap. FR-10 needs the true size to decide
    # whether a query is too broad; the user needs the capped list.
    candidate_count: int = 0
    # Exactly what went into the WHERE clause, echoed into trace.filters_applied
    # so a widened or dropped filter is visible rather than implicit (NFR-G).
    filters_applied: Dict[str, object] = field(default_factory=dict)
    relaxation_options: List[RelaxationOffer] = field(default_factory=list)
    truncated: bool = False
    # True when a specific-state search also swept in Malaysia / Multi-state
    # providers (DD open question D3). The answer says so out loud.
    state_widened: bool = False

    @property
    def is_empty(self) -> bool:
        return not self.rows

    @property
    def record_ids(self) -> List[str]:
        return [r.record_id for r in self.rows]


# ─── Query construction ─────────────────────────────────────────────────────


def _join_dimensions(stmt: Select) -> Select:
    """Inner-join every dimension table.

    Inner joins, not outer: a class whose facet does not resolve is a broken
    row, and the seed's self-check (`db/init/002_seed_corpus.sql`) already
    refuses to create one. Making the join strict here means such a row would
    disappear loudly from counts rather than appear with NULL facets and fail
    FR-12 validation deep inside the response builder.
    """
    return (
        stmt.join(Provider, ClassRecord.provider_id == Provider.provider_id)
        .join(Subject, ClassRecord.subject_id == Subject.subject_id)
        .join(Level, ClassRecord.level_id == Level.level_id)
        .join(DeliveryMode, ClassRecord.delivery_mode_id == DeliveryMode.delivery_mode_id)
        .join(
            InstructionLanguage,
            ClassRecord.instruction_lang_id == InstructionLanguage.instruction_lang_id,
        )
        .join(Curriculum, ClassRecord.curriculum_id == Curriculum.curriculum_id)
    )


def _as_values(value: object) -> List[str]:
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(v) for v in value]
    return [str(value)]


def _conditions(
    filters: Dict[str, object], settings=None
) -> Tuple[List[ColumnElement], Dict[str, object], bool]:
    """Turn gated filters into WHERE clauses.

    Returns the clause list, the filters as actually applied (for the trace),
    and whether the state filter was widened.

    Unknown facet names are dropped rather than raising. They cannot originate
    from a request — `Understanding.filters` only ever holds FILTERABLE_FACETS
    keys — so an unknown key means a programming error upstream, and failing a
    user's search is the wrong way to report one. The dropped key is visible by
    its absence from the returned `applied` dict, which lands in the trace.
    """
    settings = settings or get_settings()
    clauses: List[ColumnElement] = []
    applied: Dict[str, object] = {}
    widened = False

    for facet, raw in filters.items():
        column = FACET_COLUMNS.get(facet)
        if column is None:
            continue
        values = _as_values(raw)
        if not values:
            continue

        if facet == "state" and settings.include_cross_region_states:
            specific = [v for v in values if v not in CROSS_REGION_STATES]
            if specific:
                # A nationwide or multi-state provider genuinely can serve a
                # Selangor student, especially online. Including them is the
                # more useful default; hiding the widening would not be.
                values = list(dict.fromkeys(values + list(CROSS_REGION_STATES)))
                widened = True

        # `.in_()` for one value as well as many: one code path, and the value
        # still travels as a bound parameter either way.
        clauses.append(column.in_(values))
        applied[facet] = values[0] if len(values) == 1 else values

    return clauses, applied, widened


def _order(stmt: Select) -> Select:
    return stmt.order_by(
        Level.sort_order.asc(),
        ClassRecord.verified_on.desc(),
        ClassRecord.record_id.asc(),
    )


def _base(filters: Dict[str, object], settings=None):
    clauses, applied, widened = _conditions(filters, settings)
    stmt = _join_dimensions(select(*SELECTED_COLUMNS))
    for clause in clauses:
        stmt = stmt.where(clause)
    return stmt, clauses, applied, widened


# ─── Public API ─────────────────────────────────────────────────────────────


def count_candidates(session: Session, filters: Dict[str, object]) -> int:
    """How many classes match, ignoring the display cap.

    FR-10 turns on this number: above `clarify_threshold` the query is too broad
    to answer usefully and the response asks one narrowing question instead of
    dumping the corpus.
    """
    clauses, _, _ = _conditions(filters)
    stmt = _join_dimensions(select(func.count()).select_from(ClassRecord))
    for clause in clauses:
        stmt = stmt.where(clause)
    return int(session.execute(stmt).scalar_one())


def search(
    session: Session,
    filters: Dict[str, object],
    limit: Optional[int] = None,
    with_relaxation: bool = True,
) -> Retrieval:
    """Run the conjunctive facet search (FR-11).

    `with_relaxation` exists for the eval harness, which measures raw retrieval
    and does not need the extra queries.
    """
    settings = get_settings()
    limit = settings.max_results if limit is None else limit

    stmt, _, applied, widened = _base(filters, settings)
    rows = [
        ClassRow.from_row(row)
        for row in session.execute(_order(stmt).limit(limit + 1)).all()
    ]

    truncated = len(rows) > limit
    if truncated:
        rows = rows[:limit]
        total = count_candidates(session, filters)
    else:
        total = len(rows)

    result = Retrieval(
        rows=rows,
        candidate_count=total,
        filters_applied=applied,
        truncated=truncated,
        state_widened=widened,
    )

    if with_relaxation and not rows and filters:
        result.relaxation_options = relaxation_options(session, filters)

    return result


def relaxation_options(
    session: Session,
    filters: Dict[str, object],
    max_options: int = MAX_RELAXATION_OPTIONS,
) -> List[RelaxationOffer]:
    """FR-15: for each filter, what happens if we drop just that one?

    One filter at a time, never two. A user who hears "there is nothing in
    Sabah in Mandarin for Form 4" can act on "there are 3 Form 4 Mandarin
    classes elsewhere" or on "there are 5 Form 4 classes in Sabah in other
    languages". Telling them "there are 40 classes if you drop both" is not a
    suggestion, it is a shrug.

    Options with no results are omitted — an offer that leads nowhere is worse
    than silence. The order is most-productive-first, so the best suggestion is
    the one the narrative mentions if it only has room for one.
    """
    offers: List[RelaxationOffer] = []

    for facet in filters:
        if facet not in FACET_COLUMNS:
            continue
        reduced = {k: v for k, v in filters.items() if k != facet}
        count = count_candidates(session, reduced)
        if count <= 0:
            continue
        offers.append(
            RelaxationOffer(
                field=facet,
                current_value=", ".join(_as_values(filters[facet])),
                result_count=count,
                suggested_values=distinct_values(
                    session,
                    facet,
                    reduced,
                    exclude=_as_values(filters[facet]),
                    limit=MAX_SUGGESTED_VALUES,
                ),
            )
        )

    offers.sort(key=lambda o: (-o.result_count, o.field))
    return offers[:max_options]


def distinct_values(
    session: Session,
    facet: str,
    filters: Dict[str, object],
    exclude: Sequence[str] = (),
    limit: Optional[int] = None,
) -> List[str]:
    """Which values of `facet` actually exist under `filters`.

    This is what keeps FR-10's `available_values[]` and FR-15's
    `suggested_values[]` honest: they list what the corpus contains under the
    user's other constraints, not the full closed vocabulary. Offering "Johor"
    when there is no Johor class matching the rest of the query would be a
    dead-end suggestion dressed up as help.
    """
    column = FACET_COLUMNS.get(facet)
    if column is None:
        return []

    clauses, _, _ = _conditions(filters)
    stmt = _join_dimensions(select(column).distinct().select_from(ClassRecord))
    for clause in clauses:
        stmt = stmt.where(clause)

    excluded = set(exclude)
    values = [
        str(v) for (v,) in session.execute(stmt).all() if v is not None and str(v) not in excluded
    ]
    values = _sorted_facet_values(facet, values)
    return values[:limit] if limit else values


def _sorted_facet_values(facet: str, values: Iterable[str]) -> List[str]:
    """Level order for levels, alphabetical for everything else.

    "Form 1, Form 2, Form 10" is not a list a parent reads; DD's sort_order
    exists precisely so levels come out in teaching order.
    """
    if facet == "level":
        return sorted(values, key=lambda v: LEVEL_SORT_ORDER.get(v, 999))
    if facet == "delivery_mode":
        # Group channel-mates together, then alphabetical inside the group, so
        # the three online modes read as one option rather than three scattered.
        return sorted(values, key=lambda v: (MODE_TO_GROUP.get(v, "zz"), v))
    return sorted(values)


def get_by_ids(session: Session, record_ids: Sequence[str]) -> List[ClassRow]:
    """Fetch specific classes, order preserved as given.

    Used by FR-25's follow-up path ("kelas kedua tu online ke?"), which must
    talk about the classes already shown rather than re-running a search that
    could return a different set.
    """
    if not record_ids:
        return []
    stmt = _join_dimensions(select(*SELECTED_COLUMNS)).where(
        ClassRecord.record_id.in_(list(record_ids))
    )
    found = {r.record_id: r for r in (ClassRow.from_row(row) for row in session.execute(stmt).all())}
    return [found[rid] for rid in record_ids if rid in found]


def corpus_size(session: Session) -> int:
    """Row count for /health. A zero here means the seed did not run."""
    return int(session.execute(select(func.count()).select_from(ClassRecord)).scalar_one())


def known_city_areas(session: Session) -> List[str]:
    """The 13 city areas, read from the data rather than hardcoded.

    `city_area` is the one filterable facet the Data Dictionary does not fix as
    an enum — it is bound 1:1 to a provider. So the FR-9 gate for this facet has
    to consult the database, and `app/vocabulary.is_valid` deliberately returns
    True for facets it does not own, deferring to this function.
    """
    stmt = select(Provider.city_area).distinct().order_by(Provider.city_area)
    return [str(v) for (v,) in session.execute(stmt).all() if v]


def state_for_city_area(session: Session, city_area: str) -> Optional[str]:
    """DD: city_area determines state. Used to keep a location pair consistent."""
    stmt = select(Provider.state).where(Provider.city_area == city_area).limit(1)
    row = session.execute(stmt).first()
    return str(row[0]) if row else None


def known_providers(session: Session) -> List[str]:
    """Every provider name in the corpus.

    Feeds the second half of the FR-18 grounding check. A narrative that names a
    provider which exists in the corpus but is *not* among the retrieved rows is
    the most dangerous failure mode available — it reads as authoritative and it
    is wrong — and a keyword heuristic cannot catch names like "Warna Cemerlang"
    that contain no provider-ish word. So the validator is given the real list and
    checks against it directly.

    Cheap: one column, ~90 rows. The caller caches it for a few minutes.
    """
    stmt = select(Provider.provider_name).distinct().order_by(Provider.provider_name)
    return [str(v) for (v,) in session.execute(stmt).all() if v]
