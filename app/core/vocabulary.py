"""Closed-list vocabulary for the corpus.

Source of truth: "Data Dictionary — Korpus Kelas Tuisyen Malaysia v1",
section "Facet filter (enum tertutup)".

FR-9 requires that any value parsed out of a user's message is checked against
these lists *before* it reaches the query. Anything outside the list is refused
and turns into `state: needs_clarification` — never silently dropped, and never
passed to SQL.

Cardinalities are asserted at import time so that a typo here fails loudly
rather than quietly shrinking the searchable space.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Tuple

# ── subject · 22 values ─────────────────────────────────────────────────────
SUBJECTS: Tuple[str, ...] = (
    "Accounting",
    "Additional Mathematics",
    "Advanced Mathematics I",
    "Advanced Mathematics II",
    "Art",
    "Bahasa Cina",
    "Bahasa Melayu",
    "Biology",
    "Bookkeeping and Accounts",
    "Chemistry",
    "Chinese Language",
    "Commerce",
    "Computer and Information Technology",
    "Economics",
    "English",
    "Geography",
    "History",
    "Malaysian Language",
    "Mandarin",
    "Mathematics",
    "Physics",
    "Science",
)

# ── level · 17 values ───────────────────────────────────────────────────────
# DD: "Form 1–5 · Junior Middle 1–3 · Senior Middle 1–3 · Year 1–6"
LEVELS: Tuple[str, ...] = (
    "Year 1",
    "Year 2",
    "Year 3",
    "Year 4",
    "Year 5",
    "Year 6",
    "Form 1",
    "Form 2",
    "Form 3",
    "Form 4",
    "Form 5",
    "Junior Middle 1",
    "Junior Middle 2",
    "Junior Middle 3",
    "Senior Middle 1",
    "Senior Middle 2",
    "Senior Middle 3",
)

# DD ER diagram, LEVEL.sort_order: "Year 1 sebelum Form 1".
LEVEL_SORT_ORDER: Dict[str, int] = {name: i for i, name in enumerate(LEVELS)}

LEVEL_STAGE: Dict[str, str] = {
    **{f"Year {n}": "Primary" for n in range(1, 7)},
    **{f"Form {n}": "Secondary" for n in range(1, 6)},
    **{f"Junior Middle {n}": "Secondary" for n in range(1, 4)},
    **{f"Senior Middle {n}": "Secondary" for n in range(1, 4)},
}

# ── state · 7 values ────────────────────────────────────────────────────────
STATES: Tuple[str, ...] = (
    "Johor",
    "Malaysia",
    "Multi-state",
    "Negeri Sembilan",
    "Penang",
    "Sabah",
    "Selangor",
)

# DD footnote: "Malaysia dan Multi-state bukan negeri sebenarnya — dipakai
# untuk penyedia lintas kawasan." Open question D3 asks whether these should
# surface when a user names a specific state. Configurable, defaulting to
# inclusive so a Selangor search still sees nationwide providers.
CROSS_REGION_STATES: Tuple[str, ...] = ("Malaysia", "Multi-state")

# ── delivery_mode · 11 values ───────────────────────────────────────────────
DELIVERY_MODES: Tuple[str, ...] = (
    "Home one-to-one",
    "Home small group",
    "Hybrid",
    "Online asynchronous",
    "Online live",
    "Online one-to-one",
    "Personalised tuition",
    "Physical bootcamp",
    "Physical group",
    "Private one-to-one",
    "Small group",
)

# DD ER diagram, DELIVERY_MODE.mode_group: "online, physical, hybrid, home".
# Open question D4: user says "online" — cover all three online modes, or ask?
# We expand to the group; the applied filter is always echoed in
# trace.filters_applied so the behaviour is visible rather than implicit.
#
# Three of the eleven modes carry no channel signal in their name: "Small group"
# says class size, "Private one-to-one" and "Personalised tuition" say format.
# Any of them could be run online or in person, and the corpus does not record
# which. They are therefore grouped as "unspecified" instead of being forced
# into a channel — assigning them would be a guess of exactly the kind FR-9
# forbids. A channel filter never silently includes them; when a channel search
# comes back empty they surface through FR-15 relaxation options instead.
MODE_GROUPS: Dict[str, Tuple[str, ...]] = {
    "online": ("Online asynchronous", "Online live", "Online one-to-one"),
    "physical": ("Physical bootcamp", "Physical group"),
    "hybrid": ("Hybrid",),
    "home": ("Home one-to-one", "Home small group"),
    "unspecified": ("Personalised tuition", "Private one-to-one", "Small group"),
}

# The groups a user is allowed to select. "unspecified" is bookkeeping only.
CHANNEL_GROUPS: Tuple[str, ...] = ("online", "physical", "hybrid", "home")

MODE_TO_GROUP: Dict[str, str] = {
    mode: group for group, modes in MODE_GROUPS.items() for mode in modes
}

# Modes a channel filter can neither confirm nor rule out.
CHANNEL_AGNOSTIC_MODES: Tuple[str, ...] = MODE_GROUPS["unspecified"]

# ── instruction_language · 6 values ─────────────────────────────────────────
INSTRUCTION_LANGUAGES: Tuple[str, ...] = (
    "Chinese-English bilingual",
    "English",
    "English (DLP-style support)",
    "Malay",
    "Malay-English bilingual",
    "Mandarin Chinese",
)

# ── curriculum · 5 values ───────────────────────────────────────────────────
# DD open question D2: "KSSM / SPM" (67 records) vs "SPM / KSSM" (14) look like
# the same value with the operands swapped. Unresolved, so both are kept valid
# and CURRICULUM_ALIASES records the suspected equivalence without collapsing
# the data. Flip MERGE_CURRICULUM_ALIASES once Maggie/KahJun confirm.
CURRICULA: Tuple[str, ...] = (
    "KSSM / SPM",
    "KSSM / UASA",
    "KSSR / UASA",
    "SPM / KSSM",
    "UEC",
)

CURRICULUM_ALIASES: Dict[str, str] = {"SPM / KSSM": "KSSM / SPM"}
MERGE_CURRICULUM_ALIASES = False

# ── education_stage · 2 values ──────────────────────────────────────────────
EDUCATION_STAGES: Tuple[str, ...] = ("Primary", "Secondary")

# ── school_category · 4 values (display only) ───────────────────────────────
SCHOOL_CATEGORIES: Tuple[str, ...] = (
    "National primary (SK/SJK-ready)",
    "National-type primary (SJKC-ready)",
    "National / private secondary (SPM track)",
    "Malaysian Chinese independent secondary school",
)

# ── Facets usable as query filters (DD "Peran" = K) ─────────────────────────
FILTERABLE_FACETS: Tuple[str, ...] = (
    "subject",
    "level",
    "state",
    "city_area",
    "delivery_mode",
    "instruction_language",
    "curriculum",
    "education_stage",
)

# city_area is data-driven (13 values, bound 1:1 to state) so it is loaded from
# the database at startup rather than hardcoded here.
CLOSED_VOCABULARY: Dict[str, Tuple[str, ...]] = {
    "subject": SUBJECTS,
    "level": LEVELS,
    "state": STATES,
    "delivery_mode": DELIVERY_MODES,
    "instruction_language": INSTRUCTION_LANGUAGES,
    "curriculum": CURRICULA,
    "education_stage": EDUCATION_STAGES,
}

# ── Fields that are NULL for all 200 records (DD "Field kosong") ────────────
# fee_rm 100% null -> FR-16 · schedule 100% null -> FR-17
ALWAYS_NULL_FIELDS: FrozenSet[str] = frozenset({"fee_rm", "schedule"})

# ── FR-12: minimum fields every returned class must carry, all non-empty ────
REQUIRED_RESULT_FIELDS: Tuple[str, ...] = (
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
)

# ── FR-13: a match reason may only cite these facts ─────────────────────────
# "Alasan hanya boleh merujuk pada mode pengajaran, bahasa, lokasi, dan
# kurikulum. Harga, jadwal, atau klaim kualitas tidak boleh digunakan."
REASON_ALLOWED_FIELDS: Tuple[str, ...] = (
    "delivery_mode",
    "instruction_language",
    "state",
    "city_area",
    "curriculum",
)


def is_valid(facet: str, value: str) -> bool:
    """True if `value` is in the closed list for `facet` (FR-9 gate)."""
    allowed = CLOSED_VOCABULARY.get(facet)
    if allowed is None:
        return True  # data-driven facet (e.g. city_area), validated against DB
    return value in allowed


def values_for(facet: str) -> List[str]:
    """Closed list for a facet, for FR-10's `available_values[]`."""
    return list(CLOSED_VOCABULARY.get(facet, ()))


def _assert_cardinalities() -> None:
    """Fail fast if a list drifts from the Data Dictionary."""
    expected = {
        "subject": (SUBJECTS, 22),
        "level": (LEVELS, 17),
        "state": (STATES, 7),
        "delivery_mode": (DELIVERY_MODES, 11),
        "instruction_language": (INSTRUCTION_LANGUAGES, 6),
        "curriculum": (CURRICULA, 5),
        "education_stage": (EDUCATION_STAGES, 2),
        "school_category": (SCHOOL_CATEGORIES, 4),
    }
    for facet, (values, count) in expected.items():
        if len(values) != count:
            raise AssertionError(
                f"vocabulary drift: {facet} has {len(values)} values, "
                f"Data Dictionary says {count}"
            )
        if len(set(values)) != len(values):
            raise AssertionError(f"vocabulary drift: {facet} contains duplicates")
    if set(LEVEL_STAGE) != set(LEVELS):
        raise AssertionError("LEVEL_STAGE does not cover every level")
    if set(MODE_TO_GROUP) != set(DELIVERY_MODES):
        raise AssertionError("MODE_GROUPS does not cover every delivery_mode")


_assert_cardinalities()
