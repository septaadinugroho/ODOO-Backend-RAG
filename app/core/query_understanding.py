"""Natural language -> structured filters (FR-6 … FR-10).

The pipeline is deliberately deterministic. No LLM runs here, for three reasons:

* FR-6 sets a measurable bar ("akurasi minimal 90% pada 60 query eval"), and a
  rule-based extractor is the only version of this that can be regression-tested
  line by line.
* FR-9 requires a hard gate: a value that is not in the closed vocabulary must
  never reach the query. A gate is only trustworthy if it sits outside the
  component that might hallucinate.
* NFR-D treats user text as data. Nothing typed by a user is ever interpreted as
  an instruction; it is tokenised, matched against a dictionary, and discarded.

The LLM's only job, later in `app/narrative.py`, is to phrase an answer about
records that this module and `app/retrieval.py` already selected.

Extraction strategy
-------------------
A greedy longest-match n-gram scan (4-grams down to unigrams). Longest-first is
what makes "matematik tambahan" resolve to Additional Mathematics rather than
matching "matematik" -> Mathematics and leaving "tambahan" dangling.

Fee and schedule
----------------
Both fields are NULL for all 200 records, so neither can ever become a filter
(FR-16, FR-17). But *how* the user raised them changes the right response, so
they are classified:

  question   — "berapa harga?", "pukul berapa kelas ni?"
               The user asked for a value we do not hold. There is nothing
               honest to answer -> state `unavailable_field`.
  constraint — "bawah RM200", "yang petang je"
               The user tried to narrow. We can still serve the class list, as
               long as the response says plainly that the narrowing was not
               applied -> state `answered` with `unavailable_fields` populated.

Both paths populate `unavailable_fields` (FR-20) and neither may emit a number
or a time (FR-16, FR-17). The split is an interpretation of the PRD, recorded in
README under "Catatan interpretasi"; it is what lets the persona example "yang
petang je" stay useful without implying we filtered by afternoon.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from app.config import get_settings
from app.core.synonyms import Resolution, SynonymDictionary, get_dictionary, normalize
from app.core.vocabulary import (
    CHANNEL_GROUPS,
    FILTERABLE_FACETS,
    MODE_GROUPS,
    values_for,
)

MAX_NGRAM = 4

# FR-10: which single field to ask about when a query is too broad. Ordered by
# how much each one narrows the corpus — subject (22 values) cuts hardest, and
# asking for it first keeps the conversation to one question (scenario 2).
MISSING_FIELD_PRIORITY: Tuple[str, ...] = (
    "subject",
    "level",
    "state",
    "delivery_mode",
)

# Words that mean "a class", used to tell a vague class search ("ada kelas
# tak?") apart from a question that has nothing to do with classes (FR-22).
_CLASS_SIGNAL = frozenset(
    {
        "kelas",
        "class",
        "classes",
        "tuisyen",
        "tuition",
        "les",
        "cikgu",
        "tutor",
        "teacher",
        "guru",
        "belajar",
        "learn",
        "course",
        "kursus",
    }
)

_RM_AMOUNT = re.compile(r"\brm\s*\d+", re.IGNORECASE)
_INTERROGATIVE_FEE = re.compile(
    r"\b(berapa|brape|brp|how much|whats the (price|fee)|what is the (price|fee)"
    r"|harga\s*(nya)?\s*\?|kos\s*(nya)?\s*\?)\b",
    re.IGNORECASE,
)
_INTERROGATIVE_TIME = re.compile(
    r"\b(pukul berapa|jam berapa|bila|what time|when (is|does)|jadual(nya)?\s*\?"
    r"|schedule\s*\?|hari apa)\b",
    re.IGNORECASE,
)
_CONSTRAINT_MARKER = re.compile(
    r"\b(bawah|below|under|kurang dari|less than|maksimum|max|dalam bajet"
    r"|je|jer|sahaja|only|saja)\b",
    re.IGNORECASE,
)
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


# ─── Result ─────────────────────────────────────────────────────────────────


@dataclass
class Ambiguity:
    """A fragment the system refuses to resolve on the user's behalf."""

    fragment: str
    facet: str
    candidates: Tuple[str, ...]
    note: str = ""


@dataclass
class Understanding:
    """Everything the deterministic layer could establish about a message."""

    raw_message: str
    normalized: str = ""
    tokens: Tuple[str, ...] = ()

    # facet -> canonical value, or facet -> list of values when a mode group was
    # expanded. Only ever contains keys from FILTERABLE_FACETS, and only ever
    # values that passed the FR-9 gate.
    filters: Dict[str, object] = field(default_factory=dict)
    # facet -> the user fragment that produced it, for explaining the match.
    evidence: Dict[str, str] = field(default_factory=dict)
    # Facets accepted via fuzzy match, so the response can say "saya anggap
    # maksud anda ..." rather than pretending it was exact (FR-8).
    fuzzy_facets: Dict[str, Tuple[str, float]] = field(default_factory=dict)

    mode_group: Optional[str] = None
    ambiguities: List[Ambiguity] = field(default_factory=list)
    unresolved: List[str] = field(default_factory=list)

    intents: Set[str] = field(default_factory=set)
    # "question" | "constraint" | None
    fee_intent: Optional[str] = None
    schedule_intent: Optional[str] = None

    has_class_signal: bool = False

    # ── convenience ─────────────────────────────────────────────────────────

    @property
    def unavailable_fields(self) -> List[str]:
        """FR-20: the always-null fields the user touched, in a stable order."""
        out: List[str] = []
        if self.fee_intent:
            out.append("fee_rm")
        if self.schedule_intent:
            out.append("schedule")
        return out

    @property
    def asked_unanswerable_question(self) -> bool:
        """True when the user asked *for* a value we do not hold at all.

        Drives `state: unavailable_field` — see the module docstring.
        """
        return "question" in (self.fee_intent, self.schedule_intent)

    @property
    def is_searchable(self) -> bool:
        return bool(self.filters) or self.mode_group is not None

    @property
    def wants_quality_ranking(self) -> bool:
        """FR-22: "kelas mana paling bagus" — no quality claims in the corpus."""
        return "superlative" in self.intents

    @property
    def shows_distress(self) -> bool:
        """FR-24: respond with empathy, never with clinical advice."""
        return "distress" in self.intents

    def missing_fields(self) -> List[str]:
        return [f for f in MISSING_FIELD_PRIORITY if f not in self.filters]

    def next_missing_field(self) -> Optional[str]:
        """FR-10: exactly one field per response, highest-impact first."""
        missing = self.missing_fields()
        return missing[0] if missing else None


# ─── Helpers ────────────────────────────────────────────────────────────────


def sanitize_user_text(text: str, limit: int = 2_000) -> str:
    """Make user text safe to embed in a prompt as *data* (NFR-D).

    Strips control characters and the delimiters an injection attempt would use
    to break out of its block, then truncates. The prompt template additionally
    wraps the result in an explicit data fence and states that its contents are
    never instructions.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_CHARS.sub(" ", text)
    text = text.replace("```", "'''").replace("<|", "<").replace("|>", ">")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def _fragment_variants(fragment: str) -> List[str]:
    """Forms of a fragment worth looking up.

    Users write levels every which way — "Form 4", "form4", "f4", "tingkatan4".
    Rather than enumerate every spelling in the dictionary, each fragment is also
    tried with a space inserted at, and removed from, its letter/digit boundary.
    """
    variants = [fragment]
    split = re.sub(r"(?<=[a-z])(?=\d)", " ", fragment)
    if split != fragment:
        variants.append(split)
    joined = re.sub(r"(?<=[a-z]) (?=\d)", "", fragment)
    if joined != fragment:
        variants.append(joined)
    return variants


def _contains_phrase(normalized: str, phrase: str) -> bool:
    """Word-boundary-aware substring test on already-normalised text."""
    if not phrase:
        return False
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", normalized) is not None


# ─── Analyzer ───────────────────────────────────────────────────────────────


class QueryUnderstanding:
    def __init__(self, dictionary: Optional[SynonymDictionary] = None) -> None:
        self._dict = dictionary or get_dictionary()

    def analyze(self, message: str) -> Understanding:
        settings = get_settings()
        dictionary = self._dict
        dictionary.reload_if_stale()

        u = Understanding(raw_message=message)
        u.normalized = normalize(message)
        u.tokens = tuple(u.normalized.split())

        self._detect_intents(u, dictionary)
        self._scan(u, dictionary)
        self._resolve_leftovers(u, dictionary)
        self._apply_mode_group(u, settings)
        self._gate(u)

        u.has_class_signal = (
            bool(u.filters)
            or u.mode_group is not None
            or bool(u.ambiguities)
            or any(t in _CLASS_SIGNAL for t in u.tokens)
        )
        return u

    # ── intents ─────────────────────────────────────────────────────────────

    def _detect_intents(self, u: Understanding, dictionary: SynonymDictionary) -> None:
        for intent in ("fee", "schedule", "out_of_scope", "distress", "superlative"):
            for phrase in dictionary.intent_terms(intent):
                if _contains_phrase(u.normalized, phrase):
                    u.intents.add(intent)
                    break

        raw = u.raw_message
        if _RM_AMOUNT.search(raw):
            u.intents.add("fee")

        if "fee" in u.intents:
            u.fee_intent = self._classify_touch(
                raw,
                interrogative=_INTERROGATIVE_FEE,
                has_amount=bool(_RM_AMOUNT.search(raw)),
            )
        if "schedule" in u.intents:
            u.schedule_intent = self._classify_touch(
                raw,
                interrogative=_INTERROGATIVE_TIME,
                has_amount=False,
            )

    @staticmethod
    def _classify_touch(
        raw: str, interrogative: re.Pattern[str], has_amount: bool
    ) -> str:
        """"asking for the value" vs "trying to narrow by it"."""
        if interrogative.search(raw):
            return "question"
        if has_amount or _CONSTRAINT_MARKER.search(raw):
            return "constraint"
        # Bare mention with no question word and no bound — e.g. "kelas murah".
        # Treated as a constraint: the user is expressing a preference, and we
        # can still show classes while stating that we hold no prices.
        return "constraint"

    # ── n-gram scan ─────────────────────────────────────────────────────────

    def _scan(self, u: Understanding, dictionary: SynonymDictionary) -> None:
        tokens = list(u.tokens)
        i = 0
        n_tokens = len(tokens)

        while i < n_tokens:
            width = min(MAX_NGRAM, n_tokens - i)
            consumed = 0

            while width > 0 and consumed == 0:
                fragment = " ".join(tokens[i : i + width])

                # A delivery-mode *group* term ("online", "bersemuka") is checked
                # first: it is not itself a canonical value, so the vocabulary
                # lookup below would miss it.
                group = dictionary.mode_group_for(fragment)
                if group is not None and u.mode_group is None:
                    u.mode_group = group
                    u.evidence["delivery_mode"] = fragment
                    consumed = width
                    break

                if width == 1 and dictionary.is_generic(fragment):
                    break

                for variant in _fragment_variants(fragment):
                    res = dictionary.resolve(variant)
                    if res.is_unknown:
                        continue
                    if self._accept(u, fragment, res, dictionary):
                        consumed = width
                    break

                if consumed:
                    break
                width -= 1

            if consumed:
                i += consumed
            else:
                token = tokens[i]
                if (
                    len(token) >= 3
                    and not token.isdigit()
                    and not dictionary.is_generic(token)
                ):
                    u.unresolved.append(token)
                i += 1

    def _accept(
        self,
        u: Understanding,
        fragment: str,
        res: Resolution,
        dictionary: SynonymDictionary,
    ) -> bool:
        """Record one resolution, or record why it could not be recorded."""
        if res.kind == "ambiguous":
            if not any(a.fragment == fragment for a in u.ambiguities):
                u.ambiguities.append(
                    Ambiguity(
                        fragment=fragment,
                        facet=res.facet or "subject",
                        candidates=res.candidates,
                        note=res.note or "",
                    )
                )
            return True

        facet, value = res.facet, res.value
        if facet is None or value is None:
            return False

        facet = self._disambiguate_facet(u, fragment, facet, res)
        if facet not in FILTERABLE_FACETS:
            return False

        # First value wins. A user who names two subjects is answered on the
        # first and the second lands in `unresolved`, which the response surfaces
        # rather than silently dropping.
        if facet in u.filters:
            return True

        u.filters[facet] = value
        u.evidence[facet] = fragment
        if res.kind == "fuzzy":
            u.fuzzy_facets[facet] = (fragment, res.score)
        return True

    def _disambiguate_facet(
        self, u: Understanding, fragment: str, facet: str, res: Resolution
    ) -> str:
        """Pick between facets that share a term.

        "English" is both a subject and an instruction language. "kelas English"
        means the subject; "diajar dalam English" means the medium. The decision
        uses the words immediately before the fragment.
        """
        alternates = {
            alt.split(":", 1)[0] for alt in res.candidates if ":" in alt
        }
        if "instruction_language" not in alternates | {facet}:
            return facet

        start = u.normalized.find(fragment)
        if start <= 0:
            return facet
        preceding = u.normalized[:start].strip().split()
        if not preceding:
            return facet
        tail = " ".join(preceding[-3:])
        for hint in self._dict.language_hints():
            if _contains_phrase(tail, hint):
                return "instruction_language"
        return facet

    # ── leftovers: fuzzy pass ───────────────────────────────────────────────

    def _resolve_leftovers(
        self, u: Understanding, dictionary: SynonymDictionary
    ) -> None:
        """FR-8: give unmatched fragments a second chance, with typos allowed.

        Runs only after the exact pass, and only on unigrams plus adjacent
        pairs. Fuzzing longer spans invites nonsense matches, and the exact pass
        has already had first refusal on everything.
        """
        if not u.unresolved:
            return

        leftovers = list(u.unresolved)
        u.unresolved = []

        i = 0
        while i < len(leftovers):
            handled = False
            if i + 1 < len(leftovers):
                pair = f"{leftovers[i]} {leftovers[i + 1]}"
                res = dictionary.resolve_fuzzy(pair)
                if not res.is_unknown and self._accept(u, pair, res, dictionary):
                    i += 2
                    continue

            for variant in _fragment_variants(leftovers[i]):
                res = dictionary.resolve_fuzzy(variant)
                if res.is_unknown:
                    continue
                handled = self._accept(u, leftovers[i], res, dictionary)
                break

            if not handled:
                u.unresolved.append(leftovers[i])
            i += 1

    # ── mode group expansion ────────────────────────────────────────────────

    def _apply_mode_group(self, u: Understanding, settings) -> None:
        """DD open question D4: "online" covers three modes — expand, not guess.

        An explicit mode ("Online live") always beats the group, so a precise
        user is never widened. Channel-agnostic modes are excluded by design —
        see the MODE_GROUPS comment in app/vocabulary.py.
        """
        if u.mode_group is None or "delivery_mode" in u.filters:
            return
        if u.mode_group not in CHANNEL_GROUPS:
            u.mode_group = None
            return
        modes = MODE_GROUPS.get(u.mode_group)
        if not modes:
            return
        u.filters["delivery_mode"] = list(modes)

    # ── FR-9 gate ───────────────────────────────────────────────────────────

    def _gate(self, u: Understanding) -> None:
        """Last line of defence before anything reaches SQL.

        Every value is re-checked against the closed vocabulary. Nothing that
        fails is passed through with a warning — it is removed from `filters`
        and turned into an ambiguity carrying the legal values, which is what
        FR-10's `available_values[]` renders as a dropdown.
        """
        from app.core.vocabulary import is_valid  # local import keeps the cycle out

        for facet in list(u.filters):
            value = u.filters[facet]
            values = value if isinstance(value, list) else [value]
            bad = [v for v in values if not is_valid(facet, str(v))]
            if not bad:
                continue
            del u.filters[facet]
            u.ambiguities.append(
                Ambiguity(
                    fragment=u.evidence.get(facet, str(bad[0])),
                    facet=facet,
                    candidates=tuple(values_for(facet)),
                    note="value outside closed vocabulary (FR-9)",
                )
            )


# ─── Module-level convenience ───────────────────────────────────────────────

_analyzer: Optional[QueryUnderstanding] = None


def get_analyzer() -> QueryUnderstanding:
    global _analyzer
    if _analyzer is None:
        _analyzer = QueryUnderstanding()
    return _analyzer


def reset_analyzer() -> None:
    global _analyzer
    _analyzer = None


def analyze(message: str) -> Understanding:
    return get_analyzer().analyze(message)
