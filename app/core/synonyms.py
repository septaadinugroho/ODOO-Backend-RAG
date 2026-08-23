"""Synonym dictionary with hot reload and typo tolerance.

NFR-H: "Kamus sinonim disimpan terpisah dan bisa diperbarui tanpa deploy
ulang." The dictionary therefore lives in `data/synonyms.json`, not in code, and
this module re-reads it whenever the file's mtime changes — no restart needed.

FR-7 requires the mapping to handle at minimum: Tingkatan 4 -> Form 4,
Add Math -> Additional Mathematics, Matematik -> Mathematics, tuisyen ->
tuition, with 95% accuracy on the test dictionary.

FR-8 requires typo tolerance: "add math", "adamath", and "add maths" must all
land on Additional Mathematics. That is handled by rapidfuzz over the union of
canonical vocabulary values and dictionary terms, gated by a score threshold and
a margin — see `Resolution` for why the margin matters.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Tuple

from rapidfuzz import fuzz, process

from app.config import get_settings
from app.core.textnorm import normalize
from app.core.vocabulary import CHANNEL_GROUPS, CLOSED_VOCABULARY

# `normalize` is re-exported here because it is part of this module's public
# surface historically and half the codebase imports it from here. Its body now
# lives in app/textnorm.py so that scripts/gen_sql.py can use the very same
# function with only the standard library available.
__all__ = [
    "normalize",
    "Resolution",
    "SynonymDictionary",
    "get_dictionary",
    "reset_dictionary",
]


# ─── Resolution result ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class Resolution:
    """Outcome of trying to turn a user fragment into a canonical value.

    `kind` drives what the caller does next:

      exact / synonym / fuzzy -> use `facet` + `value` as a filter
      ambiguous               -> FR-9/FR-10: ask, offering `candidates`
      unknown                 -> not vocabulary; caller ignores or clarifies

    `ambiguous` exists because guessing is the failure mode this product is
    built to avoid. When "chinese" could mean three different subjects in the
    corpus, returning one of them silently would produce a confidently wrong
    answer — exactly what FR-9 forbids.
    """

    kind: str
    facet: Optional[str] = None
    value: Optional[str] = None
    candidates: Tuple[str, ...] = ()
    matched_term: Optional[str] = None
    score: float = 0.0
    note: Optional[str] = None

    # Sentinel, attached after the class body so the dataclass machinery does
    # not mistake it for a field.
    UNKNOWN: ClassVar["Resolution"]

    @property
    def resolved(self) -> bool:
        return self.kind in ("exact", "synonym", "fuzzy")

    @property
    def is_unknown(self) -> bool:
        return self.kind == "unknown"


Resolution.UNKNOWN = Resolution(kind="unknown")


# ─── Dictionary ─────────────────────────────────────────────────────────────


@dataclass
class _Payload:
    """Everything parsed out of one revision of the JSON file."""

    version: str = "0"
    # (facet, normalized term) -> canonical value
    by_facet: Dict[Tuple[str, str], str] = field(default_factory=dict)
    # normalized term -> [(facet, canonical value)]  — for facet-free lookup
    by_term: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)
    # normalized term -> (facet, candidates, note)
    ambiguous: Dict[str, Tuple[str, Tuple[str, ...], str]] = field(default_factory=dict)
    # normalized term -> mode_group
    mode_groups: Dict[str, str] = field(default_factory=dict)
    generic_terms: frozenset = frozenset()
    intent_terms: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    language_hints: Tuple[str, ...] = ()
    # Fuzzy search space: normalized candidate -> (facet, canonical value)
    fuzzy_space: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    fuzzy_choices: Tuple[str, ...] = ()


class SynonymDictionary:
    """Thread-safe, hot-reloading view over data/synonyms.json."""

    def __init__(self, path: str | Path | None = None) -> None:
        settings = get_settings()
        self._path = Path(path or settings.synonyms_path)
        self._lock = threading.RLock()
        self._payload = _Payload()
        self._mtime: float = -1.0
        self._last_stat: float = 0.0
        self._reload_interval = settings.synonyms_reload_seconds
        self.reload_count = 0
        self._load()

    # ── loading ─────────────────────────────────────────────────────────────

    @property
    def path(self) -> Path:
        return self._path

    @property
    def version(self) -> str:
        return self._payload.version

    def _load(self) -> None:
        """Parse the file into an immutable payload, then swap it in.

        Built off to the side and assigned in one statement so a concurrent
        request never observes a half-built dictionary.
        """
        raw: dict = {}
        mtime = -1.0
        if self._path.exists():
            mtime = self._path.stat().st_mtime
            with self._path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)

        payload = _Payload(version=str(raw.get("version", "0")))

        for entry in raw.get("entries", []):
            facet = entry["facet"]
            term = normalize(entry["term"])
            canonical = entry["canonical_value"]
            if not term:
                continue
            payload.by_facet[(facet, term)] = canonical
            payload.by_term.setdefault(term, []).append((facet, canonical))

        for entry in raw.get("ambiguous", []):
            term = normalize(entry["term"])
            if not term:
                continue
            payload.ambiguous[term] = (
                entry["facet"],
                tuple(entry.get("candidates", ())),
                entry.get("note", ""),
            )

        for entry in raw.get("mode_group_terms", []):
            payload.mode_groups[normalize(entry["term"])] = entry["group"]

        payload.generic_terms = frozenset(
            normalize(t) for t in raw.get("generic_terms", [])
        )
        payload.intent_terms = {
            intent: tuple(normalize(t) for t in terms)
            for intent, terms in raw.get("intent_terms", {}).items()
        }
        payload.language_hints = tuple(
            normalize(t) for t in raw.get("language_hint_prefixes", [])
        )

        # Fuzzy space = canonical vocabulary first, then dictionary terms.
        # Canonical values are inserted first and never overwritten, so a
        # correctly-spelled official value always wins over a synonym.
        space: Dict[str, Tuple[str, str]] = {}
        for facet, values in CLOSED_VOCABULARY.items():
            for value in values:
                space.setdefault(normalize(value), (facet, value))
        for (facet, term), canonical in payload.by_facet.items():
            space.setdefault(term, (facet, canonical))
        # Ambiguous and generic terms are excluded: fuzzing onto them would turn
        # a deliberate "ask the user" into an accidental guess.
        for term in payload.ambiguous:
            space.pop(term, None)
        for term in payload.generic_terms:
            space.pop(term, None)

        payload.fuzzy_space = space
        payload.fuzzy_choices = tuple(space)

        with self._lock:
            self._payload = payload
            self._mtime = mtime
            self._last_stat = time.monotonic()
            self.reload_count += 1

    def reload_if_stale(self) -> bool:
        """Re-read the file if its mtime moved. Returns True if reloaded.

        The stat() itself is throttled to `synonyms_reload_seconds` so a busy
        endpoint does not hit the filesystem on every request while still
        picking up an edit within a couple of seconds (NFR-H).
        """
        now = time.monotonic()
        with self._lock:
            if now - self._last_stat < self._reload_interval:
                return False
            self._last_stat = now
            try:
                current = self._path.stat().st_mtime if self._path.exists() else -1.0
            except OSError:
                return False
            if current == self._mtime:
                return False
        self._load()
        return True

    def force_reload(self) -> None:
        self._load()

    # ── lookup ──────────────────────────────────────────────────────────────

    def _p(self) -> _Payload:
        with self._lock:
            return self._payload

    def is_generic(self, fragment: str) -> bool:
        return normalize(fragment) in self._p().generic_terms

    def mode_group_for(self, fragment: str) -> Optional[str]:
        """'bersemuka' -> 'physical'. DD open question D4 lives here.

        Only the four user-selectable channels are returned; "unspecified" is
        internal bookkeeping and can never come from a user's words.
        """
        group = self._p().mode_groups.get(normalize(fragment))
        return group if group in CHANNEL_GROUPS else None

    def intent_terms(self, intent: str) -> Tuple[str, ...]:
        return self._p().intent_terms.get(intent, ())

    def language_hints(self) -> Tuple[str, ...]:
        return self._p().language_hints

    def resolve(self, fragment: str, facet: Optional[str] = None) -> Resolution:
        """Exact then synonym then ambiguous lookup. No fuzzing.

        Kept separate from `resolve_fuzzy` so callers can require certainty in
        places where a near-miss is unacceptable.
        """
        term = normalize(fragment)
        if not term:
            return Resolution.UNKNOWN
        payload = self._p()

        # 1. Exact canonical value.
        facets = (facet,) if facet else tuple(CLOSED_VOCABULARY)
        for f in facets:
            for value in CLOSED_VOCABULARY.get(f, ()):
                if normalize(value) == term:
                    return Resolution(
                        kind="exact", facet=f, value=value, matched_term=term, score=100.0
                    )

        # 2. Dictionary entry.
        if facet is not None:
            canonical = payload.by_facet.get((facet, term))
            if canonical is not None:
                return Resolution(
                    kind="synonym",
                    facet=facet,
                    value=canonical,
                    matched_term=term,
                    score=100.0,
                )
        else:
            hits = payload.by_term.get(term)
            if hits:
                # One term can legitimately sit in two facets — "english" is
                # both a subject and an instruction language. That is not the
                # kind of ambiguity FR-9 is about (the caller disambiguates
                # from surrounding words), so the first mapping is returned and
                # the alternatives are carried along in `candidates`.
                f, canonical = hits[0]
                return Resolution(
                    kind="synonym",
                    facet=f,
                    value=canonical,
                    matched_term=term,
                    score=100.0,
                    candidates=tuple(f"{hf}:{hv}" for hf, hv in hits[1:]),
                )

        # 3. Known-ambiguous term -> refuse to guess (FR-9).
        amb = payload.ambiguous.get(term)
        if amb is not None:
            amb_facet, candidates, note = amb
            if facet is None or facet == amb_facet:
                return Resolution(
                    kind="ambiguous",
                    facet=amb_facet,
                    candidates=candidates,
                    matched_term=term,
                    note=note,
                )

        return Resolution.UNKNOWN

    def resolve_fuzzy(self, fragment: str, facet: Optional[str] = None) -> Resolution:
        """FR-8: accept small typos, refuse large ones.

        Two guards keep this from becoming a guessing machine:

        * a score floor, raised for short fragments where edit distance is a
          weak signal ("seni" vs "sains" differ by two edits);
        * a margin over the runner-up, so a fragment that sits between two
          canonical values is reported `unknown` instead of being snapped onto
          whichever one happened to score a fraction higher.
        """
        direct = self.resolve(fragment, facet)
        if not direct.is_unknown:
            return direct

        term = normalize(fragment)
        if not term:
            return Resolution.UNKNOWN
        payload = self._p()
        if not payload.fuzzy_choices:
            return Resolution.UNKNOWN

        settings = get_settings()
        choices = payload.fuzzy_choices
        if facet is not None:
            choices = tuple(
                c for c in choices if payload.fuzzy_space[c][0] == facet
            )
            if not choices:
                return Resolution.UNKNOWN

        floor = (
            settings.fuzzy_min_score_short
            if len(term) <= settings.fuzzy_short_term_chars
            else settings.fuzzy_min_score
        )

        matches = process.extract(
            term, choices, scorer=fuzz.WRatio, limit=3, score_cutoff=floor
        )
        if not matches:
            return Resolution.UNKNOWN

        best_choice, best_score, _ = matches[0]
        best_facet, best_value = payload.fuzzy_space[best_choice]

        runners = [
            (c, s)
            for c, s, _ in matches[1:]
            if payload.fuzzy_space[c][1] != best_value
        ]
        if runners and best_score - runners[0][1] < settings.fuzzy_min_margin:
            # Near-tie. WRatio is deliberately length-insensitive — that is what
            # makes it good at *finding* candidates and poor at *choosing*
            # between them, because "mathematics" and "additional mathematics"
            # both look plausible for a fragment containing "math". fuzz.ratio
            # restores the length penalty, so a fragment that is a misspelling of
            # the longer value stops resembling the shorter one equally.
            #
            # Abbreviations are not solved here and should not be: "adamath" is
            # closer to "mathematics" by any edit metric. Short forms belong in
            # the dictionary (FR-7), which is also why they are searchable — the
            # fuzzy space contains dictionary terms, not just canonical values.
            tied = [(best_choice, best_score), *runners]
            rescored: List[Tuple[str, float]] = []
            seen_values: set = set()
            for choice, _ in tied:
                value = payload.fuzzy_space[choice][1]
                if value in seen_values:
                    continue
                seen_values.add(value)
                rescored.append((choice, fuzz.ratio(term, choice)))
            rescored.sort(key=lambda t: -t[1])

            decisive = (
                len(rescored) > 1
                and rescored[0][1] - rescored[1][1] >= settings.fuzzy_min_margin
            )
            if not decisive:
                return Resolution(
                    kind="ambiguous",
                    facet=best_facet,
                    candidates=tuple(
                        dict.fromkeys(
                            payload.fuzzy_space[c][1] for c, _ in rescored
                        )
                    ),
                    matched_term=term,
                    score=best_score,
                    note="fuzzy match too close to call",
                )
            best_choice, best_score = rescored[0]
            best_facet, best_value = payload.fuzzy_space[best_choice]

        return Resolution(
            kind="fuzzy",
            facet=best_facet,
            value=best_value,
            matched_term=best_choice,
            score=best_score,
        )

    # ── introspection (for scripts/load_data.py and /health) ────────────────

    def all_entries(self) -> List[Tuple[str, str, str]]:
        """(term, facet, canonical_value) triples, for mirroring into SQL."""
        payload = self._p()
        return [(term, facet, value) for (facet, term), value in payload.by_facet.items()]

    def stats(self) -> Dict[str, int]:
        payload = self._p()
        return {
            "entries": len(payload.by_facet),
            "ambiguous": len(payload.ambiguous),
            "mode_group_terms": len(payload.mode_groups),
            "generic_terms": len(payload.generic_terms),
            "fuzzy_choices": len(payload.fuzzy_choices),
            "reloads": self.reload_count,
        }


# ─── Module-level singleton ─────────────────────────────────────────────────

_dictionary: Optional[SynonymDictionary] = None
_dict_lock = threading.Lock()


def get_dictionary() -> SynonymDictionary:
    """Process-wide dictionary, reloaded lazily when the file changes."""
    global _dictionary
    if _dictionary is None:
        with _dict_lock:
            if _dictionary is None:
                _dictionary = SynonymDictionary()
    else:
        _dictionary.reload_if_stale()
    return _dictionary


def reset_dictionary() -> None:
    """Drop the singleton. Used by tests that point at a fixture file."""
    global _dictionary
    with _dict_lock:
        _dictionary = None
