"""Data-honesty guardrails (FR-16 … FR-22).

This is the module the whole product leans on. Everything else can degrade
gracefully; if this layer lets a fabricated price through, the service has done
active harm to a parent making a decision about their child's education.

It runs *after* generation and has veto power. The order matters: the LLM is
never trusted to obey an instruction in a prompt, it is checked. NFR-D — "output
model tidak pernah dieksekusi" — is the same idea applied to text: model output
is data awaiting validation, not a result.

What is enforced
----------------
FR-16  No price may appear in a narrative. `fee_rm` is NULL on 200/200 records,
       so any "RM180" is by definition invented. Target 100%.
FR-17  No day, time, or date may be asserted. `schedule` is NULL on 200/200.
       A dated question ("any class on 21/8/2026?") must produce
       `unavailable_field`, never a guessed schedule.
FR-18  Every load-bearing fact in the narrative — provider name, result count,
       URL, record id, verification date — must be traceable to a row in
       `trace.retrieved_ids`. Failure flips the response to `degraded`.
FR-19  On failure the narrative is replaced by a safe template, and `results[]`
       stays complete. The user loses the prose, never the data.
FR-20  Fields we do not hold are named explicitly and the user is pointed at the
       provider's own page.
FR-21  `notice` is never empty and always says the corpus is a web-grounded
       example set rather than a live listing.
FR-22  No superlatives. NG8: the corpus holds no quality data, so "the best
       centre" is not a claim this service is entitled to make.

Why regex, and where that is not enough
---------------------------------------
Blocking a *pattern* is the right tool for FR-16 and FR-17 because the forbidden
thing has a shape: currency and a number, a clock time, a weekday. Two of the
checks are deliberately asymmetric:

* Naming the topic is fine, stating a value is not. "Kami tidak menyimpan harga"
  must pass; "RM180 sebulan" must not. So the fee check looks for amounts, not
  for the word "harga".
* Vague words ("murah", "petang") are only blocked when the answer does *not*
  also carry an unavailability disclaimer. Echoing the user's "yang petang je"
  while saying we hold no schedule is honest; saying "kelas ini petang" is not,
  and the disclaimer is what separates them.

For FR-18, grounding is checked against the retrieved rows rather than by
pattern, because the failure mode is a plausible-looking fact, not a malformed
one. A hallucinated provider name is perfectly well-formed; what makes it wrong
is that it is not in `retrieved_ids`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Set

from app.core.retrieval import ClassRow
from app.core.vocabulary import ALWAYS_NULL_FIELDS, REASON_ALLOWED_FIELDS

# ─── FR-21: the notice that never goes out empty ────────────────────────────
# Wording is close to the PRD's own: "data merupakan contoh berasaskan rujukan
# web, bukan senarai langsung daripada penyedia."
NOTICE_MS = (
    "Maklumat kelas ini ialah contoh berasaskan rujukan web, bukan senarai "
    "langsung daripada penyedia. Sahkan butiran terkini melalui pautan penyedia."
)
NOTICE_EN = (
    "These classes are web-grounded reference examples, not live listings from "
    "the providers. Please confirm current details via the provider link."
)

# FR-20: what to add when the user asked about something we do not hold.
UNAVAILABLE_MS = {
    "fee_rm": (
        "Kami tidak menyimpan harga untuk kelas ini, jadi kami tidak boleh "
        "menyebut sebarang jumlah. Sila tanya penyedia melalui pautan di atas."
    ),
    "schedule": (
        "Kami tidak menyimpan jadual, hari, atau waktu kelas, jadi kami tidak "
        "akan meneka. Sila semak dengan penyedia melalui pautan di atas."
    ),
}
UNAVAILABLE_EN = {
    "fee_rm": (
        "We do not hold fee information for these classes, so we cannot quote "
        "any amount. Please ask the provider using the link above."
    ),
    "schedule": (
        "We do not hold schedules, days, or times for these classes, so we will "
        "not guess. Please check with the provider using the link above."
    ),
}


def notice(lang: str = "ms") -> str:
    """FR-21. Always returns a non-empty string; there is no code path without one."""
    return NOTICE_EN if lang == "en" else NOTICE_MS


def unavailable_note(fields: Sequence[str], lang: str = "ms") -> str:
    table = UNAVAILABLE_EN if lang == "en" else UNAVAILABLE_MS
    return " ".join(table[f] for f in fields if f in table)


# ─── Forbidden-value patterns ───────────────────────────────────────────────

# FR-16. An amount, in any of the shapes Malaysian pricing is written in.
FEE_VALUE_PATTERNS = (
    re.compile(r"\brm\s?\d", re.IGNORECASE),
    re.compile(r"\bmyr\s?\d", re.IGNORECASE),
    re.compile(r"\bringgit\b", re.IGNORECASE),
    re.compile(r"[$€£]\s?\d"),
    re.compile(r"\bsen\b", re.IGNORECASE),
    # "180 sebulan", "250 per month", "300/bulan", "150 se-jam"
    re.compile(
        r"\b\d{2,}\s*(?:/|per\s|se\s?-?\s?)?\s*(?:bulan|month|jam|hour|sesi|session|kelas|class)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bfee\s*(?:is|:)?\s*\d", re.IGNORECASE),
    re.compile(r"\b(?:harga|kos|yuran|bayaran)\s*(?:nya)?\s*(?:ialah|adalah|is|:)?\s*\d",
               re.IGNORECASE),
)

# FR-16, soft: a price *judgement* with no price data behind it.
FEE_CLAIM_PATTERNS = (
    re.compile(r"\b(murah|mahal|berpatutan|jimat|affordable|cheap|expensive|budget[- ]friendly)\b",
               re.IGNORECASE),
    re.compile(r"\b(free|gratis|percuma|diskaun|discount|promo)\b", re.IGNORECASE),
)

# FR-17, hard: an actual time, day, or date.
SCHEDULE_VALUE_PATTERNS = (
    re.compile(r"\b\d{1,2}\s?[:.]\s?\d{2}\b"),  # 20:00, 8.30
    re.compile(r"\b\d{1,2}\s?(?:am|pm|a\.m\.?|p\.m\.?)\b", re.IGNORECASE),
    re.compile(r"\b(?:pukul|jam)\s*\d", re.IGNORECASE),
    re.compile(
        r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday"
        r"|isnin|selasa|rabu|khamis|jumaat|sabtu|ahad|minggu)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(weekend|weekday|hujung minggu|hari bekerja)\b", re.IGNORECASE),
    re.compile(r"\b\d{1,2}\s*[/-]\s*\d{1,2}\s*[/-]\s*\d{2,4}\b"),  # 21/8/2026
    re.compile(
        r"\b\d{1,2}\s+(jan|feb|mac|mar|apr|mei|may|jun|jul|aug|ogos|sep|okt|oct"
        r"|nov|dis|dec)\w*\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:setiap|every)\s+(?:hari|day|week|minggu)\b", re.IGNORECASE),
    re.compile(r"\b\d+\s*(?:kali|times|sesi|sessions)\s*(?:se|per|a)\s*\w*(?:minggu|week)\b",
               re.IGNORECASE),
)

# FR-17, soft: a part of the day. Fine inside a disclaimer, not as a claim.
SCHEDULE_VAGUE_PATTERNS = (
    re.compile(r"\b(pagi|petang|malam|tengah hari|morning|afternoon|evening|night)\b",
               re.IGNORECASE),
)

# FR-22 / NG8: quality claims the corpus cannot support.
SUPERLATIVE_PATTERNS = (
    re.compile(r"\b(terbaik|paling (?:bagus|baik|hebat|berkesan|popular)|terhebat"
               r"|nombor satu|no\.?\s?1)\b", re.IGNORECASE),
    re.compile(r"\b(best|top(?:-| )?(?:rated|choice|pick)?|greatest|finest|leading"
               r"|most popular|highest[- ]quality|premier)\b", re.IGNORECASE),
    re.compile(r"\b(kami (?:sangat )?(?:cadangkan|syorkan|recommend)"
               r"|we (?:highly )?recommend)\b", re.IGNORECASE),
    re.compile(r"\b(dijamin|guaranteed|pasti (?:lulus|A)|sure to|will improve)\b",
               re.IGNORECASE),
    re.compile(r"\b\d(?:\.\d)?\s*(?:\/\s*5|stars?|bintang)\b", re.IGNORECASE),
)

# Phrases that mark the answer as saying "we do not have this". Their presence
# is what makes a vague time or price word acceptable.
DISCLAIMER_MARKERS = (
    re.compile(r"\b(tidak|tiada|bukan|belum|tak)\s+\w*\s*(simpan|ada|tersedia|maklumat"
               r"|menyimpan|merekod|termasuk|dapat|boleh)", re.IGNORECASE),
    re.compile(r"\b(?:we )?(?:do not|don't|cannot|can't|no)\s+\w*\s*"
               r"(?:hold|have|store|record|know|available|data|information)", re.IGNORECASE),
    re.compile(r"\btidak (?:menyimpan|tersedia|dinyatakan)\b", re.IGNORECASE),
    re.compile(r"\bnot available\b", re.IGNORECASE),
)

_PROVIDER_KEYWORDS = (
    "academy", "akademi", "tuition", "tuisyen", "centre", "center", "pusat",
    "learning", "education", "institute", "institut", "enrichment", "school",
    "sekolah", "college", "kolej", "tutors", "tutorial", "studio",
)
# Two or more capitalised words in a row. Lowercase connectors ("dan", "and")
# are deliberately *not* allowed inside a span: permitting them would glue two
# adjacent provider names into one unrecognisable string and make every mention
# look hallucinated.
_PROVIDER_SPAN = re.compile(r"\b([A-Z][\w&.'’\-]*(?:\s+[A-Z][\w&.'’\-]*){1,4})")
_URL = re.compile(r"https?://[^\s,;)\]\"'<>]+")
_RECORD_ID = re.compile(r"\bMYTC-\d{3}\b")
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_COUNT_CLAIM = re.compile(
    r"\b(\d{1,3})\s+(?:kelas|class(?:es)?|hasil|result(?:s)?|pilihan|option(?:s)?|"
    r"opsyen|item(?:s)?)\b",
    re.IGNORECASE,
)
_WORD_NUMBER = {
    "satu": 1, "dua": 2, "tiga": 3, "empat": 4, "lima": 5, "enam": 6,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
}
_WORD_COUNT_CLAIM = re.compile(
    r"\b(" + "|".join(_WORD_NUMBER) + r")\s+(?:kelas|class(?:es)?|hasil|result(?:s)?|"
    r"pilihan|option(?:s)?)\b",
    re.IGNORECASE,
)


# ─── Report ─────────────────────────────────────────────────────────────────


@dataclass
class GroundingReport:
    """Outcome of validating one narrative.

    `failures` holds stable machine-readable codes rather than prose, because
    FR-30 requires every degraded response to leave a trace that says *which*
    validation failed, and NFR-G wants that aggregatable in a dashboard.
    """

    failures: List[str] = field(default_factory=list)
    details: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    def add(self, code: str, detail: str = "") -> None:
        if code not in self.failures:
            self.failures.append(code)
        if detail:
            self.details.append(detail)

    def merge(self, other: "GroundingReport") -> None:
        for code in other.failures:
            if code not in self.failures:
                self.failures.append(code)
        self.details.extend(other.details)


# ─── Individual checks ──────────────────────────────────────────────────────


def has_disclaimer(text: str) -> bool:
    return any(p.search(text) for p in DISCLAIMER_MARKERS)


def check_fee_silence(text: str, disclaimed: Optional[bool] = None) -> GroundingReport:
    """FR-16. Target 100%: no amount, ever."""
    report = GroundingReport()
    disclaimed = has_disclaimer(text) if disclaimed is None else disclaimed

    for pattern in FEE_VALUE_PATTERNS:
        match = pattern.search(text)
        if match:
            report.add("fr16_fee_amount", f"fee-like value {match.group(0)!r}")
    if not disclaimed:
        for pattern in FEE_CLAIM_PATTERNS:
            match = pattern.search(text)
            if match:
                report.add(
                    "fr16_fee_claim",
                    f"price judgement {match.group(0)!r} with no fee data and no disclaimer",
                )
    return report


def check_schedule_silence(
    text: str, disclaimed: Optional[bool] = None
) -> GroundingReport:
    """FR-17. Days, times and dates are values we do not have."""
    report = GroundingReport()
    disclaimed = has_disclaimer(text) if disclaimed is None else disclaimed

    for pattern in SCHEDULE_VALUE_PATTERNS:
        match = pattern.search(text)
        if match:
            report.add("fr17_schedule_value", f"schedule-like value {match.group(0)!r}")
    if not disclaimed:
        for pattern in SCHEDULE_VAGUE_PATTERNS:
            match = pattern.search(text)
            if match:
                report.add(
                    "fr17_schedule_hint",
                    f"time-of-day {match.group(0)!r} asserted without a disclaimer",
                )
    return report


def check_no_superlatives(text: str) -> GroundingReport:
    """FR-22 / NG8. The corpus holds no quality signal, so neither may we."""
    report = GroundingReport()
    for pattern in SUPERLATIVE_PATTERNS:
        match = pattern.search(text)
        if match:
            report.add("fr22_quality_claim", f"unsupported claim {match.group(0)!r}")
    return report


def check_grounding(
    text: str,
    rows: Sequence[ClassRow],
    known_providers: Iterable[str] = (),
) -> GroundingReport:
    """FR-18. Every fact in the narrative must point at a retrieved row.

    `known_providers` is the full provider list from the database. It exists
    because the most likely hallucination is not an invented name but a *real*
    one from a different part of the corpus — a model that has seen "Warna
    Cemerlang" in an earlier turn is quite capable of attaching it to a class
    from another provider. An invented name is caught by the keyword heuristic; a
    misattributed real one is caught by this list.
    """
    report = GroundingReport()

    allowed_ids = {r.record_id for r in rows}
    allowed_urls = {r.provider_source_url.rstrip("/") for r in rows}
    allowed_providers = {r.provider_name for r in rows}
    allowed_titles = {r.class_title for r in rows}
    allowed_dates = {r.verified_on.isoformat() for r in rows}

    for rid in set(_RECORD_ID.findall(text)):
        if rid not in allowed_ids:
            report.add("fr18_unknown_record_id", f"{rid} is not in retrieved_ids")

    for url in set(_URL.findall(text)):
        if url.rstrip("/") not in allowed_urls:
            report.add("fr18_unknown_url", f"{url} is not a retrieved provider URL")

    for iso in set(_ISO_DATE.findall(text)):
        if iso not in allowed_dates:
            report.add(
                "fr18_unknown_date",
                f"{iso} is not the verified_on of any retrieved record",
            )

    for name in known_providers:
        if name in allowed_providers:
            continue
        if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text):
            report.add(
                "fr18_unknown_provider",
                f"provider {name!r} exists in the corpus but was not retrieved for "
                "this query",
            )

    for span in _provider_like_spans(text):
        if any(name in span for name in allowed_providers):
            continue
        # A class title can contain provider-ish words too; a title that came
        # from a retrieved row is grounded by definition.
        if any(span in title or title in span for title in allowed_titles):
            continue
        report.add(
            "fr18_unknown_provider",
            f"{span!r} reads like a provider name but matches no retrieved record",
        )

    expected = len(rows)
    for raw in _COUNT_CLAIM.findall(text):
        if int(raw) != expected:
            report.add(
                "fr18_count_mismatch",
                f"answer claims {raw} classes, {expected} were retrieved",
            )
    for word in _WORD_COUNT_CLAIM.findall(text):
        if _WORD_NUMBER[word.lower()] != expected:
            report.add(
                "fr18_count_mismatch",
                f"answer claims {word!r} classes, {expected} were retrieved",
            )

    return report


def _provider_like_spans(text: str) -> Set[str]:
    """Capitalised spans that look like the name of a tuition provider.

    Deliberately narrow. A general "is this a proper noun" test would flag
    "Additional Mathematics" and "Selangor" and make the validator useless, so a
    span only counts if it contains one of the words tuition-provider names are
    built from. That is enough to catch the failure this check exists for: a model
    naming a centre that was never retrieved.

    Spans are matched by containment rather than equality upstream, because a
    sentence-initial word is capitalised too — "Kelas Edubest Academy sesuai"
    yields the span "Kelas Edubest Academy", which is a legitimate mention of a
    retrieved provider and must not be treated as an invented one.
    """
    found: Set[str] = set()
    for match in _PROVIDER_SPAN.finditer(text):
        span = match.group(1).strip(" .,'’-")
        if len(span.split()) < 2:
            continue
        lowered = span.lower()
        if any(keyword in lowered for keyword in _PROVIDER_KEYWORDS):
            found.add(span)
    return found


def check_rows_are_null_where_promised(rows: Sequence[ClassRow]) -> GroundingReport:
    """FR-16 / FR-17 at the storage layer, and DD V8.

    The response contract types `fee_rm` and `schedule` as `None`, so a value
    arriving from the database would fail Pydantic validation with a stack trace
    instead of a sentence. Checking here turns that into a controlled degraded
    response — and, more importantly, into a log line saying the v1 assumption
    ("kosong 200/200") no longer holds and the contract needs revisiting. DD V8
    treats a *populated* value as the alarm for exactly this reason.
    """
    report = GroundingReport()
    for row in rows:
        for field_name in sorted(ALWAYS_NULL_FIELDS):
            if getattr(row, field_name, None) not in (None, ""):
                report.add(
                    "v8_unexpected_value",
                    f"{row.record_id}.{field_name} is populated; v1 assumes it is NULL "
                    "for all records (DD V8)",
                )
    return report


def validate_match_reason(reason: str, row: ClassRow) -> GroundingReport:
    """FR-13. A reason may only cite mode, language, location, and curriculum.

    Checked by construction *and* here, because the narrative model is allowed to
    rephrase a reason and a rephrase is a chance to add a fact.
    """
    report = GroundingReport()
    if not reason:
        return report

    report.merge(check_fee_silence(reason, disclaimed=False))
    report.merge(check_schedule_silence(reason, disclaimed=False))
    report.merge(check_no_superlatives(reason))

    lowered = reason.lower()
    cited = [
        f for f in REASON_ALLOWED_FIELDS
        if (row.facet(f) or "").lower() in lowered and row.facet(f)
    ]
    if not cited:
        # Not a hard failure: a reason may legitimately be phrased without
        # quoting a value verbatim ("diajar dalam dua bahasa"). Recorded so the
        # eval harness can see how often reasons stop being checkable.
        report.details.append(f"reason for {row.record_id} cites no allowed facet verbatim")
    return report


# ─── Composite ──────────────────────────────────────────────────────────────


def validate_narrative(
    text: str,
    rows: Sequence[ClassRow],
    *,
    allow_unavailable: Iterable[str] = (),
    known_providers: Iterable[str] = (),
) -> GroundingReport:
    """Run every check that applies to a generated narrative.

    `allow_unavailable` names the fields the user themselves raised. It does not
    loosen FR-16 or FR-17 — no amount or time becomes permissible — it only means
    the answer is expected to discuss the absence, so the soft patterns are
    evaluated with the disclaimer requirement in mind.
    """
    report = GroundingReport()
    if not text or not text.strip():
        report.add("empty_answer", "narrative was empty")
        return report

    disclaimed = has_disclaimer(text)
    if set(allow_unavailable) & ALWAYS_NULL_FIELDS and not disclaimed:
        # The user asked about fee or schedule and the answer did not say we do
        # not hold it. FR-20 requires the absence to be explicit.
        report.add(
            "fr20_absence_not_stated",
            "user raised an unavailable field but the answer does not say so",
        )

    report.merge(check_fee_silence(text, disclaimed=disclaimed))
    report.merge(check_schedule_silence(text, disclaimed=disclaimed))
    report.merge(check_no_superlatives(text))
    report.merge(check_grounding(text, rows, known_providers))
    report.merge(check_rows_are_null_where_promised(rows))
    return report


# ─── FR-19: the safe fallback ───────────────────────────────────────────────


def safe_answer(
    rows: Sequence[ClassRow],
    lang: str = "ms",
    unavailable_fields: Sequence[str] = (),
) -> str:
    """A narrative assembled only from record values — no model involved.

    Used whenever the generated narrative fails validation (FR-19) and whenever
    the LLM is unavailable (NFR-C: "Pencarian berbasis filter tetap berjalan").
    It is built by string concatenation from fields that came out of the
    database, so it cannot fail the guardrails that just rejected the model — the
    only numbers it contains are a count it computes itself and levels/years that
    are corpus values.

    `results[]` is untouched in both cases. The user loses the prose, not the
    data.
    """
    count = len(rows)
    if lang == "en":
        if not rows:
            body = "No class in the corpus matches every condition you gave."
        else:
            head = (
                f"{count} class{'es' if count != 1 else ''} match your criteria. "
                "Details for each one are in the list below, straight from the record."
            )
            body = head
    else:
        if not rows:
            body = "Tiada kelas dalam korpus yang memenuhi semua syarat yang anda beri."
        else:
            body = (
                f"Ada {count} kelas yang menepati kriteria anda. Butiran setiap "
                "kelas ada dalam senarai di bawah, terus daripada rekod."
            )

    extra = unavailable_note(unavailable_fields, lang)
    return f"{body} {extra}".strip() if extra else body


def enforce(
    text: str,
    rows: Sequence[ClassRow],
    lang: str = "ms",
    unavailable_fields: Sequence[str] = (),
    known_providers: Iterable[str] = (),
) -> "EnforceResult":
    """Validate a narrative and substitute a safe one if it fails (FR-18 + FR-19).

    Returns the text to send, whether the response is degraded, and the failure
    codes for the trace. Callers do not get the option of shipping an unvalidated
    narrative: this function is the only path from generated text to a response.
    """
    report = validate_narrative(
        text,
        rows,
        allow_unavailable=unavailable_fields,
        known_providers=known_providers,
    )
    if report.ok:
        return EnforceResult(answer=text, degraded=False, report=report)
    return EnforceResult(
        answer=safe_answer(rows, lang, unavailable_fields),
        degraded=True,
        report=report,
    )


@dataclass
class EnforceResult:
    answer: str
    degraded: bool
    report: GroundingReport

    @property
    def failures(self) -> List[str]:
        return list(self.report.failures)
