"""FR-16, FR-17, FR-18 and FR-22 — the guardrails, at 100% coverage.

These are the requirements the PRD singles out for total coverage (NFR-H), and
for good reason: every other requirement failing produces a worse answer, but
these failing produce a *confident wrong* answer about a child's education. A
hallucinated fee is not a formatting bug.

The tests are written as adversarial input rather than as coverage exercises.
Each one is a sentence a language model plausibly produces, and the assertion is
that the guardrail refuses it. Where a test asserts that something is *allowed*,
that matters just as much: a validator that rejects legitimate answers gets
switched off, and then it protects nothing.
"""

from __future__ import annotations

import dataclasses

import pytest

from app.core import honesty
from app.core.vocabulary import ALWAYS_NULL_FIELDS


# ─── FR-16: no amount, ever ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "Kelas ini RM250 sebulan.",
        "Fee: 300 per month for the Form 4 group.",
        "Harganya ialah 180 sebulan.",
        "Kos MYR 220 untuk setiap sesi.",
        "Bayaran 150 se-jam sahaja.",
        "It costs $80 a session.",
        "Yuran 200 sen tambahan.",
        "Ringgit tambahan dikenakan untuk bahan.",
        "fee is 250",
        "Kelas 12 kelas berharga 250/bulan.",
    ],
)
def test_fee_amounts_are_always_refused(text: str) -> None:
    report = honesty.check_fee_silence(text)
    assert not report.ok
    assert "fr16_fee_amount" in report.failures


@pytest.mark.parametrize(
    "text",
    [
        "Kelas ini sangat murah untuk pelajar Form 4.",
        "An affordable option in Selangor.",
        "Ada diskaun untuk pendaftaran awal.",
        "Kelas percuma untuk minggu pertama.",
        "This is a budget-friendly centre.",
    ],
)
def test_price_judgements_need_a_disclaimer(text: str) -> None:
    """A word like "murah" is a claim about a price we do not hold."""
    report = honesty.check_fee_silence(text)
    assert "fr16_fee_claim" in report.failures


def test_price_word_is_allowed_alongside_the_absence_statement() -> None:
    """The honest sentence must survive.

    "We do not store fees, so we cannot say whether it is cheap" contains the
    word "murah" and is exactly what FR-20 asks for. A validator that rejected it
    would push the answer toward silence instead of honesty.
    """
    text = (
        "Kami tidak menyimpan harga kelas ini, jadi kami tidak boleh kata sama ada "
        "ia murah atau mahal. Sila tanya penyedia."
    )
    assert honesty.has_disclaimer(text)
    assert honesty.check_fee_silence(text).ok


def test_amount_is_refused_even_inside_a_disclaimer() -> None:
    """The soft rule relaxes with a disclaimer. The hard rule never does."""
    text = "Kami tidak menyimpan harga, tetapi biasanya sekitar RM200 sebulan."
    report = honesty.check_fee_silence(text)
    assert "fr16_fee_amount" in report.failures


@pytest.mark.parametrize(
    "text",
    [
        "Kelas Additional Mathematics Form 4 di Selangor secara Online live.",
        "There are 2 classes for Form 5 Physics in Johor.",
        "Rekod ini disemak pada 2026-08-18.",
    ],
)
def test_ordinary_answers_pass_fee_checks(text: str) -> None:
    assert honesty.check_fee_silence(text).ok


# ─── FR-17: no day, time, or date ───────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "Kelas bermula 20:00 setiap malam.",
        "Sessions run at 8.30 in the hall.",
        "Kelas pukul 8 malam.",
        "Available 7pm onwards.",
        "Kelas diadakan setiap hari Sabtu.",
        "Classes are on Monday and Wednesday.",
        "Hujung minggu sahaja.",
        "Weekend intensive programme.",
        "Bermula 21/8/2026 di pusat tuisyen.",
        "Intake opens 15 Ogos.",
        "Kelas setiap minggu di Petaling Jaya.",
        "2 sesi seminggu untuk Form 4.",
    ],
)
def test_schedule_values_are_always_refused(text: str) -> None:
    report = honesty.check_schedule_silence(text)
    assert not report.ok
    assert "fr17_schedule_value" in report.failures


@pytest.mark.parametrize(
    "text",
    ["Kelas diadakan pada waktu petang.", "An evening class for Form 5.", "Sesi pagi."],
)
def test_time_of_day_needs_a_disclaimer(text: str) -> None:
    assert "fr17_schedule_hint" in honesty.check_schedule_silence(text).failures


def test_time_of_day_is_allowed_when_stated_as_unknown() -> None:
    text = (
        "Kami tidak menyimpan jadual kelas, jadi kami tidak tahu sama ada sesi "
        "diadakan pada waktu petang. Sila semak dengan penyedia."
    )
    assert honesty.check_schedule_silence(text).ok


def test_verified_on_date_is_not_a_schedule() -> None:
    """FR-14's date must not trip FR-17.

    `verified_on` is a fact about the record, and it is written in ISO form
    precisely so it cannot be read as a class time. If this test fails, the
    service can no longer state when a record was checked.
    """
    assert honesty.check_schedule_silence("Rekod disemak pada 2026-08-18.").ok


# ─── FR-22: no ranking, no promises ─────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "Ini kelas terbaik di Selangor.",
        "Pusat tuisyen paling popular di Johor.",
        "The best physics tuition available.",
        "Our top-rated centre for Form 5.",
        "Kami cadangkan kelas ini.",
        "We highly recommend this provider.",
        "Keputusan A dijamin.",
        "Rated 4.8/5 by parents.",
        "Nombor satu di Malaysia.",
        "This will improve your grade.",
    ],
)
def test_quality_claims_are_refused(text: str) -> None:
    report = honesty.check_no_superlatives(text)
    assert "fr22_quality_claim" in report.failures


def test_stating_that_we_do_not_rank_is_allowed() -> None:
    text = (
        "Kami tidak menilai kualiti kelas kerana korpus tiada data penilaian, "
        "jadi senarai ini tidak disusun mengikut kedudukan."
    )
    assert honesty.check_no_superlatives(text).ok


# ─── FR-18: every fact points at a retrieved record ─────────────────────────


def test_unknown_record_id_is_caught(rows) -> None:
    subset = rows[:2]
    report = honesty.check_grounding("Cuba MYTC-999 juga.", subset)
    assert "fr18_unknown_record_id" in report.failures


def test_retrieved_record_id_is_allowed(rows) -> None:
    subset = rows[:2]
    text = f"Rekod {subset[0].record_id} sepadan dengan carian anda."
    assert honesty.check_grounding(text, subset).ok


def test_unknown_url_is_caught(rows) -> None:
    report = honesty.check_grounding("Lihat https://evil.example/signup", rows[:1])
    assert "fr18_unknown_url" in report.failures


def test_retrieved_url_is_allowed(rows) -> None:
    row = rows[0]
    assert honesty.check_grounding(f"Pautan: {row.provider_source_url}", [row]).ok


def test_unknown_date_is_caught(rows) -> None:
    report = honesty.check_grounding("Disemak pada 2019-01-01.", rows[:1])
    assert "fr18_unknown_date" in report.failures


def test_verified_on_of_a_retrieved_row_is_allowed(rows) -> None:
    row = rows[0]
    text = f"Rekod disemak pada {row.verified_on.isoformat()}."
    assert honesty.check_grounding(text, [row]).ok


def test_invented_provider_name_is_caught(rows) -> None:
    """The keyword heuristic: a centre nobody retrieved."""
    report = honesty.check_grounding(
        "Cuba Cemerlang Elite Academy di Shah Alam.", rows[:1]
    )
    assert "fr18_unknown_provider" in report.failures


def test_real_but_unretrieved_provider_is_caught(db, rows) -> None:
    """The failure the keyword heuristic cannot see (FR-18, hardest case).

    The likeliest hallucination is not an invented name but a real one from
    another part of the corpus, attached to the wrong class. Only the full
    provider list catches it, which is why `retrieval.known_providers()` exists.
    """
    from app.core.retrieval import known_providers

    providers = known_providers(db)
    target = next(r for r in rows if r.provider_name != rows[0].provider_name)
    text = f"Kelas ini dikendalikan oleh {target.provider_name}."

    # Without the corpus-wide list the misattribution can slip through; with it,
    # it cannot.
    report = honesty.check_grounding(text, [rows[0]], known_providers=providers)
    assert "fr18_unknown_provider" in report.failures


def test_retrieved_provider_name_is_allowed(db, rows) -> None:
    from app.core.retrieval import known_providers

    row = rows[0]
    report = honesty.check_grounding(
        f"Kelas {row.provider_name} sesuai dengan carian anda.",
        [row],
        known_providers=known_providers(db),
    )
    assert report.ok, report.details


def test_sentence_initial_capital_is_not_a_provider(rows) -> None:
    """A false positive here would degrade every second answer.

    "Kelas Edubest Academy sesuai" yields the span "Kelas Edubest Academy",
    which contains a retrieved provider and must be accepted.
    """
    row = next(r for r in rows if "Academy" in r.provider_name)
    assert honesty.check_grounding(f"Kelas {row.provider_name} sesuai.", [row]).ok


@pytest.mark.parametrize("claim,expected_ok", [("5 kelas", False), ("2 kelas", True)])
def test_count_claims_must_match_the_retrieved_set(rows, claim, expected_ok) -> None:
    subset = rows[:2]
    report = honesty.check_grounding(f"Ada {claim} yang sepadan.", subset)
    assert report.ok is expected_ok
    if not expected_ok:
        assert "fr18_count_mismatch" in report.failures


def test_spelled_out_count_is_checked_too(rows) -> None:
    subset = rows[:2]
    assert "fr18_count_mismatch" in honesty.check_grounding(
        "Ada tiga kelas yang sepadan.", subset
    ).failures
    assert honesty.check_grounding("Ada dua kelas yang sepadan.", subset).ok


# ─── DD V8: the storage-level assumption ────────────────────────────────────


def test_fee_and_schedule_are_null_for_every_fixture_row(rows) -> None:
    assert honesty.check_rows_are_null_where_promised(rows).ok
    assert ALWAYS_NULL_FIELDS == frozenset({"fee_rm", "schedule"})


@pytest.mark.parametrize("field_name,value", [("fee_rm", 250.0), ("schedule", "Sat 10am")])
def test_a_populated_row_raises_the_v8_alarm(rows, field_name, value) -> None:
    """If the corpus ever gains a fee, this is the alarm — not a stack trace."""
    poisoned = dataclasses.replace(rows[0], **{field_name: value})
    report = honesty.check_rows_are_null_where_promised([poisoned])
    assert "v8_unexpected_value" in report.failures


# ─── FR-13: match reasons cite only the allowed facets ──────────────────────


def test_match_reason_may_cite_mode_and_language(rows) -> None:
    row = rows[0]
    reason = f"Padanan mod: {row.delivery_mode} · bahasa pengantar: {row.instruction_language}"
    assert honesty.validate_match_reason(reason, row).ok


def test_match_reason_may_not_smuggle_a_price(rows) -> None:
    row = rows[0]
    report = honesty.validate_match_reason(f"Mod {row.delivery_mode}, RM199 sebulan", row)
    assert "fr16_fee_amount" in report.failures


def test_match_reason_may_not_smuggle_a_ranking(rows) -> None:
    row = rows[0]
    report = honesty.validate_match_reason("Pilihan terbaik untuk anda", row)
    assert "fr22_quality_claim" in report.failures


def test_empty_match_reason_is_not_a_failure(rows) -> None:
    assert honesty.validate_match_reason("", rows[0]).ok


# ─── FR-19: the safe answer, and the veto that installs it ──────────────────


@pytest.mark.parametrize("lang", ["ms", "en"])
def test_safe_answer_passes_its_own_guardrails(rows, lang) -> None:
    """The fallback must never be the thing that fails validation.

    `safe_answer` is what replaces a rejected narrative. If it could itself fail,
    a degraded response would have nothing to fall back to.
    """
    text = honesty.safe_answer(rows[:3], lang)
    assert honesty.validate_narrative(text, rows[:3]).ok


@pytest.mark.parametrize("lang", ["ms", "en"])
@pytest.mark.parametrize("fields", [("fee_rm",), ("schedule",), ("fee_rm", "schedule")])
def test_safe_answer_states_the_absence(rows, lang, fields) -> None:
    text = honesty.safe_answer(rows[:2], lang, unavailable_fields=list(fields))
    assert honesty.has_disclaimer(text)
    assert honesty.validate_narrative(
        text, rows[:2], allow_unavailable=list(fields)
    ).ok


@pytest.mark.parametrize("lang", ["ms", "en"])
def test_safe_answer_with_no_rows_says_so(rows, lang) -> None:
    text = honesty.safe_answer([], lang)
    assert text.strip()
    assert honesty.validate_narrative(text, []).ok


def test_enforce_accepts_a_clean_narrative(rows) -> None:
    subset = rows[:1]
    row = subset[0]
    clean = (
        f"Ada 1 kelas yang sepadan: {row.class_title} oleh {row.provider_name} "
        f"di {row.city_area}, secara {row.delivery_mode}."
    )
    result = honesty.enforce(clean, subset)
    assert not result.degraded
    assert result.answer == clean


def test_enforce_substitutes_a_safe_answer_when_validation_fails(rows) -> None:
    subset = rows[:2]
    dirty = "Ini kelas terbaik, hanya RM150 sebulan, setiap hari Sabtu pukul 10 pagi."
    result = honesty.enforce(dirty, subset)

    assert result.degraded
    assert result.answer != dirty
    # The user loses the prose, not the data: the substitute is still a real
    # answer about the same rows.
    assert str(len(subset)) in result.answer
    assert {"fr16_fee_amount", "fr17_schedule_value", "fr22_quality_claim"} <= set(
        result.failures
    )
    # And the substitute passes what the original failed.
    assert honesty.validate_narrative(result.answer, subset).ok


def test_enforce_rejects_an_empty_narrative(rows) -> None:
    result = honesty.enforce("   ", rows[:1])
    assert result.degraded
    assert "empty_answer" in result.failures


def test_unavailable_field_must_be_acknowledged(rows) -> None:
    """FR-20: answering around the question is a failure, not a workaround."""
    evasive = "Ada 2 kelas Additional Mathematics untuk Form 4 di Selangor."
    report = honesty.validate_narrative(
        evasive, rows[:2], allow_unavailable=["fee_rm"]
    )
    assert "fr20_absence_not_stated" in report.failures


# ─── FR-21: the notice ──────────────────────────────────────────────────────


@pytest.mark.parametrize("lang,expected", [("ms", honesty.NOTICE_MS), ("en", honesty.NOTICE_EN)])
def test_notice_is_never_empty(lang, expected) -> None:
    assert honesty.notice(lang) == expected
    assert honesty.notice(lang).strip()


def test_notice_falls_back_to_malay_for_an_unknown_language() -> None:
    assert honesty.notice("zz") == honesty.NOTICE_MS


def test_unavailable_note_covers_both_fields_and_ignores_others() -> None:
    both = honesty.unavailable_note(["fee_rm", "schedule"], "ms")
    assert "harga" in both and "jadual" in both
    assert honesty.unavailable_note(["subject"], "ms") == ""
    assert honesty.unavailable_note([], "en") == ""
