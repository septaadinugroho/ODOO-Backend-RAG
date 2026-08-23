#!/usr/bin/env python3
"""FR-31 — one command runs the whole eval set and prints the M1–M9 report.

    python scripts/run_eval.py                    # against the configured LLM
    python scripts/run_eval.py --llm null         # no model: templates only
    python scripts/run_eval.py --only trap-       # just the trap queries
    python scripts/run_eval.py --metric M4        # just the cases that feed M4
    python scripts/run_eval.py --json out.json --md out.md

The harness drives the real FastAPI app in-process through `TestClient`, so what
it measures is the same code path a caller hits — auth, middleware, session
handling, guardrails and all. It does not stub the service, because the whole
point of M1 is to check what actually leaves the process.

**Two layers of judgement.** The metric table is the PRD deliverable: nine
numbers against nine targets. Underneath it, each case is PASS/FAIL/SKIP against
its own expectations, which is what tells you *where* a metric slipped. A case
whose gold records are absent from the database is SKIP, never FAIL — see the
header of eval/eval_set.yaml for why that matters right now.

**The database is the referee, not the response.** Field accuracy (M2) and filter
precision (M4) are checked by re-reading each returned record straight from
Postgres with a parameterised query and comparing field by field. Checking the
response against itself would only prove it is internally consistent.

Exit codes: 0 all targets met · 1 a target missed or a case failed · 2 the
harness could not run (no database, no eval set, missing dependency).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SET = ROOT / "eval" / "eval_set.yaml"

# NFR-H: "Eval penuh selesai dalam <= 10 minit."
RUNTIME_BUDGET_SECONDS = 600.0

# The eleven fields FR-5 says come verbatim from the record. M2 scores these.
COMPARED_FIELDS: Tuple[str, ...] = (
    "record_id",
    "class_title",
    "provider_name",
    "subject",
    "level",
    "delivery_mode",
    "instruction_language",
    "state",
    "city_area",
    "provider_source_url",
    "verified_on",
)


# ─── Report vocabulary ──────────────────────────────────────────────────────


@dataclass
class Check:
    """One assertion about one turn, tagged with the metric it feeds."""

    metric: str  # "M1".."M9", or "contract" for expectations outside the table
    label: str
    ok: bool
    detail: str = ""


@dataclass
class TurnResult:
    case_id: str
    index: int
    message: str
    status_code: int
    body: Dict[str, Any]
    latency_ms: float
    checks: List[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    @property
    def state(self) -> str:
        return str(self.body.get("state", "")) if isinstance(self.body, dict) else ""

    @property
    def failures(self) -> List[Check]:
        return [c for c in self.checks if not c.ok]


@dataclass
class CaseResult:
    case_id: str
    metrics: List[str]
    turns: List[TurnResult] = field(default_factory=list)
    skipped: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.skipped is None and all(t.ok for t in self.turns)

    @property
    def status(self) -> str:
        if self.skipped:
            return "SKIP"
        return "PASS" if self.ok else "FAIL"


@dataclass
class Metric:
    id: str
    name: str
    target_text: str
    value: Optional[float]
    unit: str  # "ratio" | "ms" | "rm"
    met: Optional[bool]
    numerator: Optional[float] = None
    denominator: Optional[float] = None
    note: str = ""

    @property
    def display(self) -> str:
        if self.value is None:
            return "n/a"
        if self.unit == "ratio":
            base = f"{self.value * 100:.1f}%"
        elif self.unit == "ms":
            base = f"{self.value:.0f} ms"
        else:
            base = f"RM{self.value:.5f}"
        if self.denominator:
            base += f"  ({self.numerator:g}/{self.denominator:g})"
        return base


# ─── Setup ──────────────────────────────────────────────────────────────────


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_eval.py",
        description="Run the Class Bot eval set and report metrics M1-M9 (FR-31).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--set",
        dest="eval_set",
        default=str(DEFAULT_SET),
        help="path to the eval set YAML (default: eval/eval_set.yaml)",
    )
    p.add_argument(
        "--llm",
        choices=("auto", "null", "vertex"),
        default="auto",
        help=(
            "auto: use LLM_PROVIDER from the environment. null: skip the model "
            "entirely, so every answer is a template and 'answered' is accepted as "
            "'degraded'. vertex: force the real provider."
        ),
    )
    p.add_argument("--only", default="", help="run cases whose id starts with this")
    p.add_argument("--metric", default="", help="run only cases feeding this metric")
    p.add_argument("--json", dest="json_out", default="", help="write the report as JSON")
    p.add_argument("--md", dest="md_out", default="", help="write the report as Markdown")
    p.add_argument("--quiet", action="store_true", help="metric table only")
    p.add_argument("--no-color", action="store_true", help="plain ASCII output")
    p.add_argument(
        "--respect-rate-limit",
        action="store_true",
        help=(
            "keep the production 60/min limit. Off by default: the eval fires "
            "~90 requests in a few seconds on one key and would 429 itself."
        ),
    )
    return p.parse_args(argv)


def prepare_env(args: argparse.Namespace) -> None:
    """Set configuration *before* app.config is imported and its cache warms."""
    if args.llm != "auto":
        os.environ["LLM_PROVIDER"] = args.llm
    if not args.respect_rate_limit:
        os.environ["RATE_LIMIT_PER_MINUTE"] = "1000000"
    # TestClient speaks http://; HTTPS enforcement would reject every request.
    os.environ["REQUIRE_HTTPS"] = "false"
    os.environ.setdefault("LOG_JSON", "false")
    os.environ.setdefault("LOG_LEVEL", "WARNING")
    # Every case must start from a clean context, and the harness holds ~50
    # sessions open at once.
    os.environ.setdefault("MAX_CONCURRENT_SESSIONS", "1000")


def load_eval_set(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:  # pragma: no cover
        raise SystemExit(
            "PyYAML is not installed. Run:  pip install pyyaml\n"
            "(it is in requirements.txt)"
        )
    if not path.exists():
        raise SystemExit(f"eval set not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict) or not data.get("cases"):
        raise SystemExit(f"{path} has no `cases:` list")
    return data


# ─── Checking one turn ──────────────────────────────────────────────────────


class Judge:
    """Holds everything a turn is judged against: the DB, the vocabulary, config.

    One instance per run. It owns a database session because M2 and M4 both need
    the authoritative record, and reopening a session per turn would make the
    latency numbers measure connection setup.
    """

    def __init__(self, db, *, llm_ready: bool) -> None:
        from app.config import get_settings
        from app.core import retrieval

        self.db = db
        self.settings = get_settings()
        self.llm_ready = llm_ready
        self.retrieval = retrieval
        self.providers = retrieval.known_providers(db)
        self._row_cache: Dict[str, Any] = {}

    # -- helpers ----------------------------------------------------------
    def rows_for(self, ids: Sequence[str]) -> List[Any]:
        missing = [i for i in ids if i not in self._row_cache]
        if missing:
            for row in self.retrieval.get_by_ids(self.db, missing):
                self._row_cache[row.record_id] = row
        return [self._row_cache[i] for i in ids if i in self._row_cache]

    def ids_present(self, ids: Sequence[str]) -> bool:
        return len(self.rows_for(list(ids))) == len(set(ids))

    def accepted_states(self, expected: object) -> List[str]:
        wanted = [expected] if isinstance(expected, str) else list(expected or [])
        wanted = [str(w) for w in wanted]
        if not self.llm_ready and "answered" in wanted and "degraded" not in wanted:
            # Without a model there is no narrative to accept, so NFR-C's degraded
            # response *is* the success case. Recorded in the report footer.
            wanted.append("degraded")
        return wanted

    # -- guardrails, run on every turn ------------------------------------
    def guardrails(self, turn: TurnResult) -> None:
        from app.core import honesty
        from app.core.schemas import ChatResponse

        body = turn.body

        # M7 — the response the client actually received still satisfies the
        # contract. Re-validating the raw JSON catches a field the service
        # populated but the schema would reject.
        if turn.status_code != 200:
            turn.checks.append(
                Check("M7", "http 200", False, f"HTTP {turn.status_code}: {body}")
            )
            return
        try:
            ChatResponse.model_validate(body)
            turn.checks.append(Check("M7", "schema valid", True))
        except Exception as exc:
            turn.checks.append(Check("M7", "schema valid", False, str(exc)[:400]))
            return

        # M6 — FR-21: the disclaimer is present on every single response. Checked
        # against the canonical wording rather than a keyword search, because the
        # requirement is that this exact statement reaches the user, not that the
        # answer happens to sound cautious.
        notice = str(body.get("notice") or "").strip()
        turn.checks.append(
            Check(
                "M6",
                "notice present",
                notice in {honesty.NOTICE_MS, honesty.NOTICE_EN},
                f"notice was {notice[:80]!r}",
            )
        )

        answer = str(body.get("answer") or "")
        trace = body.get("trace") or {}
        retrieved = [str(i) for i in (trace.get("retrieved_ids") or [])]
        rows = self.rows_for(retrieved)

        if len(rows) != len(retrieved):
            found = {r.record_id for r in rows}
            turn.checks.append(
                Check(
                    "M1",
                    "retrieved ids exist in the database",
                    False,
                    f"not in db: {sorted(set(retrieved) - found)}",
                )
            )

        # M1 — groundedness. Fee silence, schedule silence and the no-superlative
        # rule live here too: an amount or a day is precisely a value that is not
        # in any retrieved record.
        report = honesty.GroundingReport()
        report.merge(honesty.check_fee_silence(answer))
        report.merge(honesty.check_schedule_silence(answer))
        report.merge(honesty.check_no_superlatives(answer))
        report.merge(honesty.check_grounding(answer, rows, self.providers))

        failures = list(report.failures)
        if not body.get("results"):
            # States that return no records quantify something other than the
            # retrieved set: a clarification says how many candidates matched, a
            # no_result says how many classes each relaxation would return. Those
            # numbers come from SQL, not from the model, so a count mismatch
            # against zero rows is an artefact of the check, not a failure.
            failures = [f for f in failures if f != "fr18_count_mismatch"]

        turn.checks.append(
            Check(
                "M1",
                "answer grounded in retrieved records",
                not failures,
                "; ".join(failures) + (f" | {report.details[:2]}" if report.details else ""),
            )
        )

        # M2 — every returned field equals the database value (FR-5).
        for item in body.get("results") or []:
            rid = str(item.get("record_id"))
            row = self._row_cache.get(rid)
            if row is None:
                turn.checks.append(
                    Check("M2", f"{rid} exists", False, "result id not in the database")
                )
                continue
            for name in COMPARED_FIELDS:
                expected = getattr(row, name)
                actual = item.get(name)
                turn.checks.append(
                    Check(
                        "M2",
                        f"{rid}.{name}",
                        _field_equal(name, expected, actual),
                        f"db={expected!r} response={actual!r}",
                    )
                )
            # FR-16 / FR-17 at the wire: both fields must be null, always.
            for name in ("fee_rm", "schedule"):
                turn.checks.append(
                    Check(
                        "M1",
                        f"{rid}.{name} is null",
                        item.get(name) is None,
                        f"{name}={item.get(name)!r}",
                    )
                )

    # -- declared expectations --------------------------------------------
    def expectations(self, turn: TurnResult, expect: Dict[str, Any]) -> None:
        if not expect or turn.status_code != 200:
            return
        body = turn.body
        trace = body.get("trace") or {}
        results = body.get("results") or []
        answer = str(body.get("answer") or "")
        ids = [str(r.get("record_id")) for r in results]
        rows = self.rows_for(ids)

        if "state" in expect:
            accepted = self.accepted_states(expect["state"])
            turn.checks.append(
                Check(
                    "contract",
                    f"state in {accepted}",
                    turn.state in accepted,
                    f"got {turn.state!r}",
                )
            )

        if "gold_ids" in expect:  # M3, recall@5
            gold = [str(g) for g in expect["gold_ids"]]
            top5 = ids[:5]
            hit = [g for g in gold if g in top5]
            turn.checks.append(
                Check(
                    "M3",
                    f"recall@5 for {gold}",
                    len(hit) == len(gold),
                    f"missing {sorted(set(gold) - set(hit))} · returned {top5}",
                    )
            )
            turn.recall = len(hit) / len(gold) if gold else None  # type: ignore[attr-defined]

        if "exactly_ids" in expect:
            want = {str(i) for i in expect["exactly_ids"]}
            turn.checks.append(
                Check(
                    "M4",
                    f"result set == {sorted(want)}",
                    set(ids) == want,
                    f"got {sorted(ids)}",
                )
            )

        if "forbid_ids" in expect:
            leaked = [i for i in expect["forbid_ids"] if str(i) in ids]
            turn.checks.append(
                Check("M4", "no forbidden record returned", not leaked, f"leaked {leaked}")
            )

        if "constraints" in expect:
            for facet, value in (expect["constraints"] or {}).items():
                bad = [
                    r.record_id
                    for r in rows
                    if not self._row_satisfies(r, facet, str(value))
                ]
                turn.checks.append(
                    Check(
                        "M4",
                        f"every row has {facet}={value!r}",
                        not bad,
                        f"violated by {bad}",
                    )
                )

        if "min_results" in expect:
            turn.checks.append(
                Check(
                    "contract",
                    f"at least {expect['min_results']} results",
                    len(results) >= int(expect["min_results"]),
                    f"got {len(results)}",
                )
            )
        if "max_results" in expect:
            turn.checks.append(
                Check(
                    "contract",
                    f"at most {expect['max_results']} results",
                    len(results) <= int(expect["max_results"]),
                    f"got {len(results)}",
                )
            )

        if "unavailable_fields" in expect:  # M5
            declared = set(body.get("unavailable_fields") or [])
            want = {str(f) for f in expect["unavailable_fields"]}
            turn.checks.append(
                Check(
                    "M5",
                    f"declares unavailable {sorted(want)}",
                    want <= declared,
                    f"declared {sorted(declared)}",
                )
            )
            # The guardrails already forbid an amount or a time anywhere; M5 is
            # about saying so out loud rather than answering something else.
            from app.core import honesty

            turn.checks.append(
                Check(
                    "M5",
                    "answer states the absence",
                    honesty.has_disclaimer(answer + " " + str(body.get("notice") or "")),
                    "no disclaimer in the answer",
                )
            )

        if "verified_on" in expect:  # M2, FR-14
            for rid, iso in (expect["verified_on"] or {}).items():
                got = next(
                    (r.get("verified_on") for r in results if r.get("record_id") == rid),
                    None,
                )
                turn.checks.append(
                    Check("M2", f"{rid}.verified_on == {iso}", str(got) == str(iso), f"got {got!r}")
                )

        if "retained" in expect:  # M8
            applied = trace.get("filters_applied") or {}
            for facet, value in (expect["retained"] or {}).items():
                turn.checks.append(
                    Check(
                        "M8",
                        f"{facet}={value!r} retained across turns",
                        _filter_holds(applied.get(facet), str(value)),
                        f"filters_applied={applied}",
                    )
                )
        if "not_retained" in expect:  # FR-27
            applied = trace.get("filters_applied") or {}
            for facet in expect["not_retained"] or []:
                turn.checks.append(
                    Check(
                        "M8",
                        f"{facet} dropped after the subject changed",
                        facet not in applied,
                        f"filters_applied={applied}",
                    )
                )

        if expect.get("relaxation_options"):
            turn.checks.append(
                Check(
                    "contract",
                    "offers at least one relaxation",
                    bool(body.get("relaxation_options")),
                    "relaxation_options is empty",
                )
            )
        if expect.get("suggested_questions"):
            turn.checks.append(
                Check(
                    "contract",
                    "offers example questions",
                    bool(body.get("suggested_questions")),
                    "suggested_questions is empty",
                )
            )
        if "missing_field" in expect:
            turn.checks.append(
                Check(
                    "contract",
                    f"asks about {expect['missing_field']}",
                    body.get("missing_field") == expect["missing_field"],
                    f"got {body.get('missing_field')!r}",
                )
            )

        if "answer_contains_any" in expect:
            needles = [str(s).lower() for s in expect["answer_contains_any"]]
            haystack = (answer + " " + " ".join(body.get("suggested_questions") or [])).lower()
            turn.checks.append(
                Check(
                    "contract",
                    f"mentions one of {needles}",
                    any(n in haystack for n in needles),
                    f"answer={answer[:200]!r}",
                )
            )
        if "answer_must_not_contain" in expect:
            lowered = answer.lower()
            hits = [s for s in expect["answer_must_not_contain"] if str(s).lower() in lowered]
            turn.checks.append(
                Check("M1", "answer leaks nothing forbidden", not hits, f"contains {hits}")
            )

    def _row_satisfies(self, row: Any, facet: str, value: str) -> bool:
        from app.core.vocabulary import CROSS_REGION_STATES

        actual = getattr(row, facet, None)
        if actual is None:
            actual = row.facet(facet) if hasattr(row, "facet") else None
        if actual is None:
            return False
        if facet == "state" and self.settings.include_cross_region_states:
            # A nationwide provider legitimately answers a state query (FR-11).
            return str(actual) == value or str(actual) in CROSS_REGION_STATES
        return str(actual) == value


def _field_equal(name: str, expected: Any, actual: Any) -> bool:
    if actual is None:
        return expected is None
    if name == "provider_source_url":
        # Pydantic's HttpUrl normalises a bare host to a trailing slash.
        return str(expected).rstrip("/") == str(actual).rstrip("/")
    if name == "verified_on":
        return str(expected) == str(actual)
    return str(expected) == str(actual)


def _filter_holds(applied: Any, value: str) -> bool:
    if applied is None:
        return False
    if isinstance(applied, (list, tuple, set)):
        return value in {str(a) for a in applied}
    return str(applied) == value


# ─── Running ────────────────────────────────────────────────────────────────


def run_case(client, judge: Judge, case: Dict[str, Any], api_key: Optional[str]) -> CaseResult:
    case_id = str(case.get("id") or "unnamed")
    result = CaseResult(case_id=case_id, metrics=[str(m) for m in case.get("metrics") or []])

    required = [str(i) for i in case.get("requires_ids") or []]
    if required and not judge.ids_present(required):
        result.skipped = f"corpus has no {required} (load the full 200-record CSV)"
        return result

    headers = {"X-API-Key": api_key} if api_key else {}
    session_id: Optional[str] = None

    for index, turn_spec in enumerate(case.get("turns") or [], start=1):
        payload: Dict[str, Any] = {"message": str(turn_spec.get("message", "")).strip()}
        if session_id:
            payload["session_id"] = session_id

        started = time.perf_counter()
        response = client.post("/v1/chat", json=payload, headers=headers)
        latency_ms = (time.perf_counter() - started) * 1000.0

        try:
            body = response.json()
        except Exception:
            body = {"_raw": response.text[:500]}

        turn = TurnResult(
            case_id=case_id,
            index=index,
            message=payload["message"],
            status_code=response.status_code,
            body=body if isinstance(body, dict) else {"_raw": body},
            latency_ms=latency_ms,
        )
        judge.guardrails(turn)
        judge.expectations(turn, turn_spec.get("expect") or {})
        result.turns.append(turn)

        if isinstance(body, dict) and body.get("session_id"):
            session_id = str(body["session_id"])

    return result


def select_cases(cases: Sequence[Dict[str, Any]], args) -> List[Dict[str, Any]]:
    picked = list(cases)
    if args.only:
        picked = [c for c in picked if str(c.get("id", "")).startswith(args.only)]
    if args.metric:
        wanted = args.metric.upper()
        picked = [c for c in picked if wanted in {str(m).upper() for m in c.get("metrics") or []}]
    return picked


# ─── Metrics ────────────────────────────────────────────────────────────────


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile — same definition app/observability.py reports."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round(pct / 100.0 * len(ordered) + 0.5)) - 1))
    return ordered[index]


def _ratio_metric(
    mid: str,
    name: str,
    target_text: str,
    passed: int,
    total: int,
    threshold: float,
    note: str = "",
) -> Metric:
    value = (passed / total) if total else None
    return Metric(
        id=mid,
        name=name,
        target_text=target_text,
        value=value,
        unit="ratio",
        met=None if value is None else value >= threshold - 1e-9,
        numerator=passed,
        denominator=total,
        note=note,
    )


def compute_metrics(
    cases: Sequence[CaseResult], *, cost_per_query: Optional[float], llm_ready: bool
) -> List[Metric]:
    turns = [t for c in cases for t in c.turns]

    def tally(metric: str) -> Tuple[int, int]:
        checks = [c for t in turns for c in t.checks if c.metric == metric]
        return sum(1 for c in checks if c.ok), len(checks)

    # M1 is a guardrail: one bad turn is a failure, so it is scored per turn
    # rather than per check — 40 clean field assertions must not dilute one
    # hallucination.
    m1_turns = [t for t in turns if any(c.metric == "M1" for c in t.checks)]
    m1_ok = sum(1 for t in m1_turns if all(c.ok for c in t.checks if c.metric == "M1"))

    m2_ok, m2_total = tally("M2")
    m5_ok, m5_total = tally("M5")
    m6_ok, m6_total = tally("M6")
    m7_ok, m7_total = tally("M7")
    m8_ok, m8_total = tally("M8")

    recalls = [t.recall for t in turns if getattr(t, "recall", None) is not None]

    trap_cases = [c for c in cases if "M4" in c.metrics and not c.skipped]
    trap_ok = sum(
        1
        for c in trap_cases
        if all(chk.ok for t in c.turns for chk in t.checks if chk.metric == "M4")
    )

    latencies = [t.latency_ms for t in turns if t.status_code == 200]
    # NFR-A carves out the paths that never call the model: they must be fast
    # because there is nothing slow left to wait for.
    fast_states = {"no_result", "unavailable_field", "out_of_scope"}
    fast = [t.latency_ms for t in turns if t.state in fast_states and t.status_code == 200]

    metrics = [
        _ratio_metric(
            "M1",
            "Groundedness (nilai output ⊆ record retrieved)",
            "100%",
            m1_ok,
            len(m1_turns),
            1.0,
            "includes FR-16 fee silence, FR-17 schedule silence, FR-22 no ranking",
        ),
        _ratio_metric(
            "M2", "Field accuracy vs database", ">= 0.90", m2_ok, m2_total, 0.90
        ),
        Metric(
            id="M3",
            name="Recall@5",
            target_text=">= 0.95",
            value=(sum(recalls) / len(recalls)) if recalls else None,
            unit="ratio",
            met=None if not recalls else (sum(recalls) / len(recalls)) >= 0.95,
            numerator=len(recalls),
            denominator=len(recalls),
            note=f"{len(recalls)} turns declared gold ids",
        ),
        _ratio_metric(
            "M4",
            "Filter precision (trap queries)",
            "100%",
            trap_ok,
            len(trap_cases),
            1.0,
            "cases, not assertions: one leaked row fails the query",
        ),
        _ratio_metric(
            "M5", "Unanswerable field handling", "100%", m5_ok, m5_total, 1.0
        ),
        _ratio_metric("M6", "Disclaimer presence", "100%", m6_ok, m6_total, 1.0),
        _ratio_metric(
            "M7", "Schema validity", ">= 0.90", m7_ok, m7_total, 0.90
        ),
        _ratio_metric(
            "M8",
            "Multi-turn constraint retention",
            "100%",
            m8_ok,
            m8_total,
            1.0,
            "verified through trace.filters_applied",
        ),
        Metric(
            id="M9",
            name="Latency p95 (client-side, in-process)",
            target_text="<= 3000 ms",
            value=percentile(latencies, 95) if latencies else None,
            unit="ms",
            met=None if not latencies else percentile(latencies, 95) <= 3000.0,
            note=(
                f"p50 {percentile(latencies, 50):.0f} · p99 {percentile(latencies, 99):.0f} "
                f"(NFR-A p99 <= 5000) · no-LLM p95 "
                f"{percentile(fast, 95):.0f} (NFR-A <= 500)"
                if latencies
                else ""
            ),
        ),
    ]

    if cost_per_query is not None:
        metrics.append(
            Metric(
                id="NFR-E",
                name="Cost per query",
                target_text="<= RM0.05",
                value=cost_per_query,
                unit="rm",
                met=cost_per_query <= 0.05,
                note="estimated from token counts; RM0.05 is the PRD's NFR-E ceiling",
            )
        )
    if not llm_ready:
        for m in metrics:
            if m.id in {"M1", "M9"}:
                m.note = (m.note + " · " if m.note else "") + "LLM disabled: template answers"
    return metrics


# ─── Output ─────────────────────────────────────────────────────────────────


class Palette:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def ok(self, t: str) -> str:
        return self._wrap(t, "32")

    def bad(self, t: str) -> str:
        return self._wrap(t, "31")

    def warn(self, t: str) -> str:
        return self._wrap(t, "33")

    def bold(self, t: str) -> str:
        return self._wrap(t, "1")

    def dim(self, t: str) -> str:
        return self._wrap(t, "2")


def print_report(
    metrics: Sequence[Metric],
    cases: Sequence[CaseResult],
    *,
    elapsed: float,
    pal: Palette,
    quiet: bool,
    llm_ready: bool,
    corpus_records: int,
) -> None:
    out = sys.stdout.write

    out("\n" + pal.bold("Class Bot — eval report (FR-31)") + "\n")
    out(
        pal.dim(
            f"corpus {corpus_records} records · llm {'ready' if llm_ready else 'disabled'} · "
            f"{sum(len(c.turns) for c in cases)} turns in {len(cases)} cases · "
            f"{elapsed:.1f}s of the {RUNTIME_BUDGET_SECONDS:.0f}s NFR-H budget"
        )
        + "\n\n"
    )

    header = f"{'':4} {'metric':<44} {'result':<22} {'target':<12} verdict"
    out(pal.bold(header) + "\n")
    out("-" * len(header) + "\n")
    for m in metrics:
        if m.met is None:
            verdict = pal.warn("no data")
        elif m.met:
            verdict = pal.ok("PASS")
        else:
            verdict = pal.bad("FAIL")
        out(f"{m.id:<4} {m.name[:44]:<44} {m.display:<22} {m.target_text:<12} {verdict}\n")
        if m.note:
            out(pal.dim(f"     {m.note}") + "\n")

    failed = [c for c in cases if c.status == "FAIL"]
    skipped = [c for c in cases if c.status == "SKIP"]
    passed = [c for c in cases if c.status == "PASS"]

    out(
        "\n"
        + pal.bold("cases: ")
        + pal.ok(f"{len(passed)} pass")
        + " · "
        + (pal.bad(f"{len(failed)} fail") if failed else "0 fail")
        + " · "
        + pal.warn(f"{len(skipped)} skip")
        + "\n"
    )

    if skipped and not quiet:
        out("\n" + pal.warn("skipped") + "\n")
        for c in skipped:
            out(f"  {c.case_id}: {c.skipped}\n")

    if failed:
        out("\n" + pal.bad("failures") + "\n")
        for c in failed:
            for t in c.turns:
                for chk in t.failures:
                    out(
                        f"  {c.case_id} turn {t.index} [{chk.metric}] {chk.label}\n"
                        f"      message: {t.message[:90]!r}\n"
                        f"      state:   {t.state or f'HTTP {t.status_code}'}\n"
                        f"      detail:  {chk.detail[:300]}\n"
                    )

    if not quiet and passed:
        out("\n" + pal.dim("passed: " + ", ".join(c.case_id for c in passed)) + "\n")

    if elapsed > RUNTIME_BUDGET_SECONDS:
        out(
            "\n"
            + pal.bad(
                f"NFR-H: the eval took {elapsed:.0f}s, over the 10-minute budget."
            )
            + "\n"
        )


def report_dict(
    metrics: Sequence[Metric],
    cases: Sequence[CaseResult],
    *,
    elapsed: float,
    llm_ready: bool,
    corpus_records: int,
) -> Dict[str, Any]:
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": round(elapsed, 2),
        "runtime_budget_seconds": RUNTIME_BUDGET_SECONDS,
        "llm_ready": llm_ready,
        "corpus_records": corpus_records,
        "metrics": [
            {
                "id": m.id,
                "name": m.name,
                "value": m.value,
                "display": m.display,
                "unit": m.unit,
                "target": m.target_text,
                "met": m.met,
                "note": m.note,
            }
            for m in metrics
        ],
        "cases": [
            {
                "id": c.case_id,
                "status": c.status,
                "metrics": c.metrics,
                "skipped_reason": c.skipped,
                "turns": [
                    {
                        "index": t.index,
                        "message": t.message,
                        "status_code": t.status_code,
                        "state": t.state,
                        "latency_ms": round(t.latency_ms, 1),
                        "result_count": len(t.body.get("results") or []),
                        "trace": t.body.get("trace"),
                        "failures": [
                            {"metric": f.metric, "label": f.label, "detail": f.detail}
                            for f in t.failures
                        ],
                    }
                    for t in c.turns
                ],
            }
            for c in cases
        ],
    }


def write_markdown(path: Path, report: Dict[str, Any]) -> None:
    lines = [
        "# Class Bot — eval report (FR-31)",
        "",
        f"- generated: {report['generated_at']}",
        f"- corpus: {report['corpus_records']} records",
        f"- LLM: {'ready' if report['llm_ready'] else 'disabled (template answers)'}",
        f"- runtime: {report['elapsed_seconds']}s of a {report['runtime_budget_seconds']:.0f}s budget (NFR-H)",
        "",
        "| Metric | What it measures | Result | Target | Verdict |",
        "| --- | --- | --- | --- | --- |",
    ]
    for m in report["metrics"]:
        verdict = "n/a" if m["met"] is None else ("PASS" if m["met"] else "**FAIL**")
        lines.append(
            f"| {m['id']} | {m['name']} | {m['display']} | {m['target']} | {verdict} |"
        )

    statuses = [c["status"] for c in report["cases"]]
    lines += [
        "",
        f"Cases: {statuses.count('PASS')} pass · {statuses.count('FAIL')} fail · "
        f"{statuses.count('SKIP')} skip",
        "",
    ]

    failing = [c for c in report["cases"] if c["status"] == "FAIL"]
    if failing:
        lines += ["## Failures", ""]
        for c in failing:
            for t in c["turns"]:
                for f in t["failures"]:
                    lines.append(
                        f"- `{c['id']}` turn {t['index']} — **{f['metric']}** {f['label']}  \n"
                        f"  message: `{t['message']}` · state: `{t['state']}`  \n"
                        f"  {f['detail']}"
                    )
        lines.append("")

    skipped = [c for c in report["cases"] if c["status"] == "SKIP"]
    if skipped:
        lines += ["## Skipped", ""]
        lines += [f"- `{c['id']}` — {c['skipped_reason']}" for c in skipped]
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# ─── Entry point ────────────────────────────────────────────────────────────


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    prepare_env(args)

    data = load_eval_set(Path(args.eval_set))

    try:
        from fastapi.testclient import TestClient
    except ImportError:
        raise SystemExit(
            "fastapi/httpx are not installed. Run ./setup.sh, or:\n"
            "  pip install -r requirements.txt"
        )

    from app.config import get_settings
    from app.infrastructure.db import ping, session_scope
    from app.infrastructure.llm import get_provider
    from app.main import create_app
    from app.infrastructure.observability import reset_metrics
    from app.core.retrieval import corpus_size
    from app.core.sessions import reset_store

    settings = get_settings()

    if not ping():
        host = settings.database_url.rsplit("@", 1)[-1]
        print(
            f"Cannot reach the database at {host}.\n"
            "Start it and seed the corpus first:\n"
            "  docker compose up -d db\n"
            "  python scripts/init_db.py",
            file=sys.stderr,
        )
        return 2

    api_key: Optional[str] = None
    if settings.require_api_key:
        keys = list(settings.api_key_map)
        if not keys:
            print("REQUIRE_API_KEY is true but API_KEYS is empty.", file=sys.stderr)
            return 2
        api_key = keys[0]

    cases = select_cases(data["cases"], args)
    if not cases:
        print("no cases matched the filters", file=sys.stderr)
        return 2

    pal = Palette(enabled=sys.stdout.isatty() and not args.no_color)

    # A fresh process-local store and metrics window, so the numbers below
    # describe this run and nothing else.
    reset_store()
    reset_metrics()

    try:
        llm_ready = bool(get_provider().ready())
    except Exception:
        llm_ready = False

    started = time.perf_counter()
    results: List[CaseResult] = []

    app = create_app()
    with session_scope() as db:
        records = corpus_size(db)
        if records == 0:
            print(
                "The corpus is empty — every query would return no_result.\n"
                "Run: python scripts/init_db.py",
                file=sys.stderr,
            )
            return 2

        judge = Judge(db, llm_ready=llm_ready)

        with TestClient(app) as client:
            for case in cases:
                result = run_case(client, judge, case, api_key)
                results.append(result)
                if not args.quiet:
                    mark = {"PASS": ".", "FAIL": "F", "SKIP": "s"}[result.status]
                    sys.stdout.write(mark)
                    sys.stdout.flush()

            cost_per_query: Optional[float] = None
            if llm_ready:
                snap = client.get(
                    "/metrics", headers={"X-API-Key": api_key} if api_key else {}
                )
                if snap.status_code == 200:
                    cost_per_query = (snap.json().get("cost") or {}).get("avg_per_query_rm")

    elapsed = time.perf_counter() - started
    if not args.quiet:
        sys.stdout.write("\n")

    metrics = compute_metrics(results, cost_per_query=cost_per_query, llm_ready=llm_ready)
    print_report(
        metrics,
        results,
        elapsed=elapsed,
        pal=pal,
        quiet=args.quiet,
        llm_ready=llm_ready,
        corpus_records=records,
    )

    report = report_dict(
        metrics, results, elapsed=elapsed, llm_ready=llm_ready, corpus_records=records
    )
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nJSON report: {args.json_out}")
    if args.md_out:
        write_markdown(Path(args.md_out), report)
        print(f"Markdown report: {args.md_out}")

    missed = [m.id for m in metrics if m.met is False]
    failed_cases = [c.case_id for c in results if c.status == "FAIL"]
    if missed or failed_cases:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
