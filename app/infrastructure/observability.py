"""Structured logging, trace ids, and the metrics NFR-G asks for.

NFR-G is specific, so this module is too. Every log line is JSON and carries
`trace_id`, `session_id`, `state` and `latency` ("Structured logging — JSON; setiap
log memuat trace_id, session_id, state, dan latency"). The trace id comes from the
client's header when it sends one and is generated otherwise ("Menggunakan ID dari
header klien jika tersedia, atau dibuat baru oleh service"), which is what lets an
Odoo request be followed from there into here. The metrics are the five named
ones — latency, requests per state, degraded ratio, error rate, and cost/tokens —
and the three alerts fire at the stated thresholds: degraded above 2% in an hour,
errors above 5%, p95 above 5 seconds.

Two things this module is careful about.

**Logs contain no user identity** (NFR-D: "Query dicatat tanpa identitas user").
The `session_id` is random and carries none, and `redact()` strips e-mail
addresses and phone numbers from any text on its way to a log — because FR-30
requires the *query text* of a degraded response to be logged for debugging, and a
parent typing "call me at 012-345 6789" should not have that persisted for 30
days.

**Metrics are in-process.** Counters reset when the worker restarts and are not
shared between workers, so they are a live health signal and the 90-day retention
in NFR-G belongs to whatever scrapes `/metrics`. Saying so here is more useful
than implying a durability this dict does not have.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import threading
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from app.config import get_settings

TRACE_HEADERS = ("X-Trace-Id", "X-Request-Id", "X-Correlation-Id")
_SAFE_TRACE = re.compile(r"^[A-Za-z0-9._:-]{6,64}$")

# ─── Redaction (NFR-D) ──────────────────────────────────────────────────────

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# Malaysian mobile/landline shapes plus a generic long-digit run: 012-345 6789,
# +6012 3456789, 03-1234 5678.
_PHONE = re.compile(r"(?:\+?6?0\d[\d\s-]{6,12}\d)|(?:\b\d{9,}\b)")
_IC = re.compile(r"\b\d{6}-?\d{2}-?\d{4}\b")  # MyKad


def redact(text: str, limit: int = 500) -> str:
    """Strip the identifiers a class-search query should never need.

    Not a claim to catch everything — free text can hide a name in any form — but
    it removes the categories that show up in practice and that would turn a
    30-day query log into a store of personal data.
    """
    if not text:
        return ""
    cleaned = _IC.sub("[id]", text)
    cleaned = _EMAIL.sub("[email]", cleaned)
    cleaned = _PHONE.sub("[phone]", cleaned)
    return cleaned[:limit]


# ─── Trace ids ──────────────────────────────────────────────────────────────


def new_trace_id() -> str:
    return "t_" + uuid.uuid4().hex[:20]


def resolve_trace_id(header_value: Optional[str]) -> str:
    """Adopt the client's trace id when it is sane, else mint one.

    Validated rather than trusted: the value ends up in log lines, so an
    unbounded or newline-carrying string from a caller could forge log entries.
    """
    if header_value:
        candidate = header_value.strip()
        if _SAFE_TRACE.match(candidate):
            return candidate
    return new_trace_id()


# ─── JSON logging ───────────────────────────────────────────────────────────

_RESERVED = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "module", "msecs",
        "message", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName", "taskName",
    }
)


class JsonFormatter(logging.Formatter):
    """One JSON object per line, extras included verbatim.

    The four NFR-G fields are always present, as `null` when absent, so a log
    consumer can index on them without conditionals.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": getattr(record, "trace_id", None),
            "session_id": getattr(record, "session_id", None),
            "state": getattr(record, "state", None),
            "latency_ms": getattr(record, "latency_ms", None),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED or key in payload:
                continue
            if key.startswith("_"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # default=str so an unexpected object in an extra degrades to its repr
        # instead of throwing inside the logging call and losing the line.
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: Optional[str] = None, as_json: Optional[bool] = None) -> None:
    """Install the formatter on the root logger and quiet uvicorn's duplicate.

    Idempotent: called from the FastAPI lifespan, which can run more than once in
    tests.
    """
    settings = get_settings()
    level = (level or settings.log_level).upper()
    as_json = settings.log_json if as_json is None else as_json

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter()
        if as_json
        else logging.Formatter("%(asctime)s %(levelname)-7s %(name)s :: %(message)s")
    )
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)

    # uvicorn.access logs its own line per request with no trace id; ours carries
    # the same facts plus the NFR-G fields, so the duplicate is turned off.
    logging.getLogger("uvicorn.access").handlers = []
    logging.getLogger("uvicorn.access").propagate = False
    logging.getLogger("uvicorn.error").propagate = True
    # SQLAlchemy at INFO echoes every statement, including bound parameters.
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


# ─── Metrics ────────────────────────────────────────────────────────────────


@dataclass
class Sample:
    at: float
    state: str
    latency_ms: int
    degraded: bool
    error: bool
    tokens: int = 0
    cost_rm: float = 0.0


@dataclass
class Alert:
    name: str
    message: str
    value: float
    threshold: float
    extra: Dict[str, Any] = field(default_factory=dict)


class Metrics:
    """Rolling-window counters for the five NFR-G metrics.

    The window is bounded by both time and count. Time is what the alert
    definitions are written in ("dalam 1 jam"); the count cap is what stops a
    traffic spike from turning this into an unbounded list.
    """

    MAX_SAMPLES = 20_000

    def __init__(self) -> None:
        settings = get_settings()
        self.window_seconds = settings.metrics_window_seconds
        self._samples: Deque[Sample] = deque(maxlen=self.MAX_SAMPLES)
        self._states: Counter = Counter()
        self._errors: Counter = Counter()
        self._lock = threading.Lock()
        self._alerted_at: Dict[str, float] = {}
        self.started_at = time.time()
        self.total_requests = 0
        self.total_degraded = 0
        self.total_errors = 0
        self.total_tokens = 0
        self.total_cost_rm = 0.0
        self.llm_calls = 0
        self.llm_failures = 0

    # ── recording ───────────────────────────────────────────────────────────

    def observe(
        self,
        state: str,
        latency_ms: int,
        *,
        degraded: bool = False,
        error: bool = False,
        tokens: int = 0,
        cost_rm: float = 0.0,
        error_code: Optional[str] = None,
    ) -> None:
        now = time.monotonic()
        with self._lock:
            self._samples.append(
                Sample(now, state, latency_ms, degraded, error, tokens, cost_rm)
            )
            self._states[state] += 1
            if error_code:
                self._errors[error_code] += 1
            self.total_requests += 1
            self.total_degraded += int(degraded)
            self.total_errors += int(error)
            self.total_tokens += tokens
            self.total_cost_rm += cost_rm
            self._trim_locked(now)

    def observe_llm(self, ok: bool) -> None:
        with self._lock:
            self.llm_calls += 1
            if not ok:
                self.llm_failures += 1

    def _trim_locked(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._samples and self._samples[0].at < cutoff:
            self._samples.popleft()

    # ── reading ─────────────────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            self._trim_locked(time.monotonic())
            samples = list(self._samples)
            states = dict(self._states)
            errors = dict(self._errors)
            totals = {
                "requests": self.total_requests,
                "degraded": self.total_degraded,
                "errors": self.total_errors,
                "tokens": self.total_tokens,
                "cost_rm": round(self.total_cost_rm, 4),
                "llm_calls": self.llm_calls,
                "llm_failures": self.llm_failures,
            }

        window_count = len(samples)
        latencies = sorted(s.latency_ms for s in samples)
        degraded = sum(1 for s in samples if s.degraded)
        failed = sum(1 for s in samples if s.error)
        window_cost = sum(s.cost_rm for s in samples)
        settings = get_settings()

        return {
            "uptime_seconds": int(time.time() - self.started_at),
            "window_seconds": self.window_seconds,
            "window_requests": window_count,
            "requests_by_state": dict(
                sorted(
                    Counter(s.state for s in samples).items(),
                    key=lambda kv: -kv[1],
                )
            ),
            "requests_by_state_total": states,
            "errors_by_code": errors,
            "degraded_ratio": _ratio(degraded, window_count),
            "error_rate": _ratio(failed, window_count),
            "latency_ms": {
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
                "p99": _percentile(latencies, 0.99),
                "max": latencies[-1] if latencies else 0,
            },
            "cost": {
                "window_rm": round(window_cost, 4),
                "avg_per_query_rm": round(_ratio(window_cost, window_count), 5),
                "per_query_limit_rm": settings.cost_per_query_limit_rm,
                "monthly_budget_rm": settings.monthly_budget_rm,
                "process_total_rm": totals["cost_rm"],
            },
            "totals": totals,
            "alerts": [a.__dict__ for a in self.alerts()],
        }

    def alerts(self) -> List[Alert]:
        """Evaluate the three NFR-G alert conditions plus the NFR-E cost ceiling."""
        settings = get_settings()
        with self._lock:
            self._trim_locked(time.monotonic())
            samples = list(self._samples)

        count = len(samples)
        out: List[Alert] = []
        if count < settings.alert_min_samples:
            return out

        degraded_ratio = _ratio(sum(1 for s in samples if s.degraded), count)
        error_rate = _ratio(sum(1 for s in samples if s.error), count)
        p95 = _percentile(sorted(s.latency_ms for s in samples), 0.95)
        avg_cost = _ratio(sum(s.cost_rm for s in samples), count)

        if degraded_ratio > settings.alert_degraded_ratio:
            out.append(
                Alert(
                    "degraded_ratio",
                    f"degraded ratio {degraded_ratio:.1%} over the last "
                    f"{self.window_seconds}s exceeds "
                    f"{settings.alert_degraded_ratio:.1%}",
                    round(degraded_ratio, 4),
                    settings.alert_degraded_ratio,
                    {"window_requests": count},
                )
            )
        if error_rate > settings.alert_error_rate:
            out.append(
                Alert(
                    "error_rate",
                    f"error rate {error_rate:.1%} exceeds "
                    f"{settings.alert_error_rate:.1%}",
                    round(error_rate, 4),
                    settings.alert_error_rate,
                    {"window_requests": count},
                )
            )
        if p95 > settings.alert_latency_p95_ms:
            out.append(
                Alert(
                    "latency_p95",
                    f"p95 latency {p95}ms exceeds {settings.alert_latency_p95_ms}ms",
                    float(p95),
                    float(settings.alert_latency_p95_ms),
                    {"window_requests": count},
                )
            )
        if avg_cost > settings.cost_per_query_limit_rm:
            out.append(
                Alert(
                    "cost_per_query",
                    f"average cost RM{avg_cost:.4f} per query exceeds the NFR-E "
                    f"ceiling of RM{settings.cost_per_query_limit_rm:.2f}",
                    round(avg_cost, 5),
                    settings.cost_per_query_limit_rm,
                    {"window_requests": count},
                )
            )
        return out

    def emit_alerts(self, logger: Optional[logging.Logger] = None) -> List[Alert]:
        """Log any firing alert, at most once per cooldown per alert name."""
        log = logger or logging.getLogger("app.alerts")
        settings = get_settings()
        now = time.monotonic()
        fired: List[Alert] = []
        for alert in self.alerts():
            last = self._alerted_at.get(alert.name, 0.0)
            if now - last < settings.alert_cooldown_seconds:
                continue
            self._alerted_at[alert.name] = now
            fired.append(alert)
            log.warning(
                "alert: %s",
                alert.message,
                extra={
                    "alert": alert.name,
                    "value": alert.value,
                    "threshold": alert.threshold,
                    **alert.extra,
                },
            )
        return fired

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
            self._states.clear()
            self._errors.clear()
            self._alerted_at.clear()
            self.total_requests = 0
            self.total_degraded = 0
            self.total_errors = 0
            self.total_tokens = 0
            self.total_cost_rm = 0.0
            self.llm_calls = 0
            self.llm_failures = 0


def _ratio(numerator: float, denominator: float) -> float:
    return 0.0 if not denominator else numerator / denominator


def _percentile(sorted_values: List[int], q: float) -> int:
    """Nearest-rank percentile. Integer milliseconds; no interpolation needed."""
    if not sorted_values:
        return 0
    index = max(0, min(len(sorted_values) - 1, int(round(q * (len(sorted_values) - 1)))))
    return sorted_values[index]


def estimate_cost_rm(prompt_tokens: int, completion_tokens: int) -> float:
    """Token counts to ringgit, for NFR-E."""
    settings = get_settings()
    return (
        prompt_tokens / 1000.0 * settings.llm_input_price_per_1k_rm
        + completion_tokens / 1000.0 * settings.llm_output_price_per_1k_rm
    )


_metrics: Optional[Metrics] = None
_metrics_lock = threading.Lock()


def get_metrics() -> Metrics:
    global _metrics
    if _metrics is None:
        with _metrics_lock:
            if _metrics is None:
                _metrics = Metrics()
    return _metrics


def reset_metrics() -> None:
    global _metrics
    with _metrics_lock:
        _metrics = None


__all__ = [
    "Alert",
    "JsonFormatter",
    "Metrics",
    "Sample",
    "TRACE_HEADERS",
    "configure_logging",
    "estimate_cost_rm",
    "get_metrics",
    "new_trace_id",
    "redact",
    "reset_metrics",
    "resolve_trace_id",
]
