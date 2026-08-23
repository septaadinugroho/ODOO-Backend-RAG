"""The interface every LLM provider implements.

Small on purpose. The service asks the model for exactly one thing — a short
paragraph about records it has already chosen — so the interface is one method
and there is no tool calling, no streaming, and no conversation history sent to
the model. Conversation state lives in `app/sessions.py` and is applied as
filters before retrieval; replaying a transcript to the model would give it the
chance to reuse a fact from a turn whose records are no longer on the table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable


class LLMConfigurationError(RuntimeError):
    """The provider cannot be built at all — a deployment error, not a runtime one."""


@dataclass
class Prompt:
    """A system instruction plus user-supplied *data*.

    The split is not cosmetic. NFR-D: "Input user diperlakukan sebagai data,
    bukan instruksi sistem." Keeping the two apart in the type means a caller
    cannot accidentally concatenate a user's message into the instruction, and
    `app/narrative.py` fences `data` explicitly before it is sent.
    """

    system: str
    data: str
    max_output_tokens: Optional[int] = None
    temperature: Optional[float] = None


@dataclass
class LLMResult:
    """Outcome of one generation attempt.

    `error` being set is an ordinary, expected outcome — the LLM is down, or slow,
    or returned nothing usable — and the caller responds with `degraded` rather
    than an HTTP error. The only case that becomes HTTP 503 is a timeout, per
    FR-4, and that decision is made in the router, not here.
    """

    text: str = ""
    error: Optional[str] = None
    # NFR-E: cost per query is tracked from token counts × unit price.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    model: str = ""
    attempts: int = 1
    timed_out: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text.strip())

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@runtime_checkable
class LLMProvider(Protocol):
    name: str
    model: str

    def generate(self, prompt: Prompt) -> LLMResult:
        """Produce text. Must not raise for ordinary failures — return an error."""
        ...

    def ready(self) -> bool:
        """Cheap, non-network readiness check for /health."""
        ...
