"""The "no LLM" provider.

Configured with `LLM_PROVIDER=null`. Every response then comes from the safe
templates in `app/honesty.py` and `app/narrative.py`, `degraded` is true, and
`results[]` is complete and correct — which is exactly the behaviour NFR-C
specifies for an LLM outage:

    "Saat LLM down — Pencarian berbasis filter tetap berjalan; respons memakai
     degraded tanpa narasi generatif."

Two practical uses beyond outages: demoing `/docs` with no GCP project attached,
and running the eval set (FR-31) without spending tokens on the retrieval metrics
M2, M3 and M4, none of which involve the model.
"""

from __future__ import annotations

from app.infrastructure.llm.base import LLMResult, Prompt


class NullProvider:
    name = "null"
    model = "none"

    def generate(self, prompt: Prompt) -> LLMResult:  # noqa: ARG002 - interface
        return LLMResult(
            text="",
            error="llm_disabled",
            model=self.model,
            meta={"reason": "LLM_PROVIDER=null; narrative generation is switched off"},
        )

    def ready(self) -> bool:
        # Honest answer: this provider will never produce a narrative. /health
        # reports llm_ready=false so an operator can see the service is running
        # in template-only mode on purpose rather than by accident.
        return False
