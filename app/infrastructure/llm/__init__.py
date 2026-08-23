"""LLM provider abstraction.

NFR-H requires swapping the model "melalui konfigurasi, tanpa mengubah logic
utama". That is only true if the rest of the codebase never imports a vendor SDK,
so everything vendor-specific is confined to this package behind one small
interface.

Two providers ship:

* `vertex` — Google Vertex AI (Gemini) via the `google-genai` SDK.
* `null`   — answers nothing, always reports itself unavailable. Not a stub for
  tests but a supported production mode: NFR-C says filter search must keep
  working when the LLM is down, so "no LLM at all" has to be a first-class
  configuration. It is also what lets `/docs` be demoed without GCP credentials.

Failure is a return value, not an exception, in the normal path. Every call
returns `LLMResult`; a failed one carries `error` and empty `text`, which the
caller turns into `degraded` (FR-19). The one exception type here,
`LLMConfigurationError`, is for a provider that cannot be constructed at all —
that is a deployment mistake and should be loud at startup.
"""

from __future__ import annotations

from typing import Optional

from app.config import get_settings
from app.infrastructure.llm.base import LLMConfigurationError, LLMProvider, LLMResult, Prompt

_provider: Optional[LLMProvider] = None


def build_provider(name: Optional[str] = None) -> LLMProvider:
    """Construct a provider by name. Import of the SDK happens inside."""
    settings = get_settings()
    name = (name or settings.llm_provider).lower()

    if name == "vertex":
        from app.infrastructure.llm.vertex import VertexProvider

        return VertexProvider()
    if name == "null":
        from app.infrastructure.llm.null import NullProvider

        return NullProvider()
    raise LLMConfigurationError(
        f"unknown LLM_PROVIDER {name!r}; supported values are 'vertex' and 'null'"
    )


def get_provider() -> LLMProvider:
    """Process-wide provider. Built lazily so import never touches the network."""
    global _provider
    if _provider is None:
        _provider = build_provider()
    return _provider


def reset_provider() -> None:
    global _provider
    _provider = None


__all__ = [
    "LLMConfigurationError",
    "LLMProvider",
    "LLMResult",
    "Prompt",
    "build_provider",
    "get_provider",
    "reset_provider",
]
