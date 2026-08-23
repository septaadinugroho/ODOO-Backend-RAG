"""Conversation state: FR-25 through FR-28.

Multi-turn is where a filter service earns or loses trust. "yang online je" only
means something if the previous turn's subject and level are still in play, and
FR-25 requires exactly that. But carrying context is also the easiest way to
answer the wrong question, so three rules bound it:

* **FR-27 — a new subject resets everything.** Someone who switches from Add
  Maths to Physics has started a different search; keeping "Form 4" would be a
  lucky guess and keeping "Online live" would be an unlucky one. Reset is the only
  behaviour that is never surprising.
* **FR-28 — 30 minutes of inactivity and the session is gone.** Sliding window,
  refreshed on every turn.
* **FR-26 / NFR-D — `session_id` is random.** `secrets.token_urlsafe`, never
  derived from a name, e-mail, or phone number. A guessable id is a way to read
  someone else's conversation.

What is deliberately *not* stored: the user's messages. The store holds canonical
filter values, the record ids of the last result set (so a follow-up like "yang
pertama tu" can resolve), and timestamps. Nothing here identifies a person, which
is what makes the "tanpa akun murid, tanpa profil, tanpa riwayat lintas sesi"
constraint in the PRD's minor-data section true in the code and not just in the
README.

The store is in-process and single-node on purpose. The PRD's scale target is one
Odoo instance and a small web front end; a Redis dependency would add an outage
mode for no benefit. If this ever runs multi-worker, this module is the seam —
swap the dict for Redis and nothing above it changes. Note that `uvicorn
--workers N` would give each worker its own store, so run one worker (or add
sticky routing) until that swap happens.
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.config import get_settings

SESSION_ID_PREFIX = "s_"
_ID_BYTES = 16  # ~22 chars of urlsafe base64: not guessable


def new_session_id() -> str:
    """A random, opaque session id (FR-26, NFR-D).

    `secrets`, not `random`: the latter is seeded predictably and is the wrong
    tool for anything an outsider should not be able to guess.
    """
    return SESSION_ID_PREFIX + secrets.token_urlsafe(_ID_BYTES)


@dataclass
class SessionState:
    """One conversation's carried context. Holds no personal data — see above."""

    session_id: str
    created_at: float = field(default_factory=time.monotonic)
    last_seen: float = field(default_factory=time.monotonic)
    turns: int = 0

    # Canonical facet values carried forward. Same shape as Understanding.filters.
    filters: Dict[str, object] = field(default_factory=dict)
    # Which of those arrived in an earlier turn — surfaced in the trace so a
    # confusing answer can be explained by "you asked this three turns ago".
    carried: Dict[str, object] = field(default_factory=dict)
    # Result set of the previous turn, for ordinal follow-ups ("yang kedua tu").
    last_record_ids: List[str] = field(default_factory=list)
    last_state: Optional[str] = None
    lang: str = "ms"
    resets: int = 0

    def is_expired(self, ttl_seconds: float, now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else now
        return (now - self.last_seen) > ttl_seconds

    def touch(self, now: Optional[float] = None) -> None:
        self.last_seen = time.monotonic() if now is None else now

    def age_seconds(self) -> float:
        return time.monotonic() - self.created_at

    def idle_seconds(self) -> float:
        return time.monotonic() - self.last_seen

    # ── FR-25 / FR-27 ───────────────────────────────────────────────────────

    def merge(self, incoming: Dict[str, object]) -> Tuple[Dict[str, object], bool]:
        """Combine carried context with this turn's filters.

        Returns the effective filter set and whether FR-27's reset fired.

        The turn's own values always win. Someone who says "Form 5 pula" is
        correcting the level, not asking for both — and a filter service that
        ANDs a correction onto the thing being corrected returns nothing and
        looks broken.
        """
        reset = self._should_reset(incoming)
        if reset:
            self.filters = {}
            self.carried = {}
            self.last_record_ids = []
            self.resets += 1

        carried = {k: v for k, v in self.filters.items() if k not in incoming}
        effective: Dict[str, object] = dict(carried)
        effective.update(incoming)

        self.filters = dict(effective)
        self.carried = carried
        self.turns += 1
        self.touch()
        return effective, reset

    def _should_reset(self, incoming: Dict[str, object]) -> bool:
        """FR-27: a different subject means a different search.

        Only the subject triggers this. Level, mode, state and language are the
        things people refine within one search ("okay, and only the online ones"),
        so resetting on those would throw away context the user expects to keep.
        """
        old = self.filters.get("subject")
        new = incoming.get("subject")
        return bool(old and new and old != new)

    def remember_results(self, record_ids: List[str], state: str, lang: str) -> None:
        self.last_record_ids = list(record_ids)
        self.last_state = state
        self.lang = lang


class SessionStore:
    """TTL map from session id to `SessionState`.

    Locked because uvicorn runs request handlers on a thread pool for sync
    endpoints; two turns of the same conversation can genuinely overlap. The
    critical sections are dict operations, so contention is not a concern.
    """

    def __init__(
        self,
        ttl_seconds: Optional[float] = None,
        max_sessions: Optional[int] = None,
    ) -> None:
        settings = get_settings()
        self.ttl_seconds = (
            float(settings.session_ttl_seconds) if ttl_seconds is None else float(ttl_seconds)
        )
        self.max_sessions = (
            settings.max_concurrent_sessions if max_sessions is None else max_sessions
        )
        self._sessions: Dict[str, SessionState] = {}
        self._lock = threading.Lock()

    # ── lookup ──────────────────────────────────────────────────────────────

    def get(self, session_id: str) -> Optional[SessionState]:
        """Live session, or None if unknown or expired (FR-28)."""
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return None
            if state.is_expired(self.ttl_seconds):
                # Drop it here rather than waiting for the sweep: an expired
                # session must never contribute a filter to a later turn.
                self._sessions.pop(session_id, None)
                return None
            return state

    def get_or_create(self, session_id: Optional[str]) -> Tuple[SessionState, bool]:
        """Returns `(state, is_new)`.

        An unknown or expired id is *not* an error. The caller passed something
        plausible, the conversation simply is not in memory any more — either it
        timed out or the service restarted. Starting a fresh session under the id
        the client already holds keeps the client's own threading intact; the
        alternative (a 404 mid-conversation) would be worse for a chat widget.
        """
        if not session_id:
            return self._create(new_session_id()), True
        existing = self.get(session_id)
        if existing is not None:
            return existing, False
        return self._create(session_id), True

    def _create(self, session_id: str) -> SessionState:
        with self._lock:
            self._evict_locked()
            state = SessionState(session_id=session_id)
            self._sessions[session_id] = state
            return state

    # ── housekeeping ────────────────────────────────────────────────────────

    def _evict_locked(self) -> None:
        """Purge expired sessions; if still at capacity, drop the stalest one.

        A bounded store is a small denial-of-service defence: without it, a script
        sending one request per new id fills memory. Dropping the least recently
        used session costs that conversation its carried filters and nothing else.
        """
        now = time.monotonic()
        expired = [
            sid for sid, st in self._sessions.items() if st.is_expired(self.ttl_seconds, now)
        ]
        for sid in expired:
            self._sessions.pop(sid, None)

        while len(self._sessions) >= self.max_sessions:
            oldest = min(self._sessions.items(), key=lambda kv: kv[1].last_seen)[0]
            self._sessions.pop(oldest, None)

    def purge(self) -> int:
        with self._lock:
            now = time.monotonic()
            expired = [
                sid
                for sid, st in self._sessions.items()
                if st.is_expired(self.ttl_seconds, now)
            ]
            for sid in expired:
                self._sessions.pop(sid, None)
            return len(expired)

    def drop(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def count(self) -> int:
        """Live sessions. Reported by /health as `active_sessions`."""
        with self._lock:
            now = time.monotonic()
            return sum(
                1
                for st in self._sessions.values()
                if not st.is_expired(self.ttl_seconds, now)
            )

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()


_store: Optional[SessionStore] = None
_store_lock = threading.Lock()


def get_store() -> SessionStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = SessionStore()
    return _store


def reset_store() -> None:
    """Test hook: forget every session and rebuild from current settings."""
    global _store
    with _store_lock:
        _store = None


__all__ = [
    "SESSION_ID_PREFIX",
    "SessionState",
    "SessionStore",
    "get_store",
    "new_session_id",
    "reset_store",
]
