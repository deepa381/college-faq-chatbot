# utils/conversation_memory.py
# Lightweight per-session conversation memory for multi-turn context.
#
# Public API
# ----------
# get_history(session_id)                      – returns conversation history
# add_exchange(session_id, user_msg, bot_msg)  – stores a turn pair
# clear_session(session_id)                    – removes a session

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock

logger = logging.getLogger(__name__)

# Maximum exchanges (user+assistant pairs) kept per session
_MAX_EXCHANGES = 5

# Maximum number of sessions to track before evicting oldest
_MAX_SESSIONS = 512

# Sessions expire after this many seconds of inactivity (30 minutes)
_SESSION_TTL = 1800


@dataclass
class _Session:
    """Internal container for one session's conversation history."""
    history: list[dict] = field(default_factory=list)
    last_active: float = field(default_factory=time.time)


class ConversationMemory:
    """
    Thread-safe, in-memory conversation store keyed by session ID.

    Each session holds the last ``_MAX_EXCHANGES`` user/assistant pairs.
    Oldest sessions are evicted when the total count exceeds
    ``_MAX_SESSIONS``, and idle sessions are pruned on access.
    """

    def __init__(self) -> None:
        self._sessions: OrderedDict[str, _Session] = OrderedDict()
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_history(self, session_id: str) -> list[dict]:
        """
        Return conversation history for *session_id*.

        Returns a list of ``{"role": "user"|"assistant", "content": str}``
        dicts, ordered from oldest to newest.  Returns an empty list if
        the session does not exist.
        """
        with self._lock:
            self._evict_expired()
            session = self._sessions.get(session_id)
            if session is None:
                return []
            session.last_active = time.time()
            self._sessions.move_to_end(session_id)
            return list(session.history)

    def add_exchange(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """
        Append a user/assistant exchange to *session_id*.

        Creates the session if it doesn't exist.  Trims history to the
        most recent ``_MAX_EXCHANGES`` pairs (i.e. 2 * _MAX_EXCHANGES
        individual messages).
        """
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = _Session()

            session = self._sessions[session_id]
            session.history.append({"role": "user", "content": user_message})
            session.history.append({"role": "assistant", "content": assistant_message})

            # Keep only the last N exchanges (each exchange = 2 messages)
            max_messages = _MAX_EXCHANGES * 2
            if len(session.history) > max_messages:
                session.history = session.history[-max_messages:]

            session.last_active = time.time()
            self._sessions.move_to_end(session_id)

            # Evict oldest sessions if over capacity
            while len(self._sessions) > _MAX_SESSIONS:
                self._sessions.popitem(last=False)

    def clear_session(self, session_id: str) -> None:
        """Remove all history for *session_id*."""
        with self._lock:
            self._sessions.pop(session_id, None)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _evict_expired(self) -> None:
        """Remove sessions that have been idle longer than _SESSION_TTL."""
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items()
            if now - s.last_active > _SESSION_TTL
        ]
        for sid in expired:
            del self._sessions[sid]
        if expired:
            logger.debug(
                "[ConversationMemory] Evicted %d expired sessions.", len(expired)
            )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_memory = ConversationMemory()


def get_history(session_id: str) -> list[dict]:
    return _memory.get_history(session_id)


def add_exchange(session_id: str, user_message: str, assistant_message: str) -> None:
    _memory.add_exchange(session_id, user_message, assistant_message)


def clear_session(session_id: str) -> None:
    _memory.clear_session(session_id)
