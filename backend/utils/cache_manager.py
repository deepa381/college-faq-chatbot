# utils/cache_manager.py
# Pluggable response-cache layer for the RAG chatbot.
#
# Ships with an in-memory LRU backend (``InMemoryCache``).
# The active backend can be swapped to Redis (or any other store) by
# implementing the ``CacheBackend`` interface and calling
# ``set_backend()``.  Pipeline code in ``chat.py`` only talks to the
# public ``cache_get`` / ``cache_put`` / ``cache_clear`` functions —
# it never touches the backend directly.
#
# Public API
# ----------
# cache_get(key)          – return cached value or None
# cache_put(key, value)   – store a value
# cache_clear()           – flush all entries
# set_backend(backend)    – swap the active backend at runtime

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Abstract interface — implement this to plug in a different store
# ---------------------------------------------------------------------------


class CacheBackend(ABC):
    """
    Minimal interface that every cache backend must implement.

    To add Redis support later, create a ``RedisCacheBackend`` that
    inherits from this class, and call ``set_backend(RedisCacheBackend())``.
    """

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Return the cached value for *key*, or ``None`` on a miss."""

    @abstractmethod
    def put(self, key: str, value: Any) -> None:
        """Store *value* under *key*."""

    @abstractmethod
    def clear(self) -> None:
        """Remove every entry from the cache."""


# ---------------------------------------------------------------------------
# Default backend — thread-safe in-memory LRU dict
# ---------------------------------------------------------------------------

_DEFAULT_MAX_SIZE = 128


class InMemoryCache(CacheBackend):
    """
    Thread-safe, bounded LRU cache backed by an ``OrderedDict``.

    Parameters
    ----------
    max_size:
        Maximum number of entries before the oldest is evicted.
    """

    def __init__(self, max_size: int = _DEFAULT_MAX_SIZE) -> None:
        self._max_size = max_size
        self._store: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    # -- CacheBackend interface --

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                return self._store[key]
            return None

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = value
            self._store.move_to_end(key)
            if len(self._store) > self._max_size:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    # -- Introspection helpers (useful for tests / debugging) --

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._store)


# ---------------------------------------------------------------------------
# Module-level singleton & public API
# ---------------------------------------------------------------------------

_backend: CacheBackend = InMemoryCache()


def set_backend(backend: CacheBackend) -> None:
    """
    Replace the active cache backend at runtime.

    Example — switching to Redis::

        from utils.cache_manager import set_backend
        set_backend(RedisCacheBackend(host="redis", port=6379))
    """
    global _backend
    _backend = backend
    logger.info("[cache] Backend switched to %s", type(backend).__name__)


def cache_get(key: str) -> Optional[Any]:
    """Return the cached value for *key*, or ``None`` on a miss."""
    return _backend.get(key)


def cache_put(key: str, value: Any) -> None:
    """Store *value* under *key*."""
    _backend.put(key, value)


def cache_clear() -> None:
    """Flush every entry from the cache."""
    _backend.clear()
    logger.info("[cache] Cache cleared.")
