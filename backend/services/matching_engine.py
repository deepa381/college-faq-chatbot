# services/matching_engine.py
# Semantic search engine using sentence-transformers embeddings.
#
# Public API
# ----------
# RetrievalResult      – dataclass for top-K retrieval (RAG pipeline)
# MatchingEngine       – stateful engine built from a FAQStore
# build_engine(store)  – factory that creates and caches the singleton engine
# get_engine()         – returns the cached singleton

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    FALLBACK_MESSAGE,
    RAG_TOP_K, RAG_OVERVIEW_TOP_K,
    OVERVIEW_KEYWORDS, EMBEDDING_MODEL,
)
from utils.loader import FAQStore

logger = logging.getLogger(__name__)

# Module-level singleton
_engine: Optional["MatchingEngine"] = None


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """
    A single retrieved FAQ entry with its similarity score.
    Used by :meth:`MatchingEngine.retrieve_top_k` for the RAG pipeline.
    """
    entry: dict
    score: float
    question: str
    category: str

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "category": self.category,
            "score":    round(self.score, 4),
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class MatchingEngine:
    """
    Semantic search engine built once at startup from a :class:`FAQStore`.

    Uses ``sentence-transformers`` (model: ``all-MiniLM-L6-v2``) to encode
    each FAQ entry (category + question + answer) into a dense 384-dim
    vector.  At query time the user's message is encoded and compared
    against the pre-built embedding matrix using cosine similarity.

    No hard similarity threshold is applied — the top-K results are always
    returned, ensuring Gemini always receives context to work with.
    """

    def __init__(self, store: FAQStore) -> None:
        if not store.questions:
            raise ValueError("[MatchingEngine] FAQStore contains no questions to index.")

        self._store = store

        # Load the sentence-transformer model
        logger.info("[MatchingEngine] Loading embedding model: %s ...", EMBEDDING_MODEL)
        t0 = time.time()
        self._model = SentenceTransformer(EMBEDDING_MODEL)

        # Build combined corpus: category + question + answer per entry
        self._corpus = [
            f"{e.get('category', '')} {e['question']} {e['answer']}"
            for e in store.entries
        ]

        # Pre-compute embeddings for the entire FAQ corpus (done once)
        self._embeddings: np.ndarray = self._model.encode(
            self._corpus,
            show_progress_bar=False,
            normalize_embeddings=True,   # L2-normalise so dot product = cosine sim
            convert_to_numpy=True,
        )

        elapsed = time.time() - t0
        logger.info(
            "[MatchingEngine] Semantic index ready: %d entries × %d dims (%.1fs)",
            self._embeddings.shape[0],
            self._embeddings.shape[1],
            elapsed,
        )

    # ------------------------------------------------------------------
    # Properties for startup diagnostics
    # ------------------------------------------------------------------

    @property
    def num_entries(self) -> int:
        return self._embeddings.shape[0]

    @property
    def embedding_dim(self) -> int:
        return self._embeddings.shape[1]

    # ------------------------------------------------------------------
    # Overview-query detection
    # ------------------------------------------------------------------

    @staticmethod
    def is_overview_query(query: str) -> bool:
        """Return True if *query* looks like a broad / overview question."""
        q = query.lower()
        return any(kw in q for kw in OVERVIEW_KEYWORDS)

    # ------------------------------------------------------------------
    # Semantic retrieval — top K entries
    # ------------------------------------------------------------------

    def retrieve_top_k(
        self,
        query: str,
        top_k: int = RAG_TOP_K,
        is_overview: bool = False,
    ) -> list[RetrievalResult]:
        """
        Retrieve the top *top_k* FAQ entries by semantic similarity.

        Steps
        -----
        1. Encode the user query with the same embedding model.
        2. Compute cosine similarity against the pre-built FAQ embeddings.
        3. Return the top-K entries sorted by score (descending).

        No hard threshold is applied — the top-K results are always returned
        so that Gemini always receives context.

        When *is_overview* is True, ``top_k`` is raised to
        ``RAG_OVERVIEW_TOP_K`` (8) so Gemini can build a comprehensive answer.

        Parameters
        ----------
        query:
            The raw user question string.
        top_k:
            Maximum number of entries to return (default from config).
        is_overview:
            Set True for broad queries — bumps top_k.

        Returns
        -------
        list[RetrievalResult]
            Sorted descending by similarity score.
        """
        query = query.strip()
        if not query:
            return []

        # Boost top_k for overview queries
        if is_overview:
            top_k = max(top_k, RAG_OVERVIEW_TOP_K)

        # Encode the query (normalised so dot product = cosine similarity)
        query_vec: np.ndarray = self._model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        # Cosine similarity via dot product (both vectors are L2-normalised)
        scores = (self._embeddings @ query_vec.T).flatten()

        # Get top_k indices sorted by score (descending)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[RetrievalResult] = []
        for idx in top_indices:
            idx_int = int(idx)
            entry = self._store.entries[idx_int]
            results.append(RetrievalResult(
                entry=entry,
                score=float(scores[idx_int]),
                question=entry["question"],
                category=entry.get("category", "Uncategorised"),
            ))

        logger.info(
            "[MatchingEngine] retrieve: query=%r  is_overview=%s  "
            "returned=%d entries (top_k=%d, best_score=%.4f)",
            query, is_overview, len(results), top_k,
            results[0].score if results else 0.0,
        )
        return results


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

def build_engine(store: FAQStore) -> MatchingEngine:
    """
    Build a :class:`MatchingEngine` from *store*, cache it, and return it.
    Safe to call multiple times (rebuilds and replaces the singleton).
    """
    global _engine
    _engine = MatchingEngine(store)
    return _engine


def get_engine() -> MatchingEngine:
    """
    Return the cached :class:`MatchingEngine`.

    Raises
    ------
    RuntimeError
        If called before :func:`build_engine`.
    """
    if _engine is None:
        raise RuntimeError(
            "[MatchingEngine] Engine not initialised. "
            "Call build_engine(store) during application startup."
        )
    return _engine
