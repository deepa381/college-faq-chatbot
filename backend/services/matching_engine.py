# services/matching_engine.py
# Hybrid retrieval engine: semantic search (ChromaDB) + keyword search (BM25).
#
# Public API
# ----------
# RetrievalResult      – dataclass for top-K retrieval (RAG pipeline)
# MatchingEngine       – stateful hybrid engine built from a FAQStore
# build_engine(store)  – factory that creates and caches the singleton engine
# get_engine()         – returns the cached singleton

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from config import (
    FALLBACK_MESSAGE,
    RAG_TOP_K, RAG_OVERVIEW_TOP_K,
    OVERVIEW_KEYWORDS, EMBEDDING_MODEL,
    KNOWLEDGE_BASE_FILE,
    HYBRID_SEMANTIC_WEIGHT, HYBRID_KEYWORD_WEIGHT,
)
from utils.loader import FAQStore
from services.vector_store import get_vector_store
from utils.query_preprocessor import preprocess as preprocess_query

logger = logging.getLogger(__name__)

# Module-level singleton
_engine: Optional["MatchingEngine"] = None

# Candidate pool multiplier – retrieve more from each source before merging
_CANDIDATE_MULTIPLIER = 3


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """
    A single retrieved entry with its hybrid score.
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
# Query preprocessing
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "can", "could", "may", "might", "must", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "and", "but", "or", "if", "so", "yet", "not", "no", "nor",
    "it", "its", "this", "that", "these", "those", "i", "me", "my",
    "we", "our", "you", "your", "he", "him", "she", "her", "they", "them",
    "what", "which", "who", "whom", "how", "when", "where", "why",
    "about", "up", "out", "off", "over", "very", "just", "than",
})

from nltk.stem import WordNetLemmatizer as _WNL
_lemmatizer = _WNL()


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stopwords, lemmatize."""
    tokens = re.findall(r"[a-zA-Z0-9.+#]+", text.lower())
    return [_lemmatizer.lemmatize(t) for t in tokens if t not in _STOPWORDS]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class MatchingEngine:
    """
    Hybrid retrieval engine combining semantic search and keyword search.

    **Semantic search** uses ChromaDB (backed by SentenceTransformer
    ``all-MiniLM-L6-v2``) for dense vector retrieval with cosine similarity.

    **Keyword search** uses BM25 (Okapi variant) over the tokenised
    knowledge-base corpus for sparse lexical matching.

    At query time both systems are queried independently, their scores
    are normalised to [0, 1], and merged via weighted ranking::

        final_score = 0.6 × semantic_score + 0.4 × keyword_score

    Results are deduplicated by entry ID and returned sorted by
    ``final_score`` descending.
    """

    def __init__(self, store: FAQStore) -> None:
        if not store.questions:
            raise ValueError("[MatchingEngine] FAQStore contains no questions to index.")

        self._store = store

        t0 = time.time()

        # ----------------------------------------------------------
        # Load RAG knowledge base for BM25 (keyword) index
        # ----------------------------------------------------------
        with open(KNOWLEDGE_BASE_FILE, "r", encoding="utf-8") as f:
            self._kb_entries: list[dict] = json.load(f)

        # Build id → knowledge-base entry lookup
        self._kb_by_id: dict[int, dict] = {
            e["id"]: e for e in self._kb_entries
        }

        # Tokenise each knowledge-base entry for BM25
        tokenised_corpus: list[list[str]] = []
        for entry in self._kb_entries:
            variants = " ".join(entry.get("question_variants", []))
            keywords = " ".join(entry.get("keywords", []))
            text = (
                f"{entry.get('category', '')} "
                f"{entry.get('title', '')} "
                f"{entry.get('content', '')} "
                f"{variants} {keywords}"
            )
            tokenised_corpus.append(_tokenize(text))

        self._bm25 = BM25Okapi(tokenised_corpus)

        elapsed = time.time() - t0
        logger.info(
            "[MatchingEngine] BM25 keyword index ready: %d entries (%.1fs)",
            len(self._kb_entries), elapsed,
        )

    # ------------------------------------------------------------------
    # Properties for startup diagnostics
    # ------------------------------------------------------------------

    @property
    def num_entries(self) -> int:
        return len(self._kb_entries)

    @property
    def embedding_dim(self) -> int:
        return get_vector_store().embedding_dim

    # ------------------------------------------------------------------
    # Overview-query detection
    # ------------------------------------------------------------------

    @staticmethod
    def is_overview_query(query: str) -> bool:
        """Return True if *query* looks like a broad / overview question."""
        q = query.lower()
        return any(kw in q for kw in OVERVIEW_KEYWORDS)

    # ------------------------------------------------------------------
    # BM25 keyword search
    # ------------------------------------------------------------------

    def _keyword_search(
        self, query: str, top_k: int,
    ) -> list[tuple[dict, float]]:
        """
        Return top-k (entry, raw_bm25_score) pairs via BM25X keyword search.
        """
        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            idx_int = int(idx)
            raw_score = float(scores[idx_int])
            if raw_score > 0:
                results.append((self._kb_entries[idx_int], raw_score))
        return results

    # ------------------------------------------------------------------
    # Hybrid retrieval — top K entries
    # ------------------------------------------------------------------

    def retrieve_top_k(
        self,
        query: str,
        top_k: int = RAG_TOP_K,
        is_overview: bool = False,
        expand_variants: bool = True,
    ) -> list[RetrievalResult]:
        """
        Hybrid retrieval combining semantic + keyword search.

        Pipeline
        --------
        1. Preprocess the query (normalize + generate Gemini variants).
        2. Retrieve candidates from ChromaDB (semantic / vector search)
           using the original query **and** each variant.
        3. Retrieve candidates from BM25 (keyword search) using the
           normalised query **and** each variant.
        4. Normalise each score set to [0, 1].
        5. Merge and deduplicate by entry ID.
        6. Compute ``final_score = 0.6 * semantic + 0.4 * keyword``.
        7. Return top-k results sorted descending by final_score.

        Parameters
        ----------
        query:
            The raw user question string.
        top_k:
            Maximum number of entries to return.
        is_overview:
            When True, bumps top_k to RAG_OVERVIEW_TOP_K.
        expand_variants:
            When True (default), generates Gemini query variants for
            broader recall.

        Returns
        -------
        list[RetrievalResult]
            Sorted descending by hybrid score.
        """
        query = query.strip()
        if not query:
            return []

        if is_overview:
            top_k = max(top_k, RAG_OVERVIEW_TOP_K)

        # ---- Preprocessing ----
        pp = preprocess_query(query, expand_variants=expand_variants)
        variants = pp["variants"]

        # All queries to fan out across both search backends.
        # BM25's _tokenize already normalises (lowercase, stopwords, lemma)
        # so we pass the original query rather than the NLTK-normalised form.
        semantic_queries = [query] + variants
        keyword_queries = [query] + variants

        # Fetch more candidates than needed, then trim after merging
        candidate_k = top_k * _CANDIDATE_MULTIPLIER

        # ---- Semantic search via ChromaDB ----
        vs = get_vector_store()
        semantic_scores: dict[int, float] = {}
        entries_by_id: dict[int, dict] = {}

        for sq in semantic_queries:
            vector_results = vs.query_vector_store(
                sq, k=candidate_k, is_overview=False,
            )
            for vr in vector_results:
                eid = vr.entry["id"]
                # Keep the best semantic score for each entry
                if eid not in semantic_scores or vr.score > semantic_scores[eid]:
                    semantic_scores[eid] = vr.score
                    entries_by_id[eid] = vr.entry

        # ---- Keyword search via BM25 ----
        keyword_scores: dict[int, float] = {}

        for kq in keyword_queries:
            keyword_results = self._keyword_search(kq, top_k=candidate_k)
            for entry, raw_score in keyword_results:
                eid = entry["id"]
                if eid not in keyword_scores or raw_score > keyword_scores[eid]:
                    keyword_scores[eid] = raw_score
                if eid not in entries_by_id:
                    entries_by_id[eid] = entry

        # ---- Normalise scores to [0, 1] ----
        def _normalise(scores: dict[int, float]) -> dict[int, float]:
            if not scores:
                return scores
            vals = list(scores.values())
            lo, hi = min(vals), max(vals)
            rng = hi - lo
            if rng == 0:
                return {k: 1.0 for k in scores}
            return {k: (v - lo) / rng for k, v in scores.items()}

        sem_norm = _normalise(semantic_scores)
        kw_norm = _normalise(keyword_scores)

        # ---- Merge: weighted combination ----
        all_ids = set(sem_norm) | set(kw_norm)
        merged: list[tuple[int, float]] = []

        for eid in all_ids:
            s_score = sem_norm.get(eid, 0.0)
            k_score = kw_norm.get(eid, 0.0)
            final = (HYBRID_SEMANTIC_WEIGHT * s_score
                     + HYBRID_KEYWORD_WEIGHT * k_score)
            merged.append((eid, final))

        # Sort by final score descending, take top_k
        merged.sort(key=lambda x: x[1], reverse=True)
        merged = merged[:top_k]

        # ---- Build results ----
        results: list[RetrievalResult] = []
        for eid, final_score in merged:
            entry = entries_by_id.get(eid) or self._kb_by_id.get(eid, {})

            # Resolve a display question — prefer the original FAQ question
            # if available, otherwise fall back to the KB title
            faq_entry = next(
                (e for e in self._store.entries if e.get("id") == eid), None
            )
            question = (
                faq_entry["question"] if faq_entry
                else entry.get("title", "")
            )
            answer = (
                faq_entry["answer"] if faq_entry
                else entry.get("content", "")
            )

            # Ensure entry dict has the keys downstream expects
            merged_entry = {**entry}
            if faq_entry:
                merged_entry["question"] = question
                merged_entry["answer"] = answer

            results.append(RetrievalResult(
                entry=merged_entry,
                score=final_score,
                question=question,
                category=entry.get("category", "Uncategorised"),
            ))

        logger.info(
            "[MatchingEngine] hybrid_retrieve: query=%r  "
            "variants=%d  is_overview=%s  "
            "semantic_candidates=%d  keyword_candidates=%d  "
            "merged=%d  top_score=%.4f",
            query[:80],
            len(variants), is_overview,
            len(semantic_scores), len(keyword_scores),
            len(results),
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
