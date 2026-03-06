# services/rag_pipeline.py
# Central RAG pipeline orchestrator.
#
# Encapsulates the full query-to-answer flow so that route handlers
# remain thin HTTP adapters.  Every stage is delegated to a dedicated
# module under services/ or utils/, keeping components loosely coupled.
#
# Pipeline
# --------
# User Query
#   ↓  Query preprocessing   (utils.query_preprocessor)
#   ↓  Cache check            (utils.cache_manager)
#   ↓  Hybrid retrieval       (services.matching_engine)
#   │    ├─ semantic search    (services.vector_store / ChromaDB)
#   │    └─ keyword search     (BM25 / TF-IDF sparse retrieval)
#   ↓  Top-K document selection
#   ↓  Conversation memory    (utils.conversation_memory)
#   ↓  Gemini generation      (services.llm_generator)
#   ↓  Fallback response      (if Gemini fails)
#   ↓  Return response
#
# Public API
# ----------
# process_query(query, session_id) → PipelineResult

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from config import FALLBACK_MESSAGE
from utils.query_preprocessor import normalize as normalize_query
from utils.cache_manager import cache_get, cache_put
from utils.conversation_memory import get_history, add_exchange
from services.matching_engine import get_engine
from services.llm_generator import get_generator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Outcome of a single pipeline run, returned to the route handler."""
    answer: str
    retrieved_entries: list[dict] = field(default_factory=list)
    model_used: Optional[str] = None
    success: bool = True
    session_id: Optional[str] = None
    error: Optional[str] = None
    cached: bool = False

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "retrieved_entries": self.retrieved_entries,
            "model_used": self.model_used,
            "success": self.success,
            "session_id": self.session_id,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def process_query(
    query: str,
    session_id: Optional[str] = None,
) -> PipelineResult:
    """
    Run the full RAG pipeline for a single user query.

    Steps
    -----
    1. **Query preprocessing** — normalise text for cache key.
    2. **Cache check** — return immediately if a cached response exists
       (skipped when conversation history is present, since follow-up
       answers depend on prior turns).
    3. **Hybrid retrieval** — fan out across ChromaDB semantic search
       and BM25 keyword search, merge scores, select top-K documents.
    4. **Conversation memory** — fetch prior user/assistant turns for
       the session so Gemini can handle follow-up questions.
    5. **Gemini response generation** — pass retrieved context +
       conversation history to the LLM.
    6. **Fallback** — if Gemini is unavailable or fails, return the
       best retrieval-only answer.
    7. **Post-processing** — store the exchange in conversation memory
       and cache the response for future identical queries.

    Parameters
    ----------
    query:
        The raw user question (already stripped of surrounding whitespace
        by the caller).
    session_id:
        Optional session identifier for multi-turn conversations.

    Returns
    -------
    PipelineResult
    """

    # === STEP 1: Query preprocessing ===
    cache_key = normalize_query(query)

    # === STEP 2: Cache check ===
    # Fetch conversation history early to decide whether to skip cache.
    # Follow-up questions are context-dependent, so cached answers from
    # a different (or empty) conversation would be incorrect.
    conversation_history: list[dict] = []
    if session_id:
        conversation_history = get_history(session_id)

    if not conversation_history:
        cached = cache_get(cache_key)
        if cached is not None:
            logger.info("[pipeline] Cache hit for query: %r", query)
            return PipelineResult(
                answer=cached["answer"],
                retrieved_entries=cached.get("retrieved_entries", []),
                model_used=cached.get("model_used"),
                success=True,
                session_id=session_id,
                cached=True,
            )

    # === STEP 3: Hybrid retrieval (semantic + keyword → top-K) ===
    try:
        engine = get_engine()
        is_overview = engine.is_overview_query(query)
        retrieved = engine.retrieve_top_k(query, is_overview=is_overview)
    except RuntimeError as exc:
        logger.error("[pipeline] Matching engine unavailable: %s", exc)
        return PipelineResult(
            answer=FALLBACK_MESSAGE,
            success=False,
            session_id=session_id,
            error="Matching engine is not available.",
        )
    except Exception as exc:
        logger.exception("[pipeline] Retrieval error: %r", query)
        return PipelineResult(
            answer=FALLBACK_MESSAGE,
            success=False,
            session_id=session_id,
            error="An internal error occurred during retrieval.",
        )

    if not retrieved:
        return PipelineResult(
            answer=FALLBACK_MESSAGE,
            success=True,
            session_id=session_id,
        )

    # === STEP 4: Conversation memory context ===
    # (already fetched above; injected into Gemini prompt below)

    # === STEP 5: Gemini response generation ===
    try:
        generator = get_generator()
    except RuntimeError:
        logger.warning("[pipeline] Gemini unavailable — using retrieval fallback.")
        best = retrieved[0]
        return _fallback_result(retrieved, session_id)

    try:
        entries_for_llm = [r.entry for r in retrieved]
        gen_result = generator.generate(
            query=query,
            retrieved_entries=entries_for_llm,
            is_overview=is_overview,
            conversation_history=conversation_history or None,
        )
    except Exception:
        logger.exception("[pipeline] Gemini generation failed: %r", query)
        gen_result = None

    # === STEP 6: Fallback if Gemini failed ===
    if gen_result is None or not gen_result.success:
        logger.warning("[pipeline] Falling back to retrieval for: %r", query)
        return _fallback_result(retrieved, session_id)

    # === STEP 7: Post-processing — memory + cache ===
    result = PipelineResult(
        answer=gen_result.answer,
        retrieved_entries=gen_result.retrieved_entries,
        model_used=gen_result.model_used,
        success=True,
        session_id=session_id,
    )

    # Store exchange in conversation memory
    if session_id:
        add_exchange(session_id, query, result.answer)

    # Cache stateless queries only
    if not conversation_history:
        cache_put(cache_key, result.to_dict())

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fallback_result(
    retrieved: list,
    session_id: Optional[str],
) -> PipelineResult:
    """Build a PipelineResult using the best retrieval-only answer."""
    best = retrieved[0]
    return PipelineResult(
        answer=best.entry.get("answer", best.entry.get("content", FALLBACK_MESSAGE)),
        retrieved_entries=[r.to_dict() for r in retrieved],
        model_used="retrieval-fallback",
        success=True,
        session_id=session_id,
    )
