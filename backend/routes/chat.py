# routes/chat.py
# Blueprint: POST /api/chat
# RAG pipeline: retrieve top-K FAQ entries → Gemini generates grounded answer.

import logging

from flask import Blueprint, request, jsonify
from services.matching_engine import get_engine
from services.llm_generator import get_generator
from config import FALLBACK_MESSAGE

logger = logging.getLogger(__name__)
chat_bp = Blueprint("chat", __name__)


@chat_bp.route("/chat", methods=["POST"])
def chat():
    """
    POST /api/chat

    Request body (JSON):
        { "message": "<user question>" }

    Success response (200):
        {
            "answer":            str,
            "retrieved_entries":  list[dict],
            "model_used":        str,
            "success":           bool
        }

    Error responses:
        400 – missing / empty message, or non-JSON body
        500 – unexpected internal error
    """

    # --- Parse body ---
    body = request.get_json(silent=True)

    if body is None:
        return jsonify({
            "error": "Invalid request. Body must be JSON with Content-Type: application/json."
        }), 400

    # --- Validate 'message' key ---
    if "message" not in body:
        return jsonify({
            "error": 'Missing required field: "message".',
        }), 400

    raw_message = body["message"]

    if not isinstance(raw_message, str):
        return jsonify({
            "error": '"message" must be a string.',
        }), 400

    user_message = raw_message.strip()

    if not user_message:
        return jsonify({
            "error": '"message" cannot be empty or whitespace only.',
        }), 400

    # ------------------------------------------------------------------
    # STEP 1: Retrieve top-K relevant FAQ entries via semantic search
    # ------------------------------------------------------------------
    try:
        engine = get_engine()
        is_overview = engine.is_overview_query(user_message)
        retrieved = engine.retrieve_top_k(user_message, is_overview=is_overview)
    except RuntimeError as exc:
        logger.error("[chat] Matching engine unavailable: %s", exc)
        return jsonify({
            "error": "Matching engine is not available. Please try again later."
        }), 500
    except Exception as exc:
        logger.exception("[chat] Unexpected error during retrieval: %r", user_message)
        return jsonify({
            "error": "An internal error occurred during retrieval. Please try again."
        }), 500

    # ------------------------------------------------------------------
    # STEP 2: If no relevant entries found → return fallback
    # ------------------------------------------------------------------
    if not retrieved:
        return jsonify({
            "answer":           FALLBACK_MESSAGE,
            "retrieved_entries": [],
            "model_used":       None,
            "success":          True,
        }), 200

    # ------------------------------------------------------------------
    # STEP 3: Pass retrieved entries + user query to Gemini (RAG)
    # ------------------------------------------------------------------
    try:
        generator = get_generator()
    except RuntimeError:
        # Gemini not initialised — fall back to best semantic match
        logger.warning("[chat] Gemini generator not available, falling back to retrieval.")
        best = retrieved[0]
        return jsonify({
            "answer":           best.entry["answer"],
            "retrieved_entries": [r.to_dict() for r in retrieved],
            "model_used":       "retrieval-fallback",
            "success":          True,
        }), 200

    try:
        entries_for_llm = [r.entry for r in retrieved]
        result = generator.generate(
            query=user_message,
            retrieved_entries=entries_for_llm,
            is_overview=is_overview,
        )
    except Exception as exc:
        logger.exception("[chat] Gemini generation failed: %r", user_message)
        result = None

    # If Gemini failed (exception or API error), fall back to best semantic match
    if result is None or not result.success:
        logger.warning("[chat] Falling back to retrieval answer for query: %r", user_message)
        best = retrieved[0]
        return jsonify({
            "answer":           best.entry["answer"],
            "retrieved_entries": [r.to_dict() for r in retrieved],
            "model_used":       "retrieval-fallback",
            "success":          True,
        }), 200

    return jsonify(result.to_dict()), 200
