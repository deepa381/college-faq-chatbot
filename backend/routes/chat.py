# routes/chat.py
# Blueprint: POST /api/chat
# Thin HTTP handler — delegates all RAG logic to services.rag_pipeline.

import logging

from flask import Blueprint, request, jsonify
from services.rag_pipeline import process_query

logger = logging.getLogger(__name__)
chat_bp = Blueprint("chat", __name__)


@chat_bp.route("/chat", methods=["POST"])
def chat():
    """
    POST /api/chat

    Request body (JSON):
        { "message": "<user question>", "session_id": "<optional session id>" }

    Success response (200):
        {
            "answer":            str,
            "retrieved_entries":  list[dict],
            "model_used":        str,
            "success":           bool,
            "session_id":        str | null
        }

    Error responses:
        400 – missing / empty message, or non-JSON body
        500 – unexpected internal error
    """

    # --- Parse & validate request body ---
    body = request.get_json(silent=True)

    if body is None:
        return jsonify({
            "error": "Invalid request. Body must be JSON with Content-Type: application/json."
        }), 400

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

    # --- Extract optional session ID ---
    session_id = body.get("session_id")
    if not isinstance(session_id, str):
        session_id = None

    # --- Delegate to the RAG pipeline ---
    result = process_query(query=user_message, session_id=session_id)

    if result.error:
        return jsonify({"error": result.error}), 500

    return jsonify(result.to_dict()), 200
