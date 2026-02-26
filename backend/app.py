# app.py
# Entry point for the Flask application.
from dotenv import load_dotenv
import os

load_dotenv()  # This loads variables from .env file
import logging
import sys

from flask import Flask, jsonify
from flask_cors import CORS

from config import CORS_ORIGINS, GEMINI_API_KEY
from routes.chat import chat_bp
from utils.loader import load_faq, FAQStore
from services.matching_engine import build_engine

# Configure root logger so all module-level messages appear in the console
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """Application factory - creates and configures the Flask instance."""
    app = Flask(__name__)

    # ------------------------------------------------------------------ CORS
    CORS(app, resources={r"/api/*": {"origins": CORS_ORIGINS}})

    # --------------------------------------------------------- Load FAQ data
    try:
        store: FAQStore = load_faq()
    except (FileNotFoundError, ValueError, TypeError, RuntimeError) as exc:
        logger.critical("[startup] Failed to load FAQ dataset: %s", exc)
        sys.exit(1)

    app.config["FAQ_STORE"] = store
    print(
        f"[KARE FAQ] {len(store)} entries loaded "
        f"across {len(store.categories)} categories."
    )
    print(f"[KARE FAQ] Categories: {', '.join(store.categories)}")

    # ------------------------------------------------ Build semantic search engine
    try:
        engine = build_engine(store)
    except Exception as exc:
        logger.critical("[startup] Failed to build matching engine: %s", exc)
        sys.exit(1)

    print(
        f"[KARE FAQ] Semantic engine ready — "
        f"{engine.num_entries} entries × {engine.embedding_dim} dims."
    )
    print("Gemini key loaded:", bool(os.getenv("GEMINI_API_KEY")))

    # -------------------------------------------------- Initialise Gemini (RAG)
    if GEMINI_API_KEY:
        try:
            from services.llm_generator import build_generator
            generator = build_generator(api_key=GEMINI_API_KEY)
            print(
                f"[KARE FAQ] Gemini generator ready — "
                f"model: {generator._model_name}"
            )
        except Exception as exc:
            logger.warning(
                "[startup] Gemini initialisation failed: %s — "
                "falling back to TF-IDF-only mode.", exc
            )
    else:
        logger.warning(
            "[startup] GEMINI_API_KEY not set. "
            "Running in TF-IDF-only mode (no RAG generation). "
            "Set the GEMINI_API_KEY environment variable to enable RAG."
        )

    # -------------------------------------------------------- Register routes
    app.register_blueprint(chat_bp, url_prefix="/api")

    # ---------------------------------------------------- Health-check route
    @app.get("/api/health")
    def health():
        s = app.config["FAQ_STORE"]

        # Check if Gemini is available
        gemini_status = "unavailable"
        try:
            from services.llm_generator import get_generator
            get_generator()
            gemini_status = "ready"
        except RuntimeError:
            gemini_status = "not configured"

        return jsonify({
            "status":        "ok",
            "faq_entries":   len(s),
            "categories":    s.categories,
            "gemini_status": gemini_status,
        })

    # ----------------------------------------------------- Global error handlers

    @app.errorhandler(400)
    def bad_request(exc):
        return jsonify({"error": "Bad request.", "detail": str(exc)}), 400

    @app.errorhandler(404)
    def not_found(exc):
        return jsonify({"error": f"Endpoint not found: {exc}"}), 404

    @app.errorhandler(405)
    def method_not_allowed(exc):
        return jsonify({"error": "HTTP method not allowed on this endpoint."}), 405

    @app.errorhandler(500)
    def internal_error(exc):
        logger.exception("[500] Unhandled exception")
        return jsonify({"error": "An unexpected server error occurred."}), 500

    return app


if __name__ == "__main__":
    flask_app = create_app()
    flask_app.run(host="0.0.0.0", port=5000, debug=True)
