# config.py
# Centralized configuration for the Flask app.

import os

# Absolute path to the project root (one level above this file)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the FAQ dataset
FAQ_FILE = os.path.join(BASE_DIR, "kare_faq.json")

# Path to the RAG-optimized knowledge base
KNOWLEDGE_BASE_FILE = os.path.join(BASE_DIR, "kare_knowledge_base.json")

# Directory for persistent ChromaDB vector storage
VECTOR_DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# Origins allowed to call the API (React dev server)
CORS_ORIGINS = ["http://localhost:5173", "http://localhost:3000"]

# Fallback message when no FAQ entry matches the query
FALLBACK_MESSAGE = "Sorry, I could not find relevant information."

# ---------------------------------------------------------------------------
# Semantic Search Configuration
# ---------------------------------------------------------------------------

# Sentence-transformer model for semantic embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Number of top FAQ entries to retrieve for RAG context
RAG_TOP_K = 5

# Higher retrieval count for overview / broad queries
RAG_OVERVIEW_TOP_K = 8

# Keywords that signal an overview-type query (checked case-insensitively)
OVERVIEW_KEYWORDS = (
    "about", "overview", "complete", "details", "information",
    "tell me", "university", "college", "everything", "summary",
)

# ---------------------------------------------------------------------------
# Hybrid Retrieval Weights
# ---------------------------------------------------------------------------

# final_score = SEMANTIC_WEIGHT * embedding_score + KEYWORD_WEIGHT * keyword_score
HYBRID_SEMANTIC_WEIGHT = 0.6
HYBRID_KEYWORD_WEIGHT = 0.4

# ---------------------------------------------------------------------------
# Gemini / RAG Configuration
# ---------------------------------------------------------------------------

# API key — MUST be set as an environment variable
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Model to use for generation
GEMINI_MODEL = "gemini-2.0-flash"
