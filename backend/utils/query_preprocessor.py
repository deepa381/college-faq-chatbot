# utils/query_preprocessor.py
# Query preprocessing pipeline for RAG retrieval.
#
# Public API
# ----------
# normalize(query)          – lowercase, strip punctuation, remove stopwords,
#                             lemmatize (returns cleaned string)
# generate_variants(query)  – use Gemini to produce alternative phrasings
# preprocess(query)         – full pipeline: normalize + variant generation

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Optional

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NLTK bootstrap — download required data once, silently
# ---------------------------------------------------------------------------

_NLTK_RESOURCES = ["punkt_tab", "stopwords", "wordnet"]


def _ensure_nltk_data() -> None:
    """Download NLTK resources if they are not already present."""
    for res in _NLTK_RESOURCES:
        try:
            nltk.data.find(f"tokenizers/{res}" if "punkt" in res else res)
        except LookupError:
            nltk.download(res, quiet=True)


_ensure_nltk_data()

# Build the stopword set once after download
_STOPWORDS: frozenset[str] = frozenset(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()


# ---------------------------------------------------------------------------
# 1. Normalisation
# ---------------------------------------------------------------------------

def normalize(query: str) -> str:
    """
    Normalise a user query for keyword-based retrieval.

    Steps
    -----
    1. Lowercase.
    2. Remove punctuation (keep alphanumerics, spaces, and a few symbols
       like ``+``, ``#``, ``.`` that appear in course names).
    3. Tokenise.
    4. Remove English stopwords.
    5. Lemmatize each remaining token.
    6. Rejoin into a single space-separated string.

    Parameters
    ----------
    query:
        The raw user question.

    Returns
    -------
    str
        The cleaned, lemmatized query string.
    """
    text = query.lower()
    # Keep alphanumerics, whitespace, and symbols common in course names
    text = re.sub(r"[^a-z0-9\s.+#]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in _STOPWORDS]
    tokens = [_lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# 2. Gemini-based query variant generation
# ---------------------------------------------------------------------------

_VARIANT_PROMPT = (
    "You are a search-query expansion assistant for a university FAQ system.\n"
    "Given the user's question below, generate exactly 3 short alternative "
    "phrasings that preserve the original meaning but use different wording.\n\n"
    "Rules:\n"
    "- Each variant must be a concise search query (3-10 words).\n"
    "- Do NOT add information not present in the original question.\n"
    "- Output ONLY the 3 variants, one per line, with no numbering, "
    "bullets, or extra text.\n\n"
    "User question: {query}\n"
)


def generate_variants(
    query: str,
    *,
    max_variants: int = 3,
) -> list[str]:
    """
    Use Gemini to generate alternative phrasings of *query*.

    If Gemini is unavailable or the call fails, returns an empty list
    so retrieval can still proceed with the original query.

    Parameters
    ----------
    query:
        The raw (un-normalised) user question.
    max_variants:
        Maximum number of variants to return.

    Returns
    -------
    list[str]
        Up to *max_variants* alternative query strings.
    """
    try:
        from services.llm_generator import get_generator
        generator = get_generator()
    except (RuntimeError, ImportError):
        logger.debug("[query_preprocessor] Gemini unavailable — skipping variant generation.")
        return []

    prompt = _VARIANT_PROMPT.format(query=query)

    try:
        response = generator._model.generate_content(prompt)
        if not response or not response.text:
            return []

        lines = [
            line.strip()
            for line in response.text.strip().splitlines()
            if line.strip()
        ]
        # Strip any leading numbering/bullets the model may add
        cleaned: list[str] = []
        for line in lines:
            line = re.sub(r"^[\d\.\-\*\)]+\s*", "", line).strip()
            if line:
                cleaned.append(line)

        return cleaned[:max_variants]

    except Exception as exc:
        logger.warning(
            "[query_preprocessor] Variant generation failed: %s", exc
        )
        return []


# ---------------------------------------------------------------------------
# 3. Full preprocessing pipeline
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def _cached_normalize(query: str) -> str:
    """Memoised normalisation to avoid repeated work."""
    return normalize(query)


def preprocess(
    query: str,
    *,
    expand_variants: bool = True,
) -> dict:
    """
    Run the complete query preprocessing pipeline.

    Returns a dict with::

        {
            "original":   str,   # original query (stripped)
            "normalized": str,   # lowercased / lemmatized form
            "variants":   list[str],  # Gemini-generated alternatives
        }

    Parameters
    ----------
    query:
        Raw user input.
    expand_variants:
        When False, skip Gemini variant generation (useful for
        batch evaluation or when Gemini is unavailable).
    """
    query = query.strip()
    normalized = _cached_normalize(query)
    variants = generate_variants(query) if expand_variants else []

    return {
        "original": query,
        "normalized": normalized,
        "variants": variants,
    }
