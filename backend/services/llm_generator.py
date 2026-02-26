# services/llm_generator.py
# Gemini API wrapper for Retrieval-Augmented Generation (RAG).
#
# Public API
# ----------
# GeminiGenerator        – stateful wrapper around the google-generativeai SDK
# build_generator()      – factory that creates and caches the singleton
# get_generator()        – returns the cached singleton
# format_context(entries) – formats retrieved FAQ entries into structured text

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import google.generativeai as genai

from config import GEMINI_API_KEY, GEMINI_MODEL, FALLBACK_MESSAGE

logger = logging.getLogger(__name__)

# Module-level singleton
_generator: Optional["GeminiGenerator"] = None

# ---------------------------------------------------------------------------
# Grounding prompt — enforces answer fidelity to retrieved content
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are the official AI assistant for Kalasalingam Academy of Research "
    "and Education (KARE).\n\n"

    "Your task is to provide intelligent, well-structured, and informative "
    "answers to user queries.\n\n"

    "Instructions:\n"
    "1. Use the retrieved FAQ entries as your primary knowledge source.\n"
    "2. You may think analytically and combine multiple entries to create "
    "a complete and structured answer.\n"
    "3. If the user asks for overview, complete details, or general information, "
    "provide a comprehensive summary using all relevant retrieved data.\n"
    "4. You may rephrase and expand explanations for clarity, "
    "but do not introduce unrelated external facts.\n"
    "5. Organize answers logically when appropriate (introduction, academics, facilities, placements, etc.).\n"
    "6. If insufficient information is retrieved, politely state that detailed "
    "information is limited in the available data.\n\n"

    "Tone Requirements:\n"
    "- Professional and informative\n"
    "- Clear and student-friendly\n"
    "- Avoid repeating sentences exactly as written in the entries\n"
    "- Do not mention that the information comes from FAQ entries\n"
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    """
    Returned by :meth:`GeminiGenerator.generate` for every query.

    Attributes
    ----------
    answer           : The generated answer text.
    retrieved_entries : List of dicts with the FAQ entries used as context.
    model_used       : Name of the Gemini model used.
    success          : True when generation completed without error.
    error            : Error message if generation failed, else None.
    """
    answer: str
    retrieved_entries: list[dict]
    model_used: str
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dict."""
        return {
            "answer":            self.answer,
            "retrieved_entries":  self.retrieved_entries,
            "model_used":        self.model_used,
            "success":           self.success,
            "error":             self.error,
        }


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def format_context(entries: list[dict]) -> str:
    """
    Format a list of retrieved FAQ entries into a structured context string
    suitable for the Gemini prompt.

    Each entry is expected to have at least: question, answer, category.

    Parameters
    ----------
    entries:
        List of FAQ dicts from the retrieval step.

    Returns
    -------
    str
        Formatted multi-line context block.
    """
    lines: list[str] = []
    for i, entry in enumerate(entries, start=1):
        lines.append(f"--- FAQ Entry {i} ---")
        lines.append(f"Category: {entry.get('category', 'N/A')}")
        lines.append(f"Question: {entry.get('question', 'N/A')}")
        lines.append(f"Answer: {entry.get('answer', 'N/A')}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class GeminiGenerator:
    """
    Stateful Gemini API wrapper initialised once at startup.

    Uses ``google-generativeai`` to call the specified model with a
    grounding prompt that restricts answers to the provided FAQ context.
    """

    def __init__(self, api_key: str, model_name: str = GEMINI_MODEL) -> None:
        if not api_key:
            raise ValueError(
                "[GeminiGenerator] GEMINI_API_KEY is empty. "
                "Set the GEMINI_API_KEY environment variable."
            )

        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model_name)
        self._model_name = model_name

        logger.info(
            "[GeminiGenerator] Initialised with model: %s", model_name
        )

    # ------------------------------------------------------------------
    # Public method
    # ------------------------------------------------------------------

    def generate(
        self,
        query: str,
        retrieved_entries: list[dict],
        is_overview: bool = False,
    ) -> GenerationResult:
        """
        Generate a grounded answer using Gemini.

        Steps
        -----
        1. Format retrieved FAQ entries into structured context.
        2. Build the full prompt: system instructions + context + user query.
        3. Call the Gemini API.
        4. Return the generated text wrapped in a GenerationResult.

        Parameters
        ----------
        query:
            The raw user question string.
        retrieved_entries:
            List of FAQ dicts retrieved by the matching engine.
        is_overview:
            When True, adds an extra instruction encouraging Gemini to
            produce a structured, comprehensive summary.

        Returns
        -------
        GenerationResult
        """
        if not retrieved_entries:
            return GenerationResult(
                answer=FALLBACK_MESSAGE,
                retrieved_entries=[],
                model_used=self._model_name,
                success=True,
            )

        # Format context from retrieved entries
        context_block = format_context(retrieved_entries)

        # Optional overview hint
        overview_hint = ""
        if is_overview:
            overview_hint = (
                "\n=== SPECIAL INSTRUCTION ===\n"
                "The user is asking for a broad overview. Combine ALL the "
                "retrieved entries into a single, comprehensive, well-structured "
                "answer. Organise the response by topic (e.g. Introduction, "
                "Academics, Facilities, Placements, Achievements) and cover "
                "every relevant detail from the entries.\n\n"
            )

        # Build the full prompt
        prompt = (
            f"{SYSTEM_PROMPT}\n"
            f"=== RETRIEVED FAQ ENTRIES ===\n\n"
            f"{context_block}\n"
            f"{overview_hint}"
            f"=== USER QUESTION ===\n\n"
            f"{query}\n\n"
            f"=== YOUR ANSWER ===\n"
        )

        try:
            response = self._model.generate_content(prompt)

            # Extract text from the response
            if response and response.text:
                answer_text = response.text.strip()
            else:
                logger.warning(
                    "[GeminiGenerator] Empty response from Gemini for query: %r",
                    query,
                )
                answer_text = FALLBACK_MESSAGE

            # Build serialisable entry summaries for the response
            entry_summaries = [
                {
                    "question": e.get("question", ""),
                    "category": e.get("category", ""),
                }
                for e in retrieved_entries
            ]

            return GenerationResult(
                answer=answer_text,
                retrieved_entries=entry_summaries,
                model_used=self._model_name,
                success=True,
            )

        except Exception as exc:
            logger.exception(
                "[GeminiGenerator] API call failed for query: %r — %s",
                query, exc,
            )
            # Preserve entry summaries even on error so the response
            # still shows what was retrieved.
            entry_summaries = [
                {
                    "question": e.get("question", ""),
                    "category": e.get("category", ""),
                }
                for e in retrieved_entries
            ]
            return GenerationResult(
                answer=FALLBACK_MESSAGE,
                retrieved_entries=entry_summaries,
                model_used=self._model_name,
                success=False,
                error=str(exc),
            )


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

def build_generator(
    api_key: str = GEMINI_API_KEY,
    model_name: str = GEMINI_MODEL,
) -> GeminiGenerator:
    """
    Build a :class:`GeminiGenerator`, cache it, and return it.
    Safe to call multiple times (rebuilds and replaces the singleton).
    """
    global _generator
    _generator = GeminiGenerator(api_key=api_key, model_name=model_name)
    return _generator


def get_generator() -> GeminiGenerator:
    """
    Return the cached :class:`GeminiGenerator`.

    Raises
    ------
    RuntimeError
        If called before :func:`build_generator`.
    """
    if _generator is None:
        raise RuntimeError(
            "[GeminiGenerator] Generator not initialised. "
            "Call build_generator() during application startup."
        )
    return _generator
