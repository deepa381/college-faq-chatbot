# utils/loader.py
# In-memory FAQ data store.
#
# Public API
# ----------
# FAQStore          – dataclass that holds the parsed corpus
# load_faq()        – reads kare_faq.json, validates entries, returns FAQStore
# get_faq_store()   – returns the singleton FAQStore (call after load_faq())

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from config import FAQ_FILE

logger = logging.getLogger(__name__)

# Required keys that every FAQ entry must contain
_REQUIRED_KEYS: frozenset[str] = frozenset({"id", "category", "question", "answer"})

# Module-level singleton — populated by load_faq()
_store: Optional["FAQStore"] = None


@dataclass
class FAQStore:
    """
    Immutable in-memory representation of the FAQ corpus.

    Attributes
    ----------
    entries   : full list of validated FAQ dicts
    questions : list of question strings (positionally aligned with entries)
    answers   : list of answer strings   (positionally aligned with entries)
    categories: sorted list of unique category names
    """

    entries: list[dict] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)

    def __len__(self) -> int:  # noqa: D105
        return len(self.entries)

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"FAQStore(entries={len(self.entries)}, "
            f"categories={len(self.categories)})"
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _read_json(path: str) -> list:
    """
    Open *path* and parse it as JSON.

    Raises
    ------
    FileNotFoundError – file does not exist
    ValueError        – file is not valid JSON
    TypeError         – JSON root is not a list
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[FAQLoader] Dataset not found at: {path}\n"
            "Make sure kare_faq.json is in the backend/ directory."
        )

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"[FAQLoader] kare_faq.json contains invalid JSON: {exc}"
        ) from exc

    if not isinstance(data, list):
        raise TypeError(
            f"[FAQLoader] Expected a JSON array at the root, "
            f"got {type(data).__name__!r} instead."
        )

    return data


def _validate_entry(entry: dict, index: int) -> bool:
    """
    Return True if *entry* has all required keys and non-empty string values.
    Log a warning and return False if it should be skipped.
    """
    missing = _REQUIRED_KEYS - entry.keys()
    if missing:
        logger.warning(
            "[FAQLoader] Entry #%d is missing keys %s — skipped.",
            index,
            missing,
        )
        return False

    for key in ("question", "answer"):
        if not isinstance(entry[key], str) or not entry[key].strip():
            logger.warning(
                "[FAQLoader] Entry #%d has empty '%s' — skipped.",
                index,
                key,
            )
            return False

    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_faq(path: str = FAQ_FILE) -> FAQStore:
    """
    Read *path*, validate every entry, and return a populated :class:`FAQStore`.

    Also saves the result as the module-level singleton so it can be retrieved
    later via :func:`get_faq_store` without re-reading disk.

    Parameters
    ----------
    path:
        Filesystem path to the JSON dataset. Defaults to ``config.FAQ_FILE``.

    Returns
    -------
    FAQStore
        In-memory corpus ready for the matching service.

    Raises
    ------
    FileNotFoundError
        If the JSON file is missing.
    ValueError
        If the file cannot be parsed as JSON.
    TypeError
        If the JSON root is not an array.
    RuntimeError
        If zero valid entries are found after validation.
    """
    global _store

    raw: list = _read_json(path)

    entries, questions, answers, cat_set = [], [], [], set()

    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            logger.warning(
                "[FAQLoader] Item at index %d is not an object — skipped.", idx
            )
            continue

        if not _validate_entry(item, idx):
            continue

        entries.append(item)
        questions.append(item["question"].strip())
        answers.append(item["answer"].strip())
        cat_set.add(item.get("category", "Uncategorised"))

    if not entries:
        raise RuntimeError(
            "[FAQLoader] No valid FAQ entries found in the dataset. "
            "Check kare_faq.json for structural issues."
        )

    _store = FAQStore(
        entries=entries,
        questions=questions,
        answers=answers,
        categories=sorted(cat_set),
    )

    logger.info(
        "[FAQLoader] Loaded %d entries across %d categories.",
        len(_store),
        len(_store.categories),
    )
    return _store


def get_faq_store() -> FAQStore:
    """
    Return the singleton :class:`FAQStore` created by :func:`load_faq`.

    Raises
    ------
    RuntimeError
        If called before :func:`load_faq` has been executed.
    """
    if _store is None:
        raise RuntimeError(
            "[FAQLoader] FAQStore has not been initialised. "
            "Call load_faq() during application startup."
        )
    return _store
