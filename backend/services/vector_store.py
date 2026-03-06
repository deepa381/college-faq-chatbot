# services/vector_store.py
# Persistent vector database using ChromaDB for RAG retrieval.
#
# Public API
# ----------
# index_dataset()                  – loads the RAG knowledge base and indexes into ChromaDB
# query_vector_store(query, k)     – retrieves top-k similar documents
# get_vector_store()               – returns the cached singleton VectorStore

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL,
    RAG_TOP_K,
    RAG_OVERVIEW_TOP_K,
    OVERVIEW_KEYWORDS,
    VECTOR_DB_DIR,
    KNOWLEDGE_BASE_FILE,
)

logger = logging.getLogger(__name__)

# Module-level singleton
_vector_store: Optional["VectorStore"] = None

# ChromaDB collection name
_COLLECTION_NAME = "kare_knowledge_base"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class VectorSearchResult:
    """A single retrieved knowledge chunk with its similarity score."""
    entry: dict
    score: float
    title: str
    category: str

    def to_dict(self) -> dict:
        return {
            "title":    self.title,
            "category": self.category,
            "score":    round(self.score, 4),
        }


# ---------------------------------------------------------------------------
# Custom embedding function for ChromaDB
# ---------------------------------------------------------------------------

class SentenceTransformerEmbeddingFunction:
    """Wraps SentenceTransformer to conform to ChromaDB's EmbeddingFunction interface."""

    def __init__(self, model: SentenceTransformer) -> None:
        self._model = model

    def name(self) -> str:
        return "sentence-transformer-minilm"

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._encode(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self._encode(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self._encode(input)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

class VectorStore:
    """
    Persistent vector database backed by ChromaDB.

    On first run, loads the RAG knowledge base, generates embeddings with
    SentenceTransformer (all-MiniLM-L6-v2), and persists them to disk.
    On subsequent runs, ChromaDB loads directly from the persisted directory
    without re-embedding.

    Uses cosine similarity for retrieval (ChromaDB's ``hnsw:space=cosine``).
    """

    def __init__(self) -> None:
        logger.info("[VectorStore] Loading embedding model: %s ...", EMBEDDING_MODEL)
        t0 = time.time()
        self._model = SentenceTransformer(EMBEDDING_MODEL)

        # Initialise persistent ChromaDB client
        os.makedirs(VECTOR_DB_DIR, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=VECTOR_DB_DIR,
            settings=Settings(anonymized_telemetry=False),
        )

        self._embedding_fn = SentenceTransformerEmbeddingFunction(self._model)
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        elapsed = time.time() - t0
        logger.info(
            "[VectorStore] ChromaDB client ready (%.1fs). "
            "Collection '%s' has %d entries.",
            elapsed, _COLLECTION_NAME, self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_entries(self) -> int:
        return self._collection.count()

    @property
    def embedding_dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_dataset(self) -> int:
        """
        Load the RAG knowledge base and index entries into ChromaDB.

        Skips indexing if the collection already contains data (embeddings
        are generated only once and persisted to disk).

        Returns the total number of entries in the collection.
        """
        existing_count = self._collection.count()
        if existing_count > 0:
            logger.info(
                "[VectorStore] Collection already populated with %d entries. "
                "Skipping re-indexing.", existing_count,
            )
            return existing_count

        # Load knowledge base JSON
        logger.info("[VectorStore] Loading knowledge base from %s ...", KNOWLEDGE_BASE_FILE)
        with open(KNOWLEDGE_BASE_FILE, "r", encoding="utf-8") as f:
            entries: list[dict] = json.load(f)

        if not entries:
            raise ValueError("[VectorStore] Knowledge base file is empty.")

        logger.info("[VectorStore] Indexing %d entries into ChromaDB ...", len(entries))
        t0 = time.time()

        # Prepare batch data for ChromaDB
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for entry in entries:
            entry_id = str(entry["id"])

            # Build the document text for embedding: combine key fields
            # for rich semantic representation
            variants_text = " ".join(entry.get("question_variants", []))
            keywords_text = " ".join(entry.get("keywords", []))
            doc_text = (
                f"{entry.get('category', '')} "
                f"{entry.get('title', '')} "
                f"{entry.get('content', '')} "
                f"{variants_text} "
                f"{keywords_text}"
            )

            # Metadata stored alongside each vector
            metadata = {
                "id": entry["id"],
                "category": entry.get("category", ""),
                "title": entry.get("title", ""),
                "related_links": json.dumps(entry.get("related_links", [])),
            }

            ids.append(entry_id)
            documents.append(doc_text)
            metadatas.append(metadata)

        # ChromaDB has a batch size limit; add in chunks
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            self._collection.add(
                ids=ids[i:i + batch_size],
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
            )

        elapsed = time.time() - t0
        total = self._collection.count()
        logger.info(
            "[VectorStore] Indexed %d entries in %.1fs. "
            "Vectors persisted to %s.",
            total, elapsed, VECTOR_DB_DIR,
        )
        return total

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    @staticmethod
    def is_overview_query(query: str) -> bool:
        """Return True if *query* looks like a broad / overview question."""
        q = query.lower()
        return any(kw in q for kw in OVERVIEW_KEYWORDS)

    def query_vector_store(
        self,
        query: str,
        k: int = RAG_TOP_K,
        is_overview: bool = False,
    ) -> list[VectorSearchResult]:
        """
        Retrieve the top-k most similar knowledge chunks for *query*.

        Uses cosine similarity via ChromaDB's HNSW index.

        Parameters
        ----------
        query:
            The raw user question string.
        k:
            Maximum number of entries to return.
        is_overview:
            If True, bumps k to RAG_OVERVIEW_TOP_K for broader context.

        Returns
        -------
        list[VectorSearchResult]
            Sorted descending by similarity score.
        """
        query = query.strip()
        if not query:
            return []

        if is_overview:
            k = max(k, RAG_OVERVIEW_TOP_K)

        # Query ChromaDB — returns results sorted by distance (ascending)
        results = self._collection.query(
            query_texts=[query],
            n_results=min(k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        search_results: list[VectorSearchResult] = []

        if results and results["ids"] and results["ids"][0]:
            for idx, doc_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][idx]
                distance = results["distances"][0][idx]
                document = results["documents"][0][idx]

                # ChromaDB cosine distance = 1 - cosine_similarity
                similarity_score = 1.0 - distance

                # Reconstruct entry dict for downstream compatibility
                entry = {
                    "id": metadata["id"],
                    "category": metadata["category"],
                    "title": metadata["title"],
                    "content": document,
                    "related_links": json.loads(metadata.get("related_links", "[]")),
                }

                search_results.append(VectorSearchResult(
                    entry=entry,
                    score=similarity_score,
                    title=metadata.get("title", ""),
                    category=metadata.get("category", ""),
                ))

        logger.info(
            "[VectorStore] query=%r  k=%d  results=%d  top_score=%.4f",
            query[:80], k, len(search_results),
            search_results[0].score if search_results else 0.0,
        )

        return search_results


# ---------------------------------------------------------------------------
# Module-level factory & accessor (singleton pattern)
# ---------------------------------------------------------------------------

def index_dataset() -> int:
    """
    Public entry point: initialise the VectorStore and index the dataset.

    Returns the total number of indexed entries.
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store.index_dataset()


def query_vector_store(
    query: str,
    k: int = RAG_TOP_K,
) -> list[VectorSearchResult]:
    """
    Public entry point: retrieve top-k similar documents for a query.

    Raises RuntimeError if the vector store has not been initialised.
    """
    store = get_vector_store()
    is_overview = store.is_overview_query(query)
    return store.query_vector_store(query, k=k, is_overview=is_overview)


def get_vector_store() -> VectorStore:
    """Return the cached VectorStore singleton, or raise if not initialised."""
    if _vector_store is None:
        raise RuntimeError(
            "[VectorStore] Not initialised. Call index_dataset() first."
        )
    return _vector_store
