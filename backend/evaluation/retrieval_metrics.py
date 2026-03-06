# evaluation/retrieval_metrics.py
# Offline evaluation framework for measuring RAG retrieval performance.
#
# Usage (from the backend/ directory):
#   python -m evaluation.retrieval_metrics
#
# The module loads the hybrid retrieval engine, runs every query in the
# test dataset against it, and prints a performance report.

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Ensure the backend package root is importable when run as a script
# ---------------------------------------------------------------------------
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from config import RAG_TOP_K  # noqa: E402

# Path to the ground-truth test queries (sibling file)
_TEST_QUERIES_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "test_queries.json"
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    """Evaluation outcome for a single test query."""
    query: str
    expected_ids: list[int]
    retrieved_ids: list[int]
    category: str
    reciprocal_rank: float
    hit_at_1: bool
    hit_at_3: bool
    top_score: float


@dataclass
class EvaluationReport:
    """Aggregated metrics across all test queries."""
    num_queries: int
    top_1_accuracy: float
    top_3_accuracy: float
    mrr: float
    per_category: dict[str, dict]
    elapsed_seconds: float
    query_results: list[QueryResult] = field(default_factory=list)

    def print_report(self) -> None:
        """Pretty-print the evaluation report to stdout."""
        print("\n" + "=" * 60)
        print("  RAG RETRIEVAL EVALUATION REPORT")
        print("=" * 60)
        print(f"  Test queries : {self.num_queries}")
        print(f"  Top-1 Accuracy : {self.top_1_accuracy:6.1%}")
        print(f"  Top-3 Accuracy : {self.top_3_accuracy:6.1%}")
        print(f"  MRR            : {self.mrr:.4f}")
        print(f"  Eval time      : {self.elapsed_seconds:.2f}s")

        if self.per_category:
            print("\n  --- Per-Category Breakdown ---")
            header = f"  {'Category':<28} {'Top-1':>6} {'Top-3':>6} {'MRR':>7} {'n':>4}"
            print(header)
            print("  " + "-" * (len(header) - 2))
            for cat in sorted(self.per_category):
                m = self.per_category[cat]
                print(
                    f"  {cat:<28} {m['top_1']:>5.0%} {m['top_3']:>5.0%} "
                    f"{m['mrr']:>7.4f} {m['n']:>4}"
                )

        # Queries that missed at top-1
        misses = [qr for qr in self.query_results if not qr.hit_at_1]
        if misses:
            print(f"\n  --- Top-1 Misses ({len(misses)}) ---")
            for qr in misses:
                print(f"  • [{qr.category}] {qr.query[:70]}")
                print(f"    expected={qr.expected_ids}  got={qr.retrieved_ids[:5]}")

        print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Metric computation helpers
# ---------------------------------------------------------------------------

def _reciprocal_rank(expected_ids: list[int], retrieved_ids: list[int]) -> float:
    """
    Compute the reciprocal rank: 1/rank of the first relevant result.
    Returns 0.0 if no expected ID appears in retrieved_ids.
    """
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in expected_ids:
            return 1.0 / rank
    return 0.0


def _hit_at_k(expected_ids: list[int], retrieved_ids: list[int], k: int) -> bool:
    """True if any expected ID appears in the top-k retrieved IDs."""
    return any(rid in expected_ids for rid in retrieved_ids[:k])


# ---------------------------------------------------------------------------
# Core evaluation pipeline
# ---------------------------------------------------------------------------

def load_test_queries(path: Optional[str] = None) -> list[dict]:
    """Load ground-truth test queries from JSON."""
    path = path or _TEST_QUERIES_FILE
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation(
    top_k: int = RAG_TOP_K,
    test_queries_path: Optional[str] = None,
    expand_variants: bool = False,
) -> EvaluationReport:
    """
    Execute the full evaluation pipeline:

    1. Initialise the retrieval engine (FAQ store + vector store + BM25).
    2. Load test queries with ground-truth expected IDs.
    3. For each query, call ``engine.retrieve_top_k()``.
    4. Compare retrieved IDs with expected IDs.
    5. Compute Top-1 Accuracy, Top-3 Accuracy, and MRR.
    6. Return an ``EvaluationReport``.

    Parameters
    ----------
    top_k:
        Number of results to retrieve per query (default: RAG_TOP_K).
    test_queries_path:
        Path to the test queries JSON.  Defaults to the bundled
        ``test_queries.json``.
    expand_variants:
        When True, use Gemini to generate query variants for each
        test query.  Default False to keep evaluation fast and
        deterministic.

    Returns
    -------
    EvaluationReport
    """
    # ---- Lazy imports to keep module importable without heavy deps ----
    from utils.loader import load_faq
    from services.vector_store import index_dataset, get_vector_store
    from services.matching_engine import build_engine

    print("[eval] Loading FAQ store …")
    store = load_faq()

    print("[eval] Indexing vector store …")
    index_dataset()
    get_vector_store()

    print("[eval] Building hybrid retrieval engine …")
    engine = build_engine(store)

    test_queries = load_test_queries(test_queries_path)
    print(f"[eval] Running {len(test_queries)} test queries (top_k={top_k}) …")

    query_results: list[QueryResult] = []
    t0 = time.time()

    for tq in test_queries:
        query = tq["query"]
        expected_ids: list[int] = tq["expected_ids"]
        category = tq.get("category", "Unknown")

        results = engine.retrieve_top_k(
            query, top_k=top_k, expand_variants=expand_variants,
        )
        retrieved_ids = [r.entry.get("id") for r in results]

        rr = _reciprocal_rank(expected_ids, retrieved_ids)
        h1 = _hit_at_k(expected_ids, retrieved_ids, 1)
        h3 = _hit_at_k(expected_ids, retrieved_ids, 3)
        top_score = results[0].score if results else 0.0

        query_results.append(QueryResult(
            query=query,
            expected_ids=expected_ids,
            retrieved_ids=retrieved_ids,
            category=category,
            reciprocal_rank=rr,
            hit_at_1=h1,
            hit_at_3=h3,
            top_score=top_score,
        ))

    elapsed = time.time() - t0

    # ---- Aggregate metrics ----
    n = len(query_results)
    top_1_acc = sum(1 for qr in query_results if qr.hit_at_1) / n if n else 0
    top_3_acc = sum(1 for qr in query_results if qr.hit_at_3) / n if n else 0
    mrr = sum(qr.reciprocal_rank for qr in query_results) / n if n else 0

    # ---- Per-category breakdown ----
    cats: dict[str, list[QueryResult]] = {}
    for qr in query_results:
        cats.setdefault(qr.category, []).append(qr)

    per_category: dict[str, dict] = {}
    for cat, items in cats.items():
        cn = len(items)
        per_category[cat] = {
            "top_1": sum(1 for q in items if q.hit_at_1) / cn,
            "top_3": sum(1 for q in items if q.hit_at_3) / cn,
            "mrr": sum(q.reciprocal_rank for q in items) / cn,
            "n": cn,
        }

    return EvaluationReport(
        num_queries=n,
        top_1_accuracy=top_1_acc,
        top_3_accuracy=top_3_acc,
        mrr=mrr,
        per_category=per_category,
        elapsed_seconds=elapsed,
        query_results=query_results,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    report = run_evaluation()
    report.print_report()
