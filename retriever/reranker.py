"""
Cross-encoder reranker — CP4.

Wraps ``flashrank`` (``ms-marco-MiniLM-L-12-v2``) to reorder the fused
candidate chunks before they reach the Qwen context filter.

Design notes
------------
- Lazy singleton: the model is downloaded and loaded on first use, then reused.
  Cold start ≈ 150 ms; subsequent calls ≈ 5–15 ms for 20 passages.
- Graceful fallback: if flashrank is not installed (or errors), the function
  returns the original list, unmodified.  The pipeline continues working
  normally — just without cross-encoder rescoring.
- Only the first 2 000 characters of each passage are sent to the model
  (enough signal; avoids slow inference on huge chunks).

Usage::

    from retriever.reranker import rerank
    from retriever.vector_search import SearchResult

    reranked = rerank("How does referral attribution work?", results, top_k=5)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from retriever.vector_search import SearchResult

# ---------------------------------------------------------------------------
# Availability guard
# ---------------------------------------------------------------------------

def flashrank_available() -> bool:
    """Return True if the ``flashrank`` package is importable."""
    try:
        import flashrank  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Lazy singleton
# ---------------------------------------------------------------------------

_ranker = None
_RANKER_MODEL = "ms-marco-MiniLM-L-12-v2"
_RANKER_CACHE = "/tmp/flashrank_cache"


def _get_flashrank():
    """
    Return ``(ranker, RerankRequest)`` — both loaded from ``flashrank``.

    Lazy singleton: the model is loaded once and reused.
    Monkeypatch this function in tests to avoid the real flashrank import.
    """
    global _ranker
    from flashrank import Ranker, RerankRequest  # type: ignore[import]
    if _ranker is None:
        _ranker = Ranker(model_name=_RANKER_MODEL, cache_dir=_RANKER_CACHE)
    return _ranker, RerankRequest


def reset_ranker() -> None:
    """Drop the cached ranker (used in tests to ensure a clean state)."""
    global _ranker
    _ranker = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rerank(
    query: str,
    results: list["SearchResult"],
    top_k: int | None = None,
) -> list["SearchResult"]:
    """
    Rerank *results* with a cross-encoder model.

    Args:
        query:   The user's natural-language query.
        results: Candidate :class:`~retriever.vector_search.SearchResult` objects
                 produced by the hybrid retriever (post-RRF).
        top_k:   How many results to return.  Defaults to ``len(results)``.

    Returns:
        A (possibly re-ordered, possibly truncated) list of
        :class:`~retriever.vector_search.SearchResult`.
        The ``.score`` attribute is replaced with the cross-encoder score.

    Falls back silently to ``results[:top_k]`` on any error.
    """
    if not results:
        return results

    effective_top_k = top_k if top_k is not None else len(results)

    if not flashrank_available():
        return results[:effective_top_k]

    try:
        from retriever.vector_search import SearchResult

        ranker, RerankRequest = _get_flashrank()

        passages = [
            {
                "id":   i,
                "text": r.document.page_content[:2000],
            }
            for i, r in enumerate(results)
        ]

        rerank_req = RerankRequest(query=query, passages=passages)
        reranked   = ranker.rerank(rerank_req)

        # Build output preserving original SearchResult metadata
        output: list[SearchResult] = []
        for item in reranked[:effective_top_k]:
            orig = results[item["id"]]
            output.append(
                SearchResult(
                    document=orig.document,
                    score=float(item.get("score", orig.score)),
                )
            )
        return output

    except Exception:  # noqa: BLE001
        # Reranker failure is non-fatal — return original order
        return results[:effective_top_k]
