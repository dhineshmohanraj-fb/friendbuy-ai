"""
BM25 sparse index over the ChromaDB chunk corpus — CP3.

BM25 (Best Match 25) is a keyword-frequency ranking algorithm.
Unlike vector search (semantic similarity), BM25 finds chunks that
**literally contain** the query words — which catches exact function
names, error messages, env var names, and file paths that embeddings
can miss.

The index is built lazily on first use from all ChromaDB chunks and
cached as a module-level singleton.  Call ``invalidate()`` after
re-indexing so the next query rebuilds from fresh data.

Usage::

    from retriever.bm25_index import bm25_search

    results = bm25_search("CampaignService create", top_k=10)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

from retriever.vector_search import SearchResult


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

@dataclass
class _BM25State:
    index: Any           # BM25Okapi instance
    docs: list[Document]
    corpus_size: int
    built_at: float      # time.time()


_state: _BM25State | None = None


def invalidate() -> None:
    """Drop the cached BM25 index (call after re-indexing)."""
    global _state
    _state = None


# ---------------------------------------------------------------------------
# Tokeniser — code-aware
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """
    Tokenise source-code text for BM25.

    - Splits camelCase / PascalCase into sub-words
      (``createCampaign`` → ``create``, ``Campaign``)
    - Splits on underscores (``snake_case`` → ``snake``, ``case``)
    - Lower-cases everything
    - Keeps only tokens ≥ 2 characters
    """
    # Split camelCase: insert space before each uppercase sequence
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)

    # Extract all alphanumeric tokens (also splits on _, ., /, etc.)
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9]*|[0-9]+", text)
    return [t.lower() for t in tokens if len(t) >= 2]


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def _build_index() -> _BM25State | None:
    """
    Load all chunks from ChromaDB and build a BM25Okapi index.
    Returns None if rank-bm25 is not installed or ChromaDB is empty.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return None

    try:
        from indexer.embedder import load_vector_store
        db = load_vector_store()
        col = db._collection  # type: ignore[attr-defined]
        total = col.count()
        if total == 0:
            return None

        # Fetch all chunks (capped to protect RAM on M1)
        from config import get_settings
        settings = get_settings()
        limit = getattr(settings, "bm25_corpus_limit", 10_000)
        limit = min(limit, total)

        raw = col.get(include=["documents", "metadatas"], limit=limit)
        texts:     list[str]  = raw.get("documents") or []
        metadatas: list[dict] = raw.get("metadatas")  or []
        ids:       list[str]  = raw.get("ids")         or []

        if not texts:
            return None

        docs = [
            Document(page_content=t, metadata=m or {})
            for t, m in zip(texts, metadatas)
        ]

        corpus = [_tokenize(t) for t in texts]
        index  = BM25Okapi(corpus)

        return _BM25State(
            index=index,
            docs=docs,
            corpus_size=len(docs),
            built_at=time.time(),
        )

    except Exception:  # noqa: BLE001
        return None


def _get_state() -> _BM25State | None:
    """Return the (lazily built) BM25 index state."""
    global _state
    if _state is None:
        _state = _build_index()
    return _state


# ---------------------------------------------------------------------------
# Public search API
# ---------------------------------------------------------------------------

def bm25_search(
    query: str,
    top_k: int = 20,
    repo_name: str | None = None,
) -> list[SearchResult]:
    """
    Return the top-*top_k* chunks most relevant to *query* according to BM25.

    Args:
        query:     Natural-language or keyword query string.
        top_k:     Maximum number of results to return.
        repo_name: Restrict results to this repository (optional).

    Returns:
        List of :class:`SearchResult` sorted by BM25 score (highest first).
        Returns an empty list if rank-bm25 is not installed or the index
        is not yet built.
    """
    state = _get_state()
    if state is None:
        return []

    tokens = _tokenize(query)
    if not tokens:
        return []

    scores = state.index.get_scores(tokens)

    # Pair (score, doc) and sort descending
    ranked = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True,
    )

    results: list[SearchResult] = []
    for idx, score in ranked:
        if score <= 0:
            break
        if len(results) >= top_k:
            break

        doc = state.docs[idx]
        if repo_name and doc.metadata.get("repo_name") != repo_name:
            continue

        # Normalise score to [0, 1] range so it's comparable with vector scores
        # BM25 scores are unbounded; divide by (max_score + epsilon) to normalise
        results.append(SearchResult(document=doc, score=float(score)))

    # Normalise scores so the top result = 1.0
    if results:
        max_score = results[0].score
        if max_score > 0:
            results = [
                SearchResult(document=r.document, score=r.score / max_score)
                for r in results
            ]

    return results


def corpus_size() -> int:
    """Return the number of chunks currently in the BM25 index (0 if not built)."""
    state = _get_state()
    return state.corpus_size if state else 0
