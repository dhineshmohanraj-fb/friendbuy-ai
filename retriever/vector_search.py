"""Similarity search against the ChromaDB vector store."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from config import get_settings
from indexer.embedder import load_vector_store


@dataclass
class SearchResult:
    document: Document
    score: float

    @property
    def repo_name(self) -> str:
        return self.document.metadata.get("repo_name", "unknown")

    @property
    def file_path(self) -> str:
        return self.document.metadata.get("file_path", "unknown")

    @property
    def content(self) -> str:
        return self.document.page_content


def search(
    query: str,
    top_k: int | None = None,
    repo_name: str | None = None,
) -> list[SearchResult]:
    """
    Return the top-k most relevant chunks for *query*.

    Chunks with a relevance score below ``settings.min_relevance_score``
    are filtered out before returning.

    Args:
        query:     Natural-language question or keyword string.
        top_k:     How many results to return (defaults to settings.top_k_results).
        repo_name: When provided, restrict results to a single repository.

    Returns:
        List of SearchResult sorted by relevance (highest first).
    """
    settings = get_settings()
    k = top_k or settings.top_k_results
    db = load_vector_store()

    where_filter: dict | None = None
    if repo_name:
        where_filter = {"repo_name": {"$eq": repo_name}}

    raw = db.similarity_search_with_relevance_scores(
        query,
        k=k,
        filter=where_filter,
    )

    return [
        SearchResult(document=doc, score=score)
        for doc, score in raw
        if score >= settings.min_relevance_score
    ]
