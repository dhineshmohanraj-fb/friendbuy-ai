"""
Hybrid retriever — CP3.

Runs vector search, BM25, and graph traversal in parallel, then fuses
the three ranked lists using Reciprocal Rank Fusion (RRF).

                    ┌─────────────────────────┐
    query ──────── ▶│  entity extraction      │ ──▶ [CampaignService, …]
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                   ▼
        vector search         BM25 search      graph traversal
        (ChromaDB)           (rank-bm25)        (Kuzu)
        top-20               top-20             related files
              │                  │                   │
              └──────────────────┼───────────────────┘
                                 ▼
                          RRF fusion (k=60)
                                 │
                           top-k results  +  GraphContext
                                 ▼
                         HybridResult

Usage::

    from retriever.hybrid_retriever import retrieve

    result = retrieve("what does CampaignService create?")
    # result.chunks  → fused ranked list of SearchResult
    # result.graph_context.relationship_summary  → injected into Claude prompt
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from retriever.vector_search import SearchResult, search as vector_search


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class HybridResult:
    """Everything retrieve() returns."""

    chunks:         list[SearchResult]
    graph_context:  object   # GraphContext | None  (avoid hard dep on kuzu)
    vector_count:   int
    bm25_count:     int
    graph_count:    int
    query_entities: list[str]
    retrieval_ms:   float


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------

def _rrf_fuse(
    ranked_lists: list[list[SearchResult]],
    k: int = 60,
    top_k: int = 10,
) -> list[SearchResult]:
    """
    Reciprocal Rank Fusion.

    Formula: score(d) = Σ_i  1 / (k + rank_i(d))

    Documents appearing in multiple lists get additive boosts.
    Documents only in one list still get a non-zero score.

    Args:
        ranked_lists: Each sub-list is already sorted best-first.
        k:            RRF constant (60 is the literature default).
        top_k:        Return only this many results.

    Returns:
        De-duplicated, re-ranked list of :class:`SearchResult`.
    """
    scores:  dict[str, float]       = {}
    doc_map: dict[str, SearchResult] = {}

    for result_list in ranked_lists:
        for rank, result in enumerate(result_list, start=1):
            # Stable identity key: prefer the chunk's content hash from metadata,
            # fall back to a deterministic string of file + content prefix.
            doc_id = (
                result.document.metadata.get("_content_hash")
                or result.document.metadata.get("content_hash")
                or f"{result.file_path}::{result.content[:80]}"
            )
            scores[doc_id]  = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            doc_map[doc_id] = result

    sorted_ids = sorted(scores, key=lambda d: scores[d], reverse=True)

    # Attach the fused RRF score back onto each result
    fused: list[SearchResult] = []
    for doc_id in sorted_ids[:top_k]:
        r = doc_map[doc_id]
        fused.append(SearchResult(
            document=r.document,
            score=scores[doc_id],    # RRF score (not cosine — just used for ordering)
        ))
    return fused


# ---------------------------------------------------------------------------
# Graph chunk enrichment
# ---------------------------------------------------------------------------

def _fetch_graph_chunks(
    related_file_paths: list[str],
    repo_name: str | None,
    top_k: int,
) -> list[SearchResult]:
    """
    Fetch ChromaDB chunks for files the graph traversal identified as related.

    We use a metadata filter on file_path so we only pull chunks that
    actually exist in the vector store.
    """
    if not related_file_paths:
        return []

    try:
        from indexer.embedder import load_vector_store
        db  = load_vector_store()
        col = db._collection  # type: ignore[attr-defined]

        results: list[SearchResult] = []
        from langchain_core.documents import Document

        for fpath in related_file_paths[:8]:    # cap to 8 files
            try:
                raw = col.get(
                    where={"file_path": {"$eq": fpath}},
                    include=["documents", "metadatas"],
                    limit=3,                     # up to 3 chunks per file
                )
                for txt, meta in zip(
                    raw.get("documents") or [],
                    raw.get("metadatas")  or [],
                ):
                    if txt:
                        results.append(SearchResult(
                            document=Document(page_content=txt, metadata=meta or {}),
                            score=0.5,            # neutral score — ranking done by RRF
                        ))
            except Exception:  # noqa: BLE001
                continue

        return results[:top_k]

    except Exception:  # noqa: BLE001
        return []


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    repo_name: str | None = None,
    top_k: int | None = None,
    use_graph: bool = True,
    use_bm25:  bool = True,
) -> HybridResult:
    """
    Run hybrid retrieval and return fused results.

    Args:
        query:     The user's natural-language question.
        repo_name: Restrict all searches to this repo (optional).
        top_k:     How many fused results to return (defaults to settings value).
        use_graph: Include graph traversal (requires Kuzu index).
        use_bm25:  Include BM25 sparse search (requires rank-bm25).

    Returns:
        A :class:`HybridResult` with the fused chunk list and graph context.
    """
    from config import get_settings
    settings = get_settings()

    final_k    = top_k or settings.top_k_results
    vector_k   = settings.vector_top_k
    bm25_k     = settings.bm25_top_k
    rrf_k      = settings.hybrid_rrf_k
    graph_hops = settings.graph_max_hops

    t0 = time.time()

    # ------------------------------------------------------------------
    # 1. Vector search (always on)
    # ------------------------------------------------------------------
    vector_results = vector_search(query, top_k=vector_k, repo_name=repo_name)
    v_count = len(vector_results)

    # ------------------------------------------------------------------
    # 2. BM25 sparse search (optional)
    # ------------------------------------------------------------------
    bm25_results: list[SearchResult] = []
    if use_bm25 and settings.use_bm25:
        try:
            from retriever.bm25_index import bm25_search
            bm25_results = bm25_search(query, top_k=bm25_k, repo_name=repo_name)
        except Exception:  # noqa: BLE001
            pass
    b_count = len(bm25_results)

    # ------------------------------------------------------------------
    # 3. Graph traversal + guided chunk fetch (optional)
    # ------------------------------------------------------------------
    from retriever.graph_search import GraphContext
    graph_context: GraphContext = GraphContext.empty()
    graph_chunks:  list[SearchResult] = []
    query_entities: list[str] = []

    if use_graph and settings.use_graph:
        try:
            from retriever.graph_search import GraphSearcher
            with GraphSearcher() as gs:
                query_entities = gs.extract_entities(query)
                if query_entities:
                    graph_context = gs.traverse(query_entities, max_hops=graph_hops)
                    graph_chunks  = _fetch_graph_chunks(
                        graph_context.related_file_paths, repo_name, top_k=final_k
                    )
        except (ImportError, FileNotFoundError):
            # Kuzu not installed or graph not built yet — degrade gracefully
            pass
        except Exception:  # noqa: BLE001
            pass

    g_count = len(graph_chunks)

    # ------------------------------------------------------------------
    # 4. RRF fusion
    # ------------------------------------------------------------------
    all_lists: list[list[SearchResult]] = [l for l in
        [vector_results, bm25_results, graph_chunks] if l]

    if not all_lists:
        return HybridResult(
            chunks=[], graph_context=graph_context,
            vector_count=0, bm25_count=0, graph_count=0,
            query_entities=query_entities,
            retrieval_ms=(time.time() - t0) * 1000,
        )

    fused = _rrf_fuse(all_lists, k=rrf_k, top_k=final_k)

    return HybridResult(
        chunks=fused,
        graph_context=graph_context,
        vector_count=v_count,
        bm25_count=b_count,
        graph_count=g_count,
        query_entities=query_entities,
        retrieval_ms=(time.time() - t0) * 1000,
    )
