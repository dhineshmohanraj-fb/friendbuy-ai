"""
Tests for retriever/hybrid_retriever.py — CP3.

These tests exercise the RRF fusion logic and the retrieve() function
in isolation.  All external dependencies (ChromaDB, Kuzu, Ollama) are
mocked so the tests run without any running services.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from retriever.hybrid_retriever import HybridResult, _rrf_fuse
from retriever.vector_search import SearchResult


# ---------------------------------------------------------------------------
# Helpers to build fake SearchResult objects
# ---------------------------------------------------------------------------

def _sr(file_path: str, content: str, score: float = 0.8) -> SearchResult:
    doc = Document(
        page_content=content,
        metadata={"file_path": file_path, "repo_name": "test-repo"},
    )
    return SearchResult(document=doc, score=score)


# ===========================================================================
# RRF fusion
# ===========================================================================

class TestRRFFuse:
    def test_empty_lists_return_empty(self):
        assert _rrf_fuse([]) == []

    def test_single_list_preserves_order(self):
        lst = [_sr("a.py", "alpha"), _sr("b.py", "beta"), _sr("c.py", "gamma")]
        result = _rrf_fuse([lst], top_k=3)
        assert len(result) == 3
        paths = [r.file_path for r in result]
        assert paths == ["a.py", "b.py", "c.py"]

    def test_duplicate_across_lists_scores_higher(self):
        """A document in two lists should outrank one only in one list."""
        shared  = _sr("shared.py",  "shared content", score=0.7)
        unique1 = _sr("unique1.py", "only in list1",  score=0.9)
        unique2 = _sr("unique2.py", "only in list2",  score=0.9)

        list1 = [unique1, shared]
        list2 = [unique2, shared]

        result = _rrf_fuse([list1, list2], top_k=5)
        paths  = [r.file_path for r in result]

        # shared.py appears in both lists → must outrank the uniques
        assert "shared.py" in paths
        assert paths.index("shared.py") < paths.index("unique1.py")
        assert paths.index("shared.py") < paths.index("unique2.py")

    def test_top_k_respected(self):
        lst = [_sr(f"{i}.py", f"content {i}") for i in range(10)]
        result = _rrf_fuse([lst], top_k=3)
        assert len(result) == 3

    def test_scores_are_positive(self):
        lst = [_sr("a.py", "alpha"), _sr("b.py", "beta")]
        result = _rrf_fuse([lst])
        assert all(r.score > 0 for r in result)

    def test_three_lists_fused(self):
        v_list = [_sr("v.py", "vector result")]
        b_list = [_sr("b.py", "bm25 result")]
        g_list = [_sr("g.py", "graph result")]
        result = _rrf_fuse([v_list, b_list, g_list], top_k=5)
        paths  = {r.file_path for r in result}
        assert {"v.py", "b.py", "g.py"} == paths

    def test_same_doc_in_all_three_is_top(self):
        champion = _sr("champion.py", "in all three lists")
        lists = [
            [champion, _sr("other1.py", "x")],
            [champion, _sr("other2.py", "y")],
            [champion, _sr("other3.py", "z")],
        ]
        result = _rrf_fuse(lists, top_k=5)
        assert result[0].file_path == "champion.py"

    def test_deduplication_by_content_prefix(self):
        """Same content from two lists should only appear once in output."""
        doc = Document(
            page_content="duplicate content here",
            metadata={"file_path": "dup.py", "repo_name": "r"},
        )
        sr1 = SearchResult(document=doc, score=0.9)
        sr2 = SearchResult(document=doc, score=0.8)
        result = _rrf_fuse([[sr1], [sr2]], top_k=5)
        assert sum(1 for r in result if r.file_path == "dup.py") == 1


# ===========================================================================
# retrieve() — integration with mocked dependencies
# ===========================================================================

class _FakeHybrid:
    chunks         = [_sr("api/service.py", "campaign service code")]
    graph_context  = None
    vector_count   = 1
    bm25_count     = 0
    graph_count    = 0
    query_entities = []
    retrieval_ms   = 12.3


class TestRetrieve:
    def test_returns_hybrid_result(self, monkeypatch):
        """retrieve() should return a HybridResult even if graph/BM25 disabled."""
        import retriever.hybrid_retriever as hr
        monkeypatch.setattr(
            "retriever.hybrid_retriever.vector_search",
            lambda q, top_k=None, repo_name=None: [_sr("f.py", "content")],
        )
        monkeypatch.setattr(
            "retriever.hybrid_retriever._fetch_graph_chunks",
            lambda *a, **kw: [],
        )

        result = hr.retrieve("test query", use_graph=False, use_bm25=False)
        assert isinstance(result, HybridResult)
        assert len(result.chunks) >= 1

    def test_vector_only_mode(self, monkeypatch):
        import retriever.hybrid_retriever as hr
        calls = {"vector": 0, "bm25": 0}

        def fake_vector(q, top_k=None, repo_name=None):
            calls["vector"] += 1
            return [_sr("v.py", "v")]

        def fake_bm25(q, top_k=None, repo_name=None):
            calls["bm25"] += 1
            return [_sr("b.py", "b")]

        monkeypatch.setattr("retriever.hybrid_retriever.vector_search", fake_vector)
        monkeypatch.setattr("retriever.bm25_index.bm25_search", fake_bm25, raising=False)

        hr.retrieve("q", use_graph=False, use_bm25=False)
        assert calls["vector"] == 1
        assert calls["bm25"]   == 0

    def test_empty_vector_results_returns_empty_hybrid(self, monkeypatch):
        import retriever.hybrid_retriever as hr
        monkeypatch.setattr(
            "retriever.hybrid_retriever.vector_search",
            lambda *a, **kw: [],
        )
        result = hr.retrieve("query with no results", use_graph=False, use_bm25=False)
        assert result.chunks == []

    def test_retrieval_ms_positive(self, monkeypatch):
        import retriever.hybrid_retriever as hr
        monkeypatch.setattr(
            "retriever.hybrid_retriever.vector_search",
            lambda *a, **kw: [_sr("f.py", "content")],
        )
        result = hr.retrieve("q", use_graph=False, use_bm25=False)
        assert result.retrieval_ms >= 0

    def test_result_counts_match(self, monkeypatch):
        import retriever.hybrid_retriever as hr
        monkeypatch.setattr(
            "retriever.hybrid_retriever.vector_search",
            lambda *a, **kw: [_sr(f"{i}.py", f"c{i}") for i in range(5)],
        )
        result = hr.retrieve("q", use_graph=False, use_bm25=False)
        assert result.vector_count == 5
        assert result.bm25_count   == 0
        assert result.graph_count  == 0
