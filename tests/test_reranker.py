"""
Tests for retriever/reranker.py — CP4.

These tests cover:
- Graceful fallback when flashrank is not installed.
- Correct pass-through of results in fallback mode.
- top_k respected.
- Empty input handled.
- Mocked flashrank path for when it IS available.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from retriever.reranker import flashrank_available, rerank, reset_ranker, _get_flashrank
from retriever.vector_search import SearchResult


# ---------------------------------------------------------------------------
# Availability guard — defined at top before any skipif
# ---------------------------------------------------------------------------

def _flashrank_importable() -> bool:
    try:
        import flashrank  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sr(path: str, text: str, score: float = 0.7) -> SearchResult:
    doc = Document(
        page_content=text,
        metadata={"file_path": path, "repo_name": "test"},
    )
    return SearchResult(document=doc, score=score)


# ---------------------------------------------------------------------------
# Fallback mode (flashrank not installed or mock-disabled)
# ---------------------------------------------------------------------------

class TestFallback:
    def test_empty_input_returns_empty(self):
        assert rerank("query", []) == []

    def test_fallback_preserves_original_order(self, monkeypatch):
        monkeypatch.setattr("retriever.reranker.flashrank_available", lambda: False)
        results = [_sr("a.py", "alpha"), _sr("b.py", "beta"), _sr("c.py", "gamma")]
        out = rerank("query", results)
        assert [r.file_path for r in out] == ["a.py", "b.py", "c.py"]

    def test_fallback_respects_top_k(self, monkeypatch):
        monkeypatch.setattr("retriever.reranker.flashrank_available", lambda: False)
        results = [_sr(f"{i}.py", f"text{i}") for i in range(10)]
        out = rerank("query", results, top_k=3)
        assert len(out) == 3

    def test_fallback_top_k_none_returns_all(self, monkeypatch):
        monkeypatch.setattr("retriever.reranker.flashrank_available", lambda: False)
        results = [_sr(f"{i}.py", f"text{i}") for i in range(5)]
        out = rerank("query", results, top_k=None)
        assert len(out) == 5

    def test_original_scores_preserved_in_fallback(self, monkeypatch):
        monkeypatch.setattr("retriever.reranker.flashrank_available", lambda: False)
        results = [_sr("x.py", "x", score=0.42)]
        out = rerank("query", results)
        assert out[0].score == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# Mocked flashrank path
# ---------------------------------------------------------------------------

class TestMockedFlashrank:
    """
    Verify reranker logic with a fake flashrank implementation.

    We monkeypatch ``_get_flashrank`` — the single function that imports
    from the ``flashrank`` package — so tests run without it installed.
    """

    @staticmethod
    def _fake_get_flashrank():
        """Return (FakeRanker, FakeRerankRequest) — no real flashrank import."""

        class FakeRerankRequest:
            def __init__(self, query, passages):
                self.query    = query
                self.passages = passages

        class FakeRanker:
            """Reverses the passage order so we can verify the output changed."""
            def rerank(self, req):
                passages = req.passages
                n   = len(passages)
                out = []
                for rank, p in enumerate(reversed(passages)):
                    out.append({"id": p["id"], "score": (n - rank) / n, "text": p["text"]})
                return out

        return FakeRanker(), FakeRerankRequest

    def test_reranker_changes_order(self, monkeypatch):
        monkeypatch.setattr("retriever.reranker.flashrank_available", lambda: True)
        monkeypatch.setattr("retriever.reranker._get_flashrank", self._fake_get_flashrank)
        reset_ranker()

        results = [_sr("first.py", "a"), _sr("second.py", "b"), _sr("third.py", "c")]
        out = rerank("q", results)

        # Fake reverses: third.py should be ranked first
        assert out[0].file_path == "third.py"
        assert out[-1].file_path == "first.py"

    def test_reranker_top_k_respected(self, monkeypatch):
        monkeypatch.setattr("retriever.reranker.flashrank_available", lambda: True)
        monkeypatch.setattr("retriever.reranker._get_flashrank", self._fake_get_flashrank)
        reset_ranker()

        results = [_sr(f"{i}.py", f"text{i}") for i in range(8)]
        out = rerank("q", results, top_k=3)
        assert len(out) == 3

    def test_reranker_scores_updated(self, monkeypatch):
        """Output scores should come from the cross-encoder, not originals."""
        monkeypatch.setattr("retriever.reranker.flashrank_available", lambda: True)
        monkeypatch.setattr("retriever.reranker._get_flashrank", self._fake_get_flashrank)
        reset_ranker()

        results = [_sr("a.py", "alpha", score=0.1), _sr("b.py", "beta", score=0.2)]
        out = rerank("q", results)
        # Fake gives (n - rank)/n scores: b.py→1.0, a.py→0.5
        scores = {r.file_path: r.score for r in out}
        assert scores["b.py"] == pytest.approx(1.0)
        assert scores["a.py"] == pytest.approx(0.5)

    def test_reranker_exception_falls_back(self, monkeypatch):
        """If _get_flashrank raises, return the original list."""
        monkeypatch.setattr("retriever.reranker.flashrank_available", lambda: True)
        reset_ranker()

        def _bad_flashrank():
            raise RuntimeError("model download failed")

        monkeypatch.setattr("retriever.reranker._get_flashrank", _bad_flashrank)

        results = [_sr("a.py", "x"), _sr("b.py", "y")]
        out = rerank("q", results)
        assert len(out) == 2
        assert out[0].file_path == "a.py"


# ---------------------------------------------------------------------------
# Availability helper
# ---------------------------------------------------------------------------

class TestAvailability:
    def test_returns_bool(self):
        result = flashrank_available()
        assert isinstance(result, bool)

    @pytest.mark.skipif(_flashrank_importable(), reason="flashrank IS installed")
    def test_returns_false_when_not_installed(self):
        assert flashrank_available() is False
