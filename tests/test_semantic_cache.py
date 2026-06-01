"""
Tests for retriever/semantic_cache.py — CP4.

All tests use a temporary SQLite database (``tmp_path`` fixture) and a
deterministic fake embedder so Ollama is never required.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import pytest

from retriever.semantic_cache import CacheHit, SemanticCache, _cosine


# ---------------------------------------------------------------------------
# Fake embedder helpers
# ---------------------------------------------------------------------------

def _unit_vec(dim: int, idx: int) -> list[float]:
    """Build a unit vector whose first component equals idx/dim."""
    v = [0.0] * dim
    v[0] = idx / dim
    mag = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / mag for x in v]


def _make_embedder(vectors: dict[str, list[float]]):
    """Return an embedder that looks up *query* in *vectors*, else returns zeros."""
    def _emb(query: str) -> list[float] | None:
        return vectors.get(query, [0.0] * 8)
    return _emb


# Canonical 8-dim test vectors
VEC_A = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # points along dim-0
VEC_B = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # identical → cosine = 1.0
VEC_C = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # orthogonal → cosine = 0.0
VEC_NEAR = [0.99, 0.14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # cosine ≈ 0.99 to VEC_A


def _norm(v):
    mag = math.sqrt(sum(x * x for x in v))
    return [x / mag for x in v] if mag else v


VEC_NEAR = _norm(VEC_NEAR)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def cache(tmp_path: Path) -> SemanticCache:
    """A fresh SemanticCache backed by a temp SQLite file."""
    embedder = _make_embedder({
        "query A": VEC_A,
        "query B": VEC_B,
        "query C": VEC_C,
        "near A":  VEC_NEAR,
    })
    return SemanticCache(
        db_path=tmp_path / "test_cache.db",
        threshold=0.93,
        _embedder=embedder,
    )


# ---------------------------------------------------------------------------
# Cosine helper
# ---------------------------------------------------------------------------

class TestCosine:
    def test_identical_vectors(self):
        assert _cosine(VEC_A, VEC_A) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        assert _cosine(VEC_A, VEC_C) == pytest.approx(0.0, abs=1e-6)

    def test_near_identical(self):
        sim = _cosine(VEC_A, VEC_NEAR)
        assert sim > 0.93

    def test_zero_vector_returns_zero(self):
        zero = [0.0] * 8
        assert _cosine(VEC_A, zero) == 0.0


# ---------------------------------------------------------------------------
# Cache miss on empty cache
# ---------------------------------------------------------------------------

class TestLookupEmptyCache:
    def test_empty_cache_returns_none(self, cache):
        assert cache.lookup("query A") is None

    def test_empty_query_returns_none(self, cache):
        assert cache.lookup("") is None
        assert cache.lookup("   ") is None


# ---------------------------------------------------------------------------
# Store + lookup hit
# ---------------------------------------------------------------------------

_FAKE_RESULT = {
    "answer": "The answer is 42.",
    "relevant_files": ["api/service.py"],
    "input_tokens": 100,
    "output_tokens": 50,
    "vector_count": 5,
    "bm25_count": 2,
    "graph_count": 1,
    "query_entities": ["CampaignService"],
    "retrieval_ms": 123.4,
}


class TestStoreAndLookup:
    def test_store_returns_true(self, cache):
        ok = cache.store("query A", _FAKE_RESULT)
        assert ok is True

    def test_exact_match_is_a_hit(self, cache):
        cache.store("query A", _FAKE_RESULT)
        hit = cache.lookup("query A")
        assert hit is not None
        assert isinstance(hit, CacheHit)

    def test_hit_returns_correct_answer(self, cache):
        cache.store("query A", _FAKE_RESULT)
        hit = cache.lookup("query A")
        assert hit.answer == "The answer is 42."

    def test_hit_returns_correct_files(self, cache):
        cache.store("query A", _FAKE_RESULT)
        hit = cache.lookup("query A")
        assert hit.relevant_files == ["api/service.py"]

    def test_hit_increments_hit_count(self, cache):
        cache.store("query A", _FAKE_RESULT)
        h1 = cache.lookup("query A")
        h2 = cache.lookup("query A")
        assert h1.hit_count == 1
        assert h2.hit_count == 2

    def test_near_match_above_threshold_is_hit(self, cache):
        """VEC_NEAR has cosine ≈ 0.99 to VEC_A — should be a hit at threshold 0.93."""
        cache.store("query A", _FAKE_RESULT)
        hit = cache.lookup("near A")
        assert hit is not None

    def test_orthogonal_query_is_miss(self, cache):
        """VEC_C is orthogonal to VEC_A — cosine = 0.0 → miss."""
        cache.store("query A", _FAKE_RESULT)
        hit = cache.lookup("query C")
        assert hit is None

    def test_similarity_field_populated(self, cache):
        cache.store("query A", _FAKE_RESULT)
        hit = cache.lookup("query A")
        assert hit.similarity == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Invalidation
# ---------------------------------------------------------------------------

class TestInvalidate:
    def test_invalidate_removes_all_entries(self, cache):
        cache.store("query A", _FAKE_RESULT)
        cache.store("query C", _FAKE_RESULT)
        deleted = cache.invalidate()
        assert deleted == 2
        assert cache.lookup("query A") is None

    def test_invalidate_on_empty_cache_returns_zero(self, cache):
        assert cache.invalidate() == 0


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_empty(self, cache):
        s = cache.stats()
        assert s.total_entries == 0
        assert s.total_hits == 0

    def test_stats_after_store_and_hit(self, cache):
        cache.store("query A", _FAKE_RESULT)
        cache.lookup("query A")
        s = cache.stats()
        assert s.total_entries == 1
        assert s.total_hits == 1


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------

class TestEviction:
    def test_evicts_oldest_when_over_max_size(self, tmp_path):
        embedder = _make_embedder({})

        def seq_embedder(q: str) -> list[float]:
            """Return a distinct orthogonal vector per unique query."""
            idx = hash(q) % 8
            v = [0.0] * 8
            v[idx] = 1.0
            return v

        small_cache = SemanticCache(
            db_path=tmp_path / "small.db",
            threshold=0.93,
            max_size=3,
            _embedder=seq_embedder,
        )
        result = {"answer": "x", "relevant_files": [], "input_tokens": 0,
                  "output_tokens": 0, "vector_count": 0, "bm25_count": 0,
                  "graph_count": 0, "query_entities": [], "retrieval_ms": 0.0}
        for i in range(5):
            small_cache.store(f"q{i}", result)
            time.sleep(0.01)  # ensure distinct created_at ordering

        stats = small_cache.stats()
        assert stats.total_entries <= 3
