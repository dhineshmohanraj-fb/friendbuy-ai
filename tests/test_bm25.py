"""
Tests for retriever/bm25_index.py — CP3.

These tests only require rank-bm25 to be installed; no Ollama or ChromaDB.
They mock the ChromaDB layer to build an index from in-memory data.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from retriever.bm25_index import _tokenize, bm25_search, corpus_size, invalidate


# ---------------------------------------------------------------------------
# Availability guard — defined BEFORE skipif decorators reference it
# ---------------------------------------------------------------------------

def _rank_bm25_available() -> bool:
    try:
        import rank_bm25  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Fixtures — patch load_vector_store so tests are hermetic
# ---------------------------------------------------------------------------

SAMPLE_CHUNKS = [
    {
        "content": "class CampaignService:\n    def create(self, data):\n        pass",
        "meta": {"repo_name": "retailer", "file_path": "api/campaign_service.py"},
    },
    {
        "content": "def create_campaign(data: dict):\n    return CampaignService().create(data)",
        "meta": {"repo_name": "retailer", "file_path": "api/routes/campaigns.py"},
    },
    {
        "content": "class RewardService:\n    def issue(self, reward_id):\n        pass",
        "meta": {"repo_name": "retailer", "file_path": "api/reward_service.py"},
    },
    {
        "content": "STRIPE_SECRET_KEY = os.getenv('STRIPE_SECRET_KEY')",
        "meta": {"repo_name": "retailer", "file_path": "config/stripe.py"},
    },
    {
        "content": "router.post('/campaigns', create_campaign)",
        "meta": {"repo_name": "influencer", "file_path": "routes/index.js"},
    },
]


class _FakeCollection:
    def count(self):
        return len(SAMPLE_CHUNKS)

    def get(self, include=None, limit=None):
        return {
            "documents": [c["content"] for c in SAMPLE_CHUNKS],
            "metadatas": [c["meta"]    for c in SAMPLE_CHUNKS],
            "ids":       [str(i)       for i in range(len(SAMPLE_CHUNKS))],
        }


class _FakeVectorStore:
    _collection = _FakeCollection()


@pytest.fixture(autouse=True)
def patch_vector_store(monkeypatch):
    """
    Replace load_vector_store at its source (indexer.embedder) so that
    the local import inside bm25_index._build_index() picks up the fake.
    """
    # Must invalidate BEFORE the test so the fixture's fake is used on first build
    invalidate()
    monkeypatch.setattr(
        "indexer.embedder.load_vector_store",
        lambda: _FakeVectorStore(),
    )
    yield
    invalidate()   # clean up after each test


# ===========================================================================
# Tokeniser
# ===========================================================================

class TestTokenise:
    def test_camel_case_split(self):
        tokens = _tokenize("createCampaign")
        assert "create" in tokens
        assert "campaign" in tokens or "Campaign".lower() in tokens

    def test_pascal_case_split(self):
        tokens = _tokenize("CampaignService")
        assert "campaign" in tokens
        assert "service" in tokens

    def test_snake_case_split(self):
        tokens = _tokenize("create_campaign")
        assert "create" in tokens
        assert "campaign" in tokens

    def test_single_char_filtered(self):
        tokens = _tokenize("a b c def")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "def" in tokens

    def test_numbers_included(self):
        tokens = _tokenize("stripe_key_123")
        assert "123" in tokens

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_code_snippet(self):
        tokens = _tokenize("def create_campaign(data: dict) -> None:")
        assert "create" in tokens
        assert "campaign" in tokens
        assert "data" in tokens


# ===========================================================================
# BM25 search
# ===========================================================================

@pytest.mark.skipif(
    not _rank_bm25_available(),
    reason="rank-bm25 not installed",
)
class TestBM25Search:
    def test_returns_results_for_known_term(self):
        results = bm25_search("CampaignService", top_k=3)
        assert len(results) > 0

    def test_top_result_most_relevant(self):
        results = bm25_search("CampaignService create", top_k=5)
        assert results, "Expected at least one result"
        # The campaign-related files should appear somewhere in results
        all_paths = [r.file_path for r in results]
        assert any("campaign" in p.lower() for p in all_paths)

    def test_scores_normalised_to_one(self):
        results = bm25_search("CampaignService", top_k=5)
        if results:
            assert results[0].score == pytest.approx(1.0)

    def test_scores_descending(self):
        results = bm25_search("campaign", top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_repo_filter(self):
        results = bm25_search("campaign", top_k=10, repo_name="influencer")
        repos = {r.repo_name for r in results}
        assert "retailer" not in repos
        assert "influencer" in repos or len(results) == 0

    def test_no_results_for_unknown_term(self):
        # BM25 tokenises into sub-words; a truly unknown term scores 0
        results = bm25_search("xyzzy_nonexistent_zzz_abc", top_k=5)
        # All scores should be 0 after normalisation — list is empty
        # (BM25Okapi assigns 0 to tokens not in the corpus)
        assert all(r.score == 0.0 for r in results) or len(results) == 0

    def test_empty_query_returns_empty(self):
        results = bm25_search("", top_k=5)
        assert results == []

    def test_corpus_size_positive(self):
        bm25_search("anything", top_k=1)   # triggers build
        assert corpus_size() == len(SAMPLE_CHUNKS)

    def test_invalidate_resets_index(self):
        import retriever.bm25_index as bm25_mod
        bm25_search("test", top_k=1)   # build
        assert bm25_mod._state is not None   # index is built
        invalidate()
        assert bm25_mod._state is None       # state dropped — next call rebuilds


