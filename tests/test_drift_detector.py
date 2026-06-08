"""
Tests for indexer/drift_detector.py — CP5.

All tests use tmp_path and a deterministic fake embedder so Ollama is never needed.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from indexer.drift_detector import DriftDetector, DriftReport, _cosine


# ---------------------------------------------------------------------------
# Fake embedder helpers
# ---------------------------------------------------------------------------

def _unit(dim: int, direction: int) -> list[float]:
    """Return a unit vector pointing in *direction* dimension."""
    v = [0.0] * dim
    v[direction % dim] = 1.0
    return v


def _fixed_embedder(vec: list[float]):
    """Return an embedder that always returns *vec*."""
    def _emb(text: str) -> list[float]:
        return vec
    return _emb


# Fixed 8-dim test vectors
VEC_A = _unit(8, 0)   # [1, 0, 0, ...]
VEC_B = _unit(8, 1)   # [0, 1, 0, ...]  — orthogonal to VEC_A
VEC_NEAR = [0.9999, 0.0141, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

def _norm(v):
    mag = math.sqrt(sum(x*x for x in v))
    return [x/mag for x in v] if mag else v

VEC_NEAR = _norm(VEC_NEAR)


# ===========================================================================
# Cosine helper
# ===========================================================================

class TestCosine:
    def test_identical(self):
        assert _cosine(VEC_A, VEC_A) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal(self):
        assert _cosine(VEC_A, VEC_B) == pytest.approx(0.0, abs=1e-6)

    def test_near_identical(self):
        assert _cosine(VEC_A, VEC_NEAR) > 0.999


# ===========================================================================
# DriftDetector — basic lifecycle
# ===========================================================================

class TestNoFingerprint:
    def test_no_fingerprint_no_drift(self, tmp_path):
        dd = DriftDetector(db_path=tmp_path / "fp.db", _embedder=_fixed_embedder(VEC_A))
        report = dd.check_drift("nomic-embed-text")
        assert report.has_drift is False
        assert report.reason == "no_fingerprint"

    def test_has_fingerprint_false_initially(self, tmp_path):
        dd = DriftDetector(db_path=tmp_path / "fp.db", _embedder=_fixed_embedder(VEC_A))
        assert dd.has_fingerprint() is False


class TestRecordFingerprint:
    def test_record_returns_true(self, tmp_path):
        dd = DriftDetector(db_path=tmp_path / "fp.db", _embedder=_fixed_embedder(VEC_A))
        ok = dd.record_fingerprint("nomic-embed-text")
        assert ok is True

    def test_has_fingerprint_after_record(self, tmp_path):
        dd = DriftDetector(db_path=tmp_path / "fp.db", _embedder=_fixed_embedder(VEC_A))
        dd.record_fingerprint("nomic-embed-text")
        assert dd.has_fingerprint() is True

    def test_record_embedder_failure_returns_false(self, tmp_path):
        def _bad_emb(text): return None
        dd = DriftDetector(db_path=tmp_path / "fp.db", _embedder=_bad_emb)
        ok = dd.record_fingerprint("model")
        assert ok is False


# ===========================================================================
# DriftDetector — drift detection
# ===========================================================================

class TestDriftCheck:
    def test_same_model_same_embedding_no_drift(self, tmp_path):
        dd = DriftDetector(
            db_path=tmp_path / "fp.db",
            threshold=0.999,
            _embedder=_fixed_embedder(VEC_A),
        )
        dd.record_fingerprint("nomic-embed-text")
        report = dd.check_drift("nomic-embed-text")
        assert report.has_drift is False

    def test_different_model_name_drift(self, tmp_path):
        dd = DriftDetector(db_path=tmp_path / "fp.db", _embedder=_fixed_embedder(VEC_A))
        dd.record_fingerprint("nomic-embed-text")
        report = dd.check_drift("mxbai-embed-large")
        assert report.has_drift is True
        assert report.reason == "model_changed"
        assert report.stored_model  == "nomic-embed-text"
        assert report.current_model == "mxbai-embed-large"

    def test_orthogonal_embedding_drift(self, tmp_path):
        """Record VEC_A, then check with embedder returning VEC_B → cosine=0 < threshold."""
        # Record with VEC_A
        dd_record = DriftDetector(
            db_path=tmp_path / "fp.db",
            threshold=0.999,
            _embedder=_fixed_embedder(VEC_A),
        )
        dd_record.record_fingerprint("model-x")

        # Check with VEC_B (orthogonal)
        dd_check = DriftDetector(
            db_path=tmp_path / "fp.db",
            threshold=0.999,
            _embedder=_fixed_embedder(VEC_B),
        )
        report = dd_check.check_drift("model-x")
        assert report.has_drift is True
        assert report.reason == "vector_changed"
        assert report.similarity == pytest.approx(0.0, abs=1e-6)

    def test_near_identical_embedding_no_drift(self, tmp_path):
        """VEC_NEAR has cosine ≈ 0.9999 to VEC_A — above 0.999 threshold."""
        dd_record = DriftDetector(
            db_path=tmp_path / "fp.db",
            threshold=0.999,
            _embedder=_fixed_embedder(VEC_A),
        )
        dd_record.record_fingerprint("model-x")

        dd_check = DriftDetector(
            db_path=tmp_path / "fp.db",
            threshold=0.999,
            _embedder=_fixed_embedder(VEC_NEAR),
        )
        report = dd_check.check_drift("model-x")
        assert report.has_drift is False

    def test_embedder_unavailable_no_drift(self, tmp_path):
        """If re-embedding fails, skip drift check gracefully."""
        dd_record = DriftDetector(
            db_path=tmp_path / "fp.db",
            _embedder=_fixed_embedder(VEC_A),
        )
        dd_record.record_fingerprint("model-x")

        def _fail(text): return None
        dd_check = DriftDetector(db_path=tmp_path / "fp.db", _embedder=_fail)
        report = dd_check.check_drift("model-x")
        assert report.has_drift is False
        assert report.reason == "embedder_unavailable"


# ===========================================================================
# Clear
# ===========================================================================

class TestClear:
    def test_clear_removes_fingerprint(self, tmp_path):
        dd = DriftDetector(db_path=tmp_path / "fp.db", _embedder=_fixed_embedder(VEC_A))
        dd.record_fingerprint("model-x")
        assert dd.has_fingerprint() is True
        dd.clear()
        assert dd.has_fingerprint() is False

    def test_clear_then_no_drift(self, tmp_path):
        dd = DriftDetector(db_path=tmp_path / "fp.db", _embedder=_fixed_embedder(VEC_A))
        dd.record_fingerprint("model-x")
        dd.clear()
        report = dd.check_drift("model-x")
        assert report.reason == "no_fingerprint"
