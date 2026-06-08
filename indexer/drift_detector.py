"""
Embedding model drift detection — CP5.

The pipeline depends on all chunk embeddings being produced by the *same*
model.  If the model changes (e.g. you switch from ``nomic-embed-text`` to
``mxbai-embed-large``), the stored vectors are stale and similarity search
returns garbage.

How it works
------------
1. After a successful full index, :meth:`DriftDetector.record_fingerprint`
   embeds a short fixed probe string and writes the result to SQLite.

2. On every subsequent index run, :meth:`DriftDetector.check_drift` re-embeds
   the probe and compares it to the stored vector via cosine similarity.

   - Model name changed             → ``DriftReport(has_drift=True, reason="model_changed")``
   - Cosine similarity < threshold  → ``DriftReport(has_drift=True, reason="vector_changed")``
   - Otherwise                      → ``DriftReport(has_drift=False)``

3. When drift is detected the ``IndexPipeline`` prints a warning and suggests
   ``--reindex``.  The pipeline does **not** abort — the user decides.

The default cosine threshold is 0.999 (configurable via
``DRIFT_SIMILARITY_THRESHOLD``).

The probe string is fixed at module level and must never change; changing it
would make every existing fingerprint appear drifted.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


# ---------------------------------------------------------------------------
# Fixed probe — must NOT change between releases
# ---------------------------------------------------------------------------

_PROBE = "friendbuy-ai embedding drift detection probe v1"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DriftReport:
    """Result of a drift check."""

    has_drift:     bool
    reason:        str   = ""    # "model_changed" | "vector_changed" | ""
    stored_model:  str   = ""
    current_model: str   = ""
    similarity:    float = 1.0


# ---------------------------------------------------------------------------
# Cosine helper (same as semantic_cache — stdlib only)
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    dot    = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Default embedder
# ---------------------------------------------------------------------------

def _default_embedder(text: str) -> list[float] | None:
    try:
        from langchain_ollama import OllamaEmbeddings
        from config import get_settings
        s = get_settings()
        emb = OllamaEmbeddings(model=s.embedding_model, base_url=s.ollama_base_url)
        return emb.embed_query(text)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# DriftDetector
# ---------------------------------------------------------------------------

_DDL = """\
CREATE TABLE IF NOT EXISTS embedding_fingerprint (
    id            INTEGER PRIMARY KEY CHECK (id = 1),
    model_name    TEXT    NOT NULL,
    dimension     INTEGER NOT NULL,
    probe_text    TEXT    NOT NULL,
    probe_embedding TEXT  NOT NULL,   -- JSON array of floats
    recorded_at   TEXT    NOT NULL
);
"""


class DriftDetector:
    """
    Detects embedding model changes by comparing a fixed probe embedding.

    Args:
        db_path:    Path to the SQLite fingerprint store.  Defaults to
                    ``<CACHE_DIR>/drift_fingerprint.db``.
        threshold:  Minimum cosine similarity to consider vectors identical.
                    Defaults to ``settings.drift_similarity_threshold`` (0.999).
        _embedder:  Injectable callable ``(text) -> list[float] | None``.
                    Used by tests to avoid Ollama.
    """

    def __init__(
        self,
        db_path: Path | None = None,
        threshold: float | None = None,
        _embedder: Callable[[str], list[float] | None] | None = None,
    ) -> None:
        from config import get_settings
        s = get_settings()

        if db_path is None:
            db_path = Path(s.cache_dir) / "drift_fingerprint.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._db_path  = db_path
        self._threshold = threshold if threshold is not None else s.drift_similarity_threshold
        self._embedder  = _embedder or _default_embedder

        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute(_DDL)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def has_fingerprint(self) -> bool:
        """Return True if a fingerprint has been recorded."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM embedding_fingerprint"
        ).fetchone()
        return (row[0] or 0) > 0

    def record_fingerprint(self, model_name: str) -> bool:
        """
        Embed the probe string and persist the fingerprint.

        Call this after a successful full ``--reindex`` run.

        Returns True on success, False if embedding failed.
        """
        embedding = self._embedder(_PROBE)
        if embedding is None:
            return False

        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._conn.execute(
            "INSERT OR REPLACE INTO embedding_fingerprint "
            "(id, model_name, dimension, probe_text, probe_embedding, recorded_at) "
            "VALUES (1, ?, ?, ?, ?, ?)",
            (model_name, len(embedding), _PROBE, json.dumps(embedding), now),
        )
        self._conn.commit()
        return True

    def check_drift(self, current_model: str) -> DriftReport:
        """
        Compare *current_model* and its probe embedding against the stored fingerprint.

        Returns :class:`DriftReport` — ``has_drift=False`` means everything is fine.
        If no fingerprint is stored yet, returns ``has_drift=False``
        (first run; nothing to compare against).
        """
        row = self._conn.execute(
            "SELECT model_name, probe_embedding FROM embedding_fingerprint WHERE id = 1"
        ).fetchone()

        if row is None:
            return DriftReport(has_drift=False, reason="no_fingerprint")

        stored_model = row[0]
        stored_emb   = json.loads(row[1])

        # 1. Model name changed?
        if stored_model != current_model:
            return DriftReport(
                has_drift=True,
                reason="model_changed",
                stored_model=stored_model,
                current_model=current_model,
                similarity=0.0,
            )

        # 2. Re-embed the probe and compare vectors
        current_emb = self._embedder(_PROBE)
        if current_emb is None:
            # Embedder unreachable — skip drift check gracefully
            return DriftReport(has_drift=False, reason="embedder_unavailable")

        sim = _cosine(stored_emb, current_emb)

        if sim < self._threshold:
            return DriftReport(
                has_drift=True,
                reason="vector_changed",
                stored_model=stored_model,
                current_model=current_model,
                similarity=sim,
            )

        return DriftReport(
            has_drift=False,
            stored_model=stored_model,
            current_model=current_model,
            similarity=sim,
        )

    def clear(self) -> None:
        """Delete the stored fingerprint (used after --reindex)."""
        self._conn.execute("DELETE FROM embedding_fingerprint")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "DriftDetector":
        return self

    def __exit__(self, *_) -> None:
        self.close()
