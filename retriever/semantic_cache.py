"""
Semantic query cache — CP4.

On each ``ask`` call the pipeline:
  1. Embeds the incoming query with ``nomic-embed-text`` (Ollama).
  2. Computes cosine similarity against all previously-cached query embeddings
     (stored in SQLite as JSON blobs — cheap for < 10 k entries).
  3. If similarity ≥ threshold (default 0.93)  →  return the cached answer
     without running any LLM.
  4. Otherwise run the full pipeline, then store the result for future hits.

The cache lives at  ``<CACHE_DIR>/semantic_cache.db``.

Design notes
------------
- Pure SQLite (no extra ChromaDB collection) keeps the dependency surface small.
- Embeddings are 768-dim float32 lists (~3 KB each as JSON).  For a warm cache
  of 1 000 entries the linear scan over 1 000 × 768 dot-products takes ~2 ms on
  M1 — negligible compared with the LLM call it replaces.
- The embedder is *injectable* (``_embedder`` kwarg) so unit tests can pass a
  deterministic mock without Ollama.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CacheHit:
    """Returned when the semantic cache has a sufficiently similar entry."""

    cached_query:    str
    similarity:      float          # cosine similarity to the stored query
    answer:          str
    relevant_files:  list[str]
    input_tokens:    int
    output_tokens:   int
    vector_count:    int
    bm25_count:      int
    graph_count:     int
    query_entities:  list[str]
    retrieval_ms:    float
    hit_count:       int


@dataclass
class CacheStats:
    total_entries:  int
    total_hits:     int
    db_path:        str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity between two equal-length float lists."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _default_embedder(query: str) -> list[float] | None:
    """Embed *query* using the project's configured Ollama model."""
    try:
        from langchain_ollama import OllamaEmbeddings
        from config import get_settings
        s = get_settings()
        emb = OllamaEmbeddings(
            model=s.embedding_model,
            base_url=s.ollama_base_url,
        )
        return emb.embed_query(query)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# SemanticCache
# ---------------------------------------------------------------------------

_DDL = """\
CREATE TABLE IF NOT EXISTS query_cache (
    id              TEXT PRIMARY KEY,
    query_text      TEXT NOT NULL,
    embedding_json  TEXT NOT NULL,
    answer          TEXT NOT NULL,
    relevant_files  TEXT NOT NULL DEFAULT '[]',
    input_tokens    INTEGER NOT NULL DEFAULT 0,
    output_tokens   INTEGER NOT NULL DEFAULT 0,
    vector_count    INTEGER NOT NULL DEFAULT 0,
    bm25_count      INTEGER NOT NULL DEFAULT 0,
    graph_count     INTEGER NOT NULL DEFAULT 0,
    query_entities  TEXT NOT NULL DEFAULT '[]',
    retrieval_ms    REAL NOT NULL DEFAULT 0.0,
    hit_count       INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    last_hit_at     TEXT
);
"""


class SemanticCache:
    """
    SQLite-backed semantic query cache with cosine similarity lookup.

    Args:
        db_path:    Path to the SQLite file.  Defaults to
                    ``<settings.cache_dir>/semantic_cache.db``.
        threshold:  Minimum cosine similarity to count as a cache hit.
                    Defaults to ``settings.semantic_cache_threshold`` (0.93).
        max_size:   Maximum number of entries before LRU eviction.
                    Defaults to ``settings.semantic_cache_max_size`` (1 000).
        _embedder:  Callable ``(query: str) -> list[float] | None``.
                    Injected by tests to avoid Ollama dependency.
    """

    def __init__(
        self,
        db_path: Path | None = None,
        threshold: float | None = None,
        max_size: int | None = None,
        _embedder: Callable[[str], list[float] | None] | None = None,
    ) -> None:
        from config import get_settings
        s = get_settings()

        if db_path is None:
            db_path = Path(s.cache_dir) / "semantic_cache.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._db_path  = db_path
        self._threshold = threshold if threshold is not None else s.semantic_cache_threshold
        self._max_size  = max_size  if max_size  is not None else s.semantic_cache_max_size
        self._embedder  = _embedder or _default_embedder

        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute(_DDL)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(self, query: str) -> CacheHit | None:
        """
        Embed *query* and search for a near-duplicate in the cache.

        Returns :class:`CacheHit` on hit or ``None`` on miss / embedding failure.
        """
        if not query.strip():
            return None

        q_emb = self._embedder(query)
        if q_emb is None:
            return None   # Ollama unreachable — degrade gracefully

        rows = self._conn.execute(
            "SELECT id, query_text, embedding_json, answer, relevant_files, "
            "input_tokens, output_tokens, vector_count, bm25_count, graph_count, "
            "query_entities, retrieval_ms, hit_count "
            "FROM query_cache"
        ).fetchall()

        best_sim  = -1.0
        best_row: Any = None

        for row in rows:
            stored_emb = json.loads(row[2])
            sim = _cosine(q_emb, stored_emb)
            if sim > best_sim:
                best_sim = sim
                best_row = row

        if best_row is None or best_sim < self._threshold:
            return None

        # Update hit stats
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._conn.execute(
            "UPDATE query_cache SET hit_count = hit_count + 1, last_hit_at = ? WHERE id = ?",
            (now, best_row[0]),
        )
        self._conn.commit()

        return CacheHit(
            cached_query   = best_row[1],
            similarity     = best_sim,
            answer         = best_row[3],
            relevant_files = json.loads(best_row[4]),
            input_tokens   = best_row[5],
            output_tokens  = best_row[6],
            vector_count   = best_row[7],
            bm25_count     = best_row[8],
            graph_count    = best_row[9],
            query_entities = json.loads(best_row[10]),
            retrieval_ms   = best_row[11],
            hit_count      = best_row[12] + 1,
        )

    def store(self, query: str, result: Any) -> bool:
        """
        Embed *query* and persist *result* in the cache.

        *result* is expected to be a ``PipelineResult``-like object with the
        matching attributes, or a plain dict.

        Returns True on success, False if embedding failed.
        """
        if not query.strip():
            return False

        q_emb = self._embedder(query)
        if q_emb is None:
            return False

        # Accept both dataclass (PipelineResult) and dict
        def _get(attr: str, default: Any = None) -> Any:
            if isinstance(result, dict):
                return result.get(attr, default)
            return getattr(result, attr, default)

        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        entry_id = str(uuid.uuid4())

        self._conn.execute(
            "INSERT OR REPLACE INTO query_cache "
            "(id, query_text, embedding_json, answer, relevant_files, "
            "input_tokens, output_tokens, vector_count, bm25_count, graph_count, "
            "query_entities, retrieval_ms, hit_count, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)",
            (
                entry_id,
                query,
                json.dumps(q_emb),
                _get("answer", ""),
                json.dumps(_get("relevant_files", [])),
                _get("input_tokens", 0),
                _get("output_tokens", 0),
                _get("vector_count", 0),
                _get("bm25_count", 0),
                _get("graph_count", 0),
                json.dumps(_get("query_entities", [])),
                _get("retrieval_ms", 0.0),
                now,
            ),
        )
        self._conn.commit()

        # Evict oldest entries if over max_size
        self._evict_if_needed()
        return True

    def invalidate(self) -> int:
        """Delete all cached entries. Returns the number of rows deleted."""
        cur = self._conn.execute("DELETE FROM query_cache")
        self._conn.commit()
        return cur.rowcount

    def stats(self) -> CacheStats:
        """Return aggregate statistics about the cache."""
        row = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(hit_count), 0) FROM query_cache"
        ).fetchone()
        return CacheStats(
            total_entries=row[0] or 0,
            total_hits=row[1] or 0,
            db_path=str(self._db_path),
        )

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    def __enter__(self) -> "SemanticCache":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evict_if_needed(self) -> None:
        """Delete the oldest entries when the cache exceeds max_size."""
        count = self._conn.execute("SELECT COUNT(*) FROM query_cache").fetchone()[0]
        if count > self._max_size:
            excess = count - self._max_size
            self._conn.execute(
                "DELETE FROM query_cache WHERE id IN ("
                "  SELECT id FROM query_cache ORDER BY created_at ASC LIMIT ?"
                ")",
                (excess,),
            )
            self._conn.commit()
