"""
SQLite-backed document registry for incremental indexing.

Tracks which source files have been indexed, their content hashes, and
the ChromaDB chunk IDs they produced.  ``filter_changed()`` lets the
indexer skip files whose content has not changed since the last run,
making repeat ``python cli.py index`` calls cheap.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

from langchain_core.documents import Document

from config import get_settings


# ---------------------------------------------------------------------------
# Stable ID helpers (used by both this module and embedder.py)
# ---------------------------------------------------------------------------

def file_doc_id(repo_name: str, file_path: str) -> str:
    """Stable identifier for a *source file* (independent of chunk count)."""
    return hashlib.sha256(f"{repo_name}::{file_path}".encode()).hexdigest()


def chunk_doc_id(repo_name: str, file_path: str, chunk_index: int) -> str:
    """Stable ChromaDB document ID for a single chunk within a file."""
    raw = f"{repo_name}::{file_path}::{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# DeltaTracker
# ---------------------------------------------------------------------------

class DeltaTracker:
    """
    Manages a SQLite registry of indexed documents.

    The registry lives at ``{cache_dir}/indexing_registry.db``.
    Each row represents one *source file* with its SHA-256 content hash
    and the list of ChromaDB IDs that were generated from it.

    Usage::

        tracker = DeltaTracker()
        changed = tracker.filter_changed(all_documents)
        # ... embed only `changed` ...
        for doc in changed:
            tracker.register(
                doc.metadata["_doc_id"],
                doc.metadata["file_path"],
                doc.metadata["repo_name"],
                doc.metadata["_content_hash"],
                chunk_ids,
            )
    """

    def __init__(self, db_path: Path | None = None) -> None:
        settings = get_settings()
        self._db_path: Path = db_path or (settings.cache_path / "indexing_registry.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS doc_registry (
                    doc_id          TEXT PRIMARY KEY,
                    file_path       TEXT NOT NULL,
                    repo_name       TEXT NOT NULL,
                    content_hash    TEXT NOT NULL,
                    chunk_ids       TEXT NOT NULL,
                    graph_node_ids  TEXT NOT NULL DEFAULT '[]',
                    last_indexed_at TEXT NOT NULL
                )
                """
            )
            # CP2 migration: add graph_node_ids to existing CP1 databases
            try:
                conn.execute(
                    "ALTER TABLE doc_registry ADD COLUMN graph_node_ids TEXT NOT NULL DEFAULT '[]'"
                )
            except Exception:
                pass  # Column already exists — expected on fresh installs

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_file ON doc_registry(file_path)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_repo ON doc_registry(repo_name)"
            )

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    @staticmethod
    def compute_hash(content: str) -> str:
        """Return the SHA-256 hex digest of *content* (UTF-8 encoded)."""
        return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()

    def is_changed(self, doc_id: str, content_hash: str) -> bool:
        """Return True if *doc_id* is new or its stored hash differs from *content_hash*."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT content_hash FROM doc_registry WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
        if row is None:
            return True
        return str(row["content_hash"]) != content_hash

    def register(
        self,
        doc_id: str,
        file_path: str,
        repo_name: str,
        content_hash: str,
        chunk_ids: list[str],
        graph_node_ids: list[str] | None = None,  # CP2
    ) -> None:
        """Insert or update the registry entry for a document."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO doc_registry
                    (doc_id, file_path, repo_name, content_hash,
                     chunk_ids, graph_node_ids, last_indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    content_hash    = excluded.content_hash,
                    chunk_ids       = excluded.chunk_ids,
                    graph_node_ids  = excluded.graph_node_ids,
                    last_indexed_at = excluded.last_indexed_at
                """,
                (
                    doc_id,
                    file_path,
                    repo_name,
                    content_hash,
                    json.dumps(chunk_ids),
                    json.dumps(graph_node_ids or []),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def deregister(self, doc_id: str) -> None:
        """Remove a document from the registry (e.g. when a source file is deleted)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM doc_registry WHERE doc_id = ?", (doc_id,))

    def all_doc_ids(self) -> set[str]:
        """Return all doc_ids currently in the registry."""
        with self._connect() as conn:
            rows = conn.execute("SELECT doc_id FROM doc_registry").fetchall()
        return {row["doc_id"] for row in rows}

    def get_chunk_ids(self, doc_id: str) -> list[str]:
        """Return the ChromaDB chunk IDs stored for *doc_id*."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT chunk_ids FROM doc_registry WHERE doc_id = ?", (doc_id,)
            ).fetchone()
        if row is None:
            return []
        return json.loads(row["chunk_ids"])

    def get_graph_node_ids(self, doc_id: str) -> list[str]:
        """Return the Kuzu graph node IDs stored for *doc_id* (CP2)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT graph_node_ids FROM doc_registry WHERE doc_id = ?", (doc_id,)
            ).fetchone()
        if row is None:
            return []
        return json.loads(row["graph_node_ids"] or "[]")

    def update_graph_node_ids(self, doc_id: str, node_ids: list[str]) -> None:
        """Update graph_node_ids for an already-registered document (CP2)."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE doc_registry SET graph_node_ids = ? WHERE doc_id = ?",
                (json.dumps(node_ids), doc_id),
            )

    def clear_all(self) -> None:
        """Delete every entry (used during --reindex)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM doc_registry")

    def get_stats(self) -> dict:
        """Return a summary of registry state."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM doc_registry").fetchone()[0]
            repo_rows = conn.execute(
                "SELECT repo_name, COUNT(*) AS cnt FROM doc_registry GROUP BY repo_name"
            ).fetchall()
            last = conn.execute(
                "SELECT MAX(last_indexed_at) FROM doc_registry"
            ).fetchone()[0]
        return {
            "total_files": total,
            "repos": {r["repo_name"]: r["cnt"] for r in repo_rows},
            "last_indexed_at": last,
        }

    # ------------------------------------------------------------------
    # Delta computation
    # ------------------------------------------------------------------

    def filter_changed(self, documents: list[Document]) -> list[Document]:
        """
        Return the subset of *documents* that are new or have changed.

        Attaches two private metadata keys to each returned document so
        the embedder can use them without re-computing:

        - ``_doc_id``       — stable file-level ID (SHA-256 of repo+path)
        - ``_content_hash`` — SHA-256 of ``page_content``
        """
        changed: list[Document] = []
        for doc in documents:
            repo = doc.metadata.get("repo_name", "")
            fpath = doc.metadata.get("file_path", "")
            doc_id = file_doc_id(repo, fpath)
            content_hash = self.compute_hash(doc.page_content)
            if self.is_changed(doc_id, content_hash):
                doc.metadata["_doc_id"] = doc_id
                doc.metadata["_content_hash"] = content_hash
                changed.append(doc)
        return changed
