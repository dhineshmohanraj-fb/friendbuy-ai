"""
Tests for indexer/delta_tracker.py — CP2 additions.

All tests use an in-memory / temp-dir SQLite database so they don't
touch the real ``cache/`` directory.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from langchain_core.documents import Document

from indexer.delta_tracker import DeltaTracker, chunk_doc_id, file_doc_id


# ---------------------------------------------------------------------------
# Fixture: fresh tracker in a temp directory
# ---------------------------------------------------------------------------

@pytest.fixture()
def tracker(tmp_path: Path) -> DeltaTracker:
    db = tmp_path / "test_registry.db"
    return DeltaTracker(db_path=db)


# ---------------------------------------------------------------------------
# Stable ID helpers
# ---------------------------------------------------------------------------

class TestIdHelpers:
    def test_file_doc_id_stable(self):
        assert file_doc_id("repo", "path.py") == file_doc_id("repo", "path.py")

    def test_file_doc_id_unique_per_repo(self):
        assert file_doc_id("repo-a", "f.py") != file_doc_id("repo-b", "f.py")

    def test_chunk_doc_id_stable(self):
        assert chunk_doc_id("r", "f.py", 0) == chunk_doc_id("r", "f.py", 0)

    def test_chunk_doc_id_unique_per_index(self):
        assert chunk_doc_id("r", "f.py", 0) != chunk_doc_id("r", "f.py", 1)

    def test_compute_hash_stable(self):
        h1 = DeltaTracker.compute_hash("hello")
        h2 = DeltaTracker.compute_hash("hello")
        assert h1 == h2

    def test_compute_hash_unique(self):
        assert DeltaTracker.compute_hash("hello") != DeltaTracker.compute_hash("world")


# ---------------------------------------------------------------------------
# Core register / is_changed
# ---------------------------------------------------------------------------

class TestRegisterAndIsChanged:
    def test_new_doc_is_changed(self, tracker):
        doc_id = file_doc_id("repo", "new.py")
        assert tracker.is_changed(doc_id, "any-hash") is True

    def test_registered_doc_same_hash_not_changed(self, tracker):
        doc_id = file_doc_id("repo", "file.py")
        tracker.register(doc_id, "file.py", "repo", "abc123", ["chunk-1"])
        assert tracker.is_changed(doc_id, "abc123") is False

    def test_registered_doc_different_hash_is_changed(self, tracker):
        doc_id = file_doc_id("repo", "file.py")
        tracker.register(doc_id, "file.py", "repo", "old-hash", [])
        assert tracker.is_changed(doc_id, "new-hash") is True

    def test_register_updates_hash(self, tracker):
        doc_id = file_doc_id("repo", "file.py")
        tracker.register(doc_id, "file.py", "repo", "v1", [])
        tracker.register(doc_id, "file.py", "repo", "v2", [])
        assert tracker.is_changed(doc_id, "v2") is False

    def test_get_chunk_ids(self, tracker):
        doc_id = file_doc_id("repo", "file.py")
        chunk_ids = ["c1", "c2", "c3"]
        tracker.register(doc_id, "file.py", "repo", "hash", chunk_ids)
        assert tracker.get_chunk_ids(doc_id) == chunk_ids

    def test_get_chunk_ids_missing_returns_empty(self, tracker):
        assert tracker.get_chunk_ids("nonexistent") == []


# ---------------------------------------------------------------------------
# CP2 — graph_node_ids column
# ---------------------------------------------------------------------------

class TestGraphNodeIds:
    def test_default_graph_node_ids_empty(self, tracker):
        doc_id = file_doc_id("repo", "file.py")
        tracker.register(doc_id, "file.py", "repo", "hash", [])
        assert tracker.get_graph_node_ids(doc_id) == []

    def test_register_with_graph_node_ids(self, tracker):
        doc_id    = file_doc_id("repo", "file.py")
        node_ids  = ["class-id-1", "func-id-1", "func-id-2"]
        tracker.register(doc_id, "file.py", "repo", "hash", [], graph_node_ids=node_ids)
        assert tracker.get_graph_node_ids(doc_id) == node_ids

    def test_update_graph_node_ids(self, tracker):
        doc_id = file_doc_id("repo", "file.py")
        tracker.register(doc_id, "file.py", "repo", "hash", [])
        tracker.update_graph_node_ids(doc_id, ["new-id-1", "new-id-2"])
        assert tracker.get_graph_node_ids(doc_id) == ["new-id-1", "new-id-2"]

    def test_register_updates_graph_node_ids(self, tracker):
        doc_id = file_doc_id("repo", "file.py")
        tracker.register(doc_id, "file.py", "repo", "hash", [], graph_node_ids=["old"])
        tracker.register(doc_id, "file.py", "repo", "hash2", [], graph_node_ids=["new1", "new2"])
        assert tracker.get_graph_node_ids(doc_id) == ["new1", "new2"]

    def test_graph_node_ids_missing_doc_returns_empty(self, tracker):
        assert tracker.get_graph_node_ids("no-such-id") == []


# ---------------------------------------------------------------------------
# filter_changed
# ---------------------------------------------------------------------------

class TestFilterChanged:
    def _doc(self, repo: str, path: str, content: str) -> Document:
        return Document(
            page_content=content,
            metadata={"repo_name": repo, "file_path": path},
        )

    def test_all_new_docs_returned(self, tracker):
        docs = [
            self._doc("repo", "a.py", "content-a"),
            self._doc("repo", "b.py", "content-b"),
        ]
        changed = tracker.filter_changed(docs)
        assert len(changed) == 2

    def test_unchanged_doc_filtered_out(self, tracker):
        doc = self._doc("repo", "a.py", "stable content")
        doc_id = file_doc_id("repo", "a.py")
        content_hash = DeltaTracker.compute_hash("stable content")
        tracker.register(doc_id, "a.py", "repo", content_hash, [])

        changed = tracker.filter_changed([doc])
        assert len(changed) == 0

    def test_changed_doc_returned(self, tracker):
        doc = self._doc("repo", "a.py", "new content")
        doc_id = file_doc_id("repo", "a.py")
        tracker.register(doc_id, "a.py", "repo", "old-hash", [])

        changed = tracker.filter_changed([doc])
        assert len(changed) == 1

    def test_filter_attaches_private_metadata(self, tracker):
        doc = self._doc("repo", "a.py", "some content")
        changed = tracker.filter_changed([doc])
        assert "_doc_id" in changed[0].metadata
        assert "_content_hash" in changed[0].metadata

    def test_mixed_docs(self, tracker):
        unchanged = self._doc("repo", "x.py", "same")
        changed   = self._doc("repo", "y.py", "changed")

        uid = file_doc_id("repo", "x.py")
        tracker.register(uid, "x.py", "repo", DeltaTracker.compute_hash("same"), [])

        result = tracker.filter_changed([unchanged, changed])
        paths = [d.metadata["file_path"] for d in result]
        assert "y.py" in paths
        assert "x.py" not in paths


# ---------------------------------------------------------------------------
# clear_all / deregister / all_doc_ids
# ---------------------------------------------------------------------------

class TestClearAndDeregister:
    def test_clear_all_empties_registry(self, tracker):
        did = file_doc_id("r", "f.py")
        tracker.register(did, "f.py", "r", "h", [])
        tracker.clear_all()
        assert tracker.all_doc_ids() == set()

    def test_deregister_removes_entry(self, tracker):
        did = file_doc_id("r", "f.py")
        tracker.register(did, "f.py", "r", "h", [])
        tracker.deregister(did)
        assert tracker.is_changed(did, "h") is True  # gone = "changed"

    def test_all_doc_ids_returns_all(self, tracker):
        ids = [file_doc_id("r", f) for f in ("a.py", "b.py", "c.py")]
        for i, fpath in enumerate(("a.py", "b.py", "c.py")):
            tracker.register(ids[i], fpath, "r", f"hash{i}", [])
        assert tracker.all_doc_ids() == set(ids)


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_empty(self, tracker):
        stats = tracker.get_stats()
        assert stats["total_files"] == 0
        assert stats["repos"] == {}

    def test_stats_counts_per_repo(self, tracker):
        for i, (repo, fpath) in enumerate([
            ("repo-a", "f1.py"), ("repo-a", "f2.py"), ("repo-b", "f3.py")
        ]):
            did = file_doc_id(repo, fpath)
            tracker.register(did, fpath, repo, f"h{i}", [])

        stats = tracker.get_stats()
        assert stats["total_files"] == 3
        assert stats["repos"]["repo-a"] == 2
        assert stats["repos"]["repo-b"] == 1


# ---------------------------------------------------------------------------
# Schema migration (CP2 column added to existing CP1 db)
# ---------------------------------------------------------------------------

class TestSchemaMigration:
    def test_cp1_db_migrated_transparently(self, tmp_path):
        """
        Simulate a CP1 database (no graph_node_ids column) and verify
        that opening it with the CP2 DeltaTracker adds the column without
        losing data.
        """
        import sqlite3

        db_path = tmp_path / "cp1_registry.db"

        # Create old CP1 schema (no graph_node_ids)
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE doc_registry (
                doc_id          TEXT PRIMARY KEY,
                file_path       TEXT NOT NULL,
                repo_name       TEXT NOT NULL,
                content_hash    TEXT NOT NULL,
                chunk_ids       TEXT NOT NULL,
                last_indexed_at TEXT NOT NULL
            )
        """)
        conn.execute(
            "INSERT INTO doc_registry VALUES (?,?,?,?,?,?)",
            ("doc1", "file.py", "repo", "hash1", "[]", "2024-01-01"),
        )
        conn.commit()
        conn.close()

        # Open with CP2 tracker — should not raise
        tracker = DeltaTracker(db_path=db_path)
        # Old data still intact
        assert tracker.is_changed("doc1", "hash1") is False
        # New column accessible
        assert tracker.get_graph_node_ids("doc1") == []
