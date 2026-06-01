"""
Kuzu knowledge graph — schema bootstrap and node/edge upsert.

CP1: creates the FULL schema (all tables for CP2–CP4 readiness) and
     populates only Repo and File nodes + BELONGS_TO_REPO edges.

CP2 will extend upsert calls to Class, Function, APIEndpoint, and the
structural edges (IMPORT_DEP, CALLS, CONTAINS_*, METHOD_OF, etc.).
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import get_settings

# ---------------------------------------------------------------------------
# Stable repo ID helper
# ---------------------------------------------------------------------------

def repo_node_id(repo_name: str) -> str:
    """Stable SHA-256 ID for a Repo node."""
    return hashlib.sha256(repo_name.encode()).hexdigest()


# ---------------------------------------------------------------------------
# DDL — full schema (all CPs)
# ---------------------------------------------------------------------------

_NODE_TABLES: list[str] = [
    # CP1
    """CREATE NODE TABLE IF NOT EXISTS Repo(
        repo_id         STRING,
        name            STRING,
        local_path      STRING,
        last_indexed_at STRING,
        PRIMARY KEY(repo_id)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS File(
        file_id         STRING,
        repo_name       STRING,
        file_path       STRING,
        file_name       STRING,
        language        STRING,
        content_hash    STRING,
        size_bytes      INT64,
        chunk_ids       STRING,
        last_indexed_at STRING,
        PRIMARY KEY(file_id)
    )""",
    # CP2 — schema ready, nodes populated in CP2
    """CREATE NODE TABLE IF NOT EXISTS Class(
        class_id       STRING,
        name           STRING,
        qualified_name STRING,
        file_path      STRING,
        repo_name      STRING,
        start_line     INT64,
        end_line       INT64,
        docstring      STRING,
        language       STRING,
        PRIMARY KEY(class_id)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Function(
        function_id    STRING,
        name           STRING,
        qualified_name STRING,
        file_path      STRING,
        repo_name      STRING,
        start_line     INT64,
        end_line       INT64,
        is_async       BOOLEAN,
        is_method      BOOLEAN,
        docstring      STRING,
        language       STRING,
        PRIMARY KEY(function_id)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS APIEndpoint(
        endpoint_id  STRING,
        http_method  STRING,
        path_pattern STRING,
        full_path    STRING,
        framework    STRING,
        file_path    STRING,
        repo_name    STRING,
        PRIMARY KEY(endpoint_id)
    )""",
]

_REL_TABLES: list[str] = [
    # CP1
    "CREATE REL TABLE IF NOT EXISTS BELONGS_TO_REPO(FROM File TO Repo)",
    # CP2
    "CREATE REL TABLE IF NOT EXISTS CONTAINS_CLASS(FROM File TO Class)",
    "CREATE REL TABLE IF NOT EXISTS CONTAINS_FUNCTION(FROM File TO Function)",
    "CREATE REL TABLE IF NOT EXISTS METHOD_OF(FROM Function TO Class)",
    "CREATE REL TABLE IF NOT EXISTS IMPORT_DEP(FROM File TO File, imported_symbols STRING, source_line INT64)",
    "CREATE REL TABLE IF NOT EXISTS CALLS(FROM Function TO Function, confidence DOUBLE)",
    "CREATE REL TABLE IF NOT EXISTS EXPOSES(FROM File TO APIEndpoint)",
    "CREATE REL TABLE IF NOT EXISTS HANDLES(FROM APIEndpoint TO Function)",
    "CREATE REL TABLE IF NOT EXISTS INHERITS(FROM Class TO Class)",
    # CP4
    "CREATE REL TABLE IF NOT EXISTS CROSS_REPO_CALL(FROM Function TO APIEndpoint, signal_type STRING, confidence DOUBLE)",
]


# ---------------------------------------------------------------------------
# GraphBuilder
# ---------------------------------------------------------------------------

class GraphBuilder:
    """
    Manages the Kuzu knowledge graph lifecycle.

    Responsible for:
    - Schema creation (all tables, once)
    - Upserting Repo and File nodes (CP1)
    - Upserting Class / Function / APIEndpoint nodes and structural edges (CP2+)
    - Providing graph statistics

    Usage::

        with GraphBuilder() as gb:
            gb.upsert_repo("api", "/repos/api")
            gb.upsert_file(file_node_dict)
    """

    def __init__(self, db_path: Path | None = None) -> None:
        try:
            import kuzu
        except ImportError as exc:
            raise ImportError(
                "kuzu is not installed. Run: pip install kuzu>=0.6.0"
            ) from exc

        settings = get_settings()
        # Kuzu 0.10+ expects a FILE path, not a directory path.
        # We keep the directory for organisation and use <dir>/graph.db as the db file.
        graph_dir = db_path or settings.graph_db_path
        graph_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = graph_dir / "graph.db"

        self._db = kuzu.Database(str(self._db_path))
        self._conn = kuzu.Connection(self._db)
        self._init_schema()

    def __enter__(self) -> "GraphBuilder":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        """Release the Kuzu connection."""
        try:
            del self._conn
            del self._db
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        """Create all node and relationship tables if they don't exist."""
        for ddl in _NODE_TABLES:
            self._conn.execute(ddl)
        for ddl in _REL_TABLES:
            self._conn.execute(ddl)

    # ------------------------------------------------------------------
    # Repo nodes (CP1)
    # ------------------------------------------------------------------

    def upsert_repo(self, repo_name: str, local_path: str = "") -> None:
        """Insert or update a Repo node."""
        rid = repo_node_id(repo_name)
        ts  = datetime.now(timezone.utc).isoformat()

        exists = self._conn.execute(
            "MATCH (r:Repo) WHERE r.repo_id = $id RETURN r LIMIT 1",
            parameters={"id": rid},
        ).has_next()

        if exists:
            self._conn.execute(
                "MATCH (r:Repo) WHERE r.repo_id = $id "
                "SET r.local_path = $path, r.last_indexed_at = $ts",
                parameters={"id": rid, "path": local_path, "ts": ts},
            )
        else:
            self._conn.execute(
                "CREATE (:Repo {repo_id: $id, name: $name, local_path: $path, last_indexed_at: $ts})",
                parameters={"id": rid, "name": repo_name, "path": local_path, "ts": ts},
            )

    # ------------------------------------------------------------------
    # File nodes (CP1)
    # ------------------------------------------------------------------

    def upsert_file(self, file_data: dict[str, Any]) -> None:
        """
        Insert or update a File node, then ensure a BELONGS_TO_REPO edge exists.

        *file_data* should be a dict produced by ``ast_parser.parse_file_node()``.
        """
        fid  = file_data["file_id"]
        rid  = repo_node_id(file_data["repo_name"])

        exists = self._conn.execute(
            "MATCH (f:File) WHERE f.file_id = $id RETURN f LIMIT 1",
            parameters={"id": fid},
        ).has_next()

        if exists:
            self._conn.execute(
                "MATCH (f:File) WHERE f.file_id = $id "
                "SET f.content_hash = $hash, f.chunk_ids = $cids, "
                "    f.size_bytes = $sz, f.last_indexed_at = $ts",
                parameters={
                    "id":   fid,
                    "hash": file_data["content_hash"],
                    "cids": file_data["chunk_ids"],
                    "sz":   file_data["size_bytes"],
                    "ts":   file_data["last_indexed_at"],
                },
            )
        else:
            self._conn.execute(
                """CREATE (:File {
                    file_id: $fid, repo_name: $rname, file_path: $fpath,
                    file_name: $fname, language: $lang, content_hash: $hash,
                    size_bytes: $sz, chunk_ids: $cids, last_indexed_at: $ts
                })""",
                parameters={
                    "fid":   fid,
                    "rname": file_data["repo_name"],
                    "fpath": file_data["file_path"],
                    "fname": file_data["file_name"],
                    "lang":  file_data["language"],
                    "hash":  file_data["content_hash"],
                    "sz":    file_data["size_bytes"],
                    "cids":  file_data["chunk_ids"],
                    "ts":    file_data["last_indexed_at"],
                },
            )
            # Create BELONGS_TO_REPO edge only on first insert
            self._conn.execute(
                "MATCH (f:File), (r:Repo) "
                "WHERE f.file_id = $fid AND r.repo_id = $rid "
                "CREATE (f)-[:BELONGS_TO_REPO]->(r)",
                parameters={"fid": fid, "rid": rid},
            )

    # ------------------------------------------------------------------
    # Bulk helpers
    # ------------------------------------------------------------------

    def upsert_files_from_chunks(
        self,
        chunks: list,   # list[langchain_core.documents.Document]
        file_chunks_map: dict[str, list[str]],
    ) -> int:
        """
        Build and upsert File nodes from a list of Document chunks.

        *file_chunks_map* maps ``file_doc_id`` → list of ChromaDB chunk IDs.
        Returns the number of unique files upserted.
        """
        from indexer.ast_parser import parse_file_node

        seen: set[str] = set()
        count = 0

        for chunk in chunks:
            repo  = chunk.metadata.get("repo_name", "")
            fpath = chunk.metadata.get("file_path", "")
            if not fpath:
                continue

            from indexer.delta_tracker import file_doc_id
            fid = file_doc_id(repo, fpath)
            if fid in seen:
                continue
            seen.add(fid)

            cids = file_chunks_map.get(fid, [])
            file_node = parse_file_node(
                file_path=fpath,
                content=chunk.page_content,   # approximation; good enough for CP1
                repo_name=repo,
                chunk_ids=cids,
            )
            # Prefer stored content_hash from metadata (more accurate)
            if chunk.metadata.get("content_hash"):
                file_node["content_hash"] = chunk.metadata["content_hash"]
            if chunk.metadata.get("size_bytes"):
                file_node["size_bytes"] = chunk.metadata["size_bytes"]

            self.upsert_file(file_node)
            count += 1

        return count

    def clear_all(self) -> None:
        """Delete ALL nodes and edges (used during --reindex)."""
        for node in ["APIEndpoint", "Function", "Class", "File", "Repo"]:
            try:
                self._conn.execute(f"MATCH (n:{node}) DETACH DELETE n")
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def graph_stats(self) -> dict[str, int]:
        """Return a {TableName: count} dict for every node and rel table."""
        stats: dict[str, int] = {}

        for table in ["Repo", "File", "Class", "Function", "APIEndpoint"]:
            try:
                result = self._conn.execute(
                    f"MATCH (n:{table}) RETURN count(n) AS cnt"
                )
                stats[table] = result.get_next()[0] if result.has_next() else 0
            except Exception:
                stats[table] = 0

        for rel in [
            "BELONGS_TO_REPO", "CONTAINS_CLASS", "CONTAINS_FUNCTION",
            "METHOD_OF", "IMPORT_DEP", "CALLS", "EXPOSES", "HANDLES",
            "INHERITS", "CROSS_REPO_CALL",
        ]:
            try:
                result = self._conn.execute(
                    f"MATCH ()-[r:{rel}]->() RETURN count(r) AS cnt"
                )
                stats[rel] = result.get_next()[0] if result.has_next() else 0
            except Exception:
                stats[rel] = 0

        return stats
