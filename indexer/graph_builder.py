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
# Module → file path resolution helper (used by GraphBuilder._resolve_import)
# ---------------------------------------------------------------------------

def _module_to_candidates(module: str, current_path: str) -> list[str]:
    """
    Generate possible relative file paths that *module* might resolve to.

    Examples::

        "os"                    → ["os.py", "os/__init__.py"]
        ".models"               → ["<current_dir>/models.py", ...]
        "api.services.campaign" → ["api/services/campaign.py", ...]
        "./services/campaign"   → ["services/campaign.js", ...]
    """
    from pathlib import PurePosixPath
    candidates: list[str] = []
    current_dir = str(PurePosixPath(current_path).parent)

    # --- JS/TS relative (starts with ./ or ../) ---
    if module.startswith("./") or module.startswith("../"):
        base = str(PurePosixPath(current_dir) / module)
        for ext in ("", ".js", ".ts", ".jsx", ".tsx"):
            candidates.append(base + ext)
        candidates.append(base + "/index.js")
        candidates.append(base + "/index.ts")

    # --- Python relative imports (starts with one or more dots) ---
    elif module.startswith("."):
        dots     = len(module) - len(module.lstrip("."))
        mod_part = module.lstrip(".")
        base     = current_dir
        for _ in range(dots - 1):
            base = str(PurePosixPath(base).parent)
        if mod_part:
            mod_path = mod_part.replace(".", "/")
            candidates.append(f"{base}/{mod_path}.py")
            candidates.append(f"{base}/{mod_path}/__init__.py")
        else:
            candidates.append(f"{base}/__init__.py")

    # --- Python absolute dotted paths ---
    elif "." in module:
        mod_path = module.replace(".", "/")
        candidates.append(f"{mod_path}.py")
        candidates.append(f"{mod_path}/__init__.py")

    # --- Single-name modules ---
    else:
        candidates.append(f"{module}.py")
        candidates.append(f"{module}/__init__.py")

    return candidates


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
    # CP2 — Class nodes
    # ------------------------------------------------------------------

    def upsert_class(self, data: dict) -> None:
        """Insert or update a Class node."""
        cid = data["class_id"]
        exists = self._conn.execute(
            "MATCH (c:Class) WHERE c.class_id = $id RETURN c LIMIT 1",
            parameters={"id": cid},
        ).has_next()

        if exists:
            self._conn.execute(
                "MATCH (c:Class) WHERE c.class_id = $id "
                "SET c.start_line = $sl, c.end_line = $el, "
                "    c.docstring = $doc, c.content_hash = COALESCE(c.content_hash, '')",
                parameters={
                    "id":  cid,
                    "sl":  data["start_line"],
                    "el":  data["end_line"],
                    "doc": data.get("docstring", ""),
                },
            )
        else:
            self._conn.execute(
                """CREATE (:Class {
                    class_id: $id, name: $name, qualified_name: $qn,
                    file_path: $fp, repo_name: $rn,
                    start_line: $sl, end_line: $el,
                    docstring: $doc, language: $lang
                })""",
                parameters={
                    "id":   cid,
                    "name": data["name"],
                    "qn":   data.get("qualified_name", data["name"]),
                    "fp":   data["file_path"],
                    "rn":   data["repo_name"],
                    "sl":   data["start_line"],
                    "el":   data["end_line"],
                    "doc":  data.get("docstring", ""),
                    "lang": data.get("language", ""),
                },
            )

    # ------------------------------------------------------------------
    # CP2 — Function nodes
    # ------------------------------------------------------------------

    def upsert_function(self, data: dict) -> None:
        """Insert or update a Function node."""
        fid = data["function_id"]
        exists = self._conn.execute(
            "MATCH (f:Function) WHERE f.function_id = $id RETURN f LIMIT 1",
            parameters={"id": fid},
        ).has_next()

        if exists:
            self._conn.execute(
                "MATCH (f:Function) WHERE f.function_id = $id "
                "SET f.start_line = $sl, f.end_line = $el, "
                "    f.is_async = $ia, f.docstring = $doc",
                parameters={
                    "id":  fid,
                    "sl":  data["start_line"],
                    "el":  data["end_line"],
                    "ia":  data.get("is_async", False),
                    "doc": data.get("docstring", ""),
                },
            )
        else:
            self._conn.execute(
                """CREATE (:Function {
                    function_id: $id, name: $name, qualified_name: $qn,
                    file_path: $fp, repo_name: $rn,
                    start_line: $sl, end_line: $el,
                    is_async: $ia, is_method: $im,
                    docstring: $doc, language: $lang
                })""",
                parameters={
                    "id":   fid,
                    "name": data["name"],
                    "qn":   data.get("qualified_name", data["name"]),
                    "fp":   data["file_path"],
                    "rn":   data["repo_name"],
                    "sl":   data["start_line"],
                    "el":   data["end_line"],
                    "ia":   data.get("is_async", False),
                    "im":   data.get("is_method", False),
                    "doc":  data.get("docstring", ""),
                    "lang": data.get("language", ""),
                },
            )

    # ------------------------------------------------------------------
    # CP2 — APIEndpoint nodes
    # ------------------------------------------------------------------

    def upsert_api_endpoint(self, data: dict) -> None:
        """Insert or update an APIEndpoint node."""
        eid = data["endpoint_id"]
        exists = self._conn.execute(
            "MATCH (e:APIEndpoint) WHERE e.endpoint_id = $id RETURN e LIMIT 1",
            parameters={"id": eid},
        ).has_next()

        if not exists:
            self._conn.execute(
                """CREATE (:APIEndpoint {
                    endpoint_id: $id, http_method: $hm, path_pattern: $pp,
                    full_path: $fp, framework: $fw,
                    file_path: $fpath, repo_name: $rn
                })""",
                parameters={
                    "id":    eid,
                    "hm":    data["http_method"],
                    "pp":    data["path_pattern"],
                    "fp":    data.get("full_path", data["path_pattern"]),
                    "fw":    data.get("framework", ""),
                    "fpath": data["file_path"],
                    "rn":    data["repo_name"],
                },
            )

    # ------------------------------------------------------------------
    # CP2 — Symbol batch upsert (nodes + all structural edges)
    # ------------------------------------------------------------------

    def upsert_symbols_from_batch(
        self,
        file_id: str,
        node_batch,   # NodeBatch from ast_parser
        edge_batch,   # EdgeBatch from ast_parser
        repo_name: str = "",
    ) -> dict[str, int]:
        """
        Upsert Class / Function / APIEndpoint nodes and all structural edges
        from a single file's NodeBatch + EdgeBatch.

        Returns a dict of counts::

            {"classes": N, "functions": N, "endpoints": N, "edges": N}
        """
        counts = {"classes": 0, "functions": 0, "endpoints": 0, "edges": 0}

        # 1. Upsert nodes first so edges can reference them
        for cls in node_batch.classes:
            self.upsert_class(cls)
            counts["classes"] += 1

        for fn in node_batch.functions:
            self.upsert_function(fn)
            counts["functions"] += 1

        for ep in node_batch.endpoints:
            self.upsert_api_endpoint(ep)
            counts["endpoints"] += 1

        # 2. CONTAINS_CLASS edges
        for fid, cid in edge_batch.contains_class:
            if not self._edge_exists("CONTAINS_CLASS", "File", "file_id", fid,
                                     "Class", "class_id", cid):
                try:
                    self._conn.execute(
                        "MATCH (f:File), (c:Class) "
                        "WHERE f.file_id = $fid AND c.class_id = $cid "
                        "CREATE (f)-[:CONTAINS_CLASS]->(c)",
                        parameters={"fid": fid, "cid": cid},
                    )
                    counts["edges"] += 1
                except Exception:
                    pass

        # 3. CONTAINS_FUNCTION edges
        for fid, fnid in edge_batch.contains_function:
            if not self._edge_exists("CONTAINS_FUNCTION", "File", "file_id", fid,
                                     "Function", "function_id", fnid):
                try:
                    self._conn.execute(
                        "MATCH (f:File), (fn:Function) "
                        "WHERE f.file_id = $fid AND fn.function_id = $fnid "
                        "CREATE (f)-[:CONTAINS_FUNCTION]->(fn)",
                        parameters={"fid": fid, "fnid": fnid},
                    )
                    counts["edges"] += 1
                except Exception:
                    pass

        # 4. METHOD_OF edges
        for fnid, cid in edge_batch.method_of:
            if not self._edge_exists("METHOD_OF", "Function", "function_id", fnid,
                                     "Class", "class_id", cid):
                try:
                    self._conn.execute(
                        "MATCH (fn:Function), (c:Class) "
                        "WHERE fn.function_id = $fnid AND c.class_id = $cid "
                        "CREATE (fn)-[:METHOD_OF]->(c)",
                        parameters={"fnid": fnid, "cid": cid},
                    )
                    counts["edges"] += 1
                except Exception:
                    pass

        # 5. EXPOSES edges  (File → APIEndpoint)
        for fid, eid in edge_batch.exposes:
            if not self._edge_exists("EXPOSES", "File", "file_id", fid,
                                     "APIEndpoint", "endpoint_id", eid):
                try:
                    self._conn.execute(
                        "MATCH (f:File), (e:APIEndpoint) "
                        "WHERE f.file_id = $fid AND e.endpoint_id = $eid "
                        "CREATE (f)-[:EXPOSES]->(e)",
                        parameters={"fid": fid, "eid": eid},
                    )
                    counts["edges"] += 1
                except Exception:
                    pass

        # 6. HANDLES edges  (APIEndpoint → Function)
        for eid, fnid in edge_batch.handles:
            if not self._edge_exists("HANDLES", "APIEndpoint", "endpoint_id", eid,
                                     "Function", "function_id", fnid):
                try:
                    self._conn.execute(
                        "MATCH (e:APIEndpoint), (fn:Function) "
                        "WHERE e.endpoint_id = $eid AND fn.function_id = $fnid "
                        "CREATE (e)-[:HANDLES]->(fn)",
                        parameters={"eid": eid, "fnid": fnid},
                    )
                    counts["edges"] += 1
                except Exception:
                    pass

        # 7. INHERITS edges — resolve parent class name → class_id
        batch_class_map = {c["name"]: c["class_id"] for c in node_batch.classes}
        for child_cid, parent_name in edge_batch.inherits:
            parent_cid = batch_class_map.get(parent_name)
            if not parent_cid and repo_name:
                try:
                    result = self._conn.execute(
                        "MATCH (c:Class) WHERE c.name = $nm AND c.repo_name = $rn "
                        "RETURN c.class_id LIMIT 1",
                        parameters={"nm": parent_name, "rn": repo_name},
                    )
                    if result.has_next():
                        parent_cid = result.get_next()[0]
                except Exception:
                    pass

            if parent_cid and not self._edge_exists(
                "INHERITS", "Class", "class_id", child_cid,
                "Class", "class_id", parent_cid,
            ):
                try:
                    self._conn.execute(
                        "MATCH (c1:Class), (c2:Class) "
                        "WHERE c1.class_id = $cid1 AND c2.class_id = $cid2 "
                        "CREATE (c1)-[:INHERITS]->(c2)",
                        parameters={"cid1": child_cid, "cid2": parent_cid},
                    )
                    counts["edges"] += 1
                except Exception:
                    pass

        # 8. IMPORT_DEP edges — best-effort file-to-file resolution
        for imp in edge_batch.imports:
            try:
                self._upsert_import_dep(imp, file_id)
                counts["edges"] += 1
            except Exception:
                pass

        return counts

    # ------------------------------------------------------------------
    # CP2 — Edge existence check
    # ------------------------------------------------------------------

    def _edge_exists(
        self,
        rel_type: str,
        from_label: str, from_key: str, from_val: str,
        to_label: str,   to_key: str,   to_val: str,
    ) -> bool:
        """Return True if this directed relationship already exists."""
        try:
            result = self._conn.execute(
                f"MATCH (a:{from_label})-[r:{rel_type}]->(b:{to_label}) "
                f"WHERE a.{from_key} = $fv AND b.{to_key} = $tv "
                f"RETURN count(r) LIMIT 1",
                parameters={"fv": from_val, "tv": to_val},
            )
            return bool(result.has_next() and result.get_next()[0] > 0)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # CP2 — Import dependency resolution
    # ------------------------------------------------------------------

    def _upsert_import_dep(self, imp: dict, from_file_id: str) -> None:
        """
        Try to resolve an import to a File node and create an IMPORT_DEP edge.
        Silently skips if the target file cannot be found.
        """
        module = imp.get("module", "").strip()
        if not module:
            return

        target_fid = self._resolve_import_to_file(module, from_file_id)
        if not target_fid or target_fid == from_file_id:
            return

        if self._edge_exists(
            "IMPORT_DEP", "File", "file_id", from_file_id,
            "File", "file_id", target_fid,
        ):
            return

        import json as _json
        symbols_json = _json.dumps(imp.get("symbols", []))
        source_line  = imp.get("source_line", 0)

        self._conn.execute(
            "MATCH (a:File), (b:File) "
            "WHERE a.file_id = $fid AND b.file_id = $tid "
            "CREATE (a)-[:IMPORT_DEP {imported_symbols: $syms, source_line: $line}]->(b)",
            parameters={
                "fid":  from_file_id,
                "tid":  target_fid,
                "syms": symbols_json,
                "line": source_line,
            },
        )

    def _resolve_import_to_file(
        self, module: str, from_file_id: str
    ) -> str | None:
        """
        Best-effort: map a module/import string to a File node's file_id.

        Strategy:
        1. Get the current file's path and repo from Kuzu.
        2. Build candidate relative paths from the module string.
        3. Query Kuzu for a File node whose path ends with each candidate.
        """
        try:
            result = self._conn.execute(
                "MATCH (f:File) WHERE f.file_id = $id "
                "RETURN f.file_path, f.repo_name LIMIT 1",
                parameters={"id": from_file_id},
            )
            if not result.has_next():
                return None
            row          = result.get_next()
            current_path = row[0]
            repo_name    = row[1]
        except Exception:
            return None

        candidates = _module_to_candidates(module, current_path)

        for candidate in candidates:
            try:
                r = self._conn.execute(
                    "MATCH (f:File) WHERE f.repo_name = $rn "
                    "AND f.file_path ENDS WITH $fp "
                    "RETURN f.file_id LIMIT 1",
                    parameters={"rn": repo_name, "fp": candidate},
                )
                if r.has_next():
                    return r.get_next()[0]
            except Exception:
                continue
        return None

    # ------------------------------------------------------------------
    # CP2 — Delete stale symbols for a changed file
    # ------------------------------------------------------------------

    def delete_file_symbols(self, file_path: str, repo_name: str) -> None:
        """
        Delete all Class, Function, and APIEndpoint nodes that belong to
        *file_path* in *repo_name*.  Used during incremental re-indexing
        to remove stale symbols before inserting fresh ones.
        """
        for label in ("Class", "Function", "APIEndpoint"):
            try:
                self._conn.execute(
                    f"MATCH (n:{label}) "
                    "WHERE n.file_path = $fp AND n.repo_name = $rn "
                    "DETACH DELETE n",
                    parameters={"fp": file_path, "rn": repo_name},
                )
            except Exception:
                pass

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
