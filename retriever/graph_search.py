"""
Knowledge-graph traversal for hybrid retrieval — CP3.

Given entity names extracted from a user query, this module:

1. Looks up those names in Kuzu (Class, Function, APIEndpoint)
2. Traverses 1–2 hops outward to find structurally related nodes
3. Returns the file paths of related files (so their chunks can be
   fetched from ChromaDB) and a human-readable relationship summary
   (included verbatim in the Claude prompt).

Usage::

    from retriever.graph_search import GraphSearcher

    with GraphSearcher() as gs:
        entities = gs.extract_entities("what does CampaignService create?")
        context  = gs.traverse(entities, max_hops=2)

    # context.related_file_paths → paths to fetch extra chunks from ChromaDB
    # context.relationship_summary → injected into the Claude prompt
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EntityMatch:
    """A single graph node that matched a query entity name."""
    name:      str
    node_type: str   # "Class" | "Function" | "APIEndpoint"
    file_path: str
    repo_name: str
    extra:     dict = field(default_factory=dict)   # http_method, path, …


@dataclass
class GraphContext:
    """Everything the graph search found for a query."""
    entities_found:      list[EntityMatch]
    related_file_paths:  list[str]   # unique, for ChromaDB chunk fetch
    relationship_summary: str         # injected into the Claude prompt
    query_entities:      list[str]   # raw entity names we searched for

    @classmethod
    def empty(cls) -> "GraphContext":
        return cls([], [], "", [])

    def is_empty(self) -> bool:
        return len(self.entities_found) == 0


# ---------------------------------------------------------------------------
# Name cache (avoids repeated Kuzu full-table scans per query)
# ---------------------------------------------------------------------------

_name_cache: dict[str, list[str]] = {}   # repo_name → list[str]
_cache_ts: float = 0.0
_CACHE_TTL = 120.0   # seconds


def _get_known_names(conn) -> list[str]:
    """Return all entity names from the graph (cached for 2 minutes)."""
    global _name_cache, _cache_ts

    if time.time() - _cache_ts < _CACHE_TTL and _name_cache:
        return [n for names in _name_cache.values() for n in names]

    names: list[str] = []
    for label, col in [
        ("Class",       "name"),
        ("Function",    "name"),
        ("APIEndpoint", "path_pattern"),
    ]:
        try:
            result = conn.execute(f"MATCH (n:{label}) RETURN DISTINCT n.{col} LIMIT 2000")
            while result.has_next():
                row = result.get_next()
                val = row[0]
                if val and isinstance(val, str) and len(val) > 1:
                    names.append(val)
        except Exception:  # noqa: BLE001
            pass

    _name_cache = {"_all": names}
    _cache_ts   = time.time()
    return names


# ---------------------------------------------------------------------------
# GraphSearcher
# ---------------------------------------------------------------------------

class GraphSearcher:
    """
    Read-only Kuzu connection for graph traversal.

    Use as a context manager::

        with GraphSearcher() as gs:
            ctx = gs.traverse(["CampaignService"], max_hops=2)
    """

    def __init__(self) -> None:
        try:
            import kuzu
        except ImportError as exc:
            raise ImportError(
                "kuzu is not installed. Run: pip install kuzu>=0.6.0"
            ) from exc

        from config import get_settings
        settings = get_settings()

        graph_dir  = settings.graph_db_path
        db_file    = graph_dir / "graph.db"

        if not db_file.exists():
            raise FileNotFoundError(
                f"Graph DB not found at {db_file}. "
                "Run `python cli.py index` first."
            )

        self._db   = kuzu.Database(str(db_file))
        self._conn = kuzu.Connection(self._db)

    def __enter__(self) -> "GraphSearcher":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        try:
            del self._conn
            del self._db
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Entity extraction from natural-language query
    # ------------------------------------------------------------------

    def extract_entities(self, query: str) -> list[str]:
        """
        Find code entity names mentioned in *query*.

        Strategy: look up all known names in the graph and check if any
        appear as whole words inside the query (case-insensitive).
        Also detect common code-name patterns directly in the query:
        ``ClassName.method()``, ``/api/path``, ``HTTP_METHOD``.
        """
        found: set[str] = set()

        # 1. Pattern-based: CamelCase words  (ClassName, ServiceName)
        for match in re.finditer(r"\b([A-Z][a-zA-Z0-9]{2,})\b", query):
            found.add(match.group(1))

        # 2. Pattern-based: snake_case identifiers  (create_campaign)
        for match in re.finditer(r"\b([a-z][a-z0-9_]{3,})\b", query):
            candidate = match.group(1)
            if "_" in candidate or candidate.endswith(
                ("_service", "_controller", "_model", "_handler", "_util")
            ):
                found.add(candidate)

        # 3. Pattern-based: /api/path patterns
        for match in re.finditer(r"(/[a-zA-Z0-9_/-]+)", query):
            found.add(match.group(1))

        # 4. Graph lookup: check known names against the query
        try:
            known = _get_known_names(self._conn)
            query_lower = query.lower()
            for name in known:
                # whole-word match (avoid matching "get" inside "budget")
                pattern = r"\b" + re.escape(name.lower()) + r"\b"
                if len(name) > 3 and re.search(pattern, query_lower):
                    found.add(name)
        except Exception:  # noqa: BLE001
            pass

        return sorted(found)

    # ------------------------------------------------------------------
    # Graph traversal
    # ------------------------------------------------------------------

    def traverse(
        self,
        entity_names: list[str],
        max_hops: int = 2,
    ) -> GraphContext:
        """
        Traverse the knowledge graph starting from *entity_names*.

        Returns a :class:`GraphContext` with:
        - matched nodes
        - file paths of structurally related files
        - a human-readable relationship summary for the Claude prompt
        """
        if not entity_names:
            return GraphContext.empty()

        matches = self._find_nodes(entity_names)
        if not matches:
            return GraphContext.empty()

        related_paths = self._get_related_files(matches, max_hops)
        summary       = self._build_summary(matches)

        return GraphContext(
            entities_found=matches,
            related_file_paths=related_paths,
            relationship_summary=summary,
            query_entities=entity_names,
        )

    # ------------------------------------------------------------------
    # Node lookup
    # ------------------------------------------------------------------

    def _find_nodes(self, names: list[str]) -> list[EntityMatch]:
        """Find Class, Function, and APIEndpoint nodes matching *names*."""
        matches: list[EntityMatch] = []
        seen: set[str] = set()

        for name in names:
            # --- Classes ---
            try:
                r = self._conn.execute(
                    "MATCH (c:Class) WHERE c.name = $n "
                    "RETURN c.class_id, c.name, c.file_path, c.repo_name, "
                    "       c.start_line, c.end_line LIMIT 5",
                    parameters={"n": name},
                )
                while r.has_next():
                    row = r.get_next()
                    key = f"Class:{row[0]}"
                    if key not in seen:
                        seen.add(key)
                        matches.append(EntityMatch(
                            name=row[1], node_type="Class",
                            file_path=row[2] or "", repo_name=row[3] or "",
                            extra={"start_line": row[4], "end_line": row[5]},
                        ))
            except Exception:  # noqa: BLE001
                pass

            # --- Functions ---
            try:
                r = self._conn.execute(
                    "MATCH (f:Function) WHERE f.name = $n "
                    "RETURN f.function_id, f.name, f.qualified_name, "
                    "       f.file_path, f.repo_name, f.is_async, f.is_method LIMIT 10",
                    parameters={"n": name},
                )
                while r.has_next():
                    row = r.get_next()
                    key = f"Function:{row[0]}"
                    if key not in seen:
                        seen.add(key)
                        matches.append(EntityMatch(
                            name=row[1], node_type="Function",
                            file_path=row[3] or "", repo_name=row[4] or "",
                            extra={
                                "qualified_name": row[2],
                                "is_async": row[5],
                                "is_method": row[6],
                            },
                        ))
            except Exception:  # noqa: BLE001
                pass

            # --- API Endpoints (path_pattern match) ---
            try:
                r = self._conn.execute(
                    "MATCH (e:APIEndpoint) "
                    "WHERE e.path_pattern CONTAINS $n OR e.path_pattern = $n "
                    "RETURN e.endpoint_id, e.http_method, e.path_pattern, "
                    "       e.file_path, e.repo_name, e.framework LIMIT 5",
                    parameters={"n": name},
                )
                while r.has_next():
                    row = r.get_next()
                    key = f"APIEndpoint:{row[0]}"
                    if key not in seen:
                        seen.add(key)
                        matches.append(EntityMatch(
                            name=row[2], node_type="APIEndpoint",
                            file_path=row[3] or "", repo_name=row[4] or "",
                            extra={
                                "http_method": row[1],
                                "framework":   row[5],
                            },
                        ))
            except Exception:  # noqa: BLE001
                pass

        return matches

    # ------------------------------------------------------------------
    # Related file discovery (1–2 hop expansion)
    # ------------------------------------------------------------------

    def _get_related_files(
        self, matches: list[EntityMatch], max_hops: int
    ) -> list[str]:
        """
        Collect file paths structurally related to the matched nodes.

        Hop 1: files that contain the matched nodes
        Hop 2: files that the hop-1 files import from (IMPORT_DEP)
        """
        paths: set[str] = set()

        # Always include the files that directly contain the matched nodes
        for m in matches:
            if m.file_path:
                paths.add(m.file_path)

        if max_hops < 2:
            return sorted(paths)

        # Hop 2: follow IMPORT_DEP edges from those files
        try:
            for m in matches:
                if not m.file_path:
                    continue
                r = self._conn.execute(
                    "MATCH (f:File)-[:IMPORT_DEP]->(g:File) "
                    "WHERE f.file_path = $fp "
                    "RETURN g.file_path LIMIT 10",
                    parameters={"fp": m.file_path},
                )
                while r.has_next():
                    row = r.get_next()
                    if row[0]:
                        paths.add(row[0])
        except Exception:  # noqa: BLE001
            pass

        # Also: for matched Classes, find files that import the class
        for m in [x for x in matches if x.node_type == "Class"]:
            try:
                r = self._conn.execute(
                    "MATCH (f:File)-[:CONTAINS_CLASS]->(c:Class {name: $n}) "
                    "WITH f "
                    "MATCH (g:File)-[:IMPORT_DEP]->(f) "
                    "RETURN g.file_path LIMIT 5",
                    parameters={"n": m.name},
                )
                while r.has_next():
                    row = r.get_next()
                    if row[0]:
                        paths.add(row[0])
            except Exception:  # noqa: BLE001
                pass

        # For matched Functions, find what endpoints handle them
        for m in [x for x in matches if x.node_type == "Function"]:
            try:
                r = self._conn.execute(
                    "MATCH (e:APIEndpoint)-[:HANDLES]->(f:Function {name: $n}) "
                    "RETURN e.file_path LIMIT 5",
                    parameters={"n": m.name},
                )
                while r.has_next():
                    row = r.get_next()
                    if row[0]:
                        paths.add(row[0])
            except Exception:  # noqa: BLE001
                pass

        return sorted(paths)

    # ------------------------------------------------------------------
    # Human-readable summary builder
    # ------------------------------------------------------------------

    def _build_summary(self, matches: list[EntityMatch]) -> str:
        """
        Build a structured summary of graph findings for inclusion in
        the Claude prompt.
        """
        if not matches:
            return ""

        lines: list[str] = ["### Graph relationships found\n"]

        classes   = [m for m in matches if m.node_type == "Class"]
        functions = [m for m in matches if m.node_type == "Function"]
        endpoints = [m for m in matches if m.node_type == "APIEndpoint"]

        if classes:
            lines.append("**Classes:**")
            for c in classes[:5]:
                base = f"- `{c.name}` [{c.node_type}]"
                if c.repo_name:
                    base += f" in `{c.repo_name}`"
                if c.file_path:
                    base += f" → `{c.file_path}`"
                # Attach methods
                methods = self._get_methods_of_class(c.name)
                if methods:
                    base += f"\n  - Methods: {', '.join(f'`{m}`' for m in methods[:8])}"
                lines.append(base)
            lines.append("")

        if functions:
            lines.append("**Functions / Methods:**")
            for f in functions[:8]:
                qn = f.extra.get("qualified_name", f.name)
                async_flag = " *(async)*" if f.extra.get("is_async") else ""
                method_flag = " *(method)*" if f.extra.get("is_method") else ""
                line = f"- `{qn}`{async_flag}{method_flag}"
                if f.repo_name:
                    line += f" in `{f.repo_name}`"
                lines.append(line)
            lines.append("")

        if endpoints:
            lines.append("**API Endpoints:**")
            for e in endpoints[:8]:
                method = e.extra.get("http_method", "?")
                fw     = e.extra.get("framework", "")
                line   = f"- `{method} {e.name}`"
                if fw:
                    line += f" [{fw}]"
                if e.repo_name:
                    line += f" in `{e.repo_name}`"
                lines.append(line)

        return "\n".join(lines)

    def _get_methods_of_class(self, class_name: str) -> list[str]:
        """Return method names for a class (best-effort)."""
        try:
            r = self._conn.execute(
                "MATCH (f:Function)-[:METHOD_OF]->(c:Class {name: $n}) "
                "RETURN f.name LIMIT 12",
                parameters={"n": class_name},
            )
            names = []
            while r.has_next():
                names.append(r.get_next()[0])
            return names
        except Exception:  # noqa: BLE001
            return []
