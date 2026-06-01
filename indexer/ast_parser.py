"""
Tree-sitter structural pass over source files.

CP1: File node extraction + symbol stubs for the splitter.
CP2: Full symbol extraction — Class, Function, APIEndpoint nodes and the
     structural edges (CONTAINS_CLASS, CONTAINS_FUNCTION, METHOD_OF,
     IMPORT_DEP, EXPOSES, HANDLES, INHERITS) returned as NodeBatch / EdgeBatch.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from indexer.delta_tracker import DeltaTracker, file_doc_id

# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

_SUFFIX_TO_LANG: dict[str, str] = {
    ".py":   "python",
    ".js":   "javascript",
    ".jsx":  "javascript",
    ".ts":   "typescript",
    ".tsx":  "tsx",
    ".go":   "go",
    ".rb":   "ruby",
    ".md":   "markdown",
    ".mdx":  "markdown",
    ".yml":  "yaml",
    ".yaml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".html": "html",
    ".css":  "css",
    ".scss": "scss",
    ".sh":   "shell",
    ".sql":  "sql",
    ".txt":  "text",
    ".rst":  "text",
}


def detect_language(suffix: str) -> str:
    """Return a human-readable language name for a file suffix."""
    return _SUFFIX_TO_LANG.get(suffix.lower(), "text")


# ---------------------------------------------------------------------------
# CP2 — Stable node ID helpers
# ---------------------------------------------------------------------------

def class_node_id(repo_name: str, file_path: str, class_name: str) -> str:
    """Stable SHA-256 ID for a Class node."""
    return hashlib.sha256(
        f"{repo_name}::{file_path}::class::{class_name}".encode()
    ).hexdigest()


def function_node_id(repo_name: str, file_path: str, qualified_name: str) -> str:
    """Stable SHA-256 ID for a Function node."""
    return hashlib.sha256(
        f"{repo_name}::{file_path}::function::{qualified_name}".encode()
    ).hexdigest()


def endpoint_node_id(
    repo_name: str, file_path: str, http_method: str, path_pattern: str
) -> str:
    """Stable SHA-256 ID for an APIEndpoint node."""
    return hashlib.sha256(
        f"{repo_name}::{file_path}::endpoint::{http_method}::{path_pattern}".encode()
    ).hexdigest()


# ---------------------------------------------------------------------------
# CP2 — NodeBatch and EdgeBatch dataclasses
# ---------------------------------------------------------------------------

@dataclass
class NodeBatch:
    """All graph nodes extracted from a single source file."""

    classes:   list[dict[str, Any]] = field(default_factory=list)
    functions: list[dict[str, Any]] = field(default_factory=list)
    endpoints: list[dict[str, Any]] = field(default_factory=list)

    def total(self) -> int:
        return len(self.classes) + len(self.functions) + len(self.endpoints)

    def all_node_ids(self) -> list[str]:
        """Return every node ID in this batch (used to track stale nodes)."""
        ids: list[str] = []
        ids.extend(c["class_id"]    for c in self.classes)
        ids.extend(f["function_id"] for f in self.functions)
        ids.extend(e["endpoint_id"] for e in self.endpoints)
        return ids


@dataclass
class EdgeBatch:
    """All graph edges extracted from a single source file."""

    # (file_id, class_id)
    contains_class:    list[tuple[str, str]] = field(default_factory=list)
    # (file_id, function_id)
    contains_function: list[tuple[str, str]] = field(default_factory=list)
    # (function_id, class_id)
    method_of:         list[tuple[str, str]] = field(default_factory=list)
    # (child_class_id, parent_class_name) — resolved at upsert time
    inherits:          list[tuple[str, str]] = field(default_factory=list)
    # (file_id, endpoint_id)
    exposes:           list[tuple[str, str]] = field(default_factory=list)
    # (endpoint_id, function_id)
    handles:           list[tuple[str, str]] = field(default_factory=list)
    # Raw import info — resolved to IMPORT_DEP edges in graph_builder
    imports: list[dict[str, Any]] = field(default_factory=list)

    def total(self) -> int:
        return (
            len(self.contains_class) + len(self.contains_function)
            + len(self.method_of) + len(self.inherits)
            + len(self.exposes) + len(self.handles)
            + len(self.imports)
        )


# ---------------------------------------------------------------------------
# CP1 — File node extraction (unchanged)
# ---------------------------------------------------------------------------

def parse_file_node(
    file_path: str,
    content: str,
    repo_name: str,
    chunk_ids: list[str] | None = None,
) -> dict[str, Any]:
    """
    Build a File node dict for the knowledge graph.

    Keys map 1-to-1 to the ``File`` Kuzu node table columns.
    """
    path = Path(file_path)
    return {
        "file_id":         file_doc_id(repo_name, file_path),
        "repo_name":       repo_name,
        "file_path":       file_path,
        "file_name":       path.name,
        "language":        detect_language(path.suffix),
        "content_hash":    DeltaTracker.compute_hash(content),
        "size_bytes":      len(content.encode("utf-8", errors="replace")),
        "chunk_ids":       json.dumps(chunk_ids or []),
        "last_indexed_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# CP2 — Main entry point: full symbol extraction
# ---------------------------------------------------------------------------

def extract_file_symbols(
    file_path: str,
    content: str,
    repo_name: str,
) -> tuple[NodeBatch, EdgeBatch]:
    """
    Extract all graph nodes (Class, Function, APIEndpoint) and edges from
    a single source file.

    Returns a (NodeBatch, EdgeBatch) tuple.  Both are always returned even
    if tree-sitter is not installed (they will be empty).
    """
    suffix = Path(file_path).suffix.lower()
    fid = file_doc_id(repo_name, file_path)

    try:
        if suffix == ".py":
            return _extract_python_full(content, file_path, repo_name, fid)
        elif suffix in (".js", ".jsx"):
            return _extract_js_full(content, file_path, repo_name, fid, tsx=False)
        elif suffix == ".tsx":
            return _extract_ts_full(content, file_path, repo_name, fid, tsx=True)
        elif suffix == ".ts":
            return _extract_ts_full(content, file_path, repo_name, fid, tsx=False)
    except Exception:  # noqa: BLE001
        pass

    return NodeBatch(), EdgeBatch()


# ===========================================================================
# CP1 — Symbol stubs used by splitter.py (preserved, do not remove)
# ===========================================================================

_PY_SYMBOL_TYPES: frozenset[str] = frozenset({
    "function_definition",
    "class_definition",
    "decorated_definition",
})

_JS_SYMBOL_TYPES: frozenset[str] = frozenset({
    "function_declaration",
    "class_declaration",
    "generator_function_declaration",
})


def extract_python_symbols(content: str) -> list[dict[str, Any]]:
    """Extract top-level function and class definitions (CP1, used by splitter)."""
    try:
        import tree_sitter_python as tspython
        from tree_sitter import Language, Parser
    except ImportError:
        return []
    try:
        source = content.encode("utf-8", errors="replace")
        parser = Parser(Language(tspython.language()))
        tree = parser.parse(source)
        return _collect_python_symbols(tree.root_node, source)
    except Exception:  # noqa: BLE001
        return []


def extract_js_symbols(content: str, tsx: bool = False) -> list[dict[str, Any]]:
    """Extract top-level JS/TS symbols (CP1, used by splitter)."""
    try:
        if tsx:
            import tree_sitter_typescript as tsts
            from tree_sitter import Language, Parser
            lang = Language(tsts.language_tsx())
        else:
            import tree_sitter_javascript as tsjs
            from tree_sitter import Language, Parser
            lang = Language(tsjs.language())
        parser = Parser(lang)
    except ImportError:
        return []
    try:
        source = content.encode("utf-8", errors="replace")
        tree = parser.parse(source)
        return _collect_js_symbols(tree.root_node, source)
    except Exception:  # noqa: BLE001
        return []


def extract_ts_symbols(content: str, tsx: bool = False) -> list[dict[str, Any]]:
    """Extract top-level TypeScript symbols (CP1, used by splitter)."""
    try:
        import tree_sitter_typescript as tsts
        from tree_sitter import Language, Parser
        lang = Language(tsts.language_tsx() if tsx else tsts.language_typescript())
        parser = Parser(lang)
    except ImportError:
        return []
    try:
        source = content.encode("utf-8", errors="replace")
        tree = parser.parse(source)
        return _collect_js_symbols(tree.root_node, source)
    except Exception:  # noqa: BLE001
        return []


# ---------------------------------------------------------------------------
# CP1 internals — Python (splitter compat)
# ---------------------------------------------------------------------------

def _collect_python_symbols(root, source: bytes) -> list[dict[str, Any]]:
    symbols = []
    for node in root.children:
        sym = None
        if node.type in ("function_definition", "class_definition"):
            sym = _extract_py_symbol(node, source)
        elif node.type == "decorated_definition":
            inner = next(
                (c for c in node.children
                 if c.type in ("function_definition", "class_definition")),
                None,
            )
            if inner:
                sym = _extract_py_symbol(node, source, name_from=inner)
        if sym:
            symbols.append(sym)
    return symbols


def _extract_py_symbol(node, source: bytes, name_from=None) -> dict[str, Any]:
    target = name_from or node
    name_node = target.child_by_field_name("name")
    name = name_node.text.decode("utf-8") if name_node else "<anonymous>"
    text = source[node.start_byte: node.end_byte].decode("utf-8", errors="replace")
    symbol_type = "class" if "class" in target.type else "function"
    return {
        "name":       name,
        "type":       symbol_type,
        "text":       text,
        "start_line": node.start_point[0] + 1,
        "end_line":   node.end_point[0] + 1,
        "docstring":  _py_docstring(target, source),
    }


def _py_docstring(node, source: bytes) -> str:
    """Extract the first string literal from a function/class body (docstring)."""
    body = node.child_by_field_name("body")
    if not body:
        return ""
    for child in body.children:
        if child.type == "expression_statement" and child.children:
            expr = child.children[0]
            if expr.type == "string":
                raw = expr.text.decode("utf-8", errors="replace")
                return raw.strip('"""').strip("'''").strip('"').strip("'").strip()[:400]
    return ""


# ---------------------------------------------------------------------------
# CP1 internals — JS/TS (splitter compat)
# ---------------------------------------------------------------------------

def _collect_js_symbols(root, source: bytes) -> list[dict[str, Any]]:
    symbols = []
    for node in root.children:
        if node.type in _JS_SYMBOL_TYPES:
            symbols.append(_extract_js_symbol(node, source))
        elif node.type == "export_statement":
            for child in node.children:
                if child.type in _JS_SYMBOL_TYPES:
                    symbols.append(_extract_js_symbol(node, source, name_from=child))
                    break
    return symbols


def _extract_js_symbol(node, source: bytes, name_from=None) -> dict[str, Any]:
    target = name_from or node
    name_node = target.child_by_field_name("name")
    name = name_node.text.decode("utf-8") if name_node else "<anonymous>"
    text = source[node.start_byte: node.end_byte].decode("utf-8", errors="replace")
    symbol_type = "class" if "class" in target.type else "function"
    return {
        "name":       name,
        "type":       symbol_type,
        "text":       text,
        "start_line": node.start_point[0] + 1,
        "end_line":   node.end_point[0] + 1,
        "docstring":  "",
    }


# ===========================================================================
# CP2 — Full Python extraction
# ===========================================================================

_HTTP_METHODS: frozenset[str] = frozenset(
    {"get", "post", "put", "patch", "delete", "head", "options"}
)


def _extract_python_full(
    content: str,
    file_path: str,
    repo_name: str,
    file_id: str,
) -> tuple[NodeBatch, EdgeBatch]:
    """Walk a Python AST and extract all CP2 nodes + edges."""
    try:
        import tree_sitter_python as tspython
        from tree_sitter import Language, Parser
    except ImportError:
        return NodeBatch(), EdgeBatch()

    source = content.encode("utf-8", errors="replace")
    parser = Parser(Language(tspython.language()))
    tree = parser.parse(source)

    nodes = NodeBatch()
    edges = EdgeBatch()

    for node in tree.root_node.children:
        if node.type == "class_definition":
            _py_extract_class(node, source, file_path, repo_name, file_id, nodes, edges)
        elif node.type == "function_definition":
            _py_extract_function(node, source, file_path, repo_name, file_id, nodes, edges)
        elif node.type == "decorated_definition":
            _py_extract_decorated(node, source, file_path, repo_name, file_id, nodes, edges)
        elif node.type in ("import_statement", "import_from_statement"):
            _py_extract_import(node, source, file_id, edges)

    return nodes, edges


def _py_extract_class(
    node,
    source: bytes,
    file_path: str,
    repo_name: str,
    file_id: str,
    nodes: NodeBatch,
    edges: EdgeBatch,
    outer_node=None,
) -> None:
    """Extract a class definition and its methods."""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return

    class_name = name_node.text.decode("utf-8")
    cid = class_node_id(repo_name, file_path, class_name)
    effective = outer_node or node

    # Base classes for INHERITS edges
    bases: list[str] = []
    superclasses = node.child_by_field_name("superclasses")
    if superclasses:
        for child in superclasses.children:
            if child.type in ("identifier", "attribute"):
                base = child.text.decode("utf-8", errors="replace").strip()
                if base and base not in (",", "(", ")"):
                    bases.append(base)

    nodes.classes.append({
        "class_id":       cid,
        "name":           class_name,
        "qualified_name": class_name,
        "file_path":      file_path,
        "repo_name":      repo_name,
        "start_line":     effective.start_point[0] + 1,
        "end_line":       effective.end_point[0] + 1,
        "docstring":      _py_docstring(node, source),
        "language":       "python",
    })
    edges.contains_class.append((file_id, cid))

    for base in bases:
        edges.inherits.append((cid, base))

    # Methods in class body
    body = node.child_by_field_name("body")
    if not body:
        return

    for child in body.children:
        if child.type == "function_definition":
            _py_extract_method(
                child, source, file_path, repo_name, file_id,
                class_name, cid, nodes, edges,
            )
        elif child.type == "decorated_definition":
            inner_fn = next(
                (c for c in child.children if c.type == "function_definition"), None
            )
            if inner_fn:
                _py_extract_method(
                    inner_fn, source, file_path, repo_name, file_id,
                    class_name, cid, nodes, edges, outer_node=child,
                )


def _py_extract_method(
    node,
    source: bytes,
    file_path: str,
    repo_name: str,
    file_id: str,
    class_name: str,
    class_id: str,
    nodes: NodeBatch,
    edges: EdgeBatch,
    outer_node=None,
) -> None:
    """Extract a method as a Function node + METHOD_OF / CONTAINS_FUNCTION edges."""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return

    method_name  = name_node.text.decode("utf-8")
    qualified    = f"{class_name}.{method_name}"
    fid          = function_node_id(repo_name, file_path, qualified)
    effective    = outer_node or node
    is_async     = any(c.type == "async" for c in node.children)

    nodes.functions.append({
        "function_id":    fid,
        "name":           method_name,
        "qualified_name": qualified,
        "file_path":      file_path,
        "repo_name":      repo_name,
        "start_line":     effective.start_point[0] + 1,
        "end_line":       effective.end_point[0] + 1,
        "is_async":       is_async,
        "is_method":      True,
        "docstring":      _py_docstring(node, source),
        "language":       "python",
    })
    edges.contains_function.append((file_id, fid))
    edges.method_of.append((fid, class_id))


def _py_extract_function(
    node,
    source: bytes,
    file_path: str,
    repo_name: str,
    file_id: str,
    nodes: NodeBatch,
    edges: EdgeBatch,
    outer_node=None,
) -> str | None:
    """Extract a top-level function. Returns function_id or None."""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return None

    func_name = name_node.text.decode("utf-8")
    fid       = function_node_id(repo_name, file_path, func_name)
    effective = outer_node or node
    is_async  = any(c.type == "async" for c in node.children)

    nodes.functions.append({
        "function_id":    fid,
        "name":           func_name,
        "qualified_name": func_name,
        "file_path":      file_path,
        "repo_name":      repo_name,
        "start_line":     effective.start_point[0] + 1,
        "end_line":       effective.end_point[0] + 1,
        "is_async":       is_async,
        "is_method":      False,
        "docstring":      _py_docstring(node, source),
        "language":       "python",
    })
    edges.contains_function.append((file_id, fid))
    return fid


def _py_extract_decorated(
    node,
    source: bytes,
    file_path: str,
    repo_name: str,
    file_id: str,
    nodes: NodeBatch,
    edges: EdgeBatch,
) -> None:
    """Handle decorated_definition — detects FastAPI/Flask HTTP routes."""
    inner = next(
        (c for c in node.children
         if c.type in ("function_definition", "class_definition")),
        None,
    )
    if not inner:
        return

    # Collect HTTP-route decorators
    route_infos: list[dict] = []
    for child in node.children:
        if child.type == "decorator":
            info = _py_parse_http_decorator(child, source)
            if info:
                route_infos.append(info)

    if inner.type == "class_definition":
        _py_extract_class(inner, source, file_path, repo_name, file_id,
                          nodes, edges, outer_node=node)

    elif inner.type == "function_definition":
        fn_id = _py_extract_function(inner, source, file_path, repo_name, file_id,
                                     nodes, edges, outer_node=node)

        if fn_id and route_infos:
            for route in route_infos:
                eid = endpoint_node_id(
                    repo_name, file_path,
                    route["http_method"], route["path_pattern"],
                )
                nodes.endpoints.append({
                    "endpoint_id":  eid,
                    "http_method":  route["http_method"],
                    "path_pattern": route["path_pattern"],
                    "full_path":    route["path_pattern"],
                    "framework":    route["framework"],
                    "file_path":    file_path,
                    "repo_name":    repo_name,
                })
                edges.exposes.append((file_id, eid))
                edges.handles.append((eid, fn_id))


def _py_parse_http_decorator(decorator_node, source: bytes) -> dict | None:
    """
    Return ``{http_method, path_pattern, framework}`` if this decorator
    is an HTTP route decorator (FastAPI @router.get, Flask @app.route, etc.).
    """
    call_node = next(
        (c for c in decorator_node.children if c.type == "call"), None
    )
    if not call_node:
        return None

    func_node = call_node.child_by_field_name("function")
    if not func_node or func_node.type != "attribute":
        return None

    attr_node = func_node.child_by_field_name("attribute")
    if not attr_node:
        return None

    method_name = attr_node.text.decode("utf-8", errors="replace").lower()

    # FastAPI-style: @router.get / @app.post / etc.
    if method_name in _HTTP_METHODS:
        http_method = method_name.upper()
        framework = "fastapi"

    # Flask-style: @app.route("/path", methods=["GET", "POST"])
    elif method_name == "route":
        http_method = "GET"
        framework = "flask"
        args_node = call_node.child_by_field_name("arguments")
        if args_node:
            for child in args_node.children:
                if child.type == "keyword_argument":
                    kw_name = child.child_by_field_name("name")
                    if kw_name and kw_name.text.decode() == "methods":
                        kw_val = child.child_by_field_name("value")
                        if kw_val:
                            methods_raw = kw_val.text.decode("utf-8").strip("[]")
                            methods = [
                                m.strip().strip("\"'").upper()
                                for m in methods_raw.split(",")
                                if m.strip().strip("\"'")
                            ]
                            if methods:
                                http_method = ",".join(methods)
    else:
        return None

    # Path string — first positional argument
    args_node = call_node.child_by_field_name("arguments")
    if not args_node:
        return None

    path: str | None = None
    for child in args_node.children:
        if child.type == "string":
            raw = child.text.decode("utf-8", errors="replace")
            # Strip all quote variants and f-string prefix
            cleaned = raw.lstrip("fFrRbB").strip("\"'")
            if cleaned:
                path = cleaned
                break

    if not path:
        return None

    return {"http_method": http_method, "path_pattern": path, "framework": framework}


def _py_extract_import(
    node, source: bytes, file_id: str, edges: EdgeBatch
) -> None:
    """Record an import statement in EdgeBatch.imports (resolved later)."""
    text = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace").strip()
    line = node.start_point[0] + 1

    if node.type == "import_statement":
        # "import os, sys"  /  "import os as operating_system"
        modules_part = text[7:].strip()  # strip leading "import "
        for m in modules_part.split(","):
            mod = m.split(" as ")[0].strip()
            if mod:
                edges.imports.append({
                    "from_file_id": file_id,
                    "module":       mod,
                    "symbols":      [],
                    "source_line":  line,
                    "raw":          text,
                })

    elif node.type == "import_from_statement":
        # "from os.path import join, exists"
        match = re.match(r"from\s+([\.\w]*)\s+import\s+(.*)", text, re.DOTALL)
        if match:
            module  = match.group(1).strip()
            sym_txt = match.group(2).strip().strip("()")
            symbols = [
                s.split(" as ")[0].strip().rstrip(",").strip()
                for s in re.split(r"[,\n]", sym_txt)
                if s.strip() and s.strip() != "*"
            ]
            if module:
                edges.imports.append({
                    "from_file_id": file_id,
                    "module":       module,
                    "symbols":      [s for s in symbols if s],
                    "source_line":  line,
                    "raw":          text,
                })


# ===========================================================================
# CP2 — Full JS / TS / TSX extraction
# ===========================================================================

_JS_HTTP_METHODS: frozenset[str] = frozenset(
    {"get", "post", "put", "patch", "delete", "head", "options", "all", "use"}
)


def _extract_js_full(
    content: str,
    file_path: str,
    repo_name: str,
    file_id: str,
    tsx: bool = False,
) -> tuple[NodeBatch, EdgeBatch]:
    """Walk a JS/JSX AST and extract all CP2 nodes + edges."""
    try:
        if tsx:
            import tree_sitter_typescript as tsts
            from tree_sitter import Language, Parser
            lang = Language(tsts.language_tsx())
        else:
            import tree_sitter_javascript as tsjs
            from tree_sitter import Language, Parser
            lang = Language(tsjs.language())
        parser = Parser(lang)
    except ImportError:
        return NodeBatch(), EdgeBatch()

    source = content.encode("utf-8", errors="replace")
    tree = parser.parse(source)

    nodes = NodeBatch()
    edges = EdgeBatch()
    _js_walk_toplevel(tree.root_node, source, file_path, repo_name, file_id, nodes, edges)
    return nodes, edges


def _extract_ts_full(
    content: str,
    file_path: str,
    repo_name: str,
    file_id: str,
    tsx: bool = False,
) -> tuple[NodeBatch, EdgeBatch]:
    """Walk a TS/TSX AST and extract all CP2 nodes + edges."""
    try:
        import tree_sitter_typescript as tsts
        from tree_sitter import Language, Parser
        lang = Language(tsts.language_tsx() if tsx else tsts.language_typescript())
        parser = Parser(lang)
    except ImportError:
        return NodeBatch(), EdgeBatch()

    source = content.encode("utf-8", errors="replace")
    tree = parser.parse(source)

    nodes = NodeBatch()
    edges = EdgeBatch()
    _js_walk_toplevel(tree.root_node, source, file_path, repo_name, file_id, nodes, edges)
    return nodes, edges


def _js_walk_toplevel(
    root,
    source: bytes,
    file_path: str,
    repo_name: str,
    file_id: str,
    nodes: NodeBatch,
    edges: EdgeBatch,
) -> None:
    """Dispatch top-level AST nodes to appropriate extractors."""
    for node in root.children:
        t = node.type

        if t == "class_declaration":
            _js_extract_class(node, source, file_path, repo_name, file_id, nodes, edges)

        elif t in ("function_declaration", "generator_function_declaration"):
            _js_extract_function(node, source, file_path, repo_name, file_id, nodes, edges)

        elif t == "export_statement":
            _js_handle_export(node, source, file_path, repo_name, file_id, nodes, edges)

        elif t == "lexical_declaration":
            _js_extract_const_function(node, source, file_path, repo_name, file_id, nodes, edges)

        elif t == "expression_statement":
            _js_detect_express_route(node, source, file_path, repo_name, file_id, nodes, edges)

        elif t == "import_statement":
            _js_extract_import(node, source, file_id, edges)


def _js_extract_class(
    node,
    source: bytes,
    file_path: str,
    repo_name: str,
    file_id: str,
    nodes: NodeBatch,
    edges: EdgeBatch,
    outer_node=None,
) -> None:
    """Extract a JS/TS class declaration."""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return

    class_name = name_node.text.decode("utf-8", errors="replace")
    cid        = class_node_id(repo_name, file_path, class_name)
    effective  = outer_node or node

    # Heritage clause (extends)
    bases: list[str] = []
    heritage = node.child_by_field_name("heritage")
    if heritage:
        for child in heritage.children:
            if child.type == "identifier":
                bases.append(child.text.decode("utf-8", errors="replace"))

    nodes.classes.append({
        "class_id":       cid,
        "name":           class_name,
        "qualified_name": class_name,
        "file_path":      file_path,
        "repo_name":      repo_name,
        "start_line":     effective.start_point[0] + 1,
        "end_line":       effective.end_point[0] + 1,
        "docstring":      "",
        "language":       "javascript",
    })
    edges.contains_class.append((file_id, cid))
    for base in bases:
        edges.inherits.append((cid, base))

    # Methods
    body = node.child_by_field_name("body")
    if body:
        for child in body.children:
            if child.type in ("method_definition", "public_field_definition"):
                _js_extract_method(
                    child, source, file_path, repo_name, file_id,
                    class_name, cid, nodes, edges,
                )


def _js_extract_method(
    node,
    source: bytes,
    file_path: str,
    repo_name: str,
    file_id: str,
    class_name: str,
    class_id: str,
    nodes: NodeBatch,
    edges: EdgeBatch,
) -> None:
    """Extract a class method as a Function node."""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return

    method_name = name_node.text.decode("utf-8", errors="replace")
    qualified   = f"{class_name}.{method_name}"
    fid         = function_node_id(repo_name, file_path, qualified)
    is_async    = any(c.text == b"async" for c in node.children if c.type != "comment")

    nodes.functions.append({
        "function_id":    fid,
        "name":           method_name,
        "qualified_name": qualified,
        "file_path":      file_path,
        "repo_name":      repo_name,
        "start_line":     node.start_point[0] + 1,
        "end_line":       node.end_point[0] + 1,
        "is_async":       is_async,
        "is_method":      True,
        "docstring":      "",
        "language":       "javascript",
    })
    edges.contains_function.append((file_id, fid))
    edges.method_of.append((fid, class_id))


def _js_extract_function(
    node,
    source: bytes,
    file_path: str,
    repo_name: str,
    file_id: str,
    nodes: NodeBatch,
    edges: EdgeBatch,
    outer_node=None,
    name_override: str | None = None,
    language: str = "javascript",
) -> str | None:
    """Extract a named function declaration. Returns function_id or None."""
    if name_override:
        func_name = name_override
    else:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return None
        func_name = name_node.text.decode("utf-8", errors="replace")

    fid       = function_node_id(repo_name, file_path, func_name)
    effective = outer_node or node
    is_async  = any(c.type == "async" for c in node.children)

    nodes.functions.append({
        "function_id":    fid,
        "name":           func_name,
        "qualified_name": func_name,
        "file_path":      file_path,
        "repo_name":      repo_name,
        "start_line":     effective.start_point[0] + 1,
        "end_line":       effective.end_point[0] + 1,
        "is_async":       is_async,
        "is_method":      False,
        "docstring":      "",
        "language":       language,
    })
    edges.contains_function.append((file_id, fid))
    return fid


def _js_handle_export(
    node,
    source: bytes,
    file_path: str,
    repo_name: str,
    file_id: str,
    nodes: NodeBatch,
    edges: EdgeBatch,
) -> None:
    """Handle export_statement — unwrap and dispatch inner declarations."""
    for child in node.children:
        if child.type == "class_declaration":
            _js_extract_class(child, source, file_path, repo_name, file_id,
                              nodes, edges, outer_node=node)
        elif child.type in ("function_declaration", "generator_function_declaration"):
            _js_extract_function(child, source, file_path, repo_name, file_id,
                                 nodes, edges, outer_node=node)
        elif child.type == "lexical_declaration":
            _js_extract_const_function(child, source, file_path, repo_name, file_id,
                                       nodes, edges)


def _js_extract_const_function(
    node,
    source: bytes,
    file_path: str,
    repo_name: str,
    file_id: str,
    nodes: NodeBatch,
    edges: EdgeBatch,
) -> None:
    """Extract ``const foo = () => {}`` or ``const foo = function() {}``."""
    for child in node.children:
        if child.type != "variable_declarator":
            continue
        name_node = child.child_by_field_name("name")
        val_node  = child.child_by_field_name("value")
        if not name_node or not val_node:
            continue
        if val_node.type in ("arrow_function", "function", "generator_function"):
            func_name = name_node.text.decode("utf-8", errors="replace")
            _js_extract_function(
                val_node, source, file_path, repo_name, file_id,
                nodes, edges, outer_node=node, name_override=func_name,
            )


def _js_detect_express_route(
    node,
    source: bytes,
    file_path: str,
    repo_name: str,
    file_id: str,
    nodes: NodeBatch,
    edges: EdgeBatch,
) -> None:
    """
    Detect Express.js route registrations::

        app.get('/path', handler)
        router.post('/path', async (req, res) => { ... })
    """
    call_node = next(
        (c for c in node.children if c.type == "call_expression"), None
    )
    if not call_node:
        return

    func_node = call_node.child_by_field_name("function")
    if not func_node or func_node.type != "member_expression":
        return

    prop_node = func_node.child_by_field_name("property")
    if not prop_node:
        return

    method_name = prop_node.text.decode("utf-8", errors="replace").lower()
    if method_name not in _JS_HTTP_METHODS:
        return

    args_node = call_node.child_by_field_name("arguments")
    if not args_node:
        return

    path: str | None = None
    handler_name: str | None = None
    arg_idx = 0

    for child in args_node.children:
        if child.type in (",", "(", ")"):
            continue
        if arg_idx == 0 and child.type == "string":
            raw = child.text.decode("utf-8", errors="replace")
            path = raw.strip("\"'`")
        elif arg_idx >= 1 and child.type == "identifier":
            handler_name = child.text.decode("utf-8", errors="replace")
        arg_idx += 1

    if not path:
        return

    http_method = method_name.upper()
    if http_method in ("USE", "ALL"):
        http_method = "ALL"

    eid = endpoint_node_id(repo_name, file_path, http_method, path)
    nodes.endpoints.append({
        "endpoint_id":  eid,
        "http_method":  http_method,
        "path_pattern": path,
        "full_path":    path,
        "framework":    "express",
        "file_path":    file_path,
        "repo_name":    repo_name,
    })
    edges.exposes.append((file_id, eid))

    # Link to named handler if already extracted
    if handler_name:
        for fn in nodes.functions:
            if fn["name"] == handler_name:
                edges.handles.append((eid, fn["function_id"]))
                break


def _js_extract_import(
    node, source: bytes, file_id: str, edges: EdgeBatch
) -> None:
    """Extract an ES6 import statement."""
    text = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace").strip()
    line = node.start_point[0] + 1

    from_match = re.search(r"from\s+[\"'`](.*?)[\"'`]", text)
    if not from_match:
        side_effect = re.search(r"import\s+[\"'`](.*?)[\"'`]", text)
        if side_effect:
            edges.imports.append({
                "from_file_id": file_id,
                "module":       side_effect.group(1),
                "symbols":      [],
                "source_line":  line,
                "raw":          text,
            })
        return

    module = from_match.group(1)

    named_match = re.search(r"import\s*\{([^}]+)\}", text)
    symbols: list[str] = []
    if named_match:
        symbols = [
            s.split(" as ")[0].strip()
            for s in named_match.group(1).split(",")
            if s.strip()
        ]

    edges.imports.append({
        "from_file_id": file_id,
        "module":       module,
        "symbols":      symbols,
        "source_line":  line,
        "raw":          text,
    })
