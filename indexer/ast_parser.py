"""
Tree-sitter structural pass over source files.

CP1: extracts File-level node dicts for the knowledge graph.
CP2 will extend this to extract Class, Function, and APIEndpoint nodes.
"""

from __future__ import annotations

import json
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
# CP1 — File node extraction
# ---------------------------------------------------------------------------

def parse_file_node(
    file_path: str,
    content: str,
    repo_name: str,
    chunk_ids: list[str] | None = None,
) -> dict[str, Any]:
    """
    Build a File node dict suitable for inserting into the knowledge graph.

    This is the CP1 contract — called once per source file after embedding.

    Args:
        file_path:  Relative path (e.g. ``api/services/campaign.py``).
        content:    Full file text (used for hash + size computation).
        repo_name:  Repository the file belongs to.
        chunk_ids:  ChromaDB document IDs produced from this file.

    Returns:
        A dict whose keys map 1-to-1 to the ``File`` Kuzu node table columns.
    """
    path = Path(file_path)
    return {
        "file_id":        file_doc_id(repo_name, file_path),
        "repo_name":      repo_name,
        "file_path":      file_path,
        "file_name":      path.name,
        "language":       detect_language(path.suffix),
        "content_hash":   DeltaTracker.compute_hash(content),
        "size_bytes":     len(content.encode("utf-8", errors="replace")),
        "chunk_ids":      json.dumps(chunk_ids or []),
        "last_indexed_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# CP1 — Tree-sitter symbol extraction (used by splitter; stubs for CP2)
# ---------------------------------------------------------------------------

# Python node types that represent a standalone, indexable symbol
_PY_SYMBOL_TYPES: frozenset[str] = frozenset({
    "function_definition",
    "class_definition",
    "decorated_definition",
})

# JS/TS node types for standalone symbols
_JS_SYMBOL_TYPES: frozenset[str] = frozenset({
    "function_declaration",
    "class_declaration",
    "generator_function_declaration",
})


def extract_python_symbols(content: str) -> list[dict[str, Any]]:
    """
    Extract top-level function and class definitions from Python source.

    Returns a list of dicts with keys:
    ``name``, ``type``, ``text``, ``start_line``, ``end_line``, ``docstring``.
    Returns an empty list if tree-sitter is not installed or parsing fails.
    """
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
    """
    Extract top-level function and class declarations from JS/TS source.

    Args:
        content: File text.
        tsx:     If True, parse as TSX; otherwise plain JS.
    """
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
    """Extract top-level symbols from TypeScript source."""
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
# Internal helpers — Python
# ---------------------------------------------------------------------------

def _collect_python_symbols(root, source: bytes) -> list[dict[str, Any]]:
    symbols = []
    for node in root.children:
        sym = None
        if node.type in ("function_definition", "class_definition"):
            sym = _extract_py_symbol(node, source)
        elif node.type == "decorated_definition":
            # decorated_definition wraps the actual function/class
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
    """Extract the first string literal from a function/class body (the docstring)."""
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
# Internal helpers — JS/TS
# ---------------------------------------------------------------------------

def _collect_js_symbols(root, source: bytes) -> list[dict[str, Any]]:
    symbols = []
    for node in root.children:
        if node.type in _JS_SYMBOL_TYPES:
            symbols.append(_extract_js_symbol(node, source))
        elif node.type == "export_statement":
            # export function foo() {}  or  export class Foo {}
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
        "docstring":  "",    # JS JSDoc is in comments — handled in CP2
    }
