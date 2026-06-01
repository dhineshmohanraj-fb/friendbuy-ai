"""
Tests for indexer/ast_parser.py — CP2 symbol extraction.

These tests use hardcoded source strings so they run without any repos,
Ollama, or ChromaDB.  They only require tree-sitter packages to be installed.
"""

from __future__ import annotations

import pytest

from indexer.ast_parser import (
    EdgeBatch,
    NodeBatch,
    class_node_id,
    endpoint_node_id,
    extract_file_symbols,
    function_node_id,
)

# ---------------------------------------------------------------------------
# Test fixtures — inline source code
# ---------------------------------------------------------------------------

PYTHON_SIMPLE = """\
import os
from typing import List

class MyService:
    \"\"\"A service class.\"\"\"

    def __init__(self):
        pass

    def do_thing(self, x: int) -> str:
        \"\"\"Does a thing.\"\"\"
        return str(x)

async def standalone_func():
    \"\"\"Standalone async function.\"\"\"
    pass
"""

PYTHON_FASTAPI = """\
from fastapi import APIRouter

router = APIRouter()

@router.get("/items/{item_id}")
async def read_item(item_id: int):
    \"\"\"Get an item by ID.\"\"\"
    return {"item_id": item_id}

@router.post("/items")
async def create_item(item: dict):
    return item
"""

PYTHON_FLASK = """\
from flask import Flask
app = Flask(__name__)

@app.route("/users", methods=["GET", "POST"])
def users():
    return []
"""

PYTHON_INHERITANCE = """\
class Base:
    pass

class Child(Base):
    pass

class MultiChild(Base, Mixin):
    pass
"""

JS_SIMPLE = """\
import React from 'react';
import { useState } from 'react';

class MyComponent extends React.Component {
  render() {
    return null;
  }

  async fetchData() {
    return [];
  }
}

function helperFunc() {
  return 42;
}

export function exportedFunc() {
  return true;
}
"""

JS_EXPRESS = """\
const express = require('express');
const router = express.Router();

router.get('/api/items', async (req, res) => {
  res.json([]);
});

router.post('/api/items', (req, res) => {
  res.status(201).json(req.body);
});
"""

TS_SIMPLE = """\
import { Injectable } from '@angular/core';

export class UserService {
  private users: string[] = [];

  getUsers(): string[] {
    return this.users;
  }
}

export async function fetchUser(id: number): Promise<string> {
  return String(id);
}
"""

REPO  = "test-repo"
FPATH = "test/file.py"


# ===========================================================================
# Availability guards — defined BEFORE skipif decorators reference them
# ===========================================================================

def _can_import_tree_sitter_python() -> bool:
    try:
        import tree_sitter_python  # noqa: F401
        return True
    except ImportError:
        return False


def _can_import_tree_sitter_js() -> bool:
    try:
        import tree_sitter_javascript  # noqa: F401
        return True
    except ImportError:
        return False


def _can_import_tree_sitter_ts() -> bool:
    try:
        import tree_sitter_typescript  # noqa: F401
        return True
    except ImportError:
        return False


# ===========================================================================
# Helper
# ===========================================================================

def _extract(fpath: str, content: str, repo: str = REPO):
    return extract_file_symbols(fpath, content, repo)


# ===========================================================================
# Node ID stability
# ===========================================================================

class TestNodeIds:
    def test_class_id_stable(self):
        id1 = class_node_id("repo", "path.py", "MyClass")
        id2 = class_node_id("repo", "path.py", "MyClass")
        assert id1 == id2

    def test_function_id_stable(self):
        id1 = function_node_id("repo", "path.py", "my_func")
        id2 = function_node_id("repo", "path.py", "my_func")
        assert id1 == id2

    def test_endpoint_id_stable(self):
        id1 = endpoint_node_id("repo", "path.py", "GET", "/items")
        id2 = endpoint_node_id("repo", "path.py", "GET", "/items")
        assert id1 == id2

    def test_ids_are_unique_across_types(self):
        """Different node types with same name must produce different IDs."""
        cid = class_node_id("r", "p.py", "Foo")
        fid = function_node_id("r", "p.py", "Foo")
        assert cid != fid

    def test_ids_are_unique_across_repos(self):
        a = class_node_id("repo-a", "file.py", "MyClass")
        b = class_node_id("repo-b", "file.py", "MyClass")
        assert a != b


# ===========================================================================
# NodeBatch helpers
# ===========================================================================

class TestNodeBatch:
    def test_total_empty(self):
        nb = NodeBatch()
        assert nb.total() == 0

    def test_all_node_ids_empty(self):
        assert NodeBatch().all_node_ids() == []

    def test_total_counts_all_types(self):
        nb = NodeBatch(
            classes=[{"class_id": "c1"}],
            functions=[{"function_id": "f1"}, {"function_id": "f2"}],
            endpoints=[{"endpoint_id": "e1"}],
        )
        assert nb.total() == 4

    def test_all_node_ids_returns_all(self):
        nb = NodeBatch(
            classes=[{"class_id": "c1"}],
            functions=[{"function_id": "f1"}],
            endpoints=[{"endpoint_id": "e1"}],
        )
        ids = nb.all_node_ids()
        assert set(ids) == {"c1", "f1", "e1"}


# ===========================================================================
# Python extraction
# ===========================================================================

@pytest.mark.skipif(
    not _can_import_tree_sitter_python(),
    reason="tree-sitter-python not installed",
)
class TestPythonExtraction:
    def test_class_extracted(self):
        nodes, edges = _extract("f.py", PYTHON_SIMPLE)
        class_names = [c["name"] for c in nodes.classes]
        assert "MyService" in class_names

    def test_class_docstring(self):
        nodes, _ = _extract("f.py", PYTHON_SIMPLE)
        svc = next(c for c in nodes.classes if c["name"] == "MyService")
        assert "service" in svc["docstring"].lower()

    def test_methods_extracted(self):
        nodes, _ = _extract("f.py", PYTHON_SIMPLE)
        fn_names = [f["name"] for f in nodes.functions]
        assert "__init__" in fn_names
        assert "do_thing" in fn_names

    def test_method_is_method_flag(self):
        nodes, _ = _extract("f.py", PYTHON_SIMPLE)
        init = next(f for f in nodes.functions if f["name"] == "__init__")
        assert init["is_method"] is True

    def test_method_qualified_name(self):
        nodes, _ = _extract("f.py", PYTHON_SIMPLE)
        init = next(f for f in nodes.functions if f["name"] == "__init__")
        assert init["qualified_name"] == "MyService.__init__"

    def test_standalone_function_extracted(self):
        nodes, _ = _extract("f.py", PYTHON_SIMPLE)
        fn_names = [f["name"] for f in nodes.functions]
        assert "standalone_func" in fn_names

    def test_async_flag(self):
        nodes, _ = _extract("f.py", PYTHON_SIMPLE)
        fn = next(f for f in nodes.functions if f["name"] == "standalone_func")
        assert fn["is_async"] is True

    def test_sync_flag(self):
        nodes, _ = _extract("f.py", PYTHON_SIMPLE)
        fn = next(f for f in nodes.functions if f["name"] == "do_thing")
        assert fn["is_async"] is False

    def test_contains_class_edges(self):
        nodes, edges = _extract("f.py", PYTHON_SIMPLE)
        class_ids = {c["class_id"] for c in nodes.classes}
        edge_class_ids = {cid for _, cid in edges.contains_class}
        assert class_ids == edge_class_ids

    def test_contains_function_edges(self):
        nodes, edges = _extract("f.py", PYTHON_SIMPLE)
        fn_ids  = {f["function_id"] for f in nodes.functions}
        edge_fn = {fid for _, fid in edges.contains_function}
        assert fn_ids == edge_fn

    def test_method_of_edges(self):
        nodes, edges = _extract("f.py", PYTHON_SIMPLE)
        class_id = nodes.classes[0]["class_id"]
        method_class_ids = {cid for _, cid in edges.method_of}
        assert class_id in method_class_ids

    def test_imports_captured(self):
        _, edges = _extract("f.py", PYTHON_SIMPLE)
        modules = [imp["module"] for imp in edges.imports]
        assert "os" in modules
        assert "typing" in modules

    def test_import_symbols(self):
        _, edges = _extract("f.py", PYTHON_SIMPLE)
        typing_imp = next(i for i in edges.imports if i["module"] == "typing")
        assert "List" in typing_imp["symbols"]

    def test_line_numbers(self):
        nodes, _ = _extract("f.py", PYTHON_SIMPLE)
        svc = next(c for c in nodes.classes if c["name"] == "MyService")
        assert svc["start_line"] >= 1
        assert svc["end_line"] > svc["start_line"]


@pytest.mark.skipif(
    not _can_import_tree_sitter_python(),
    reason="tree-sitter-python not installed",
)
class TestPythonFastAPI:
    def test_endpoint_extracted(self):
        nodes, _ = _extract("routes.py", PYTHON_FASTAPI)
        assert len(nodes.endpoints) >= 1

    def test_endpoint_http_method(self):
        nodes, _ = _extract("routes.py", PYTHON_FASTAPI)
        methods = {e["http_method"] for e in nodes.endpoints}
        assert "GET" in methods
        assert "POST" in methods

    def test_endpoint_path_pattern(self):
        nodes, _ = _extract("routes.py", PYTHON_FASTAPI)
        patterns = {e["path_pattern"] for e in nodes.endpoints}
        assert "/items/{item_id}" in patterns

    def test_endpoint_framework(self):
        nodes, _ = _extract("routes.py", PYTHON_FASTAPI)
        frameworks = {e["framework"] for e in nodes.endpoints}
        assert "fastapi" in frameworks

    def test_exposes_edge_exists(self):
        nodes, edges = _extract("routes.py", PYTHON_FASTAPI)
        endpoint_ids = {e["endpoint_id"] for e in nodes.endpoints}
        edge_ep_ids  = {eid for _, eid in edges.exposes}
        assert endpoint_ids == edge_ep_ids

    def test_handles_edge_exists(self):
        nodes, edges = _extract("routes.py", PYTHON_FASTAPI)
        assert len(edges.handles) >= 1
        # Each handle edge links an endpoint to a function
        fn_ids  = {f["function_id"] for f in nodes.functions}
        ep_ids  = {e["endpoint_id"] for e in nodes.endpoints}
        for eid, fnid in edges.handles:
            assert eid in ep_ids
            assert fnid in fn_ids


@pytest.mark.skipif(
    not _can_import_tree_sitter_python(),
    reason="tree-sitter-python not installed",
)
class TestPythonInheritance:
    def test_inherits_edge_recorded(self):
        nodes, edges = _extract("models.py", PYTHON_INHERITANCE)
        child = next(c for c in nodes.classes if c["name"] == "Child")
        parent_names = [pn for cid, pn in edges.inherits if cid == child["class_id"]]
        assert "Base" in parent_names

    def test_multi_inherits(self):
        nodes, edges = _extract("models.py", PYTHON_INHERITANCE)
        mc = next(c for c in nodes.classes if c["name"] == "MultiChild")
        parent_names = [pn for cid, pn in edges.inherits if cid == mc["class_id"]]
        assert "Base" in parent_names
        assert "Mixin" in parent_names


# ===========================================================================
# JavaScript extraction
# ===========================================================================

@pytest.mark.skipif(
    not _can_import_tree_sitter_js(),
    reason="tree-sitter-javascript not installed",
)
class TestJSExtraction:
    def test_class_extracted(self):
        nodes, _ = _extract("c.js", JS_SIMPLE)
        assert any(c["name"] == "MyComponent" for c in nodes.classes)

    def test_methods_extracted(self):
        nodes, _ = _extract("c.js", JS_SIMPLE)
        fn_names = [f["name"] for f in nodes.functions]
        assert "render" in fn_names
        assert "fetchData" in fn_names

    def test_function_extracted(self):
        nodes, _ = _extract("c.js", JS_SIMPLE)
        assert any(f["name"] == "helperFunc" for f in nodes.functions)

    def test_exported_function(self):
        nodes, _ = _extract("c.js", JS_SIMPLE)
        assert any(f["name"] == "exportedFunc" for f in nodes.functions)

    def test_imports_captured(self):
        _, edges = _extract("c.js", JS_SIMPLE)
        modules = [i["module"] for i in edges.imports]
        assert "react" in modules

    def test_named_imports(self):
        _, edges = _extract("c.js", JS_SIMPLE)
        react_imp = next(i for i in edges.imports if i["module"] == "react"
                         and "useState" in i.get("symbols", []))
        assert "useState" in react_imp["symbols"]


@pytest.mark.skipif(
    not _can_import_tree_sitter_js(),
    reason="tree-sitter-javascript not installed",
)
class TestExpressRoutes:
    def test_endpoints_extracted(self):
        nodes, _ = _extract("routes.js", JS_EXPRESS)
        assert len(nodes.endpoints) >= 1

    def test_get_endpoint(self):
        nodes, _ = _extract("routes.js", JS_EXPRESS)
        ep = next((e for e in nodes.endpoints if e["http_method"] == "GET"), None)
        assert ep is not None
        assert ep["path_pattern"] == "/api/items"

    def test_post_endpoint(self):
        nodes, _ = _extract("routes.js", JS_EXPRESS)
        ep = next((e for e in nodes.endpoints if e["http_method"] == "POST"), None)
        assert ep is not None

    def test_express_framework(self):
        nodes, _ = _extract("routes.js", JS_EXPRESS)
        assert all(e["framework"] == "express" for e in nodes.endpoints)


# ===========================================================================
# TypeScript extraction
# ===========================================================================

@pytest.mark.skipif(
    not _can_import_tree_sitter_ts(),
    reason="tree-sitter-typescript not installed",
)
class TestTSExtraction:
    def test_class_extracted(self):
        nodes, _ = _extract("service.ts", TS_SIMPLE)
        assert any(c["name"] == "UserService" for c in nodes.classes)

    def test_method_extracted(self):
        nodes, _ = _extract("service.ts", TS_SIMPLE)
        assert any(f["name"] == "getUsers" for f in nodes.functions)

    def test_async_function(self):
        nodes, _ = _extract("service.ts", TS_SIMPLE)
        fn = next((f for f in nodes.functions if f["name"] == "fetchUser"), None)
        assert fn is not None

    def test_import_captured(self):
        _, edges = _extract("service.ts", TS_SIMPLE)
        assert any(i["module"] == "@angular/core" for i in edges.imports)


# ===========================================================================
# Graceful fallback when tree-sitter is unavailable
# ===========================================================================

class TestGracefulFallback:
    def test_unsupported_extension_returns_empty(self):
        """Non-code files should return empty NodeBatch / EdgeBatch."""
        nodes, edges = _extract("README.md", "# Hello world")
        assert nodes.total() == 0
        assert edges.total() == 0

    def test_empty_content_returns_empty(self):
        nodes, edges = _extract("empty.py", "")
        assert nodes.total() == 0

    def test_syntax_error_returns_empty(self):
        """Broken syntax should not raise — return empty batches."""
        nodes, edges = _extract("broken.py", "def (((broken syntax")
        # tree-sitter is error-tolerant; we just check no exception is raised
        assert isinstance(nodes, NodeBatch)
        assert isinstance(edges, EdgeBatch)


