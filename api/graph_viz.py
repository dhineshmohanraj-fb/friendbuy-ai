"""
Knowledge Graph Visualizer — CP5 bonus.

Two endpoints:
  GET /graph/ui           — interactive D3.js force-graph browser (HTML)
  GET /graph/viz/data     — nodes + edges as JSON for the viewer

The viewer looks and feels like Obsidian Graph View:
  • Dark canvas with a force-directed layout
  • Colour-coded nodes by type (Repo / File / Class / Function / APIEndpoint)
  • Edge colours by relationship type
  • Search / filter sidebar
  • Click a node → detail panel slides in
  • Zoom / pan / drag
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# Kuzu query helper
# ---------------------------------------------------------------------------

def _fetchall(result) -> list:
    """
    Kuzu QueryResult does NOT have .fetchall().
    Iterate with has_next() / get_next() instead.
    """
    rows = []
    while result.has_next():
        rows.append(result.get_next())
    return rows


# ---------------------------------------------------------------------------
# Graph data extraction
# ---------------------------------------------------------------------------

def get_graph_data(
    repo_name: str | None = None,
    max_nodes:  int        = 600,
    show_functions: bool   = True,
) -> dict:
    """
    Query Kuzu and return ``{"nodes": [...], "edges": [...], "stats": {...}}``.

    Nodes have keys: id, label, type, repo, meta (dict of extra info).
    Edges have keys: source, target, type.
    """
    try:
        from indexer.graph_builder import GraphBuilder
    except ImportError:
        return {"nodes": [], "edges": [], "stats": {}, "error": "Kuzu not installed — run pip install kuzu>=0.6.0"}

    nodes: list[dict] = []
    edges: list[dict] = []
    seen:  set[str]   = set()

    def add_node(nid: str, label: str, ntype: str, repo: str, meta: dict) -> None:
        if nid and nid not in seen:
            seen.add(nid)
            nodes.append({"id": nid, "label": label, "type": ntype, "repo": repo, "meta": meta})

    repo_filter_cypher = f" WHERE n.repo_name = '{repo_name}'" if repo_name else ""

    # Actual schema (from graph_builder.py DDL):
    #   Repo    → repo_id, name, local_path
    #   File    → file_id, repo_name, file_path, file_name, language
    #   Class   → class_id, name, qualified_name, file_path, repo_name, start_line
    #   Function→ function_id, name, qualified_name, file_path, repo_name, is_async, is_method, start_line
    #   APIEndpoint → endpoint_id, http_method, path_pattern, full_path, framework, file_path, repo_name

    try:
        with GraphBuilder() as gb:
            c = gb._conn

            # ── Repos ──────────────────────────────────────────────────
            for row in _fetchall(c.execute("MATCH (n:Repo) RETURN n.repo_id, n.name, n.local_path")):
                rname = row[1] or ""
                if repo_name and rname != repo_name:
                    continue
                add_node(row[0], rname, "Repo", rname, {"path": row[2] or ""})

            # ── Files ──────────────────────────────────────────────────
            # File has repo_name as a direct column — filter cleanly
            if repo_name:
                fq = (f"MATCH (n:File) WHERE n.repo_name = '{repo_name}' "
                      f"RETURN n.file_id, n.file_path, n.file_name, n.language, n.repo_name "
                      f"LIMIT {max_nodes}")
            else:
                fq = (f"MATCH (n:File) "
                      f"RETURN n.file_id, n.file_path, n.file_name, n.language, n.repo_name "
                      f"LIMIT {max_nodes}")
            for row in _fetchall(c.execute(fq)):
                fid   = row[0] or ""
                fpath = row[1] or ""
                fname = row[2] or ""
                lang  = row[3] or ""
                rname = row[4] or ""
                label = fname or (fpath.split("/")[-1] if fpath else fid[:8])
                add_node(fid, label, "File", rname, {"path": fpath, "language": lang})

            # ── Classes ────────────────────────────────────────────────
            cq = (f"MATCH (n:Class){repo_filter_cypher} "
                  f"RETURN n.class_id, n.name, n.qualified_name, n.file_path, n.repo_name, n.start_line "
                  f"LIMIT {max_nodes}")
            for row in _fetchall(c.execute(cq)):
                cid, name, qname, fpath, rname, line = row
                add_node(cid or "", name or "", "Class", rname or "", {
                    "qualified_name": qname or name or "",
                    "file_path": fpath or "",
                    "line": line or 0,
                })

            # ── Functions (optional — can be noisy on large codebases) ──
            if show_functions:
                fq2 = (f"MATCH (n:Function){repo_filter_cypher} "
                       f"RETURN n.function_id, n.name, n.file_path, n.repo_name, "
                       f"n.is_method, n.is_async, n.start_line "
                       f"LIMIT {max_nodes}")
                for row in _fetchall(c.execute(fq2)):
                    fid2, name, fpath, rname, is_m, is_a, line = row
                    add_node(fid2 or "", name or "", "Function", rname or "", {
                        "file_path": fpath or "",
                        "is_method": bool(is_m),
                        "is_async":  bool(is_a),
                        "line": line or 0,
                    })

            # ── API Endpoints ──────────────────────────────────────────
            eq = (f"MATCH (n:APIEndpoint){repo_filter_cypher} "
                  f"RETURN n.endpoint_id, n.http_method, n.path_pattern, "
                  f"n.framework, n.file_path, n.repo_name "
                  f"LIMIT {max_nodes}")
            for row in _fetchall(c.execute(eq)):
                eid, method, pattern, fw, fpath, rname = row
                label = f"{method or 'GET'} {pattern or (eid or '')[:14]}"
                add_node(eid or "", label, "APIEndpoint", rname or "", {
                    "method":    method   or "",
                    "path":      pattern  or "",
                    "framework": fw       or "",
                    "file_path": fpath    or "",
                })

            # ── Edges ──────────────────────────────────────────────────
            def add_edges(query: str, etype: str) -> None:
                try:
                    for row in _fetchall(c.execute(query)):
                        src, tgt = row[0], row[1]
                        if src and tgt and src in seen and tgt in seen:
                            edges.append({"source": src, "target": tgt, "type": etype})
                except Exception:
                    pass   # skip edge type gracefully if query fails

            # Note: Repo PK is repo_id (not node_id)
            add_edges("MATCH (a:File)-[:BELONGS_TO_REPO]->(b:Repo) RETURN a.file_id, b.repo_id",          "BELONGS_TO_REPO")
            add_edges("MATCH (a:File)-[:CONTAINS_CLASS]->(b:Class) RETURN a.file_id, b.class_id",         "CONTAINS_CLASS")
            add_edges("MATCH (a:File)-[:CONTAINS_FUNCTION]->(b:Function) RETURN a.file_id, b.function_id","CONTAINS_FUNCTION")
            add_edges("MATCH (a:Function)-[:METHOD_OF]->(b:Class) RETURN a.function_id, b.class_id",      "METHOD_OF")
            add_edges("MATCH (a:Class)-[:INHERITS]->(b:Class) RETURN a.class_id, b.class_id",             "INHERITS")
            add_edges("MATCH (a:File)-[:EXPOSES]->(b:APIEndpoint) RETURN a.file_id, b.endpoint_id",       "EXPOSES")
            add_edges("MATCH (a:APIEndpoint)-[:HANDLES]->(b:Function) RETURN a.endpoint_id, b.function_id","HANDLES")
            add_edges("MATCH (a:File)-[:IMPORT_DEP]->(b:File) RETURN a.file_id, b.file_id",               "IMPORT_DEP")

    except Exception as exc:  # noqa: BLE001
        return {"nodes": nodes, "edges": edges, "stats": {}, "error": str(exc)}

    stats = {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "by_type": {},
    }
    for n in nodes:
        stats["by_type"][n["type"]] = stats["by_type"].get(n["type"], 0) + 1

    return {"nodes": nodes, "edges": edges, "stats": stats}


# ---------------------------------------------------------------------------
# HTML viewer (self-contained single-page app)
# ---------------------------------------------------------------------------

GRAPH_VIEWER_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>friendbuy-ai · Knowledge Graph</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:        #0d1117;
    --bg2:       #161b22;
    --bg3:       #21262d;
    --border:    #30363d;
    --text:      #c9d1d9;
    --text-dim:  #8b949e;
    --accent:    #58a6ff;
    --repo:      #4da6ff;
    --file:      #6e7f8a;
    --class:     #4ade80;
    --func:      #fb923c;
    --endpoint:  #f472b6;
    --edge-dim:  #1f2937;
    --sidebar-w: 260px;
    --detail-w:  300px;
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 13px;
    overflow: hidden;
    height: 100vh;
    display: flex;
    flex-direction: column;
  }

  /* ── Top bar ─────────────────────────────────────────────────────── */
  #topbar {
    height: 44px;
    background: var(--bg2);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    padding: 0 14px;
    gap: 12px;
    flex-shrink: 0;
    z-index: 100;
  }
  #topbar .logo {
    font-weight: 700;
    font-size: 14px;
    color: var(--accent);
    letter-spacing: -0.3px;
  }
  #topbar .logo span { color: var(--text-dim); font-weight: 400; }
  #search {
    flex: 1;
    max-width: 340px;
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    padding: 5px 10px;
    font-size: 13px;
    outline: none;
  }
  #search:focus { border-color: var(--accent); }
  #topbar .stats {
    margin-left: auto;
    color: var(--text-dim);
    font-size: 12px;
    white-space: nowrap;
  }
  .tag {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 2px 8px;
    font-size: 11px;
    cursor: pointer;
    transition: background 0.15s;
    user-select: none;
  }
  .tag:hover { background: #2d333b; }
  .tag.active { background: #1f6feb33; border-color: var(--accent); color: var(--accent); }
  .tag .dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  /* ── Main layout ─────────────────────────────────────────────────── */
  #main {
    flex: 1;
    display: flex;
    overflow: hidden;
  }

  /* ── Sidebar ─────────────────────────────────────────────────────── */
  #sidebar {
    width: var(--sidebar-w);
    background: var(--bg2);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    flex-shrink: 0;
  }
  .sidebar-section {
    padding: 12px 14px;
    border-bottom: 1px solid var(--border);
  }
  .sidebar-section h3 {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--text-dim);
    margin-bottom: 8px;
  }
  .filter-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 0;
    cursor: pointer;
    border-radius: 4px;
  }
  .filter-row:hover { background: var(--bg3); padding: 4px 6px; margin: 0 -6px; }
  .filter-row input[type=checkbox] { accent-color: var(--accent); cursor: pointer; }
  .filter-row .dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .filter-row label {
    flex: 1;
    cursor: pointer;
    font-size: 12px;
  }
  .filter-row .count {
    font-size: 11px;
    color: var(--text-dim);
    background: var(--bg3);
    border-radius: 10px;
    padding: 1px 6px;
    min-width: 24px;
    text-align: center;
  }
  #repo-filter {
    width: 100%;
    background: var(--bg3);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 6px;
    padding: 5px 8px;
    font-size: 12px;
    outline: none;
    margin-top: 4px;
    cursor: pointer;
  }
  #physics-toggle {
    width: 100%;
    background: var(--bg3);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
    cursor: pointer;
    text-align: left;
    transition: background 0.15s;
    margin-top: 4px;
  }
  #physics-toggle:hover { background: #2d333b; }
  .slider-row {
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-top: 6px;
  }
  .slider-row label {
    font-size: 11px;
    color: var(--text-dim);
    display: flex;
    justify-content: space-between;
  }
  input[type=range] { width: 100%; accent-color: var(--accent); cursor: pointer; }

  /* ── Canvas area ─────────────────────────────────────────────────── */
  #canvas-wrap {
    flex: 1;
    position: relative;
    overflow: hidden;
    background: var(--bg);
  }
  #canvas-wrap svg {
    width: 100%;
    height: 100%;
    display: block;
  }
  .grid-bg {
    stroke: #ffffff06;
    stroke-width: 1;
  }
  .node circle {
    cursor: grab;
    transition: filter 0.2s;
  }
  .node circle:active { cursor: grabbing; }
  .node text {
    pointer-events: none;
    font-size: 10px;
    fill: var(--text-dim);
    text-anchor: middle;
    dominant-baseline: hanging;
  }
  .link {
    stroke-opacity: 0.5;
    stroke-width: 1;
  }
  .node.faded circle  { opacity: 0.12; }
  .node.faded text    { opacity: 0.06; }
  .link.faded         { stroke-opacity: 0.04; }
  .node.highlighted circle { filter: drop-shadow(0 0 6px currentColor); }

  /* ── Tooltip ─────────────────────────────────────────────────────── */
  #tooltip {
    position: absolute;
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 10px;
    font-size: 12px;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.15s;
    max-width: 240px;
    z-index: 200;
    line-height: 1.5;
  }
  #tooltip strong { color: var(--accent); }
  #tooltip .tt-type {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--text-dim);
  }

  /* ── Detail panel ────────────────────────────────────────────────── */
  #detail {
    width: 0;
    background: var(--bg2);
    border-left: 1px solid var(--border);
    overflow: hidden;
    transition: width 0.2s ease;
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
  }
  #detail.open { width: var(--detail-w); }
  #detail-inner { padding: 14px; overflow-y: auto; flex: 1; }
  #detail h2 { font-size: 14px; font-weight: 600; margin-bottom: 4px; word-break: break-all; }
  #detail .d-type {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--text-dim);
    margin-bottom: 12px;
  }
  #detail .d-section { margin-top: 12px; }
  #detail .d-section h4 {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--text-dim);
    margin-bottom: 6px;
  }
  #detail .d-row {
    display: flex;
    justify-content: space-between;
    padding: 3px 0;
    border-bottom: 1px solid var(--bg3);
    font-size: 12px;
  }
  #detail .d-row .k { color: var(--text-dim); flex-shrink: 0; margin-right: 8px; }
  #detail .d-row .v { color: var(--text); word-break: break-all; text-align: right; }
  #detail .d-chip {
    display: inline-block;
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 2px 6px;
    font-size: 11px;
    margin: 2px 2px 0 0;
    color: var(--text-dim);
    max-width: 100%;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  #detail-close {
    position: absolute;
    top: 8px; right: 8px;
    background: none;
    border: none;
    color: var(--text-dim);
    font-size: 18px;
    cursor: pointer;
    line-height: 1;
    padding: 2px 6px;
    border-radius: 4px;
  }
  #detail-close:hover { background: var(--bg3); color: var(--text); }

  /* ── Loading overlay ─────────────────────────────────────────────── */
  #loading {
    position: absolute;
    inset: 0;
    background: var(--bg);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 16px;
    z-index: 300;
  }
  #loading .spinner {
    width: 40px; height: 40px;
    border: 3px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  #loading p { color: var(--text-dim); font-size: 13px; }

  /* ── Error banner ────────────────────────────────────────────────── */
  #error-banner {
    display: none;
    position: absolute;
    top: 12px; left: 50%; transform: translateX(-50%);
    background: #490202;
    border: 1px solid #f85149;
    border-radius: 6px;
    padding: 8px 14px;
    font-size: 12px;
    color: #ff8080;
    z-index: 400;
    max-width: 460px;
    text-align: center;
  }
</style>
</head>
<body>

<!-- ── Top bar ─────────────────────────────────────────────── -->
<div id="topbar">
  <div class="logo">friendbuy<span>-ai</span> · Knowledge Graph</div>
  <input id="search" type="text" placeholder="Search nodes…" autocomplete="off">
  <div id="type-tags"></div>
  <div class="stats" id="top-stats">Loading…</div>
</div>

<!-- ── Main ───────────────────────────────────────────────── -->
<div id="main">

  <!-- Sidebar -->
  <div id="sidebar">
    <div class="sidebar-section">
      <h3>Repos</h3>
      <select id="repo-filter"><option value="">All repos</option></select>
    </div>
    <div class="sidebar-section">
      <h3>Node Types</h3>
      <div id="node-filters"></div>
    </div>
    <div class="sidebar-section">
      <h3>Edge Types</h3>
      <div id="edge-filters"></div>
    </div>
    <div class="sidebar-section">
      <h3>Layout</h3>
      <button id="physics-toggle">⏸ Pause physics</button>
      <div class="slider-row">
        <label>Link distance <span id="dist-val">80</span></label>
        <input type="range" id="link-dist" min="20" max="300" value="80">
      </div>
      <div class="slider-row">
        <label>Repulsion <span id="rep-val">-120</span></label>
        <input type="range" id="repulsion" min="-600" max="-20" value="-120">
      </div>
    </div>
    <div class="sidebar-section" style="margin-top: auto; border-top: 1px solid var(--border);">
      <div style="font-size:11px; color:var(--text-dim); line-height:1.6;">
        <b style="color:var(--text)">Controls</b><br>
        Scroll — zoom<br>
        Drag canvas — pan<br>
        Drag node — move<br>
        Click node — details<br>
        Double-click — focus
      </div>
    </div>
  </div>

  <!-- Canvas -->
  <div id="canvas-wrap">
    <div id="loading"><div class="spinner"></div><p>Loading graph…</p></div>
    <div id="error-banner"></div>
    <div id="tooltip"></div>
    <svg id="graph-svg"></svg>
  </div>

  <!-- Detail panel -->
  <div id="detail">
    <div style="position:relative; height:100%; display:flex; flex-direction:column;">
      <button id="detail-close" onclick="closeDetail()">✕</button>
      <div id="detail-inner"></div>
    </div>
  </div>

</div>

<script>
// ═══════════════════════════════════════════════════════════════════
// Config
// ═══════════════════════════════════════════════════════════════════
const NODE_COLORS = {
  Repo:        '#4da6ff',
  File:        '#6e7f8a',
  Class:       '#4ade80',
  Function:    '#fb923c',
  APIEndpoint: '#f472b6',
};
const EDGE_COLORS = {
  BELONGS_TO_REPO:    '#2a4a6b',
  CONTAINS_CLASS:     '#1a3a2a',
  CONTAINS_FUNCTION:  '#1a2a1a',
  METHOD_OF:          '#4c1d95',
  INHERITS:           '#78350f',
  EXPOSES:            '#14532d',
  HANDLES:            '#1e3a5f',
  IMPORT_DEP:         '#1f2937',
  CROSS_REPO_CALL:    '#7f1d1d',
};
const NODE_RADIUS = {
  Repo: 18, File: 9, Class: 13, Function: 7, APIEndpoint: 11,
};

// ═══════════════════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════════════════
let allNodes = [], allEdges = [], graphStats = {};
let enabledNodeTypes = new Set(Object.keys(NODE_COLORS));
let enabledEdgeTypes = new Set(Object.keys(EDGE_COLORS));
let selectedRepo = '';
let searchTerm   = '';
let simulation, svg, gAll, linkSel, nodeSel;
let physicsRunning = true;
let selectedNode = null;

// ═══════════════════════════════════════════════════════════════════
// Fetch & bootstrap
// ═══════════════════════════════════════════════════════════════════
async function fetchData(repo = '') {
  const url = `/graph/viz/data${repo ? '?repo=' + encodeURIComponent(repo) : ''}`;
  const res  = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function init() {
  try {
    const data = await fetchData();
    if (data.error) throw new Error(data.error);
    allNodes   = data.nodes;
    allEdges   = data.edges;
    graphStats = data.stats || {};
    setupUI();
    buildGraph();
    document.getElementById('loading').style.display = 'none';
  } catch (e) {
    document.getElementById('loading').style.display = 'none';
    const err = document.getElementById('error-banner');
    err.textContent = '⚠ ' + e.message;
    err.style.display = 'block';
  }
}

// ═══════════════════════════════════════════════════════════════════
// UI setup
// ═══════════════════════════════════════════════════════════════════
function setupUI() {
  // Top stats
  document.getElementById('top-stats').textContent =
    `${allNodes.length} nodes · ${allEdges.length} edges`;

  // Repo dropdown
  const repos = [...new Set(allNodes.map(n => n.repo).filter(Boolean))].sort();
  const sel = document.getElementById('repo-filter');
  repos.forEach(r => {
    const o = document.createElement('option');
    o.value = r; o.textContent = r;
    sel.appendChild(o);
  });
  sel.addEventListener('change', async () => {
    selectedRepo = sel.value;
    document.getElementById('loading').style.display = 'flex';
    try {
      const data = await fetchData(selectedRepo);
      allNodes = data.nodes; allEdges = data.edges; graphStats = data.stats || {};
      document.getElementById('top-stats').textContent =
        `${allNodes.length} nodes · ${allEdges.length} edges`;
      rebuildGraph();
    } catch(e) {}
    document.getElementById('loading').style.display = 'none';
  });

  // Node type filters
  const nf = document.getElementById('node-filters');
  Object.entries(NODE_COLORS).forEach(([type, color]) => {
    const cnt = (graphStats.by_type || {})[type] || 0;
    const row = document.createElement('div');
    row.className = 'filter-row';
    row.innerHTML = `
      <input type="checkbox" id="nf-${type}" checked>
      <span class="dot" style="background:${color}"></span>
      <label for="nf-${type}">${type}</label>
      <span class="count">${cnt}</span>`;
    row.querySelector('input').addEventListener('change', e => {
      if (e.target.checked) enabledNodeTypes.add(type);
      else enabledNodeTypes.delete(type);
      applyFilters();
    });
    nf.appendChild(row);
  });

  // Edge type filters
  const ef = document.getElementById('edge-filters');
  Object.entries(EDGE_COLORS).forEach(([type, color]) => {
    const row = document.createElement('div');
    row.className = 'filter-row';
    row.innerHTML = `
      <input type="checkbox" id="ef-${type}" checked>
      <span class="dot" style="background:${color}; border-radius:2px; width:16px; height:4px;"></span>
      <label for="ef-${type}" style="font-size:10px;">${type}</label>`;
    row.querySelector('input').addEventListener('change', e => {
      if (e.target.checked) enabledEdgeTypes.add(type);
      else enabledEdgeTypes.delete(type);
      applyFilters();
    });
    ef.appendChild(row);
  });

  // Search
  const searchEl = document.getElementById('search');
  searchEl.addEventListener('input', () => {
    searchTerm = searchEl.value.trim().toLowerCase();
    applyFilters();
  });

  // Physics toggle
  const physBtn = document.getElementById('physics-toggle');
  physBtn.addEventListener('click', () => {
    physicsRunning = !physicsRunning;
    if (physicsRunning) { simulation.alpha(0.3).restart(); physBtn.textContent = '⏸ Pause physics'; }
    else                { simulation.stop(); physBtn.textContent = '▶ Resume physics'; }
  });

  // Sliders
  const distEl = document.getElementById('link-dist');
  distEl.addEventListener('input', () => {
    document.getElementById('dist-val').textContent = distEl.value;
    if (simulation) {
      simulation.force('link').distance(+distEl.value);
      simulation.alpha(0.3).restart();
    }
  });
  const repEl = document.getElementById('repulsion');
  repEl.addEventListener('input', () => {
    document.getElementById('rep-val').textContent = repEl.value;
    if (simulation) {
      simulation.force('charge').strength(+repEl.value);
      simulation.alpha(0.3).restart();
    }
  });
}

// ═══════════════════════════════════════════════════════════════════
// D3 graph
// ═══════════════════════════════════════════════════════════════════
function buildGraph() {
  const wrap = document.getElementById('canvas-wrap');
  const W = wrap.clientWidth, H = wrap.clientHeight;

  svg = d3.select('#graph-svg');
  svg.selectAll('*').remove();

  // Zoom
  const zoom = d3.zoom()
    .scaleExtent([0.05, 8])
    .on('zoom', e => gAll.attr('transform', e.transform));
  svg.call(zoom);

  // Grid background
  const defs = svg.append('defs');
  const pat = defs.append('pattern')
    .attr('id', 'grid').attr('width', 40).attr('height', 40)
    .attr('patternUnits', 'userSpaceOnUse');
  pat.append('path').attr('d', 'M 40 0 L 0 0 0 40')
    .attr('fill', 'none').attr('stroke', '#ffffff04').attr('stroke-width', 1);
  svg.append('rect').attr('width', '100%').attr('height', '100%').attr('fill', 'url(#grid)');

  gAll = svg.append('g');

  // Arrow markers
  ['default', 'highlight'].forEach(id => {
    defs.append('marker')
      .attr('id', 'arrow-' + id)
      .attr('viewBox', '0 -4 8 8')
      .attr('refX', 20).attr('refY', 0)
      .attr('markerWidth', 6).attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-4L8,0L0,4')
      .attr('fill', id === 'highlight' ? '#58a6ff' : '#30363d');
  });

  rebuildGraph(W, H);

  // Center on initial load
  const initialScale = 0.7;
  svg.call(zoom.transform, d3.zoomIdentity
    .translate(W / 2, H / 2)
    .scale(initialScale));
}

function rebuildGraph(W, H) {
  const wrap = document.getElementById('canvas-wrap');
  W = W || wrap.clientWidth;
  H = H || wrap.clientHeight;

  if (simulation) simulation.stop();

  const nodes = allNodes.filter(n => enabledNodeTypes.has(n.type));
  const nodeIds = new Set(nodes.map(n => n.id));
  const links  = allEdges
    .filter(e => enabledEdgeTypes.has(e.type) && nodeIds.has(e.source) && nodeIds.has(e.target))
    .map(e => ({ ...e }));   // shallow copy so D3 can mutate source/target

  // Init positions
  nodes.forEach(n => {
    if (n.x === undefined) {
      n.x = (Math.random() - 0.5) * W * 0.6;
      n.y = (Math.random() - 0.5) * H * 0.6;
    }
  });

  gAll.selectAll('.link').remove();
  gAll.selectAll('.node').remove();

  // Links
  linkSel = gAll.append('g').attr('class', 'links-g').selectAll('.link')
    .data(links, d => `${d.source}-${d.target}-${d.type}`)
    .join('line')
    .attr('class', 'link')
    .attr('stroke', d => EDGE_COLORS[d.type] || '#30363d')
    .attr('stroke-width', d => d.type === 'CROSS_REPO_CALL' ? 2 : 1)
    .attr('stroke-opacity', d => d.type === 'IMPORT_DEP' ? 0.25 : 0.55)
    .attr('marker-end', 'url(#arrow-default)');

  // Nodes
  const nodeG = gAll.append('g').attr('class', 'nodes-g').selectAll('.node')
    .data(nodes, d => d.id)
    .join('g')
    .attr('class', 'node')
    .call(d3.drag()
      .on('start', dragStart)
      .on('drag',  dragging)
      .on('end',   dragEnd))
    .on('mouseover', showTooltip)
    .on('mousemove', moveTooltip)
    .on('mouseout',  hideTooltip)
    .on('click',     clickNode)
    .on('dblclick',  focusNode);

  nodeG.append('circle')
    .attr('r', d => NODE_RADIUS[d.type] || 8)
    .attr('fill', d => NODE_COLORS[d.type] || '#888')
    .attr('stroke', d => darken(NODE_COLORS[d.type] || '#888'))
    .attr('stroke-width', 1.5);

  nodeG.append('text')
    .attr('dy', d => (NODE_RADIUS[d.type] || 8) + 4)
    .text(d => truncate(d.label, 18));

  nodeSel = nodeG;

  // Simulation
  simulation = d3.forceSimulation(nodes)
    .force('link',   d3.forceLink(links).id(d => d.id).distance(80).strength(0.4))
    .force('charge', d3.forceManyBody().strength(-120).theta(0.9))
    .force('center', d3.forceCenter(0, 0).strength(0.05))
    .force('collide', d3.forceCollide(d => (NODE_RADIUS[d.type] || 8) + 3))
    .alphaDecay(0.025)
    .on('tick', ticked);

  if (!physicsRunning) simulation.stop();
}

function ticked() {
  linkSel
    .attr('x1', d => d.source.x)
    .attr('y1', d => d.source.y)
    .attr('x2', d => d.target.x)
    .attr('y2', d => d.target.y);

  // Get current zoom scale to hide labels when zoomed out
  const transform = d3.zoomTransform(svg.node());
  const showLabels = transform.k > 0.5;

  nodeSel
    .attr('transform', d => `translate(${d.x},${d.y})`)
    .select('text')
    .attr('visibility', showLabels ? 'visible' : 'hidden');
}

// ═══════════════════════════════════════════════════════════════════
// Filters & search
// ═══════════════════════════════════════════════════════════════════
function applyFilters() {
  if (!nodeSel) return;

  const term = searchTerm;
  const hasSearch = term.length > 0;

  nodeSel.classed('faded', d => {
    if (!enabledNodeTypes.has(d.type)) return true;
    if (hasSearch && !d.label.toLowerCase().includes(term) &&
        !(d.meta.path || '').toLowerCase().includes(term)) return true;
    return false;
  });

  if (linkSel) {
    linkSel
      .attr('visibility', d => enabledEdgeTypes.has(d.type) ? 'visible' : 'hidden')
      .classed('faded', d => {
        if (!enabledEdgeTypes.has(d.type)) return true;
        if (hasSearch) {
          const sLbl = (d.source.label || '').toLowerCase();
          const tLbl = (d.target.label || '').toLowerCase();
          if (!sLbl.includes(term) && !tLbl.includes(term)) return true;
        }
        return false;
      });
  }

  simulation && simulation.alpha(0.05).restart();
}

// ═══════════════════════════════════════════════════════════════════
// Drag
// ═══════════════════════════════════════════════════════════════════
function dragStart(event, d) {
  if (!event.active) simulation.alphaTarget(0.3).restart();
  d.fx = d.x; d.fy = d.y;
  d3.select(event.sourceEvent.target.closest('.node')).raise();
}
function dragging(event, d) { d.fx = event.x; d.fy = event.y; }
function dragEnd(event, d) {
  if (!event.active) simulation.alphaTarget(0);
  d.fx = null; d.fy = null;
}

// ═══════════════════════════════════════════════════════════════════
// Tooltip
// ═══════════════════════════════════════════════════════════════════
function showTooltip(event, d) {
  const tt = document.getElementById('tooltip');
  tt.innerHTML = `
    <div class="tt-type">${d.type}</div>
    <strong>${d.label}</strong>
    ${d.repo ? `<br><span style="color:var(--text-dim);font-size:11px;">repo: ${d.repo}</span>` : ''}
    ${d.meta.path ? `<br><span style="color:var(--text-dim);font-size:10px;">${d.meta.path}</span>` : ''}
  `;
  tt.style.opacity = '1';
}
function moveTooltip(event) {
  const tt = document.getElementById('tooltip');
  const wrap = document.getElementById('canvas-wrap').getBoundingClientRect();
  let x = event.clientX - wrap.left + 14;
  let y = event.clientY - wrap.top  - 8;
  if (x + 250 > wrap.width)  x -= 270;
  if (y + 100 > wrap.height) y -= 90;
  tt.style.left = x + 'px';
  tt.style.top  = y + 'px';
}
function hideTooltip() {
  document.getElementById('tooltip').style.opacity = '0';
}

// ═══════════════════════════════════════════════════════════════════
// Click → detail panel
// ═══════════════════════════════════════════════════════════════════
function clickNode(event, d) {
  event.stopPropagation();
  selectedNode = d;
  openDetail(d);

  nodeSel.classed('highlighted', n => n.id === d.id);
  nodeSel.classed('faded',       n => n.id !== d.id);
  if (linkSel) linkSel.classed('faded', l =>
    l.source.id !== d.id && l.target.id !== d.id);
}
svg && svg.on('click', () => {
  if (selectedNode) {
    selectedNode = null;
    nodeSel && nodeSel.classed('highlighted faded', false);
    linkSel  && linkSel.classed('faded', false);
    closeDetail();
    applyFilters();
  }
});

function focusNode(event, d) {
  event.stopPropagation();
  const connectedIds = new Set([d.id]);
  allEdges.forEach(e => {
    const sid = typeof e.source === 'object' ? e.source.id : e.source;
    const tid = typeof e.target === 'object' ? e.target.id : e.target;
    if (sid === d.id) connectedIds.add(tid);
    if (tid === d.id) connectedIds.add(sid);
  });
  nodeSel.classed('faded',       n => !connectedIds.has(n.id));
  nodeSel.classed('highlighted', n => n.id === d.id);
  if (linkSel) linkSel.classed('faded', l =>
    !connectedIds.has(typeof l.source === 'object' ? l.source.id : l.source) ||
    !connectedIds.has(typeof l.target === 'object' ? l.target.id : l.target));
}

function openDetail(d) {
  const panel  = document.getElementById('detail');
  const inner  = document.getElementById('detail-inner');
  const color  = NODE_COLORS[d.type] || '#888';

  // Connected nodes
  const connected = [];
  allEdges.forEach(e => {
    const sid = typeof e.source === 'object' ? e.source.id : e.source;
    const tid = typeof e.target === 'object' ? e.target.id : e.target;
    if (sid === d.id) {
      const n = allNodes.find(x => x.id === tid);
      if (n) connected.push({ dir: '→', rel: e.type, node: n });
    } else if (tid === d.id) {
      const n = allNodes.find(x => x.id === sid);
      if (n) connected.push({ dir: '←', rel: e.type, node: n });
    }
  });

  inner.innerHTML = `
    <h2 style="color:${color}">${escHtml(d.label)}</h2>
    <div class="d-type">${d.type} ${d.repo ? '· ' + escHtml(d.repo) : ''}</div>

    <div class="d-section">
      <h4>Properties</h4>
      ${Object.entries(d.meta || {})
          .filter(([k,v]) => v !== '' && v !== 0 && v !== false && v !== null && v !== undefined)
          .map(([k,v]) => `<div class="d-row"><span class="k">${k}</span><span class="v">${escHtml(String(v))}</span></div>`)
          .join('') || '<div style="color:var(--text-dim);font-size:11px;">No metadata</div>'}
    </div>

    ${connected.length > 0 ? `
    <div class="d-section">
      <h4>Connections (${connected.length})</h4>
      ${connected.slice(0, 20).map(c => `
        <div style="margin-bottom:4px;">
          <span style="font-size:10px;color:${NODE_COLORS[c.node.type]};">${c.node.type}</span>
          <span style="font-size:10px;color:var(--text-dim);"> ${c.dir} ${escHtml(c.rel)}</span><br>
          <span class="d-chip" title="${escHtml(c.node.label)}">${escHtml(c.node.label)}</span>
        </div>`).join('')}
      ${connected.length > 20 ? `<div style="color:var(--text-dim);font-size:11px;">…and ${connected.length - 20} more</div>` : ''}
    </div>` : ''}
  `;

  panel.classList.add('open');
}

function closeDetail() {
  document.getElementById('detail').classList.remove('open');
  selectedNode = null;
  nodeSel && nodeSel.classed('highlighted faded', false);
  linkSel  && linkSel.classed('faded', false);
}

// ═══════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════
function truncate(s, n) { return s && s.length > n ? s.slice(0, n) + '…' : (s || ''); }
function darken(hex) {
  const n = parseInt(hex.slice(1), 16);
  const r = Math.max(0, (n >> 16) - 40);
  const g = Math.max(0, ((n >> 8) & 0xff) - 40);
  const b = Math.max(0, (n & 0xff) - 40);
  return '#' + [r,g,b].map(x => x.toString(16).padStart(2,'0')).join('');
}
function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// Handle window resize
window.addEventListener('resize', () => {
  if (simulation) simulation.alpha(0.1).restart();
});

// ── Boot ─────────────────────────────────────────────────────────
init();
</script>
</body>
</html>"""
