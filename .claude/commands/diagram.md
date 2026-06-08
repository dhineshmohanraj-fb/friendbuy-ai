# /diagram — Architecture Diagram Generator

Generate or refresh the **friendbuy-ai** production architecture as a draw.io XML diagram.

## What this skill does

1. Reads the current project structure (config.py, pipeline/, retriever/, api/, indexer/, eval/, observability/)
2. Produces a **rich, dark-theme draw.io XML** file at `docs/architecture.drawio`
3. Prints how to open it

## Design rules (follow these every time)

### Canvas
- Size: **1920 × 1080** (16:9 landscape, fits any screen)
- Background: `#0d1117` (GitHub dark)
- Shadow enabled, grid disabled

### Color palette — one color per system layer

| Layer | Fill | Stroke | Text |
|-------|------|--------|------|
| Header / Title | `#161b22` | `#30363d` | `#58a6ff` |
| Source Repos | `#0e2a1e` | `#22c55e` | `#86efac` |
| Index Pipeline | `#0d1f3d` | `#3b82f6` | `#93c5fd` |
| Storage (ChromaDB) | `#1a2000` | `#84cc16` | `#a3e635` |
| Storage (Kuzu) | `#1a0020` | `#a855f7` | `#c084fc` |
| Storage (SQLite) | `#1a1000` | `#f59e0b` | `#fde68a` |
| Query Pipeline | `#1a000a` | `#f43f5e` | `#fda4af` |
| Semantic Cache | `#1a1000` | `#f59e0b` | `#fbbf24` |
| Reranker / RRF | `#1a0a1a` | `#a855f7` | `#d8b4fe` |
| Qwen (local LLM) | `#1a1000` | `#f97316` | `#fdba74` |
| Claude API | `#0d1f3d` | `#818cf8` | `#c7d2fe` |
| FastAPI Server | `#0d1a2a` | `#4f86c6` | `#7dd3fc` |
| Eval Harness | `#1a0a2a` | `#9333ea` | `#d8b4fe` |
| Observability | `#0a1a0a` | `#16a34a` | `#4ade80` |
| Graph Viewer | `#0a1a0a` | `#22c55e` | `#4ade80` |

### Shape conventions
- **Swimlane containers** for each layer (`startSize=35`, `rounded=1`, `arcSize=3`)
- **Database cylinders** (`shape=mxgraph.flowchart.database`) for ChromaDB, Kuzu, SQLite
- **Rounded rectangles** for pipeline steps and API endpoints
- **Diamonds** (`rhombus`) for decision points (cache hit/miss)
- **Ellipses** for start/end nodes (User input, Answer output)
- **Dashed arrows** for feedback paths (store-result-back-to-cache)
- **Solid colored arrows** matching the source layer's stroke color

### Layout (pixel coordinates)
```
[20,80]   Source Repos        w=200  h=340
[240,80]  Index Pipeline      w=260  h=340
[520,80]  Storage Layer       w=300  h=340
[840,80]  FastAPI Server      w=280  h=340
[1140,80] Observability       w=260  h=820

[20,440]  Query Pipeline      w=800  h=460
[840,440] Eval Harness        w=280  h=460
```

### Typography
- Section headers: `fontSize=13, fontStyle=1 (bold)`
- Step labels: `fontSize=11`
- Sub-labels / code: `fontSize=10, fontFamily=Courier New`
- All text: white variants (see color palette above)

## Output

Save the generated XML to `docs/architecture.drawio`.

Then print:
```
✅  docs/architecture.drawio written

Open with one of:
  • draw.io Desktop app  →  File › Open
  • VS Code extension    →  right-click › Open with draw.io
  • Browser (free)       →  https://app.diagrams.net  then File › Open
```

## When to regenerate

Run `/diagram` again whenever:
- A new checkpoint (CP6+) is added
- New storage backends are introduced
- The API surface changes significantly
- The pipeline steps change order
