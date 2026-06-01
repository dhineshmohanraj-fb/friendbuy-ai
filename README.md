# friendbuy-ai

A production-grade **Hybrid Vector + Graph RAG** pipeline that lets you query the Friendbuy codebase using natural language. A local **Qwen** model (Ollama) retrieves and curates context; **Claude** (Anthropic API) reasons over it and produces precise, code-aware answers. A **Kuzu knowledge graph** stores structural relationships between classes, functions, and API endpoints — enabling questions like *"what calls this endpoint?"* or *"which services inherit from BaseService?"*.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        friendbuy-ai  (CP2)                                  │
│                                                                             │
│  ./repos/          ┌────────────────────────────────────────────────────┐  │
│  ├── api/          │  IndexPipeline                                     │  │
│  ├── payments/     │  1. load_repos     → Documents (one per file)      │  │
│  └── widgets/      │  2. delta_filter   → skip unchanged files (SQLite) │  │
│                    │  3. ast_splitter   → tree-sitter chunks            │  │
│                    │  4. embed_and_store→ ChromaDB  + Repo/File nodes   │  │
│                    │  5. extract_symbols→ Class/Function/Endpoint nodes │  │
│                    └───────────┬────────────────────┬───────────────────┘  │
│                                │                    │                       │
│                          ChromaDB             Kuzu Graph                    │
│                      (vector embeddings)   (structural edges)               │
│                                │                    │                       │
│  User question                 │                    │                       │
│       └──────────▶  Retriever ─┴────────────────────┘                      │
│                    1. vector_search   (nomic-embed-text similarity)         │
│                    2. context_filter  (Qwen cleans + summarises)            │
│                    3. Claude API      (reasoning + answer)                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## What's in the knowledge graph

After indexing, Kuzu holds a structural map of your codebase:

| Node type | What it represents |
|-----------|-------------------|
| `Repo` | A top-level repository folder |
| `File` | Every source file indexed |
| `Class` | Every Python/JS/TS class |
| `Function` | Every function and method |
| `APIEndpoint` | FastAPI / Express routes (`@router.get`, `app.post`, …) |

| Edge | Meaning |
|------|---------|
| `BELONGS_TO_REPO` | File lives in Repo |
| `CONTAINS_CLASS` | File defines Class |
| `CONTAINS_FUNCTION` | File defines Function |
| `METHOD_OF` | Function is a method of Class |
| `INHERITS` | Class extends another Class |
| `EXPOSES` | File registers an APIEndpoint |
| `HANDLES` | APIEndpoint is handled by Function |
| `IMPORT_DEP` | File imports symbols from another File |

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | **3.11 or 3.12** *(not 3.13+)* | [python.org](https://python.org) |
| Ollama | latest | `brew install ollama` |
| Anthropic API key | — | [console.anthropic.com](https://console.anthropic.com) |

> **Why 3.11 / 3.12?** The `kuzu` graph database has no pre-built wheel for Python 3.13+. Everything else works on any modern Python.

### Pull the required Ollama models

```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

---

## Installation

```bash
# 1. Clone / enter this repo
cd friendbuy-ai

# 2. Create venv on Python 3.11 or 3.12
python3.12 -m venv .venv          # or python3.11
source .venv/bin/activate         # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Open .env and set ANTHROPIC_API_KEY=sk-ant-…

# 5. Install the test runner (optional)
pip install pytest
```

---

## Adding repos

Drop any cloned Friendbuy repos into the `./repos/` directory:

```bash
cd repos/
git clone git@github.com:friendbuy/api.git
git clone git@github.com:friendbuy/payments-service.git
git clone git@github.com:friendbuy/widgets.git
```

Each top-level folder inside `./repos/` becomes a named "repo" that you can scope queries to.

---

## CLI Usage

### Build the knowledge base

```bash
# First run — indexes all repos, builds vector + graph stores
python cli.py index

# Wipe everything and rebuild from scratch
python cli.py index --reindex

# Vector index only (skip Kuzu graph)
python cli.py index --no-graph
```

On subsequent runs without `--reindex`, only **changed files** are re-embedded and re-extracted (delta tracking). Unchanged files are skipped automatically.

### Ask questions

```bash
# Ask a question across the entire codebase
python cli.py ask "How does the referral tracking flow work?"

# Scope the search to a single repo
python cli.py ask "Where is the Stripe webhook handler?" --repo payments-service

# Multi-word questions (always quote them)
python cli.py ask "What database migrations exist for the users table?"
```

### Show statistics

```bash
# Vector index stats (chunks per repo)
python cli.py stats

# Knowledge graph stats (nodes + edges)
python cli.py graph-stats
```

---

## FastAPI server (optional)

```bash
# Start the API server (default: http://localhost:8000)
python -m api.server

# Or with auto-reload for development
uvicorn api.server:app --reload --port 8000
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Liveness check |
| `GET`  | `/ready` | Readiness probe (ChromaDB + Ollama) |
| `GET`  | `/stats` | Index statistics |
| `POST` | `/index?reindex=false` | Trigger (re-)indexing (async, returns `job_id`) |
| `GET`  | `/index/status/{job_id}` | Poll indexing job status |
| `POST` | `/ask` | Ask a question |

#### Example — ask via curl

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How does referral attribution work?", "repo": null}'
```

#### Example — trigger index and poll

```bash
JOB=$(curl -s -X POST "http://localhost:8000/index" | jq -r .job_id)
curl "http://localhost:8000/index/status/$JOB"
```

---

## Running tests

```bash
# Run all tests (no Ollama / ChromaDB needed)
pytest tests/ -v

# Run a single test file
pytest tests/test_ast_parser.py -v
pytest tests/test_delta_tracker.py -v
```

---

## Project structure

```
friendbuy-ai/
├── .env.example              ← copy to .env and fill in your API key
├── requirements.txt
├── config.py                 ← all settings (reads from .env)
├── cli.py                    ← CLI entry point
│
├── indexer/
│   ├── repo_loader.py        ← walks ./repos/, returns LangChain Documents
│   ├── splitter.py           ← tree-sitter AST-aware chunking (CP1)
│   ├── ast_parser.py         ← full symbol extraction: NodeBatch/EdgeBatch (CP2)
│   ├── embedder.py           ← nomic-embed-text → ChromaDB + Repo/File graph nodes
│   ├── delta_tracker.py      ← SQLite registry: skip unchanged files
│   └── graph_builder.py      ← Kuzu graph: upsert all node/edge types (CP2)
│
├── pipeline/
│   ├── index_pipeline.py     ← unified orchestrator: load→delta→chunk→embed→graph (CP2)
│   └── query_pipeline.py     ← vector search → Qwen filter → Claude answer
│
├── retriever/
│   ├── vector_search.py      ← ChromaDB similarity search
│   └── context_filter.py     ← Qwen (local) curates + summarises chunks
│
├── api/
│   └── server.py             ← FastAPI server
│
├── tests/
│   ├── test_ast_parser.py    ← 48 tests for symbol extraction (CP2)
│   └── test_delta_tracker.py ← 28 tests for delta tracking (CP2)
│
├── repos/                    ← drop cloned Friendbuy repos here
├── cache/                    ← SQLite delta-tracking registry (gitignored)
├── friendbuy-knowledge-base/ ← ChromaDB vector store (gitignored)
└── friendbuy-graph-db/       ← Kuzu graph database (gitignored)
```

---

## Configuration reference

All options live in `.env` (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | *(required)* | Your Anthropic API key |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `LOCAL_MODEL` | `qwen2.5:3b` | Local Qwen model for context filtering |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `CHROMA_PERSIST_DIR` | `./friendbuy-knowledge-base` | Where ChromaDB stores data |
| `REPOS_DIR` | `./repos` | Where you drop cloned repos |
| `TOP_K_RESULTS` | `5` | Chunks returned per query |
| `MIN_RELEVANCE_SCORE` | `0.30` | Minimum similarity score to include a chunk |
| `CLAUDE_MODEL` | `claude-sonnet-4-5` | Claude model for answers |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Character overlap between chunks |
| `EMBED_BATCH_SIZE` | `100` | Chunks per Ollama embed call |
| `FILE_SIZE_CAP_BYTES` | `512000` | Skip files larger than 500 KB |
| `GRAPH_DB_DIR` | `./friendbuy-graph-db` | Where Kuzu stores the graph |
| `USE_GRAPH` | `true` | Set `false` to disable graph entirely |
| `CACHE_DIR` | `./cache` | Where the delta-tracking SQLite DB lives |

---

## Checkpoint roadmap

| CP | Status | What it delivers |
|----|--------|-----------------|
| **CP0** | ✅ Done | Stable IDs · delta tracking · atomic reindex · relevance filter · async API |
| **CP1** | ✅ Done | Tree-sitter AST chunking · Kuzu schema · Repo + File graph nodes |
| **CP2** | ✅ Done | Full symbol extraction · Class/Function/APIEndpoint nodes · structural edges · unified pipeline · tests |
| CP3 | Planned | Hybrid retrieval: RRF fusion of vector + BM25 + graph traversal |
| CP4 | Planned | Cross-repo inference · semantic cache · CALLS edge population |
| CP5 | Planned | Eval harness · observability · LLM-as-judge scoring |

---

## Memory notes (M1 Air, 8 GB RAM)

- Embeddings are batched in groups of 100 to avoid OOM.
- `qwen2.5:3b` uses ~2 GB RAM; `nomic-embed-text` uses ~300 MB.
- ChromaDB stores vectors on disk — only loaded chunks stay in RAM.
- Kuzu stores the graph on disk — negligible RAM overhead.
- Quit other heavy apps (browser tabs, Docker) before indexing large repos.
- If you run out of memory during indexing, reduce `CHUNK_SIZE` to `500`.
