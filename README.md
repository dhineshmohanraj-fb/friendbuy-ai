# friendbuy-ai

A two-layer RAG pipeline that lets you query the Friendbuy codebase using
natural language.  A local **Qwen** model (via Ollama) retrieves and
curates context; **Claude** (via the Anthropic API) reasons over it and
produces precise, code-aware answers.

```
┌──────────────────────────────────────────────────────────────────────┐
│                        friendbuy-ai pipeline                         │
│                                                                      │
│  ./repos/          ┌──────────┐   chunks   ┌──────────────────────┐ │
│  ├── api/          │  Indexer │──────────▶ │  ChromaDB (local)    │ │
│  ├── payments/     │  (once)  │            │  nomic-embed-text    │ │
│  └── widgets/      └──────────┘            └──────────┬───────────┘ │
│                                                        │             │
│  User question                                         │ top-k chunks│
│       │            ┌──────────────────────────────────▼───────────┐ │
│       └──────────▶ │  Retriever                                   │ │
│                    │  1. vector_search  → similarity search        │ │
│                    │  2. context_filter → Qwen (local) cleans it  │ │
│                    └──────────────────────────┬────────────────────┘ │
│                                               │ clean context        │
│                    ┌──────────────────────────▼────────────────────┐ │
│                    │  Claude (claude-opus-4-5)                     │ │
│                    │  Anthropic API — reasoning + code generation  │ │
│                    └──────────────────────────┬────────────────────┘ │
│                                               │ answer               │
│                                           CLI / API                  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.11+ | [python.org](https://python.org) |
| Ollama | latest | `brew install ollama` |
| Anthropic API key | — | [console.anthropic.com](https://console.anthropic.com) |

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

# 2. Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Open .env and set ANTHROPIC_API_KEY=sk-ant-…
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

Each top-level folder inside `./repos/` becomes a named "repo" that you
can scope queries to.

---

## CLI Usage

### Build the knowledge base

```bash
# Index everything inside ./repos/
python cli.py index

# Wipe the existing index and rebuild from scratch
python cli.py index --reindex
```

### Ask questions

```bash
# Ask a question across the entire codebase
python cli.py ask "How does the referral tracking flow work?"

# Scope the search to a single repo
python cli.py ask "Where is the Stripe webhook handler?" --repo payments-service

# Multi-word questions (always quote them)
python cli.py ask "What database migrations exist for the users table?"
```

### Show index statistics

```bash
python cli.py stats
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
| `GET` | `/health` | Liveness check |
| `GET` | `/stats` | Index statistics |
| `POST` | `/index?reindex=false` | Trigger (re-)indexing |
| `POST` | `/ask` | Ask a question |

#### Example — ask via curl

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How does referral attribution work?", "repo": null}'
```

#### Example — scope to a repo

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "List all API endpoints", "repo": "api"}'
```

---

## Project structure

```
friendbuy-ai/
├── .env.example          ← copy to .env and fill in your API key
├── requirements.txt
├── config.py             ← all settings (reads from .env)
├── cli.py                ← CLI entry point
│
├── indexer/
│   ├── repo_loader.py    ← walks ./repos/, returns LangChain Documents
│   ├── splitter.py       ← language-aware chunking
│   └── embedder.py       ← nomic-embed-text → ChromaDB
│
├── retriever/
│   ├── vector_search.py  ← similarity search with optional repo filter
│   └── context_filter.py ← Qwen (local) curates + summarises chunks
│
├── pipeline/
│   └── query_pipeline.py ← orchestrates retriever + Claude API call
│
├── api/
│   └── server.py         ← FastAPI server
│
└── repos/                ← drop cloned Friendbuy repos here
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
| `CLAUDE_MODEL` | `claude-opus-4-5` | Claude model for answers |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Character overlap between chunks |

---

## Memory notes (M1 Air, 8 GB RAM)

- Embeddings are batched in groups of 100 to avoid OOM.
- `qwen2.5:3b` uses ~2 GB RAM; `nomic-embed-text` uses ~300 MB.
- ChromaDB stores vectors on disk — only loaded chunks stay in RAM.
- Quit other heavy apps (browser tabs, Docker) before indexing large repos.
- If you run out of memory during indexing, reduce `CHUNK_SIZE` to `500`.
# friendbuy-ai
