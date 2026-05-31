"""FastAPI server exposing the friendbuy-ai RAG pipeline over HTTP."""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import get_settings


# ---------------------------------------------------------------------------
# In-memory index job tracker (reset on process restart — acceptable for CP0)
# ---------------------------------------------------------------------------

_index_jobs: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    query: str
    repo: str | None = None
    top_k: int | None = None


class AskResponse(BaseModel):
    answer: str
    relevant_files: list[str]
    input_tokens: int
    output_tokens: int
    request_id: str


class IndexAcceptedResponse(BaseModel):
    status: str
    job_id: str
    message: str


class IndexJobStatus(BaseModel):
    status: str                  # accepted | running | completed | failed
    chunks_indexed: int | None = None
    error: str | None = None
    started_at: str | None = None
    completed_at: str | None = None


class StatsResponse(BaseModel):
    total_chunks: int
    total_repos: int
    unique_files: int
    repos: dict[str, int]
    indexed_at: str | None


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    ollama_url: str
    claude_model: str


class ReadyResponse(BaseModel):
    status: str          # "ready" | "not_ready"
    checks: dict[str, str]


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    if not settings.anthropic_api_key:
        import warnings
        warnings.warn(
            "ANTHROPIC_API_KEY is not set — /ask endpoint will return 503.",
            stacklevel=1,
        )
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="friendbuy-ai",
    description="RAG knowledge pipeline for the Friendbuy codebase",
    version="0.2.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Middleware: attach X-Request-ID to every response
# ---------------------------------------------------------------------------

@app.middleware("http")
async def add_request_id(request: Request, call_next) -> Response:
    request_id = str(uuid.uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ---------------------------------------------------------------------------
# System endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    """Liveness check — always 200 if the process is up."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
        ollama_url=settings.ollama_base_url,
        claude_model=settings.claude_model,
    )


@app.get("/ready", tags=["system"])
async def ready() -> JSONResponse:
    """
    Readiness probe.

    Returns 200 when the ChromaDB index exists **and** Ollama is reachable.
    Returns 503 otherwise so load balancers / k8s can withhold traffic.
    """
    settings = get_settings()
    checks: dict[str, str] = {}

    # --- ChromaDB index ---
    checks["chroma_index"] = (
        "ok" if settings.chroma_path.exists() else "not_built — run 'python cli.py index'"
    )

    # --- Ollama ---
    try:
        import httpx
        httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=3).raise_for_status()
        checks["ollama"] = "ok"
    except Exception as exc:
        checks["ollama"] = f"unreachable ({exc})"

    # --- Anthropic API key ---
    checks["anthropic_api_key"] = "configured" if settings.anthropic_api_key else "missing"

    is_ready = (
        checks["chroma_index"] == "ok"
        and checks["ollama"] == "ok"
    )

    return JSONResponse(
        status_code=200 if is_ready else 503,
        content=ReadyResponse(
            status="ready" if is_ready else "not_ready",
            checks=checks,
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# Stats endpoint
# ---------------------------------------------------------------------------

@app.get("/stats", response_model=StatsResponse, tags=["index"])
async def stats() -> StatsResponse:
    """Return knowledge-base index statistics."""
    try:
        from indexer.embedder import load_vector_store

        db = load_vector_store()
        collection = db._collection  # type: ignore[attr-defined]
        count = collection.count()
        meta_result = collection.get(include=["metadatas"])
        all_meta = meta_result.get("metadatas") or []

        repo_counts: dict[str, int] = {}
        file_set: set[str] = set()
        for m in all_meta:
            if m:
                repo = m.get("repo_name", "unknown")
                repo_counts[repo] = repo_counts.get(repo, 0) + 1
                fp = m.get("file_path")
                if fp:
                    file_set.add(fp)

        col_meta = collection.metadata or {}
        return StatsResponse(
            total_chunks=count,
            total_repos=len(repo_counts),
            unique_files=len(file_set),
            repos=repo_counts,
            indexed_at=col_meta.get("indexed_at"),
        )
    except SystemExit:
        raise HTTPException(
            status_code=503,
            detail="Index not found. Run 'python cli.py index' first.",
        )


# ---------------------------------------------------------------------------
# Index endpoints
# ---------------------------------------------------------------------------

async def _run_index_bg(job_id: str, reindex: bool) -> None:
    """Background task: runs the full indexing pipeline in a thread pool."""
    _index_jobs[job_id]["status"] = "running"
    _index_jobs[job_id]["started_at"] = datetime.now(timezone.utc).isoformat()

    try:
        from indexer.embedder import embed_and_store
        from indexer.repo_loader import load_repos
        from indexer.splitter import split_documents

        docs = await asyncio.to_thread(load_repos)
        if not docs:
            _index_jobs[job_id].update({"status": "failed", "error": "No documents found in repos dir."})
            return

        chunks = await asyncio.to_thread(split_documents, docs)
        await asyncio.to_thread(embed_and_store, chunks, reindex)

        _index_jobs[job_id].update(
            {
                "status": "completed",
                "chunks_indexed": len(chunks),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    except SystemExit as exc:
        # SystemExit is raised by embedder/loader on service errors
        _index_jobs[job_id].update({"status": "failed", "error": f"Service error (code {exc.code}). Check server logs."})
    except Exception as exc:
        _index_jobs[job_id].update({"status": "failed", "error": str(exc)})


@app.post("/index", response_model=IndexAcceptedResponse, status_code=202, tags=["index"])
async def trigger_index(
    background_tasks: BackgroundTasks,
    reindex: bool = False,
) -> IndexAcceptedResponse:
    """
    Start (re-)indexing of the repos directory.

    The job runs in the background — returns immediately with a
    ``job_id`` you can poll via ``GET /index/status/{job_id}``.
    """
    job_id = str(uuid.uuid4())
    _index_jobs[job_id] = {"status": "accepted"}
    background_tasks.add_task(_run_index_bg, job_id, reindex)
    return IndexAcceptedResponse(
        status="accepted",
        job_id=job_id,
        message=f"Indexing started. Poll GET /index/status/{job_id} for progress.",
    )


@app.get("/index/status/{job_id}", response_model=IndexJobStatus, tags=["index"])
async def index_status(job_id: str) -> IndexJobStatus:
    """Poll the status of an indexing job."""
    if job_id not in _index_jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return IndexJobStatus(**_index_jobs[job_id])


# ---------------------------------------------------------------------------
# Ask endpoint
# ---------------------------------------------------------------------------

@app.post("/ask", response_model=AskResponse, tags=["query"])
async def ask(body: AskRequest, request: Request) -> AskResponse:
    """Answer a question using the full RAG pipeline."""
    settings = get_settings()
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

    if not settings.anthropic_api_key:
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY is not configured on the server.",
        )

    try:
        from pipeline.query_pipeline import run

        result = await asyncio.to_thread(
            run,
            body.query,
            body.repo,
            body.top_k,
        )
        return AskResponse(
            answer=result.answer,
            relevant_files=result.relevant_files,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            request_id=request_id,
        )
    except SystemExit as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Pipeline service error (code {exc.code}). Check server logs.",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def start(host: str | None = None, port: int | None = None, reload: bool = False) -> None:
    settings = get_settings()
    uvicorn.run(
        "api.server:app",
        host=host or settings.api_host,
        port=port or settings.api_port,
        reload=reload,
    )


if __name__ == "__main__":
    start()
