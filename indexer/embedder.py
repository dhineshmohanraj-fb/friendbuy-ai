"""Embed document chunks with nomic-embed-text (Ollama) and persist in ChromaDB."""

from __future__ import annotations

import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from rich.console import Console
from rich.table import Table

from config import get_settings
from indexer.delta_tracker import DeltaTracker, chunk_doc_id, file_doc_id

# Lazy console — not a module-level global so it doesn't pollute server logs
def _console() -> Console:
    return Console()


# ---------------------------------------------------------------------------
# Ollama connectivity
# ---------------------------------------------------------------------------

def _check_ollama(retries: int = 3) -> None:
    """
    Verify Ollama is reachable, with exponential backoff.

    Raises:
        SystemExit: if Ollama remains unreachable after *retries* attempts.
    """
    import httpx

    settings = get_settings()
    last_exc: Exception | None = None

    for attempt in range(retries):
        try:
            httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5).raise_for_status()
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < retries - 1:
                wait = 2 ** attempt  # 1s, 2s
                _console().print(
                    f"  [yellow]Ollama unreachable (attempt {attempt + 1}/{retries}), "
                    f"retrying in {wait}s…[/yellow]"
                )
                time.sleep(wait)

    _console().print(
        "\n[bold red]Error:[/bold red] Cannot reach Ollama at "
        f"[cyan]{get_settings().ollama_base_url}[/cyan] after {retries} attempts.\n"
        "Make sure Ollama is running:  [bold]ollama serve[/bold]\n"
        f"Detail: {last_exc}"
    )
    raise SystemExit(1) from last_exc


def _make_embeddings() -> OllamaEmbeddings:
    """Return an OllamaEmbeddings instance after confirming Ollama is up."""
    _check_ollama()
    s = get_settings()
    return OllamaEmbeddings(
        model=s.embedding_model,
        base_url=s.ollama_base_url,
    )


# ---------------------------------------------------------------------------
# Atomic reindex helper
# ---------------------------------------------------------------------------

def _atomic_replace(target: Path, source: Path) -> None:
    """
    Atomically replace *target* directory with *source*.

    Strategy:
      1. Rename existing target → ``{target}.old``
      2. Rename source          → target
      3. Delete ``{target}.old``

    If step 2 fails, ``{target}.old`` is restored.
    """
    old = target.parent / f"{target.name}.old"
    if old.exists():
        shutil.rmtree(old)

    if target.exists():
        target.rename(old)

    try:
        source.rename(target)
    except Exception:
        # Restore previous index so the system remains usable
        if old.exists():
            old.rename(target)
        raise

    if old.exists():
        shutil.rmtree(old)


# ---------------------------------------------------------------------------
# Module-level Chroma singleton (avoid re-opening on every query)
# ---------------------------------------------------------------------------

_vector_store: Chroma | None = None


def _invalidate_store() -> None:
    """Drop the cached Chroma instance (called after reindex)."""
    global _vector_store
    _vector_store = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_and_store(
    chunks: list[Document],
    reindex: bool = False,
) -> Chroma:
    """
    Embed *chunks* with Ollama and upsert them into ChromaDB.

    Uses stable SHA-256 chunk IDs so repeated runs without ``--reindex``
    are idempotent (no duplicate chunks accumulate).

    When ``reindex=True``:
      1. Embeddings are written to a *temporary* directory.
      2. On success, the temp dir atomically replaces the live index.
      3. If anything fails mid-way the previous index is intact.

    Args:
        chunks:   Pre-split Document objects (must have chunk_index in metadata).
        reindex:  If True, rebuild from scratch atomically.

    Returns:
        The populated Chroma vector store instance.
    """
    global _vector_store

    settings = get_settings()
    con = _console()

    if not chunks:
        con.print("[yellow]No chunks to embed.[/yellow]")
        raise SystemExit(0)

    embeddings = _make_embeddings()

    # Determine write target
    live_path = settings.chroma_path
    if reindex:
        write_path = live_path.parent / f"{live_path.name}.new"
        if write_path.exists():
            shutil.rmtree(write_path)
        write_path.mkdir(parents=True, exist_ok=True)
        con.print(f"[yellow]Reindex:[/yellow] writing to temp dir [dim]{write_path}[/dim]")
    else:
        live_path.mkdir(parents=True, exist_ok=True)
        write_path = live_path

    # Count chunks per repo for the summary table
    repo_counts: dict[str, int] = {}
    for chunk in chunks:
        repo = chunk.metadata.get("repo_name", "unknown")
        repo_counts[repo] = repo_counts.get(repo, 0) + 1

    con.print(
        f"\n[bold]Embedding[/bold] {len(chunks):,} chunks "
        f"using [cyan]{settings.embedding_model}[/cyan] …"
    )

    db: Chroma | None = None
    batch_size = settings.embed_batch_size

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]

        # Assign stable IDs to each chunk in this batch
        ids = [
            chunk_doc_id(
                chunk.metadata.get("repo_name", ""),
                chunk.metadata.get("file_path", ""),
                chunk.metadata.get("chunk_index", idx),
            )
            for idx, chunk in enumerate(batch, start=i)
        ]

        if db is None:
            db = Chroma.from_documents(
                documents=batch,
                ids=ids,
                embedding=embeddings,
                persist_directory=str(write_path),
                collection_name=settings.chroma_collection_name,
                collection_metadata={
                    "indexed_at": datetime.now(timezone.utc).isoformat(),
                    "embedding_model": settings.embedding_model,
                },
            )
        else:
            db.add_documents(batch, ids=ids)

        done = min(i + batch_size, len(chunks))
        con.print(f"  [dim]Embedded {done:,} / {len(chunks):,} chunks[/dim]")

    # Atomic swap when reindexing
    if reindex and write_path != live_path:
        con.print("[dim]Swapping temp index into place…[/dim]")
        _atomic_replace(live_path, write_path)
        con.print(f"[green]✓[/green] Index updated at [dim]{live_path}[/dim]")

    # Invalidate the cached store so the next query picks up fresh data
    _invalidate_store()

    # Update delta tracker — mark all indexed files as "current"
    tracker = DeltaTracker()
    if reindex:
        tracker.clear_all()

    # Build a map: file_doc_id → list of chunk IDs
    file_chunks: dict[str, list[str]] = {}
    for idx, chunk in enumerate(chunks):
        repo = chunk.metadata.get("repo_name", "")
        fpath = chunk.metadata.get("file_path", "")
        fdid = file_doc_id(repo, fpath)
        cid = chunk_doc_id(repo, fpath, chunk.metadata.get("chunk_index", idx))
        file_chunks.setdefault(fdid, []).append(cid)

    for chunk in chunks:
        repo = chunk.metadata.get("repo_name", "")
        fpath = chunk.metadata.get("file_path", "")
        fdid = chunk.metadata.get("_doc_id") or file_doc_id(repo, fpath)
        content_hash = chunk.metadata.get("_content_hash") or DeltaTracker.compute_hash(chunk.page_content)
        tracker.register(
            doc_id=fdid,
            file_path=fpath,
            repo_name=repo,
            content_hash=content_hash,
            chunk_ids=file_chunks.get(fdid, []),
        )

    # Print summary table
    table = Table(title="Indexing complete", show_header=True, header_style="bold cyan")
    table.add_column("Repo", style="cyan")
    table.add_column("Chunks", justify="right", style="green")
    for repo, count in sorted(repo_counts.items()):
        table.add_row(repo, str(count))
    table.add_row("[bold]Total[/bold]", f"[bold]{len(chunks):,}[/bold]")
    con.print(table)

    return db  # type: ignore[return-value]


def load_vector_store() -> Chroma:
    """
    Return the ChromaDB vector store, using a module-level singleton
    to avoid re-opening the database and re-pinging Ollama on every query.

    Raises:
        SystemExit: if no index has been built yet.
    """
    global _vector_store

    if _vector_store is not None:
        return _vector_store

    settings = get_settings()

    if not settings.chroma_path.exists():
        _console().print(
            "\n[bold red]No index found.[/bold red] "
            "Run [bold]python cli.py index[/bold] first to build the knowledge base."
        )
        raise SystemExit(1)

    embeddings = _make_embeddings()
    _vector_store = Chroma(
        persist_directory=str(settings.chroma_path),
        embedding_function=embeddings,
        collection_name=settings.chroma_collection_name,
    )
    return _vector_store
