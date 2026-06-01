"""
Unified indexing pipeline — CP2.

Orchestrates the full index flow:
  1. Load source files from ``repos/``
  2. Delta-filter to only changed / new files
  3. AST-aware chunking (tree-sitter)
  4. Embed & persist in ChromaDB  +  upsert Repo/File graph nodes
  5. Full symbol extraction  →  upsert Class / Function / APIEndpoint nodes
     and structural edges in Kuzu

Usage::

    from pipeline.index_pipeline import IndexPipeline

    result = IndexPipeline().run(reindex=False)
    print(result)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from langchain_core.documents import Document
from rich.console import Console


def _console() -> Console:
    return Console()


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class IndexResult:
    """Summary of what happened during one index run."""

    total_files_scanned: int  = 0
    changed_files:       int  = 0
    skipped_files:       int  = 0
    total_chunks:        int  = 0
    ast_chunks:          int  = 0
    char_chunks:         int  = 0
    graph:               dict = field(default_factory=dict)   # node / edge counts
    elapsed_seconds:     float = 0.0

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"IndexResult("
            f"files={self.changed_files}/{self.total_files_scanned}, "
            f"chunks={self.total_chunks} [{self.ast_chunks} AST / {self.char_chunks} char], "
            f"graph={self.graph}, "
            f"elapsed={self.elapsed_seconds:.1f}s)"
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class IndexPipeline:
    """
    End-to-end indexing orchestrator.

    Instantiate and call ``run()``.  All heavy work runs synchronously so
    callers can wrap it in ``asyncio.to_thread()`` for non-blocking operation.
    """

    def run(
        self,
        reindex: bool = False,
        no_graph: bool = False,
    ) -> IndexResult:
        """
        Execute the full indexing pipeline.

        Args:
            reindex:  If True, wipe all existing indexes and rebuild from scratch.
            no_graph: If True, skip Kuzu graph updates (vector index only).

        Returns:
            An :class:`IndexResult` with per-phase counts and timing.
        """
        from config import get_settings
        from indexer.delta_tracker import DeltaTracker
        from indexer.embedder import embed_and_store
        from indexer.repo_loader import load_repos
        from indexer.splitter import split_documents

        start   = time.time()
        con     = _console()
        settings = get_settings()

        # ------------------------------------------------------------------
        # Step 1: Load all source files
        # ------------------------------------------------------------------
        con.print("[bold]Step 1/4[/bold]  Loading files from repos…")
        documents: list[Document] = load_repos()
        total_files = len(documents)

        if not documents:
            con.print("[red]No documents found in repos/.[/red]")
            return IndexResult(elapsed_seconds=time.time() - start)

        # ------------------------------------------------------------------
        # Step 2: Delta filter — only changed / new files
        # ------------------------------------------------------------------
        tracker = DeltaTracker()
        if reindex:
            tracker.clear_all()
            changed_docs = documents
        else:
            changed_docs = tracker.filter_changed(documents)

        skipped   = total_files - len(changed_docs)
        con.print(
            f"  → [cyan]{len(changed_docs):,}[/cyan] changed, "
            f"[dim]{skipped:,}[/dim] unchanged"
        )

        if not changed_docs:
            con.print("[green]✓ Everything up-to-date — nothing to embed.[/green]")
            return IndexResult(
                total_files_scanned=total_files,
                changed_files=0,
                skipped_files=skipped,
                elapsed_seconds=time.time() - start,
            )

        # ------------------------------------------------------------------
        # Step 3: Split into embedding-ready chunks
        # ------------------------------------------------------------------
        con.print(
            f"\n[bold]Step 2/4[/bold]  Chunking "
            f"{len(changed_docs):,} changed files…"
        )
        chunks = split_documents(changed_docs)
        ast_chunks  = sum(1 for c in chunks if "symbol_name" in c.metadata)
        char_chunks = len(chunks) - ast_chunks
        con.print(
            f"  → {len(chunks):,} chunks  "
            f"([cyan]{ast_chunks:,}[/cyan] AST-aware, "
            f"[dim]{char_chunks:,}[/dim] character-split)"
        )

        # ------------------------------------------------------------------
        # Step 4: Embed & store in ChromaDB (also upserts Repo + File nodes)
        # ------------------------------------------------------------------
        con.print("\n[bold]Step 3/4[/bold]  Embedding & storing in ChromaDB…")
        if reindex:
            # Override USE_GRAPH for the embed_and_store call so its internal
            # _update_graph does the clear + Repo/File upsert correctly.
            pass

        # Pass no_graph by temporarily overriding the setting
        original_use_graph = settings.use_graph
        if no_graph:
            import os
            os.environ["USE_GRAPH"] = "false"
            from config import get_settings as _gs
            _gs.cache_clear()

        try:
            embed_and_store(chunks, reindex=reindex)
        finally:
            if no_graph and not original_use_graph:
                pass  # Already false — no restore needed
            elif no_graph:
                import os
                os.environ["USE_GRAPH"] = "true"
                from config import get_settings as _gs
                _gs.cache_clear()

        # ------------------------------------------------------------------
        # Step 5: Full symbol extraction → Class/Function/APIEndpoint nodes
        # ------------------------------------------------------------------
        graph_counts: dict = {}
        if not no_graph and settings.use_graph:
            con.print("\n[bold]Step 4/4[/bold]  Extracting symbols → knowledge graph…")
            graph_counts, file_node_map = _populate_symbol_graph(
                changed_docs, reindex=reindex
            )
            con.print(
                f"  → classes: [cyan]{graph_counts.get('classes', 0):,}[/cyan]  "
                f"functions: [cyan]{graph_counts.get('functions', 0):,}[/cyan]  "
                f"endpoints: [cyan]{graph_counts.get('endpoints', 0):,}[/cyan]  "
                f"edges: [cyan]{graph_counts.get('edges', 0):,}[/cyan]"
            )

            # Persist graph_node_ids in the delta tracker
            from indexer.delta_tracker import file_doc_id
            for doc in changed_docs:
                repo  = doc.metadata.get("repo_name", "")
                fpath = doc.metadata.get("file_path", "")
                if not fpath:
                    continue
                fid      = file_doc_id(repo, fpath)
                node_ids = file_node_map.get(fid, [])
                if node_ids:
                    tracker.update_graph_node_ids(fid, node_ids)

        elapsed = time.time() - start
        con.print(f"\n[bold green]✓ Done in {elapsed:.1f}s[/bold green]")

        return IndexResult(
            total_files_scanned=total_files,
            changed_files=len(changed_docs),
            skipped_files=skipped,
            total_chunks=len(chunks),
            ast_chunks=ast_chunks,
            char_chunks=char_chunks,
            graph=graph_counts,
            elapsed_seconds=elapsed,
        )


# ---------------------------------------------------------------------------
# Symbol extraction helper
# ---------------------------------------------------------------------------

def _populate_symbol_graph(
    documents: list[Document],
    reindex: bool,
) -> tuple[dict[str, int], dict[str, list[str]]]:
    """
    Extract Class / Function / APIEndpoint nodes + edges for every document
    and upsert them into Kuzu.

    Args:
        documents: Changed source-file Documents (one per file).
        reindex:   If True the graph was already cleared by embed_and_store;
                   we must NOT clear it again.

    Returns:
        (aggregate_counts, file_id → [node_id, ...])
    """
    agg: dict[str, int] = {
        "classes": 0, "functions": 0, "endpoints": 0, "edges": 0
    }
    file_node_map: dict[str, list[str]] = {}

    try:
        from indexer.ast_parser import extract_file_symbols
        from indexer.delta_tracker import file_doc_id
        from indexer.graph_builder import GraphBuilder
    except ImportError:
        return agg, file_node_map

    try:
        with GraphBuilder() as gb:
            for doc in documents:
                repo  = doc.metadata.get("repo_name", "")
                fpath = doc.metadata.get("file_path", "")
                if not fpath:
                    continue

                fid = file_doc_id(repo, fpath)

                # Incremental: delete stale symbols before upserting fresh ones.
                # On full reindex the graph was already cleared by embed_and_store,
                # so skip the per-file delete (no stale nodes exist).
                if not reindex:
                    gb.delete_file_symbols(fpath, repo)

                node_batch, edge_batch = extract_file_symbols(
                    fpath, doc.page_content, repo
                )

                if node_batch.total() == 0:
                    continue

                counts = gb.upsert_symbols_from_batch(
                    fid, node_batch, edge_batch, repo
                )
                for k, v in counts.items():
                    agg[k] = agg.get(k, 0) + v

                file_node_map[fid] = node_batch.all_node_ids()

    except ImportError:
        _console().print(
            "[yellow]Warning:[/yellow] Kuzu not installed — skipping symbol graph."
        )
    except Exception as exc:  # noqa: BLE001
        _console().print(
            f"[yellow]Warning:[/yellow] Symbol graph extraction failed (non-fatal): {exc}"
        )

    return agg, file_node_map
