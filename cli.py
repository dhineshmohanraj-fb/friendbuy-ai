#!/usr/bin/env python3
"""
friendbuy-ai CLI

Usage:
    python cli.py index                              # Index all repos
    python cli.py index --reindex                   # Wipe and rebuild index
    python cli.py ask "your question"               # Ask a question
    python cli.py ask "your question" --repo NAME   # Scope to one repo
    python cli.py stats                             # Show index statistics
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.status import Status
from rich.table import Table

console = Console()


# ---------------------------------------------------------------------------
# index command
# ---------------------------------------------------------------------------

def cmd_index(reindex: bool, no_graph: bool = False) -> None:
    from pipeline.index_pipeline import IndexPipeline

    console.print(Rule("[bold cyan]friendbuy-ai — Indexer[/bold cyan]"))

    if reindex:
        console.print("[yellow]--reindex flag set: existing index will be wiped.[/yellow]\n")
    if no_graph:
        console.print("[dim]--no-graph: skipping knowledge graph update.[/dim]\n")

    result = IndexPipeline().run(reindex=reindex, no_graph=no_graph)

    if result.changed_files == 0 and result.total_files_scanned > 0:
        return  # Pipeline already printed "up-to-date" message

    # Graph summary table
    if result.graph:
        from rich.table import Table
        tbl = Table(title="Graph nodes indexed", show_header=True, header_style="bold cyan")
        tbl.add_column("Type", style="cyan")
        tbl.add_column("Count", justify="right", style="green")
        for label in ("classes", "functions", "endpoints"):
            cnt = result.graph.get(label, 0)
            if cnt:
                tbl.add_row(label.capitalize(), f"{cnt:,}")
        edge_cnt = result.graph.get("edges", 0)
        if edge_cnt:
            tbl.add_row("[dim]edges created[/dim]", f"[dim]{edge_cnt:,}[/dim]")
        if tbl.row_count:
            console.print(tbl)


# ---------------------------------------------------------------------------
# ask command
# ---------------------------------------------------------------------------

def cmd_ask(
    question: str,
    repo: str | None,
    no_graph: bool = False,
    no_bm25: bool = False,
) -> None:
    from pipeline.query_pipeline import run

    console.print(Rule("[bold cyan]friendbuy-ai — Query[/bold cyan]"))
    console.print(f"[dim]Question:[/dim] {question}")
    if repo:
        console.print(f"[dim]Scoped to repo:[/dim] [cyan]{repo}[/cyan]")
    flags = []
    if no_graph:
        flags.append("no-graph")
    if no_bm25:
        flags.append("no-bm25")
    if flags:
        console.print(f"[dim]Retrieval flags:[/dim] {', '.join(flags)}")
    console.print()

    with Status("[bold]Retrieving context…[/bold]", spinner="dots", console=console):
        result = run(
            query=question,
            repo_name=repo,
            use_graph=not no_graph,
            use_bm25=not no_bm25,
        )

    # CP4: cache hit indicator
    if result.cache_hit:
        console.print(
            f"[bold green]⚡ Cache hit[/bold green] "
            f"[dim](similarity {result.cache_similarity:.3f})[/dim]"
        )

    # CP3 retrieval breakdown
    retrieval_parts = [f"vector:[cyan]{result.vector_count}[/cyan]"]
    if result.bm25_count:
        retrieval_parts.append(f"BM25:[cyan]{result.bm25_count}[/cyan]")
    if result.graph_count:
        retrieval_parts.append(f"graph:[cyan]{result.graph_count}[/cyan]")
    if result.query_entities:
        entities_str = ", ".join(f"[cyan]{e}[/cyan]" for e in result.query_entities[:4])
        retrieval_parts.append(f"entities:{entities_str}")
    if not result.cache_hit:
        console.print(
            f"[dim]Retrieval ({result.retrieval_ms:.0f}ms):[/dim] "
            + "  ".join(retrieval_parts)
        )
    console.print()

    # Files used
    if result.relevant_files:
        console.print("[bold]Context files used:[/bold]")
        for f in result.relevant_files:
            console.print(f"  [cyan]•[/cyan] {f}")
        console.print()

    # Answer panel
    console.print(
        Panel(
            Markdown(result.answer),
            title="[bold green]Claude's Answer[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )

    # Token usage
    console.print(
        f"\n[dim]Token usage — "
        f"input: [cyan]{result.input_tokens:,}[/cyan]  "
        f"output: [cyan]{result.output_tokens:,}[/cyan]  "
        f"total: [cyan]{result.input_tokens + result.output_tokens:,}[/cyan][/dim]"
    )


# ---------------------------------------------------------------------------
# stats command
# ---------------------------------------------------------------------------

def cmd_stats() -> None:
    from config import settings
    from indexer.embedder import load_vector_store

    console.print(Rule("[bold cyan]friendbuy-ai — Index Stats[/bold cyan]"))

    db = load_vector_store()
    collection = db._collection  # access underlying chromadb collection

    count = collection.count()
    metadata = collection.metadata or {}

    # Fetch metadata in batches to avoid ChromaDB "too many SQL variables" error
    BATCH = 2000
    repo_counts: dict[str, int] = {}
    file_set: set[str] = set()
    offset = 0
    while True:
        batch = collection.get(include=["metadatas"], limit=BATCH, offset=offset)
        metas = batch.get("metadatas") or []
        if not metas:
            break
        for m in metas:
            if m:
                repo = m.get("repo_name", "unknown")
                repo_counts[repo] = repo_counts.get(repo, 0) + 1
                fp = m.get("file_path")
                if fp:
                    file_set.add(fp)
        offset += BATCH
        if len(metas) < BATCH:
            break

    table = Table(show_header=True, header_style="bold cyan", title="Knowledge Base Stats")
    table.add_column("Repo", style="cyan")
    table.add_column("Chunks", justify="right", style="green")

    for repo, cnt in sorted(repo_counts.items()):
        table.add_row(repo, f"{cnt:,}")

    table.add_section()
    table.add_row("[bold]Total repos[/bold]", str(len(repo_counts)))
    table.add_row("[bold]Total chunks[/bold]", f"{count:,}")
    table.add_row("[bold]Unique files[/bold]", f"{len(file_set):,}")

    console.print(table)

    if "indexed_at" in metadata:
        console.print(f"\n[dim]Last indexed: {metadata['indexed_at']}[/dim]")

    console.print(f"[dim]Persist dir: {settings.chroma_path.resolve()}[/dim]")


# ---------------------------------------------------------------------------
# graph-stats command
# ---------------------------------------------------------------------------

def cmd_graph_stats() -> None:
    from config import get_settings

    console.print(Rule("[bold cyan]friendbuy-ai — Graph Stats[/bold cyan]"))

    settings = get_settings()
    if not settings.graph_db_path.exists():
        console.print(
            "[yellow]No graph index found.[/yellow] "
            "Run [bold]python cli.py index[/bold] to build it."
        )
        return

    try:
        from indexer.graph_builder import GraphBuilder

        with GraphBuilder() as gb:
            stats = gb.graph_stats()

        # Nodes table
        node_table = Table(title="Graph Nodes", show_header=True, header_style="bold cyan")
        node_table.add_column("Node type", style="cyan")
        node_table.add_column("Count", justify="right", style="green")
        for label in ["Repo", "File", "Class", "Function", "APIEndpoint"]:
            node_table.add_row(label, f"{stats.get(label, 0):,}")
        console.print(node_table)

        # Edges table
        edge_table = Table(title="Graph Edges", show_header=True, header_style="bold cyan")
        edge_table.add_column("Relationship", style="cyan")
        edge_table.add_column("Count", justify="right", style="green")
        for rel in ["BELONGS_TO_REPO", "CONTAINS_CLASS", "CONTAINS_FUNCTION",
                    "METHOD_OF", "IMPORT_DEP", "CALLS", "EXPOSES",
                    "HANDLES", "INHERITS", "CROSS_REPO_CALL"]:
            cnt = stats.get(rel, 0)
            style = "green" if cnt > 0 else "dim"
            edge_table.add_row(rel, f"[{style}]{cnt:,}[/{style}]")
        console.print(edge_table)

        console.print(f"\n[dim]Graph DB: {settings.graph_db_path.resolve()}[/dim]")

    except ImportError:
        console.print(
            "[red]Kuzu not installed.[/red] "
            "Run [bold]pip install kuzu>=0.6.0[/bold]"
        )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="friendbuy-ai — RAG knowledge pipeline for the Friendbuy codebase",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # index
    p_index = sub.add_parser("index", help="Index repos into ChromaDB")
    p_index.add_argument(
        "--reindex",
        action="store_true",
        help="Wipe existing index and rebuild from scratch",
    )
    p_index.add_argument(
        "--no-graph",
        action="store_true",
        dest="no_graph",
        help="Skip knowledge graph update (vector index only)",
    )

    # ask
    p_ask = sub.add_parser("ask", help="Ask a question about the codebase")
    p_ask.add_argument("question", type=str, help="Your question")
    p_ask.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Scope the search to a single repo by name",
    )
    p_ask.add_argument(
        "--no-graph",
        action="store_true",
        dest="no_graph",
        help="Disable graph traversal — vector + BM25 only",
    )
    p_ask.add_argument(
        "--no-bm25",
        action="store_true",
        dest="no_bm25",
        help="Disable BM25 sparse search — vector (+ graph) only",
    )

    # stats
    sub.add_parser("stats", help="Show vector index statistics")

    # graph-stats
    sub.add_parser("graph-stats", help="Show knowledge graph statistics (CP1)")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "index":
            cmd_index(reindex=args.reindex, no_graph=args.no_graph)
        elif args.command == "ask":
            cmd_ask(
                question=args.question,
                repo=args.repo,
                no_graph=getattr(args, "no_graph", False),
                no_bm25=getattr(args, "no_bm25", False),
            )
        elif args.command == "stats":
            cmd_stats()
        elif args.command == "graph-stats":
            cmd_graph_stats()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    main()
