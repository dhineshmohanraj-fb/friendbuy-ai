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

def cmd_index(reindex: bool) -> None:
    from indexer.embedder import embed_and_store
    from indexer.repo_loader import load_repos
    from indexer.splitter import split_documents

    console.print(Rule("[bold cyan]friendbuy-ai — Indexer[/bold cyan]"))

    if reindex:
        console.print("[yellow]--reindex flag set: existing index will be wiped.[/yellow]\n")

    console.print("[bold]Step 1/3[/bold]  Loading files from repos…\n")
    documents = load_repos()

    if not documents:
        console.print("[red]No documents loaded. Aborting.[/red]")
        sys.exit(1)

    console.print(f"\n[bold]Step 2/3[/bold]  Splitting {len(documents):,} documents into chunks…")
    chunks = split_documents(documents)
    console.print(f"  → {len(chunks):,} chunks created\n")

    console.print("[bold]Step 3/3[/bold]  Embedding & storing in ChromaDB…\n")
    embed_and_store(chunks, reindex=reindex)

    console.print("\n[bold green]✓ Indexing complete![/bold green]")


# ---------------------------------------------------------------------------
# ask command
# ---------------------------------------------------------------------------

def cmd_ask(question: str, repo: str | None) -> None:
    from pipeline.query_pipeline import run

    console.print(Rule("[bold cyan]friendbuy-ai — Query[/bold cyan]"))
    console.print(f"[dim]Question:[/dim] {question}")
    if repo:
        console.print(f"[dim]Scoped to repo:[/dim] [cyan]{repo}[/cyan]")
    console.print()

    with Status("[bold]Retrieving context…[/bold]", spinner="dots", console=console):
        result = run(query=question, repo_name=repo)

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

    # Fetch all metadata to compute per-repo stats
    all_meta = collection.get(include=["metadatas"])["metadatas"] or []
    repo_counts: dict[str, int] = {}
    file_set: set[str] = set()
    for m in all_meta:
        if m:
            repo = m.get("repo_name", "unknown")
            repo_counts[repo] = repo_counts.get(repo, 0) + 1
            fp = m.get("file_path")
            if fp:
                file_set.add(fp)

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

    # ask
    p_ask = sub.add_parser("ask", help="Ask a question about the codebase")
    p_ask.add_argument("question", type=str, help="Your question")
    p_ask.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Scope the search to a single repo by name",
    )

    # stats
    sub.add_parser("stats", help="Show index statistics")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "index":
            cmd_index(reindex=args.reindex)
        elif args.command == "ask":
            cmd_ask(question=args.question, repo=args.repo)
        elif args.command == "stats":
            cmd_stats()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    main()
