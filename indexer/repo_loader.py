"""Walk REPOS_DIR and return LangChain Document objects for all supported files."""

from pathlib import Path
from typing import Generator

from langchain_core.documents import Document
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from config import SKIP_DIRS, SUPPORTED_EXTENSIONS, SUPPORTED_FILENAMES, settings


def _is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS or path.name in SUPPORTED_FILENAMES


def _iter_repo_files(repo_path: Path) -> Generator[Path, None, None]:
    """Yield all supported files under *repo_path*, skipping ignored dirs."""
    for item in repo_path.rglob("*"):
        if item.is_dir():
            continue
        # Skip if any parent component is a skip dir
        if any(part in SKIP_DIRS for part in item.parts):
            continue
        if _is_supported(item):
            yield item


def _read_safe(path: Path) -> str | None:
    """Return file text, or None if binary / unreadable."""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None


def load_repos(repos_dir: Path | None = None) -> list[Document]:
    """
    Load all supported source files from every subdirectory of *repos_dir*.

    Each subdirectory is treated as a separate repo.  Returns a flat list of
    LangChain Documents with rich metadata.
    """
    base = repos_dir or settings.repos_path
    base = Path(base)

    if not base.exists():
        base.mkdir(parents=True, exist_ok=True)

    repo_dirs = [d for d in base.iterdir() if d.is_dir() and d.name not in SKIP_DIRS]

    if not repo_dirs:
        from rich.console import Console
        Console().print(
            f"[yellow]Warning:[/yellow] No repo folders found in [bold]{base}[/bold]. "
            "Drop cloned repos there and re-run."
        )
        return []

    documents: list[Document] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        repo_task = progress.add_task("Scanning repos...", total=len(repo_dirs))

        for repo_dir in repo_dirs:
            repo_name = repo_dir.name
            files = list(_iter_repo_files(repo_dir))

            if not files:
                progress.console.print(
                    f"[yellow]⚠ Skipping[/yellow] [bold]{repo_name}[/bold] — no supported files found."
                )
                progress.advance(repo_task)
                continue

            file_task = progress.add_task(
                f"  [cyan]{repo_name}[/cyan]", total=len(files)
            )

            loaded = 0
            for file_path in files:
                content = _read_safe(file_path)
                progress.advance(file_task)

                if content is None or not content.strip():
                    continue

                rel_path = file_path.relative_to(base)
                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "repo_name": repo_name,
                            "file_path": str(rel_path),
                            "file_name": file_path.name,
                            "file_type": file_path.suffix.lower() or file_path.name,
                        },
                    )
                )
                loaded += 1

            progress.console.print(
                f"  [green]✓[/green] [bold]{repo_name}[/bold] — {loaded} files loaded"
            )
            progress.advance(repo_task)

    return documents
