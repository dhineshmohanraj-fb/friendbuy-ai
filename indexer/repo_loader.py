"""Walk REPOS_DIR and return LangChain Document objects for all supported files."""

from __future__ import annotations

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

from config import SKIP_DIRS, SUPPORTED_EXTENSIONS, SUPPORTED_FILENAMES, get_settings
from indexer.delta_tracker import DeltaTracker


def _is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS or path.name in SUPPORTED_FILENAMES


def _iter_repo_files(repo_path: Path) -> Generator[Path, None, None]:
    """Yield all supported files under *repo_path*, skipping ignored dirs."""
    for item in repo_path.rglob("*"):
        if item.is_dir():
            continue
        if any(part in SKIP_DIRS for part in item.parts):
            continue
        if _is_supported(item):
            yield item


def _read_safe(path: Path, size_cap: int) -> str | None:
    """
    Return file text, or None if the file is binary, unreadable, or oversized.

    Files larger than *size_cap* bytes are skipped to protect M1 RAM during
    embedding.
    """
    try:
        size = path.stat().st_size
        if size > size_cap:
            return None
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None


def _get_git_metadata(file_abs_path: Path, repo_root: Path) -> dict:
    """
    Return the last-commit metadata for *file_abs_path* via gitpython.

    Returns an empty dict silently if the directory is not a git repo or
    gitpython is not installed.
    """
    try:
        import git  # gitpython

        repo = git.Repo(repo_root, search_parent_directories=False)
        rel = file_abs_path.relative_to(repo_root)
        commits = list(repo.iter_commits(paths=str(rel), max_count=1))
        if commits:
            c = commits[0]
            return {
                "git_commit_sha":       c.hexsha[:8],
                "git_last_modified_at": c.committed_datetime.isoformat(),
                "git_last_modified_by": c.author.name,
            }
    except Exception:  # noqa: BLE001
        pass
    return {}


def load_repos(repos_dir: Path | None = None) -> list[Document]:
    """
    Load all supported source files from every subdirectory of *repos_dir*.

    CP1 additions vs CP0:
    - Files larger than ``settings.file_size_cap_bytes`` are silently skipped.
    - ``content_hash`` and ``size_bytes`` are added to every Document's metadata
      (used by the delta tracker and graph builder).
    - Optional git metadata (last commit SHA, author, timestamp) is attached
      when the repo directory is a valid git repository.

    Each subdirectory is treated as a separate repo.  Returns a flat list of
    LangChain Documents with rich metadata.
    """
    settings = get_settings()
    base = Path(repos_dir or settings.repos_path)

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
    cap = settings.file_size_cap_bytes

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        repo_task = progress.add_task("Scanning repos…", total=len(repo_dirs))

        for repo_dir in repo_dirs:
            repo_name = repo_dir.name
            files     = list(_iter_repo_files(repo_dir))

            if not files:
                progress.console.print(
                    f"[yellow]⚠ Skipping[/yellow] [bold]{repo_name}[/bold] "
                    "— no supported files found."
                )
                progress.advance(repo_task)
                continue

            file_task = progress.add_task(
                f"  [cyan]{repo_name}[/cyan]", total=len(files)
            )

            loaded = skipped_size = 0

            for file_path in files:
                content = _read_safe(file_path, cap)
                progress.advance(file_task)

                if content is None:
                    skipped_size += 1
                    continue
                if not content.strip():
                    continue

                rel_path     = file_path.relative_to(base)
                content_hash = DeltaTracker.compute_hash(content)
                git_meta     = _get_git_metadata(file_path, repo_dir)

                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "repo_name":    repo_name,
                            "file_path":    str(rel_path),
                            "file_name":    file_path.name,
                            "file_type":    file_path.suffix.lower() or file_path.name,
                            # CP1 additions
                            "content_hash": content_hash,
                            "size_bytes":   len(content.encode("utf-8", errors="replace")),
                            **git_meta,
                        },
                    )
                )
                loaded += 1

            summary = f"  [green]✓[/green] [bold]{repo_name}[/bold] — {loaded} files loaded"
            if skipped_size:
                summary += f" [dim]({skipped_size} skipped: over {cap // 1024} KB)[/dim]"
            progress.console.print(summary)
            progress.advance(repo_task)

    return documents
