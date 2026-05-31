"""Use a local Qwen model (via Ollama) to clean and summarise retrieved chunks."""

from __future__ import annotations

import json
import re

from langchain_ollama import OllamaLLM
from rich.console import Console

from config import get_settings
from retriever.vector_search import SearchResult

_FILTER_PROMPT = """\
You are a code-context curator. You are given a user question and a list of \
code/documentation chunks retrieved from a codebase.

Your tasks:
1. Remove chunks that are clearly irrelevant to the question.
2. Stitch the relevant chunks into a clean, coherent context paragraph (max 800 words).
3. List the most relevant file paths.

Respond ONLY with valid JSON in exactly this shape:
{{
  "summary": "<clean context paragraph>",
  "relevant_files": ["<file_path>", ...],
  "kept_chunk_indices": [<int>, ...]
}}

User question:
{question}

Retrieved chunks (index: content):
{chunks}
"""


def _console() -> Console:
    return Console()


def _build_chunk_text(results: list[SearchResult]) -> str:
    parts = []
    for i, r in enumerate(results):
        parts.append(f"[{i}] ({r.file_path})\n{r.content[:600]}")
    return "\n\n---\n\n".join(parts)


def _get_llm() -> OllamaLLM:
    """Return a local Qwen LLM, raising SystemExit with a clear message if Ollama is down."""
    import httpx

    settings = get_settings()
    try:
        httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5).raise_for_status()
    except Exception as exc:
        _console().print(
            f"\n[bold red]Error:[/bold red] Ollama not reachable at "
            f"[cyan]{settings.ollama_base_url}[/cyan].\n"
            "Start it with: [bold]ollama serve[/bold]\n"
            f"Detail: {exc}"
        )
        raise SystemExit(1) from exc

    return OllamaLLM(
        model=settings.local_model,
        base_url=settings.ollama_base_url,
        temperature=settings.qwen_temperature,
    )


def _extract_json(raw: str) -> dict:
    """
    Robustly extract a JSON object from raw LLM output.

    Handles all common wrapping patterns:
    - Plain JSON
    - ```json ... ``` fences (any case, with/without language tag)
    - Single-backtick inline code
    - JSON embedded in surrounding prose
    """
    text = raw.strip()

    # 1. Code-fence block: ```[json] ... ```
    fence = re.search(r"```(?:[Jj][Ss][Oo][Nn])?\s*([\s\S]*?)\s*```", text)
    if fence:
        text = fence.group(1).strip()

    # 2. Inline backtick: `{ ... }`
    elif text.startswith("`") and text.endswith("`"):
        text = text[1:-1].strip()

    # 3. Pull the outermost {...} block from prose
    brace = re.search(r"\{[\s\S]*\}", text)
    if brace:
        text = brace.group(0)

    return json.loads(text)


def filter_and_summarise(
    query: str,
    results: list[SearchResult],
) -> dict:
    """
    Use Qwen locally to curate the retrieved chunks.

    Args:
        query:   The original user question.
        results: Raw vector-search results.

    Returns:
        A dict with keys: ``summary``, ``relevant_files``, ``raw_chunks``.
    """
    if not results:
        return {"summary": "", "relevant_files": [], "raw_chunks": []}

    llm = _get_llm()
    chunk_text = _build_chunk_text(results)
    prompt = _FILTER_PROMPT.format(question=query, chunks=chunk_text)

    try:
        raw_output: str = llm.invoke(prompt)
        parsed: dict = _extract_json(raw_output)
    except (json.JSONDecodeError, ValueError, IndexError):
        _console().print(
            "[yellow]Warning:[/yellow] Qwen returned unparseable JSON — using all raw chunks as fallback."
        )
        return {
            "summary": "\n\n".join(r.content for r in results),
            "relevant_files": list(dict.fromkeys(r.file_path for r in results)),
            "raw_chunks": results,
        }

    kept_indices: list[int] = parsed.get("kept_chunk_indices", list(range(len(results))))
    kept_results = [results[i] for i in kept_indices if i < len(results)]

    return {
        "summary": parsed.get("summary", ""),
        "relevant_files": parsed.get("relevant_files", []),
        "raw_chunks": kept_results,
    }
