"""End-to-end query pipeline: retrieve → filter → Claude."""

from __future__ import annotations

from dataclasses import dataclass

import anthropic
from rich.console import Console

from config import get_settings
from retriever.context_filter import filter_and_summarise
from retriever.vector_search import SearchResult, search

def _console() -> Console:
    return Console()

console = _console()   # kept for any direct references; use _console() in new code

_SYSTEM_PROMPT = """\
You are an expert software engineer with deep knowledge of the Friendbuy codebase.
You answer questions precisely and concisely, citing the relevant file paths when \
applicable.  When writing code, prefer the languages and patterns already used in the \
codebase.  If the provided context is insufficient, say so clearly rather than guessing.
"""


@dataclass
class PipelineResult:
    answer: str
    relevant_files: list[str]
    input_tokens: int
    output_tokens: int
    raw_chunks: list[SearchResult]


def run(
    query: str,
    repo_name: str | None = None,
    top_k: int | None = None,
    stream: bool = False,
) -> PipelineResult:
    """
    Full RAG pipeline: vector search → Qwen filter → Claude answer.

    Args:
        query:     The user's question.
        repo_name: Restrict vector search to this repo (optional).
        top_k:     Override default number of retrieved chunks.
        stream:    If True, print the Claude response to stdout as it streams.

    Returns:
        A PipelineResult containing the answer and usage metadata.
    """
    settings = get_settings()

    if not settings.anthropic_api_key:
        console.print(
            "\n[bold red]Error:[/bold red] ANTHROPIC_API_KEY is not set.\n"
            "Add it to your [bold].env[/bold] file and restart."
        )
        raise SystemExit(1)

    # --- 1. Vector search ------------------------------------------------
    results: list[SearchResult] = search(query, top_k=top_k, repo_name=repo_name)

    if not results:
        return PipelineResult(
            answer="No relevant context found in the knowledge base for your query.",
            relevant_files=[],
            input_tokens=0,
            output_tokens=0,
            raw_chunks=[],
        )

    # --- 2. Local Qwen filter / summarise --------------------------------
    context_data = filter_and_summarise(query, results)
    summary: str = context_data["summary"]
    relevant_files: list[str] = context_data["relevant_files"]
    raw_chunks: list[SearchResult] = context_data["raw_chunks"]

    # --- 3. Build Claude prompt ------------------------------------------
    user_message = (
        "## Context from Friendbuy codebase\n\n"
        f"{summary}\n\n"
        "## Relevant files\n"
        + "\n".join(f"- {f}" for f in relevant_files)
        + f"\n\n## Question\n\n{query}"
    )

    # --- 4. Call Claude --------------------------------------------------
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    if stream:
        answer_parts: list[str] = []
        input_tokens = 0
        output_tokens = 0

        with client.messages.stream(
            model=settings.claude_model,
            max_tokens=settings.claude_max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        ) as streamer:
            for text in streamer.text_stream:
                print(text, end="", flush=True)
                answer_parts.append(text)
            final = streamer.get_final_message()
            input_tokens = final.usage.input_tokens
            output_tokens = final.usage.output_tokens

        print()  # newline after streamed output
        answer = "".join(answer_parts)
    else:
        response = client.messages.create(
            model=settings.claude_model,
            max_tokens=settings.claude_max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        answer = response.content[0].text
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

    return PipelineResult(
        answer=answer,
        relevant_files=relevant_files,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        raw_chunks=raw_chunks,
    )
