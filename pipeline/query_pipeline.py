"""
End-to-end query pipeline: hybrid retrieve → Qwen filter → Claude.

CP3 changes vs CP2:
- Uses ``hybrid_retriever.retrieve()`` instead of plain vector search.
  Result is a fusion of dense vector + BM25 sparse + graph traversal (RRF).
- Graph relationship summary is prepended to the Qwen curation prompt and
  injected into the Claude context block.
- Per-query trace written to ``cache/query_traces.jsonl``.

CP4 changes vs CP3:
- Semantic cache lookup before any LLM call (cosine threshold 0.93).
  Cache HIT → return stored answer immediately.
  Cache MISS → run pipeline, then store result.
- FlashRank cross-encoder reranker applied after RRF fusion.
- ``cache_hit`` / ``cache_similarity`` added to ``PipelineResult``.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import anthropic
from rich.console import Console

from config import get_settings
from observability.logger import get_logger, log
from retriever.context_filter import filter_and_summarise
from retriever.vector_search import SearchResult

_logger = get_logger("query_pipeline")


def _console() -> Console:
    return Console()


_SYSTEM_PROMPT = """\
You are an expert software engineer with deep knowledge of the Friendbuy codebase.
You answer questions precisely and concisely, citing the relevant file paths when \
applicable.  When writing code, prefer the languages and patterns already used in the \
codebase.  If the provided context is insufficient, say so clearly rather than guessing.
"""


@dataclass
class PipelineResult:
    answer:         str
    relevant_files: list[str]
    input_tokens:   int
    output_tokens:  int
    raw_chunks:     list[SearchResult]
    # CP3 additions
    vector_count:   int   = 0
    bm25_count:     int   = 0
    graph_count:    int   = 0
    query_entities: list[str] = None   # type: ignore[assignment]
    retrieval_ms:   float = 0.0
    # CP4 additions
    cache_hit:        bool  = False
    cache_similarity: float = 0.0

    def __post_init__(self) -> None:
        if self.query_entities is None:
            self.query_entities = []


# ---------------------------------------------------------------------------
# Trace logging
# ---------------------------------------------------------------------------

def _write_trace(trace: dict) -> None:
    """Append a query trace record to ``cache/query_traces.jsonl``."""
    try:
        from config import get_settings
        trace_path = Path(get_settings().cache_dir) / "query_traces.jsonl"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(trace) + "\n")
    except Exception:  # noqa: BLE001
        pass   # trace failure must never affect the user response


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run(
    query: str,
    repo_name: str | None = None,
    top_k: int | None = None,
    stream: bool = False,
    use_graph: bool = True,
    use_bm25:  bool = True,
) -> PipelineResult:
    """
    Full CP3 RAG pipeline:
    hybrid_retrieve → Qwen filter (+ graph summary) → Claude answer.

    Args:
        query:     The user's question.
        repo_name: Restrict retrieval to this repo (optional).
        top_k:     Override default number of retrieved chunks.
        stream:    If True, stream the Claude response to stdout.
        use_graph: Include Kuzu graph traversal in retrieval.
        use_bm25:  Include BM25 sparse search in retrieval.

    Returns:
        A :class:`PipelineResult` with the answer and metadata.
    """
    settings   = get_settings()
    query_id   = str(uuid.uuid4())
    t_start    = time.time()

    log(_logger, "info", "query.start",
        query_id=query_id, query=query[:120], repo_name=repo_name)

    if not settings.anthropic_api_key:
        _console().print(
            "\n[bold red]Error:[/bold red] ANTHROPIC_API_KEY is not set.\n"
            "Add it to your [bold].env[/bold] file and restart."
        )
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # 0. Semantic cache lookup  (CP4)
    # ------------------------------------------------------------------
    if settings.use_semantic_cache:
        try:
            from retriever.semantic_cache import SemanticCache
            _cache = SemanticCache()
            hit = _cache.lookup(query)
            if hit:
                log(_logger, "info", "cache.hit",
                    query_id=query_id, similarity=round(hit.similarity, 4),
                    cached_query=hit.cached_query[:80])
                return PipelineResult(
                    answer=hit.answer,
                    relevant_files=hit.relevant_files,
                    input_tokens=hit.input_tokens,
                    output_tokens=hit.output_tokens,
                    raw_chunks=[],
                    vector_count=hit.vector_count,
                    bm25_count=hit.bm25_count,
                    graph_count=hit.graph_count,
                    query_entities=hit.query_entities,
                    retrieval_ms=hit.retrieval_ms,
                    cache_hit=True,
                    cache_similarity=hit.similarity,
                )
        except Exception:  # noqa: BLE001
            pass   # cache failure must never break the pipeline

    # ------------------------------------------------------------------
    # 1. Hybrid retrieval  (vector + BM25 + graph)
    # ------------------------------------------------------------------
    t_retrieval = time.time()
    try:
        from retriever.hybrid_retriever import retrieve as hybrid_retrieve
        hybrid = hybrid_retrieve(
            query=query,
            repo_name=repo_name,
            top_k=top_k,
            use_graph=use_graph,
            use_bm25=use_bm25,
        )
        results         = hybrid.chunks
        graph_ctx       = hybrid.graph_context
        v_count         = hybrid.vector_count
        b_count         = hybrid.bm25_count
        g_count         = hybrid.graph_count
        query_entities  = hybrid.query_entities
    except Exception:  # noqa: BLE001
        # Fallback to plain vector search if hybrid fails
        from retriever.vector_search import search
        results        = search(query, top_k=top_k, repo_name=repo_name)
        graph_ctx      = None
        v_count        = len(results)
        b_count        = g_count = 0
        query_entities = []

    retrieval_ms = (time.time() - t_retrieval) * 1000
    log(_logger, "info", "retrieval.done",
        query_id=query_id, vector=v_count, bm25=b_count, graph=g_count,
        entities=query_entities, retrieval_ms=round(retrieval_ms, 1))

    if not results:
        return PipelineResult(
            answer="No relevant context found in the knowledge base for your query.",
            relevant_files=[],
            input_tokens=0, output_tokens=0,
            raw_chunks=[],
            vector_count=v_count, bm25_count=b_count, graph_count=g_count,
            query_entities=query_entities,
            retrieval_ms=retrieval_ms,
        )

    # ------------------------------------------------------------------
    # 1b. Cross-encoder reranking  (CP4)
    # ------------------------------------------------------------------
    if settings.use_reranker and results:
        try:
            from retriever.reranker import rerank as _rerank
            results = _rerank(query, results, top_k=settings.top_k_results)
        except Exception:  # noqa: BLE001
            pass   # reranker failure is non-fatal

    # ------------------------------------------------------------------
    # 2. Qwen context curation
    # ------------------------------------------------------------------
    context_data   = filter_and_summarise(query, results)
    summary:  str  = context_data["summary"]
    relevant_files = context_data["relevant_files"]
    raw_chunks     = context_data["raw_chunks"]

    # ------------------------------------------------------------------
    # 3. Build Claude prompt — inject graph summary when available
    # ------------------------------------------------------------------
    graph_section = ""
    if graph_ctx and not graph_ctx.is_empty() and graph_ctx.relationship_summary:
        graph_section = (
            "\n## Structural relationships (from knowledge graph)\n\n"
            + graph_ctx.relationship_summary
            + "\n"
        )

    user_message = (
        "## Context from Friendbuy codebase\n\n"
        f"{summary}\n"
        f"{graph_section}"
        "\n## Relevant files\n"
        + "\n".join(f"- {f}" for f in relevant_files)
        + f"\n\n## Question\n\n{query}"
    )

    # ------------------------------------------------------------------
    # 4. Claude API call
    # ------------------------------------------------------------------
    t_llm  = time.time()
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    if stream:
        answer_parts: list[str] = []
        input_tokens = output_tokens = 0

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
            input_tokens  = final.usage.input_tokens
            output_tokens = final.usage.output_tokens

        print()
        answer = "".join(answer_parts)
    else:
        response = client.messages.create(
            model=settings.claude_model,
            max_tokens=settings.claude_max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        answer        = response.content[0].text
        input_tokens  = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

    llm_ms = (time.time() - t_llm) * 1000
    log(_logger, "info", "llm.done",
        query_id=query_id, model=settings.claude_model,
        input_tokens=input_tokens, output_tokens=output_tokens,
        llm_ms=round(llm_ms, 1))

    # ------------------------------------------------------------------
    # 5. Trace log
    # ------------------------------------------------------------------
    _write_trace({
        "query_id":           query_id,
        "query":              query,
        "repo_name":          repo_name,
        "vector_chunks":      v_count,
        "bm25_chunks":        b_count,
        "graph_chunks":       g_count,
        "graph_entities":     query_entities,
        "total_fused_chunks": len(results),
        "relevant_files":     relevant_files,
        "retrieval_ms":       round(retrieval_ms, 1),
        "llm_ms":             round(llm_ms, 1),
        "input_tokens":       input_tokens,
        "output_tokens":      output_tokens,
        "timestamp":          time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })

    final_result = PipelineResult(
        answer=answer,
        relevant_files=relevant_files,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        raw_chunks=raw_chunks,
        vector_count=v_count,
        bm25_count=b_count,
        graph_count=g_count,
        query_entities=query_entities,
        retrieval_ms=retrieval_ms,
        cache_hit=False,
        cache_similarity=0.0,
    )

    # ------------------------------------------------------------------
    # 6. Store result in semantic cache  (CP4)
    # ------------------------------------------------------------------
    if settings.use_semantic_cache:
        try:
            from retriever.semantic_cache import SemanticCache
            SemanticCache().store(query, final_result)
        except Exception:  # noqa: BLE001
            pass

    return final_result
