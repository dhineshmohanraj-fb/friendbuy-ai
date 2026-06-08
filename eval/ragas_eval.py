"""
Eval harness — CP5.

Runs the friendbuy-ai pipeline against a set of golden questions and scores
each answer with Claude as an LLM judge.

Scoring dimensions (1–5 integer each)
--------------------------------------
- **faithfulness**:  Every claim is grounded in the retrieved context.
- **completeness**:  All parts of the question are addressed.
- **relevance**:     The answer addresses exactly what was asked.

Plus a heuristic score:
- **file_recall**:   Fraction of ``expected_files`` that appeared in the
                     retrieved context (0.0 – 1.0).

Usage
-----
::

    # Full eval (calls Claude for scoring)
    python -m eval.ragas_eval

    # Custom question file + output path
    python -m eval.ragas_eval \\
        --questions eval/golden_questions.jsonl \\
        --output    eval/results.jsonl

    # Skip LLM judge (heuristic scores only — free, fast)
    python -m eval.ragas_eval --dry-run

    # Scope to a single repo
    python -m eval.ragas_eval --repo payments-service
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EvalQuestion:
    id:             str
    question:       str
    expected_files: list[str] = field(default_factory=list)
    tags:           list[str] = field(default_factory=list)
    difficulty:     str       = "medium"
    notes:          str       = ""


@dataclass
class JudgeScores:
    faithfulness: float   # 1-5
    completeness: float   # 1-5
    relevance:    float   # 1-5
    explanation:  str     = ""

    @property
    def mean(self) -> float:
        return (self.faithfulness + self.completeness + self.relevance) / 3


@dataclass
class EvalResult:
    question_id:    str
    question:       str
    answer:         str
    retrieved_files: list[str]
    expected_files: list[str]
    file_recall:    float
    judge_scores:   JudgeScores | None
    retrieval_ms:   float
    llm_ms:         float
    total_ms:       float
    cache_hit:      bool
    input_tokens:   int
    output_tokens:  int
    tags:           list[str] = field(default_factory=list)
    error:          str | None = None


@dataclass
class EvalSummary:
    total:               int
    successful:          int
    failed:              int
    mean_faithfulness:   float
    mean_completeness:   float
    mean_relevance:      float
    mean_file_recall:    float
    mean_retrieval_ms:   float
    cache_hit_rate:      float
    total_input_tokens:  int
    total_output_tokens: int


# ---------------------------------------------------------------------------
# LLM-as-judge prompts
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """\
You are an evaluator for a RAG (Retrieval-Augmented Generation) system.
Score the quality of its answers objectively.
Respond with ONLY a valid JSON object — no markdown fences, no preamble.
"""

_JUDGE_PROMPT = """\
Question asked: {question}

Answer produced by the system:
{answer}

Score on three dimensions using an integer from 1 to 5:
  faithfulness  — Every claim is grounded in the retrieved context.
                  (5 = fully grounded, no hallucinations; 1 = mostly hallucinated)
  completeness  — All aspects of the question are addressed.
                  (5 = fully complete; 1 = very incomplete)
  relevance     — The answer addresses exactly what was asked.
                  (5 = perfectly on-topic; 1 = off-topic)

Return exactly this JSON (integers only for scores):
{{"faithfulness": 4, "completeness": 3, "relevance": 5, "explanation": "brief reason"}}
"""


# ---------------------------------------------------------------------------
# Public functions (importable + testable)
# ---------------------------------------------------------------------------

def load_questions(path: Path) -> list[EvalQuestion]:
    """
    Load golden questions from a JSONL file.

    Each line must be a JSON object with at minimum ``"id"`` and ``"question"``.

    Raises:
        FileNotFoundError: if *path* does not exist.
        ValueError: if a line is malformed.
    """
    questions: list[EvalQuestion] = []
    with path.open(encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {lineno}: invalid JSON — {exc}") from exc
            if "id" not in obj or "question" not in obj:
                raise ValueError(f"Line {lineno}: missing 'id' or 'question' field")
            questions.append(EvalQuestion(
                id             = str(obj["id"]),
                question       = str(obj["question"]),
                expected_files = list(obj.get("expected_files") or []),
                tags           = list(obj.get("tags") or []),
                difficulty     = str(obj.get("difficulty", "medium")),
                notes          = str(obj.get("notes", "")),
            ))
    return questions


def compute_file_recall(
    retrieved_files: list[str],
    expected_files:  list[str],
) -> float:
    """
    Fraction of *expected_files* that appear in *retrieved_files*.

    Returns 1.0 when *expected_files* is empty (nothing expected → perfect recall).
    Matching is done on the *basename* of each path so ``api/service.py``
    matches ``service.py`` and vice-versa.
    """
    if not expected_files:
        return 1.0

    def _baseset(paths: list[str]) -> set[str]:
        return {Path(p).name for p in paths}

    retrieved_bases = _baseset(retrieved_files)
    expected_bases  = _baseset(expected_files)

    hits = len(expected_bases & retrieved_bases)
    return hits / len(expected_bases)


def parse_judge_scores(response_text: str) -> JudgeScores:
    """
    Parse a Claude judge response into a :class:`JudgeScores` object.

    Accepts raw JSON (with or without markdown fences).
    Returns neutral scores (3.0) on parse failure so the eval run continues.
    """
    text = response_text.strip()

    # Strip markdown fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text  = "\n".join(
            line for line in lines
            if not line.startswith("```")
        ).strip()

    try:
        obj = json.loads(text)
        return JudgeScores(
            faithfulness = float(obj.get("faithfulness", 3)),
            completeness = float(obj.get("completeness", 3)),
            relevance    = float(obj.get("relevance", 3)),
            explanation  = str(obj.get("explanation", "")),
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return JudgeScores(faithfulness=3.0, completeness=3.0, relevance=3.0,
                           explanation="parse_failed")


def judge_answer(
    question: str,
    answer: str,
    api_key: str,
    model: str = "claude-haiku-4-5",
) -> JudgeScores:
    """
    Call Claude to score an answer.

    Uses ``claude-haiku-4-5`` by default (fast + cheap for eval runs).
    Falls back to neutral scores (3.0) if the API call fails.
    """
    try:
        import anthropic
        client   = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=256,
            system=_JUDGE_SYSTEM,
            messages=[{
                "role": "user",
                "content": _JUDGE_PROMPT.format(
                    question=question,
                    answer=answer[:3000],   # truncate for cost control
                ),
            }],
        )
        return parse_judge_scores(response.content[0].text)
    except Exception:  # noqa: BLE001
        return JudgeScores(faithfulness=3.0, completeness=3.0, relevance=3.0,
                           explanation="judge_failed")


def compute_aggregate_stats(results: list[EvalResult]) -> EvalSummary:
    """Compute aggregate statistics across all eval results."""
    successful = [r for r in results if r.error is None]
    failed     = [r for r in results if r.error is not None]

    scored = [r for r in successful if r.judge_scores is not None]

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return EvalSummary(
        total               = len(results),
        successful          = len(successful),
        failed              = len(failed),
        mean_faithfulness   = _mean([r.judge_scores.faithfulness for r in scored]),
        mean_completeness   = _mean([r.judge_scores.completeness for r in scored]),
        mean_relevance      = _mean([r.judge_scores.relevance    for r in scored]),
        mean_file_recall    = _mean([r.file_recall for r in successful]),
        mean_retrieval_ms   = _mean([r.retrieval_ms for r in successful]),
        cache_hit_rate      = _mean([1.0 if r.cache_hit else 0.0 for r in successful]),
        total_input_tokens  = sum(r.input_tokens  for r in successful),
        total_output_tokens = sum(r.output_tokens for r in successful),
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_eval(
    questions:   list[EvalQuestion],
    output_path: Path,
    repo_name:   str | None = None,
    dry_run:     bool       = False,
    pipeline_fn: Callable   | None = None,
    judge_fn:    Callable   | None = None,
) -> list[EvalResult]:
    """
    Run the eval harness.

    Args:
        questions:   Golden questions to evaluate.
        output_path: Where to write per-question JSONL results.
        repo_name:   Scope retrieval to this repo (optional).
        dry_run:     If True, skip LLM judge scoring.
        pipeline_fn: Override the default ``pipeline.query_pipeline.run``.
                     Signature: ``(query, repo_name) -> PipelineResult``.
        judge_fn:    Override the default :func:`judge_answer`.
                     Signature: ``(question, answer, api_key) -> JudgeScores``.

    Returns:
        List of :class:`EvalResult` (also written to *output_path*).
    """
    from config import get_settings
    settings = get_settings()

    if pipeline_fn is None:
        from pipeline.query_pipeline import run as _run
        def pipeline_fn(q, r):  # type: ignore[misc]
            return _run(query=q, repo_name=r)

    if judge_fn is None:
        judge_fn = judge_answer  # type: ignore[assignment]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[EvalResult] = []

    with output_path.open("w", encoding="utf-8") as out_f:
        for q in questions:
            t0 = time.time()
            error = None
            answer = ""
            retrieved_files: list[str] = []
            retrieval_ms = llm_ms = 0.0
            cache_hit = False
            input_tokens = output_tokens = 0

            try:
                pr = pipeline_fn(q.question, repo_name)
                answer          = pr.answer
                retrieved_files = pr.relevant_files
                retrieval_ms    = pr.retrieval_ms
                input_tokens    = pr.input_tokens
                output_tokens   = pr.output_tokens
                cache_hit       = getattr(pr, "cache_hit", False)
                llm_ms = (time.time() - t0) * 1000 - retrieval_ms
            except Exception as exc:  # noqa: BLE001
                error = str(exc)

            file_recall = compute_file_recall(retrieved_files, q.expected_files)

            judge_scores: JudgeScores | None = None
            if not dry_run and not error and settings.anthropic_api_key:
                judge_scores = judge_fn(q.question, answer, settings.anthropic_api_key)

            total_ms = (time.time() - t0) * 1000

            result = EvalResult(
                question_id     = q.id,
                question        = q.question,
                answer          = answer,
                retrieved_files = retrieved_files,
                expected_files  = q.expected_files,
                file_recall     = file_recall,
                judge_scores    = judge_scores,
                retrieval_ms    = retrieval_ms,
                llm_ms          = llm_ms,
                total_ms        = total_ms,
                cache_hit       = cache_hit,
                input_tokens    = input_tokens,
                output_tokens   = output_tokens,
                tags            = q.tags,
                error           = error,
            )
            results.append(result)

            # Serialize judge_scores separately (not auto-handled by asdict)
            row = asdict(result)
            row["judge_scores"] = asdict(judge_scores) if judge_scores else None
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m eval.ragas_eval",
        description="friendbuy-ai eval harness — LLM-as-judge scoring",
    )
    p.add_argument(
        "--questions",
        type=Path,
        default=Path("eval/golden_questions.jsonl"),
        help="Path to golden questions JSONL (default: eval/golden_questions.jsonl)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("eval/results.jsonl"),
        help="Path for output results JSONL (default: eval/results.jsonl)",
    )
    p.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Scope retrieval to a single repo",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Skip LLM judge — use heuristic (file recall) scoring only",
    )
    return p


def _print_summary(summary: EvalSummary) -> None:
    try:
        from rich.console import Console
        from rich.table import Table

        con = Console()
        t   = Table(title="Eval Summary", show_header=True, header_style="bold cyan")
        t.add_column("Metric", style="cyan")
        t.add_column("Value", justify="right", style="green")

        t.add_row("Questions",        str(summary.total))
        t.add_row("Successful",       str(summary.successful))
        t.add_row("Failed",           str(summary.failed))
        t.add_row("Mean faithfulness", f"{summary.mean_faithfulness:.2f} / 5")
        t.add_row("Mean completeness", f"{summary.mean_completeness:.2f} / 5")
        t.add_row("Mean relevance",    f"{summary.mean_relevance:.2f} / 5")
        t.add_row("Mean file recall",  f"{summary.mean_file_recall:.1%}")
        t.add_row("Mean retrieval ms", f"{summary.mean_retrieval_ms:.0f}")
        t.add_row("Cache hit rate",    f"{summary.cache_hit_rate:.1%}")
        t.add_row("Total input tok.",  f"{summary.total_input_tokens:,}")
        t.add_row("Total output tok.", f"{summary.total_output_tokens:,}")
        con.print(t)
    except ImportError:
        print(f"\nEval complete: {summary.successful}/{summary.total} successful")
        print(f"  faithfulness={summary.mean_faithfulness:.2f}  "
              f"completeness={summary.mean_completeness:.2f}  "
              f"relevance={summary.mean_relevance:.2f}  "
              f"file_recall={summary.mean_file_recall:.1%}")


if __name__ == "__main__":
    args = _build_parser().parse_args()
    qs   = load_questions(args.questions)
    res  = run_eval(
        questions   = qs,
        output_path = args.output,
        repo_name   = args.repo,
        dry_run     = args.dry_run,
    )
    summary = compute_aggregate_stats(res)
    _print_summary(summary)
