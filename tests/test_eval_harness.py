"""
Tests for eval/ragas_eval.py — CP5.

All tests are hermetic: no Ollama, no Claude API, no filesystem pipeline calls.
The pipeline and judge functions are mocked where needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from eval.ragas_eval import (
    EvalQuestion,
    EvalResult,
    EvalSummary,
    JudgeScores,
    compute_aggregate_stats,
    compute_file_recall,
    load_questions,
    parse_judge_scores,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _make_result(
    *,
    file_recall:  float = 1.0,
    faithfulness: float = 4.0,
    completeness: float = 4.0,
    relevance:    float = 4.0,
    retrieval_ms: float = 100.0,
    cache_hit:    bool  = False,
    error:        str | None = None,
    input_tokens:  int = 200,
    output_tokens: int = 100,
) -> EvalResult:
    scores = JudgeScores(faithfulness, completeness, relevance) if error is None else None
    return EvalResult(
        question_id    = "qX",
        question       = "test question",
        answer         = "test answer",
        retrieved_files = [],
        expected_files  = [],
        file_recall    = file_recall,
        judge_scores   = scores,
        retrieval_ms   = retrieval_ms,
        llm_ms         = 500.0,
        total_ms       = 600.0,
        cache_hit      = cache_hit,
        input_tokens   = input_tokens,
        output_tokens  = output_tokens,
        error          = error,
    )


# ===========================================================================
# load_questions
# ===========================================================================

class TestLoadQuestions:
    def test_loads_valid_jsonl(self, tmp_path):
        p = tmp_path / "q.jsonl"
        _write_jsonl(p, [
            {"id": "q1", "question": "How does X work?", "tags": ["x"]},
            {"id": "q2", "question": "What is Y?"},
        ])
        qs = load_questions(p)
        assert len(qs) == 2
        assert qs[0].id == "q1"
        assert qs[1].question == "What is Y?"

    def test_defaults_populated(self, tmp_path):
        p = tmp_path / "q.jsonl"
        _write_jsonl(p, [{"id": "q1", "question": "Q?"}])
        q = load_questions(p)[0]
        assert q.expected_files == []
        assert q.tags == []
        assert q.difficulty == "medium"
        assert q.notes == ""

    def test_skips_blank_lines(self, tmp_path):
        p = tmp_path / "q.jsonl"
        p.write_text('\n{"id":"q1","question":"Q?"}\n\n')
        assert len(load_questions(p)) == 1

    def test_raises_on_missing_id(self, tmp_path):
        p = tmp_path / "q.jsonl"
        _write_jsonl(p, [{"question": "no id here"}])
        with pytest.raises(ValueError, match="missing"):
            load_questions(p)

    def test_raises_on_invalid_json(self, tmp_path):
        p = tmp_path / "q.jsonl"
        p.write_text("not json\n")
        with pytest.raises(ValueError, match="invalid JSON"):
            load_questions(p)

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_questions(tmp_path / "missing.jsonl")


# ===========================================================================
# compute_file_recall
# ===========================================================================

class TestFileRecall:
    def test_perfect_recall(self):
        assert compute_file_recall(["api/service.py"], ["api/service.py"]) == 1.0

    def test_partial_recall(self):
        r = compute_file_recall(["a.py"], ["a.py", "b.py"])
        assert r == pytest.approx(0.5)

    def test_zero_recall(self):
        r = compute_file_recall(["c.py"], ["a.py", "b.py"])
        assert r == pytest.approx(0.0)

    def test_empty_expected_returns_one(self):
        assert compute_file_recall(["any.py"], []) == 1.0

    def test_basename_matching(self):
        # "api/service.py" should match expected "service.py"
        assert compute_file_recall(["api/service.py"], ["service.py"]) == 1.0


# ===========================================================================
# parse_judge_scores
# ===========================================================================

class TestParseJudgeScores:
    def test_valid_json(self):
        text = '{"faithfulness": 4, "completeness": 3, "relevance": 5, "explanation": "ok"}'
        s = parse_judge_scores(text)
        assert s.faithfulness == 4.0
        assert s.completeness == 3.0
        assert s.relevance    == 5.0
        assert s.explanation  == "ok"

    def test_markdown_fenced(self):
        text = "```json\n{\"faithfulness\": 4, \"completeness\": 4, \"relevance\": 4}\n```"
        s = parse_judge_scores(text)
        assert s.faithfulness == 4.0

    def test_partial_json_uses_defaults(self):
        text = '{"faithfulness": 5}'
        s = parse_judge_scores(text)
        assert s.faithfulness == 5.0
        assert s.completeness == 3.0   # default
        assert s.relevance    == 3.0   # default

    def test_invalid_json_returns_neutral(self):
        s = parse_judge_scores("not json at all")
        assert s.faithfulness == 3.0
        assert s.explanation  == "parse_failed"

    def test_mean_property(self):
        s = JudgeScores(faithfulness=5.0, completeness=3.0, relevance=4.0)
        assert s.mean == pytest.approx(4.0)


# ===========================================================================
# compute_aggregate_stats
# ===========================================================================

class TestAggregateStats:
    def test_empty_results(self):
        summary = compute_aggregate_stats([])
        assert summary.total == 0
        assert summary.successful == 0
        assert summary.failed == 0

    def test_all_successful(self):
        results = [_make_result(faithfulness=4.0, completeness=3.0, relevance=5.0)]
        s = compute_aggregate_stats(results)
        assert s.total == 1
        assert s.successful == 1
        assert s.failed == 0
        assert s.mean_faithfulness == pytest.approx(4.0)

    def test_failed_counted(self):
        results = [_make_result(), _make_result(error="oops")]
        s = compute_aggregate_stats(results)
        assert s.failed == 1

    def test_cache_hit_rate(self):
        results = [
            _make_result(cache_hit=True),
            _make_result(cache_hit=False),
        ]
        s = compute_aggregate_stats(results)
        assert s.cache_hit_rate == pytest.approx(0.5)

    def test_token_totals(self):
        results = [
            _make_result(input_tokens=100, output_tokens=50),
            _make_result(input_tokens=200, output_tokens=80),
        ]
        s = compute_aggregate_stats(results)
        assert s.total_input_tokens  == 300
        assert s.total_output_tokens == 130
