"""Tests for reasoning-trace write-back (#4528) — the final SDK-side stage of
the RAG epic's agentic loop.

Built against #4525's `rag_loop` module (merged, PR #4558), #4526's fan-out
(merged, PR #4562), and #4527's `synthesis` module (merged, PR #4552).
Constructs real `LoopResult`/`FanoutResult`/`SynthesisResult` objects (the
same types those modules' own real execution produces) rather than mocking
them, then asserts the exact `rag_store_answer` MCP payload
`build_reasoning_trace`/`store_reasoning_trace` builds from them — mirroring
`test_mcp_wrapper_arg_names.py`'s `httpx.MockTransport` capture pattern for
the wire-call assertions.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from relata import McpClient
from relata.mcp import AsyncMcpClient
from relata.models import RagHit, RagQueryResponse
from relata.rag_loop import (
    CorrectiveGradingResult,
    FanoutResult,
    GateDecision,
    HeuristicGateResult,
    HitGrade,
    LoopIteration,
    LoopResult,
    SubAgentResult,
    SubAgentStrategy,
)
from relata.rag_trace import (
    astore_reasoning_trace,
    build_reasoning_trace,
    store_reasoning_trace,
)
from relata.synthesis import Citation, SynthesisResult, SynthesizedSentence

BASE = "http://localhost:9090"


def _hit(chunk_id: str, *, relevance_confidence: float | None = 0.7, **overrides: Any) -> RagHit:
    base: dict[str, Any] = {
        "bm25_score": 1.0,
        "vector_score": 1.0,
        "rerank_score": None,
        "chunk_id": chunk_id,
        "report_id": "doc-1",
        "text": f"text for {chunk_id}",
        "section_path": ["3", "3.2"],
        "page_start": 4,
        "page_end": 5,
        "prev_chunk_id": None,
        "next_chunk_id": None,
        "entity_ids": [],
        "relevance_confidence": relevance_confidence,
    }
    base.update(overrides)
    return RagHit(**base)


def _citation(chunk_id: str, **overrides: Any) -> Citation:
    base = {
        "chunk_id": chunk_id,
        "report_id": "doc-1",
        "section_path": ["3", "3.2"],
        "page_start": 4,
        "page_end": 5,
    }
    base.update(overrides)
    return Citation(**base)


def _two_iteration_loop() -> LoopResult:
    """A loop that evaluated once (grading found one incorrect hit) and
    then converged with confidence — the shape the corrective-retrieval
    tier (#4525) produces when it forces exactly one re-query."""
    hit_a = _hit("ch_1", relevance_confidence=0.4)
    hit_b = _hit("ch_2", relevance_confidence=0.2)
    hit_c = _hit("ch_3", relevance_confidence=0.9)

    it0 = LoopIteration(
        query="original query",
        response=RagQueryResponse(hits=[hit_a, hit_b]),
        gate=HeuristicGateResult(score=0.7, decision=GateDecision.EVALUATE),
        grading=CorrectiveGradingResult(
            grades=[HitGrade.CORRECT, HitGrade.INCORRECT], fraction_correct=0.5
        ),
        confidence=0.5,
    )
    it1 = LoopIteration(
        query="original query, refined",
        response=RagQueryResponse(hits=[hit_c]),
        gate=HeuristicGateResult(score=0.92, decision=GateDecision.PASS),
        grading=None,
        confidence=0.92,
    )
    return LoopResult(
        response=it1.response,
        iterations=[it0, it1],
        llm_calls=1,
        stopped_reason="heuristic_pass",
    )


def _synthesis_result() -> SynthesisResult:
    sentence = SynthesizedSentence(
        text="Answer text grounded in [ch_3].", citations=[_citation("ch_3")], supported=True
    )
    return SynthesisResult(
        answer="Answer text grounded in [ch_3].",
        sentences=[sentence],
        citations=[_citation("ch_3")],
        unsupported_count=0,
    )


# ---------------------------------------------------------------------------
# build_reasoning_trace — pure payload-building assertions
# ---------------------------------------------------------------------------


def test_iteration_count_and_final_confidence() -> None:
    loop_result = _two_iteration_loop()
    args = build_reasoning_trace("q", loop_result, _synthesis_result())
    assert args["num_iterations"] == 2
    assert args["confidence"] == 0.92  # the LAST iteration's confidence, not the first


def test_iteration_trace_carries_per_iteration_confidence_and_query() -> None:
    loop_result = _two_iteration_loop()
    args = build_reasoning_trace("q", loop_result, _synthesis_result())
    trace = args["iteration_trace"]
    assert len(trace) == 2
    assert trace[0]["query"] == "original query"
    assert trace[0]["confidence"] == 0.5
    assert trace[0]["gate_decision"] == "evaluate"
    assert trace[0]["fraction_correct"] == 0.5
    assert trace[1]["query"] == "original query, refined"
    assert trace[1]["confidence"] == 0.92
    assert trace[1]["gate_decision"] == "pass"
    assert "fraction_correct" not in trace[1]  # no grading ran on the passing iteration


def test_evidence_gaps_include_hits_graded_not_correct() -> None:
    loop_result = _two_iteration_loop()
    args = build_reasoning_trace("q", loop_result, _synthesis_result())
    gaps = args["evidence_gaps"]
    assert len(gaps) == 1
    assert gaps[0]["chunk_id"] == "ch_2"
    assert gaps[0]["grade"] == "incorrect"
    assert gaps[0]["reason"] == "corrective_grading"
    # The CORRECT-graded hit (ch_1) must not show up as a gap.
    assert all(g.get("chunk_id") != "ch_1" for g in gaps)


def test_evidence_gaps_include_fanout_excluded_strategies() -> None:
    loop_result = _two_iteration_loop()
    winner = SubAgentResult(
        strategy=SubAgentStrategy(name="hybrid"),
        response=RagQueryResponse(hits=[_hit("ch_3")]),
        confidence=0.9,
    )
    excluded = SubAgentResult(
        strategy=SubAgentStrategy(name="lexical"),
        response=RagQueryResponse(hits=[]),
        confidence=0.05,
    )
    fanout_result = FanoutResult(
        winner=winner, merged_response=winner.response, included=[winner], excluded=[excluded]
    )
    args = build_reasoning_trace(
        "q", loop_result, _synthesis_result(), fanout_result=fanout_result
    )
    gaps = args["evidence_gaps"]
    fanout_gaps = [g for g in gaps if g.get("reason") == "fanout_low_confidence"]
    assert len(fanout_gaps) == 1
    assert fanout_gaps[0]["strategy"] == "lexical"
    assert fanout_gaps[0]["confidence"] == 0.05


def test_evidence_gaps_empty_when_nothing_discarded() -> None:
    it = LoopIteration(
        query="q",
        response=RagQueryResponse(hits=[_hit("ch_1")]),
        gate=HeuristicGateResult(score=0.9, decision=GateDecision.PASS),
        confidence=0.9,
    )
    loop_result = LoopResult(
        response=it.response, iterations=[it], llm_calls=0, stopped_reason="heuristic_pass"
    )
    args = build_reasoning_trace("q", loop_result, _synthesis_result())
    assert args["evidence_gaps"] == []


def test_sources_built_from_synthesis_citations_with_relevance_score() -> None:
    loop_result = _two_iteration_loop()
    args = build_reasoning_trace("q", loop_result, _synthesis_result())
    sources = args["sources"]
    assert len(sources) == 1
    assert sources[0]["id"] == "ch_3"
    assert sources[0]["source"] == "doc-1"
    assert sources[0]["score"] == 0.9  # from hit_c's relevance_confidence
    assert sources[0]["page"] == 4
    assert sources[0]["sectionPath"] == ["3", "3.2"]


def test_faithfulness_computed_from_unsupported_fraction() -> None:
    loop_result = _two_iteration_loop()
    sentence_ok = SynthesizedSentence(text="ok [ch_3].", citations=[_citation("ch_3")])
    sentence_bad = SynthesizedSentence(
        text="unsupported claim [unsupported]", citations=[], supported=False
    )
    synthesis_result = SynthesisResult(
        answer="ok [ch_3]. unsupported claim [unsupported]",
        sentences=[sentence_ok, sentence_bad],
        citations=[_citation("ch_3")],
        unsupported_count=1,
    )
    args = build_reasoning_trace("q", loop_result, synthesis_result)
    assert args["faithfulness"] == 0.5


def test_faithfulness_is_sentinel_when_no_sentences() -> None:
    loop_result = _two_iteration_loop()
    empty_synthesis = SynthesisResult(answer="", sentences=[], citations=[], unsupported_count=0)
    args = build_reasoning_trace("q", loop_result, empty_synthesis)
    assert args["faithfulness"] == -1.0


def test_optional_case_id_purpose_duration_are_omitted_when_unset() -> None:
    loop_result = _two_iteration_loop()
    args = build_reasoning_trace("q", loop_result, _synthesis_result())
    assert "case_id" not in args
    assert "purpose" not in args
    assert "duration_ms" not in args


def test_optional_case_id_purpose_duration_are_sent_when_given() -> None:
    loop_result = _two_iteration_loop()
    args = build_reasoning_trace(
        "q",
        loop_result,
        _synthesis_result(),
        case_id="case-42",
        purpose="analytics",
        duration_ms=1234,
    )
    assert args["case_id"] == "case-42"
    assert args["purpose"] == "analytics"
    assert args["duration_ms"] == 1234


# ---------------------------------------------------------------------------
# store_reasoning_trace / astore_reasoning_trace — wire-call assertions
# ---------------------------------------------------------------------------


def _capture_sync() -> tuple[McpClient, dict[str, Any]]:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/mcp/tools/call":
            seen.update(json.loads(request.content))
        result = {
            "stored": "RagAnswer",
            "source_rows": 1,
            "has_iteration_trace": True,
            "evidence_gap_count": 1,
        }
        body = {"content": [{"type": "text", "text": json.dumps(result)}], "isError": False}
        return httpx.Response(200, json=body)

    client = McpClient(BASE, transport=httpx.MockTransport(handler))
    return client, seen


def test_store_reasoning_trace_calls_rag_store_answer_with_full_payload() -> None:
    client, seen = _capture_sync()
    loop_result = _two_iteration_loop()
    result = store_reasoning_trace(
        client, "q", loop_result, _synthesis_result(), purpose="analytics"
    )
    assert seen["name"] == "rag_store_answer"
    args = seen["arguments"]
    assert args["question"] == "q"
    assert args["answer"] == "Answer text grounded in [ch_3]."
    assert args["num_iterations"] == 2
    assert len(args["iteration_trace"]) == 2
    assert len(args["evidence_gaps"]) == 1
    assert args["purpose"] == "analytics"
    # The (mocked) server's response is passed straight through.
    assert result["stored"] == "RagAnswer"
    assert result["has_iteration_trace"] is True


@pytest.mark.asyncio
async def test_astore_reasoning_trace_calls_rag_store_answer_with_full_payload() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/mcp/tools/call":
            seen.update(json.loads(request.content))
        result = {"stored": "RagAnswer", "source_rows": 1}
        body = {"content": [{"type": "text", "text": json.dumps(result)}], "isError": False}
        return httpx.Response(200, json=body)

    client = AsyncMcpClient(BASE, transport=httpx.MockTransport(handler))
    loop_result = _two_iteration_loop()
    result = await astore_reasoning_trace(client, "q", loop_result, _synthesis_result())
    assert seen["name"] == "rag_store_answer"
    args = seen["arguments"]
    assert args["num_iterations"] == 2
    assert result["stored"] == "RagAnswer"
