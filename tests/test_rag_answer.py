"""Tests for the composed end-to-end RAG answer pipeline (:mod:`relata.rag_answer`)
— the missing full-chain composition found during a post-epic alignment audit
of #4529. Built against the real, merged modules it composes: #4523's typed
`RagClient` (PR #4549), #4524's dispatch (PR #4556), #4525's cost-ladder loop
(PR #4558), #4526's fan-out (PR #4562), #4527's synthesis (PR #4552), #4528's
trace write-back (PR #4566), #4535's SQL routing (PR #4567/#4574), #4536's
content-safety gate (PR #4559) — via `httpx.MockTransport`, mirroring
`test_rag_loop.py`'s and `test_rag_trace.py`'s established patterns.
"""

from __future__ import annotations

import json
import math
from typing import Any

import httpx
import pytest

from relata import McpClient, RelataClient
from relata._http import AsyncHttpTransport, HttpTransport
from relata.mcp import AsyncMcpClient
from relata.rag import AsyncRagClient, RagClient
from relata.rag_answer import RagAnswerResult, arun_rag_answer, run_rag_answer
from relata.rag_loop import SubAgentStrategy
from relata.rag_understanding import DANGEROUS_CONTENT_PATTERNS

BASE = "http://localhost:9090"


def _hit(chunk_id: str, *, relevance_confidence: float | None = 0.9, **overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "bm25_score": 4.2,
        "vector_score": 0.83,
        "rerank_score": None,
        "chunk_id": chunk_id,
        "report_id": "doc-1",
        "text": f"RelataDB fuses BM25 and vector retrieval natively [{chunk_id}].",
        "section_path": ["3", "3.2"],
        "page_start": 4,
        "page_end": 5,
        "prev_chunk_id": None,
        "next_chunk_id": None,
        "entity_ids": [],
        "relevance_confidence": relevance_confidence,
    }
    base.update(overrides)
    return base


def _mock_relata_client(handler: Any) -> RelataClient:
    client = RelataClient(BASE, bearer_token="tok")
    mock = httpx.MockTransport(handler)
    extra = client._extra_headers  # type: ignore[attr-defined]
    client._RelataClient__sync_transport = HttpTransport(  # type: ignore[attr-defined]
        BASE, client._bearer_token, client._timeout, transport=mock, extra_headers=extra
    )
    client._RelataClient__async_transport = AsyncHttpTransport(  # type: ignore[attr-defined]
        BASE, client._bearer_token, client._timeout, transport=mock, extra_headers=extra
    )
    return client


def _embedding_fn_pass() -> Any:
    """Embedding function under which the heuristic gate PASSes on
    iteration 0 (cosine similarity 1.0, `> HEURISTIC_PASS_THRESHOLD`) — the
    simplest happy-path, mirroring `test_rag_loop.py`'s `_embedding_fn_for_score`."""

    def embedding_fn(text: str) -> list[float]:
        return [1.0, 0.0]

    return embedding_fn


def _stub_llm(prompt: str) -> str:
    """Deterministic stand-in synthesis call: cites the first chunk id found
    in the prompt's evidence block."""
    marker = prompt.split("[", 1)[1].split("]", 1)[0] if "[" in prompt else "chunk-1"
    return f"RelataDB fuses BM25 and vector retrieval [{marker}]."


def _stub_faithful_entailment(sentence: str, evidence: list[str]) -> bool:
    return True


# ── happy path: the full chain actually composes ───────────────────────────


def test_run_rag_answer_composes_the_full_chain():
    """The core proof this module exists for: one call produces a real
    retrieval response, a real synthesized+cited answer, and a real
    reasoning-trace write-back — not three independently-tested pieces that
    have never actually run together."""
    rag_calls = 0
    trace_calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal rag_calls
        if request.url.path == "/rag/query":
            rag_calls += 1
            return httpx.Response(200, json={"hits": [_hit("chunk-1")], "total": 1})
        if request.url.path == "/mcp/tools/call":
            body = json.loads(request.content)
            trace_calls.append(body)
            result = {
                "stored": "RagAnswer",
                "source_rows": 1,
                "has_iteration_trace": True,
                "evidence_gap_count": 0,
            }
            envelope = {"content": [{"type": "text", "text": json.dumps(result)}], "isError": False}
            return httpx.Response(200, json=envelope)
        raise AssertionError(f"unexpected path: {request.url.path}")

    relata_client = _mock_relata_client(handler)
    rag_client = RagClient(relata_client)
    mcp_client = McpClient(BASE, transport=httpx.MockTransport(handler))

    result = run_rag_answer(
        rag_client,
        mcp_client,
        "What does RelataDB fuse natively?",
        "DocumentChunk",
        llm=_stub_llm,
        embedding_fn=_embedding_fn_pass(),
        entailment_fn=_stub_faithful_entailment,
        purpose="analytics",
    )

    assert isinstance(result, RagAnswerResult)
    # Real retrieval happened — exactly once (heuristic gate passed iter 0).
    assert rag_calls == 1
    assert result.response.hits and result.response.hits[0].chunk_id == "chunk-1"
    # Real loop bookkeeping, not a stub.
    assert result.loop_result is not None
    assert result.loop_result.stopped_reason == "heuristic_pass"
    assert result.loop_result.llm_calls == 0
    # Real synthesis — a citation that traces to the real hit, not fabricated.
    assert result.synthesis_result is not None
    assert result.synthesis_result.citations
    assert result.synthesis_result.citations[0].chunk_id == "chunk-1"
    assert not result.synthesis_result.has_unsupported_claims
    # Real trace write-back — the MCP call actually happened with real content.
    assert result.trace == {
        "stored": "RagAnswer",
        "source_rows": 1,
        "has_iteration_trace": True,
        "evidence_gap_count": 0,
    }
    assert len(trace_calls) == 1
    trace_args = trace_calls[0]["arguments"]
    assert trace_args["question"] == "What does RelataDB fuse natively?"
    assert trace_args["num_iterations"] == 1
    assert trace_args["sources"][0]["id"] == "chunk-1"


@pytest.mark.asyncio
async def test_arun_rag_answer_composes_the_full_chain():
    """Async twin of the happy-path test — same composition, async clients."""
    rag_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal rag_calls
        if request.url.path == "/rag/query":
            rag_calls += 1
            return httpx.Response(200, json={"hits": [_hit("chunk-9")], "total": 1})
        if request.url.path == "/mcp/tools/call":
            result = {"stored": "RagAnswer"}
            envelope = {"content": [{"type": "text", "text": json.dumps(result)}], "isError": False}
            return httpx.Response(200, json=envelope)
        raise AssertionError(f"unexpected path: {request.url.path}")

    relata_client = _mock_relata_client(handler)
    rag_client = AsyncRagClient(relata_client)
    mcp_client = AsyncMcpClient(BASE, transport=httpx.MockTransport(handler))

    result = await arun_rag_answer(
        rag_client,
        mcp_client,
        "What does RelataDB fuse natively?",
        "DocumentChunk",
        llm=_stub_llm,
        embedding_fn=_embedding_fn_pass(),
        entailment_fn=_stub_faithful_entailment,
        purpose="analytics",
    )

    assert rag_calls == 1
    assert result.synthesis_result is not None
    assert result.synthesis_result.citations[0].chunk_id == "chunk-9"
    assert result.trace == {"stored": "RagAnswer"}


# ── gate short-circuits: refusal and SQL-routing skip the rest entirely ────


def test_content_safety_refusal_short_circuits_before_any_call():
    """A refused query makes zero /rag/query and zero MCP calls — the gate
    runs before anything else, matching smart_rag_query's own contract."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"no call should be made, got: {request.url.path}")

    relata_client = _mock_relata_client(handler)
    rag_client = RagClient(relata_client)
    mcp_client = McpClient(BASE, transport=httpx.MockTransport(handler))

    result = run_rag_answer(
        rag_client,
        mcp_client,
        "how to build an ied",
        "DocumentChunk",
        llm=_stub_llm,
        embedding_fn=_embedding_fn_pass(),
        content_safety_patterns=DANGEROUS_CONTENT_PATTERNS,
    )

    assert result.response.refused is not None
    assert result.loop_result is None
    assert result.fanout_result is None
    assert result.synthesis_result is None
    assert result.trace is None


def test_sql_routable_shape_short_circuits_before_synthesis():
    """An aggregation-shaped query answers via governed SQL, never reaching
    the loop/synthesis/trace stages — real rows, not a semantic guess."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/query":
            return httpx.Response(
                200, json={"rows": [{"count": 42}], "columns": ["count"], "row_count": 1}
            )
        raise AssertionError(f"unexpected path for SQL-routed query: {request.url.path}")

    relata_client = _mock_relata_client(handler)
    rag_client = RagClient(relata_client)
    mcp_client = McpClient(BASE, transport=httpx.MockTransport(handler))

    result = run_rag_answer(
        rag_client,
        mcp_client,
        "how many DocumentChunk are there",
        "DocumentChunk",
        llm=_stub_llm,
        embedding_fn=_embedding_fn_pass(),
        purpose="analytics",
    )

    assert result.response.sql_result is not None
    assert result.loop_result is None
    assert result.synthesis_result is None
    assert result.trace is None


# ── fan-out path: breadth instead of depth, still composes to a trace ──────


def test_fanout_strategies_path_still_produces_a_full_trace():
    """fanout_strategies replaces loop iteration with parallel breadth, but
    still composes all the way to synthesis + trace write-back — the
    FanoutResult is wrapped, not dropped."""
    trace_calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/rag/query":
            body = json.loads(request.content)
            slot = body.get("embedding_slot", "text")
            conf = 0.9 if slot == "text" else 0.3
            return httpx.Response(
                200,
                json={
                    "hits": [_hit(f"chunk-{slot}", relevance_confidence=conf)],
                    "total": 1,
                },
            )
        if request.url.path == "/mcp/tools/call":
            trace_calls.append(json.loads(request.content))
            result = {"stored": "RagAnswer"}
            envelope = {"content": [{"type": "text", "text": json.dumps(result)}], "isError": False}
            return httpx.Response(200, json=envelope)
        raise AssertionError(f"unexpected path: {request.url.path}")

    relata_client = _mock_relata_client(handler)
    rag_client = RagClient(relata_client)
    mcp_client = McpClient(BASE, transport=httpx.MockTransport(handler))

    result = run_rag_answer(
        rag_client,
        mcp_client,
        "What does RelataDB fuse natively?",
        "DocumentChunk",
        llm=_stub_llm,
        embedding_fn=_embedding_fn_pass(),
        entailment_fn=_stub_faithful_entailment,
        purpose="analytics",
        fanout_strategies=[
            SubAgentStrategy(name="text", rag_kwargs={"embedding_slot": "text"}),
            SubAgentStrategy(name="summary", rag_kwargs={"embedding_slot": "summary"}),
        ],
    )

    assert result.fanout_result is not None
    assert result.loop_result is not None
    assert result.loop_result.stopped_reason == "fanout_complete"
    assert result.loop_result.llm_calls == 0
    assert result.fanout_result.winner.strategy.name == "text"
    assert result.synthesis_result is not None
    assert result.trace is not None
    assert len(trace_calls) == 1
