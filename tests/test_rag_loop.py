"""Tests for the heuristic gate + corrective retrieval grading + loop
confidence cost ladder (#4525).

Built against #4523's typed `RagClient`/`AsyncRagClient` (merged, PR #4549)
and #4524's `rag_understanding` module (merged, PR #4556), via
`httpx.MockTransport` — no live `/rag/query` server required, mirroring the
approach `test_rag_understanding.py` uses.
"""

from __future__ import annotations

import json
import math
from typing import Any

import httpx
import pytest

from relata import RelataClient
from relata._http import AsyncHttpTransport, HttpTransport
from relata.models import RagQueryResponse
from relata.rag import AsyncRagClient, RagClient
from relata.rag_loop import (
    CORRECTIVE_FRACTION_CORRECT_FLOOR,
    HEURISTIC_PASS_THRESHOLD,
    HEURISTIC_RETRY_THRESHOLD,
    LOOP_CONFIDENCE_THRESHOLD,
    MAX_ITERATIONS,
    GateDecision,
    HitGrade,
    arun_agentic_loop,
    grade_hits,
    heuristic_gate,
    run_agentic_loop,
)

BASE = "http://localhost:9090"


def _hit(chunk_id: str, **overrides: Any) -> dict[str, Any]:
    base = {
        "bm25_score": 1.0,
        "vector_score": 1.0,
        "rerank_score": None,
        "chunk_id": chunk_id,
        "report_id": "doc-1",
        "text": f"text for {chunk_id}",
        "section_path": [],
        "page_start": 1,
        "page_end": 1,
        "prev_chunk_id": None,
        "next_chunk_id": None,
        "entity_ids": [],
    }
    base.update(overrides)
    return base


def _mock_client(handler: Any) -> RelataClient:
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


def _embedding_fn_for_score(score: float):
    """Return an embedding function under which any query text's embedding
    vs. any hit text's embedding (hit text always starts with "text for ",
    per `_hit()` above) has cosine similarity exactly `score`."""
    query_vec = [1.0, 0.0]
    orthogonal_component = math.sqrt(max(0.0, 1.0 - score * score))
    hit_vec = [score, orthogonal_component]

    def embedding_fn(text: str) -> list[float]:
        return hit_vec if text.startswith("text for ") else query_vec

    return embedding_fn


# ── heuristic gate — threshold defaults, verified against the ticket ───────


def test_heuristic_gate_threshold_defaults_match_verified_values():
    assert HEURISTIC_PASS_THRESHOLD == 0.85
    assert HEURISTIC_RETRY_THRESHOLD == 0.60
    assert LOOP_CONFIDENCE_THRESHOLD == 0.85
    assert MAX_ITERATIONS == 5


def test_heuristic_gate_passes_above_threshold():
    result = heuristic_gate([1.0, 0.0], [[1.0, 0.0], [1.0, 0.0]])
    assert result.score == pytest.approx(1.0)
    assert result.decision is GateDecision.PASS


def test_heuristic_gate_retries_below_threshold():
    result = heuristic_gate([1.0, 0.0], [[0.0, 1.0]])
    assert result.score == pytest.approx(0.0)
    assert result.decision is GateDecision.RETRY


def test_heuristic_gate_evaluates_between_thresholds():
    result = heuristic_gate([1.0, 0.0], [[0.7, math.sqrt(1 - 0.7**2)]])
    assert result.score == pytest.approx(0.7)
    assert result.decision is GateDecision.EVALUATE


def test_heuristic_gate_no_hits_retries():
    result = heuristic_gate([1.0, 0.0], [])
    assert result.score == 0.0
    assert result.decision is GateDecision.RETRY


def test_heuristic_gate_boundary_values_fall_into_evaluate():
    # Strict inequalities: exactly at either bar is neither an auto-pass nor
    # an auto-retry — it must go to the (cheap-model) evaluator, not be
    # silently treated as confident or worthless.
    pass_boundary = heuristic_gate([1.0, 0.0], [[0.85, math.sqrt(1 - 0.85**2)]])
    assert pass_boundary.decision is GateDecision.EVALUATE
    retry_boundary = heuristic_gate([1.0, 0.0], [[0.60, math.sqrt(1 - 0.60**2)]])
    assert retry_boundary.decision is GateDecision.EVALUATE


# ── corrective retrieval grading ────────────────────────────────────────────


def test_grade_hits_computes_fraction_correct():
    hits = RagQueryResponse.model_validate(
        {"hits": [_hit("c1"), _hit("c2"), _hit("c3")]}
    ).hits

    def grader_fn(query, graded_hits):
        return [HitGrade.CORRECT, HitGrade.INCORRECT, HitGrade.AMBIGUOUS]

    result = grade_hits("q", hits, grader_fn=grader_fn)
    assert result.fraction_correct == pytest.approx(1.0 / 3.0)
    assert result.grades == [HitGrade.CORRECT, HitGrade.INCORRECT, HitGrade.AMBIGUOUS]


def test_grade_hits_needs_requery_below_floor():
    hits = RagQueryResponse.model_validate({"hits": [_hit("c1"), _hit("c2")]}).hits

    def grader_fn(query, graded_hits):
        return [HitGrade.INCORRECT, HitGrade.INCORRECT]

    result = grade_hits("q", hits, grader_fn=grader_fn)
    assert result.fraction_correct < CORRECTIVE_FRACTION_CORRECT_FLOOR
    assert result.needs_requery is True


def test_grade_hits_does_not_need_requery_above_floor():
    hits = RagQueryResponse.model_validate({"hits": [_hit("c1"), _hit("c2")]}).hits

    def grader_fn(query, graded_hits):
        return [HitGrade.CORRECT, HitGrade.CORRECT]

    result = grade_hits("q", hits, grader_fn=grader_fn)
    assert result.needs_requery is False


def test_grade_hits_empty_hits_short_circuits():
    result = grade_hits("q", [], grader_fn=lambda q, h: [])
    assert result.fraction_correct == 0.0
    assert result.grades == []


def test_grade_hits_rejects_mismatched_grade_count():
    hits = RagQueryResponse.model_validate({"hits": [_hit("c1"), _hit("c2")]}).hits
    with pytest.raises(ValueError):
        grade_hits("q", hits, grader_fn=lambda q, h: [HitGrade.CORRECT])


# ── run_agentic_loop — the zero-LLM-calls property (AC #2) ─────────────────


def test_iter0_heuristic_pass_makes_zero_llm_calls():
    query_call_count = 0

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal query_call_count
        query_call_count += 1
        return httpx.Response(200, json={"hits": [_hit("c1"), _hit("c2")]})

    grader_calls: list[Any] = []
    hyde_calls: list[Any] = []

    def grader_fn(query, hits):
        grader_calls.append((query, hits))
        return [HitGrade.CORRECT for _ in hits]

    def hypothesis_fn(q: str) -> str:
        hyde_calls.append(q)
        return "hypothetical answer"

    client = _mock_client(handler)
    rag = RagClient.from_client(client)
    result = run_agentic_loop(
        rag,
        "What is RelataDB?",
        "DocumentChunk",
        purpose="research",
        embedding_fn=_embedding_fn_for_score(1.0),  # clears the pass bar
        grader_fn=grader_fn,
        hypothesis_fn=hypothesis_fn,
    )

    assert result.stopped_reason == "heuristic_pass"
    assert result.llm_calls == 0
    assert query_call_count == 1  # exactly one /rag/query call, nothing more
    assert grader_calls == []  # grader_fn never invoked
    assert hyde_calls == []  # hypothesis_fn never invoked
    assert len(result.iterations) == 1
    assert result.iterations[0].confidence == pytest.approx(1.0)


# ── run_agentic_loop — MAX_ITERATIONS hard cap under adversarial input ─────


def test_heuristic_retry_is_bounded_by_max_iterations():
    query_call_count = 0

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal query_call_count
        query_call_count += 1
        return httpx.Response(200, json={"hits": [_hit("c1")]})

    client = _mock_client(handler)
    rag = RagClient.from_client(client)
    # Score 0.0 never clears the retry floor — an adversarial embedding_fn
    # that always forces GateDecision.RETRY.
    result = run_agentic_loop(
        rag,
        "adversarial query",
        "DocumentChunk",
        purpose="research",
        embedding_fn=_embedding_fn_for_score(0.0),
        max_iterations=3,
    )

    assert result.stopped_reason == "max_iterations"
    assert query_call_count == 3
    assert len(result.iterations) == 3
    assert result.llm_calls == 0  # RETRY never reaches the evaluator


def test_corrective_grading_low_fraction_correct_is_bounded_by_max_iterations():
    query_call_count = 0

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal query_call_count
        query_call_count += 1
        return httpx.Response(200, json={"hits": [_hit("c1"), _hit("c2")]})

    # An adversarial grader that never reports confidence.
    def grader_fn(query, hits):
        return [HitGrade.INCORRECT for _ in hits]

    client = _mock_client(handler)
    rag = RagClient.from_client(client)
    result = run_agentic_loop(
        rag,
        "adversarial query",
        "DocumentChunk",
        purpose="research",
        embedding_fn=_embedding_fn_for_score(0.7),  # lands in EVALUATE
        grader_fn=grader_fn,
        max_iterations=3,
    )

    assert result.stopped_reason == "max_iterations"
    assert query_call_count == 3
    assert result.llm_calls == 3  # one grading call per iteration, no HyDE


def test_no_grader_configured_stops_immediately_on_evaluate():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"hits": [_hit("c1")]})

    client = _mock_client(handler)
    rag = RagClient.from_client(client)
    result = run_agentic_loop(
        rag,
        "q",
        "DocumentChunk",
        purpose="research",
        embedding_fn=_embedding_fn_for_score(0.7),
    )
    assert result.stopped_reason == "no_grader_configured"
    assert result.llm_calls == 0
    assert len(result.iterations) == 1


# ── run_agentic_loop — corrective grading reaching loop confidence ─────────


def test_corrective_grading_confident_stop():
    query_call_count = 0

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal query_call_count
        query_call_count += 1
        return httpx.Response(200, json={"hits": [_hit("c1"), _hit("c2")]})

    def grader_fn(query, hits):
        return [HitGrade.CORRECT for _ in hits]

    client = _mock_client(handler)
    rag = RagClient.from_client(client)
    result = run_agentic_loop(
        rag,
        "q",
        "DocumentChunk",
        purpose="research",
        embedding_fn=_embedding_fn_for_score(0.7),
        grader_fn=grader_fn,
    )
    assert result.stopped_reason == "confident"
    assert query_call_count == 1
    assert result.llm_calls == 1
    assert result.iterations[-1].confidence == pytest.approx(1.0)
    assert result.iterations[-1].confidence >= LOOP_CONFIDENCE_THRESHOLD


# ── run_agentic_loop — web-search fallback + HyDE requery refinement ───────


def test_web_search_fallback_used_when_needs_requery():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"hits": [_hit("c1")]})

    def grader_fn(query, hits):
        return [HitGrade.INCORRECT for _ in hits]

    fallback_calls: list[str] = []

    def web_search_fallback(q: str) -> RagQueryResponse:
        fallback_calls.append(q)
        return RagQueryResponse.model_validate({"hits": [_hit("web-1")]})

    client = _mock_client(handler)
    rag = RagClient.from_client(client)
    result = run_agentic_loop(
        rag,
        "q",
        "DocumentChunk",
        purpose="research",
        embedding_fn=_embedding_fn_for_score(0.7),
        grader_fn=grader_fn,
        web_search_fallback=web_search_fallback,
        max_iterations=5,
    )
    assert result.stopped_reason == "web_search_fallback"
    assert fallback_calls == ["q"]
    assert [h.chunk_id for h in result.response.hits] == ["web-1"]
    assert result.llm_calls == 1  # the one grading call; fallback isn't an LLM call


def test_hyde_refines_query_on_requery_and_is_called_at_most_once_per_retry():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"hits": [_hit("c1")]})

    def grader_fn(query, hits):
        return [HitGrade.INCORRECT for _ in hits]

    hyde_calls: list[str] = []

    def hypothesis_fn(q: str) -> str:
        hyde_calls.append(q)
        return "a refined hypothetical answer, not numeric-intent"

    client = _mock_client(handler)
    rag = RagClient.from_client(client)
    result = run_agentic_loop(
        rag,
        "How does hybrid retrieval work?",
        "DocumentChunk",
        purpose="research",
        embedding_fn=_embedding_fn_for_score(0.7),
        grader_fn=grader_fn,
        hypothesis_fn=hypothesis_fn,
        max_iterations=2,
    )
    assert result.stopped_reason == "max_iterations"
    assert len(result.iterations) == 2
    # HyDE only fires once — after iteration 0's low-confidence grading, not
    # again on the last iteration (which returns before requerying further).
    assert hyde_calls == ["How does hybrid retrieval work?"]
    assert result.llm_calls == 3  # 2 grading calls + 1 HyDE call
    assert result.iterations[1].query == "a refined hypothetical answer, not numeric-intent"


def test_run_agentic_loop_rejects_non_positive_max_iterations():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"hits": []})

    client = _mock_client(handler)
    rag = RagClient.from_client(client)
    with pytest.raises(ValueError):
        run_agentic_loop(
            rag,
            "q",
            "DocumentChunk",
            purpose="research",
            embedding_fn=_embedding_fn_for_score(1.0),
            max_iterations=0,
        )


# ── async variant ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_arun_agentic_loop_heuristic_pass_zero_llm_calls():
    query_call_count = 0

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal query_call_count
        query_call_count += 1
        return httpx.Response(200, json={"hits": [_hit("c1")]})

    grader_calls: list[Any] = []

    def grader_fn(query, hits):
        grader_calls.append((query, hits))
        return [HitGrade.CORRECT for _ in hits]

    client = _mock_client(handler)
    rag = AsyncRagClient.from_client(client)
    result = await arun_agentic_loop(
        rag,
        "What is RelataDB?",
        "DocumentChunk",
        purpose="research",
        embedding_fn=_embedding_fn_for_score(1.0),
        grader_fn=grader_fn,
    )
    assert result.stopped_reason == "heuristic_pass"
    assert result.llm_calls == 0
    assert query_call_count == 1
    assert grader_calls == []


@pytest.mark.asyncio
async def test_arun_agentic_loop_bounded_by_max_iterations():
    query_call_count = 0

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal query_call_count
        query_call_count += 1
        return httpx.Response(200, json={"hits": [_hit("c1")]})

    client = _mock_client(handler)
    rag = AsyncRagClient.from_client(client)
    result = await arun_agentic_loop(
        rag,
        "adversarial query",
        "DocumentChunk",
        purpose="research",
        embedding_fn=_embedding_fn_for_score(0.0),
        max_iterations=3,
    )
    assert result.stopped_reason == "max_iterations"
    assert query_call_count == 3
