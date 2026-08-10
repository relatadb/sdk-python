"""Tests for sub-agent fan-out + deterministic (non-LLM) merge (#4526).

Built against #4525's `rag_loop` module (merged, PR #4558) and #4523's typed
`RagClient`/`AsyncRagClient`, via `httpx.MockTransport` — no live
`/rag/query` server required, mirroring `test_rag_loop.py`'s approach.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from relata import RelataClient
from relata._http import AsyncHttpTransport, HttpTransport
from relata.rag import AsyncRagClient, RagClient
from relata.rag_loop import (
    LOW_CONFIDENCE_FLOOR,
    MAX_FANOUT_STRATEGIES,
    MERGE_THRESHOLD,
    MIN_FANOUT_STRATEGIES,
    SubAgentStrategy,
    arun_subagent_fanout,
    run_subagent_fanout,
)

BASE = "http://localhost:9090"


def _hit(chunk_id: str, *, relevance_confidence: float | None, **overrides: Any) -> dict[str, Any]:
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
        "relevance_confidence": relevance_confidence,
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


def _strategies() -> list[SubAgentStrategy]:
    return [
        SubAgentStrategy("lexical", {"search_mode": "lexical"}),
        SubAgentStrategy("dense", {"search_mode": "dense"}),
        SubAgentStrategy("hybrid", {"search_mode": "hybrid"}),
    ]


# ── verified default thresholds ─────────────────────────────────────────────


def test_fanout_threshold_defaults_match_verified_values():
    assert LOW_CONFIDENCE_FLOOR == 0.20
    assert MERGE_THRESHOLD == 0.50
    assert MIN_FANOUT_STRATEGIES == 2
    assert MAX_FANOUT_STRATEGIES == 5


# ── deterministic winner selection (AC #1) ──────────────────────────────────


def test_winner_is_strict_argmax_by_confidence_no_llm_call():
    confidences = {"lexical": 0.9, "dense": 0.3, "hybrid": 0.1}

    def handler(req: httpx.Request) -> httpx.Response:
        mode = json.loads(req.content)["search_mode"]
        return httpx.Response(
            200, json={"hits": [_hit(f"{mode}-1", relevance_confidence=confidences[mode])]}
        )

    client = _mock_client(handler)
    rag = RagClient.from_client(client)
    result = run_subagent_fanout(rag, "q", "DocumentChunk", _strategies(), purpose="research")

    # Deterministic argmax — no arbitration callable exists anywhere in this
    # call, so "no LLM call in the merge path" is provable by construction:
    # run_subagent_fanout's signature accepts no LLM-shaped callable at all.
    assert result.winner.strategy.name == "lexical"
    assert result.winner.confidence == pytest.approx(0.9)


def test_ties_broken_by_strategy_declaration_order():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"hits": [_hit("c1", relevance_confidence=0.9)]})

    client = _mock_client(handler)
    rag = RagClient.from_client(client)
    strategies = _strategies()  # lexical, dense, hybrid — all score 0.9
    result = run_subagent_fanout(rag, "q", "DocumentChunk", strategies, purpose="research")
    assert result.winner.strategy.name == "lexical"


# ── LOW_CONFIDENCE_FLOOR — excluded as noise before any expensive step ─────


def test_low_confidence_strategies_excluded_as_noise():
    confidences = {"lexical": 0.9, "dense": 0.1, "hybrid": 0.05}

    def handler(req: httpx.Request) -> httpx.Response:
        mode = json.loads(req.content)["search_mode"]
        return httpx.Response(
            200, json={"hits": [_hit(f"{mode}-1", relevance_confidence=confidences[mode])]}
        )

    client = _mock_client(handler)
    rag = RagClient.from_client(client)
    result = run_subagent_fanout(rag, "q", "DocumentChunk", _strategies(), purpose="research")

    assert {r.strategy.name for r in result.excluded} == {"dense", "hybrid"}
    assert result.included == [result.winner]


def test_all_strategies_below_floor_falls_back_to_best_scoring():
    confidences = {"lexical": 0.15, "dense": 0.1, "hybrid": 0.05}

    def handler(req: httpx.Request) -> httpx.Response:
        mode = json.loads(req.content)["search_mode"]
        return httpx.Response(
            200, json={"hits": [_hit(f"{mode}-1", relevance_confidence=confidences[mode])]}
        )

    client = _mock_client(handler)
    rag = RagClient.from_client(client)
    result = run_subagent_fanout(rag, "q", "DocumentChunk", _strategies(), purpose="research")

    assert result.winner.strategy.name == "lexical"
    assert len(result.excluded) == 2


# ── MERGE_THRESHOLD — supporting evidence folded in via RRF merge ──────────


def test_survivor_above_merge_threshold_is_folded_in():
    confidences = {"lexical": 0.9, "dense": 0.6, "hybrid": 0.05}

    def handler(req: httpx.Request) -> httpx.Response:
        mode = json.loads(req.content)["search_mode"]
        return httpx.Response(
            200, json={"hits": [_hit(f"{mode}-1", relevance_confidence=confidences[mode])]}
        )

    client = _mock_client(handler)
    rag = RagClient.from_client(client)
    result = run_subagent_fanout(rag, "q", "DocumentChunk", _strategies(), purpose="research")

    assert {r.strategy.name for r in result.included} == {"lexical", "dense"}
    merged_ids = {h.chunk_id for h in result.merged_response.hits}
    assert merged_ids == {"lexical-1", "dense-1"}


def test_survivor_below_merge_threshold_is_not_folded_in():
    confidences = {"lexical": 0.9, "dense": 0.4, "hybrid": 0.3}

    def handler(req: httpx.Request) -> httpx.Response:
        mode = json.loads(req.content)["search_mode"]
        return httpx.Response(
            200, json={"hits": [_hit(f"{mode}-1", relevance_confidence=confidences[mode])]}
        )

    client = _mock_client(handler)
    rag = RagClient.from_client(client)
    result = run_subagent_fanout(rag, "q", "DocumentChunk", _strategies(), purpose="research")

    assert result.included == [result.winner]
    assert result.merged_response is result.winner.response
    assert {h.chunk_id for h in result.merged_response.hits} == {"lexical-1"}


# ── missing relevance_confidence (pre-#4520 server) defaults to 0.0 ────────


def test_missing_relevance_confidence_defaults_to_zero_and_is_excluded():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"hits": [_hit("c1", relevance_confidence=None)]})

    client = _mock_client(handler)
    rag = RagClient.from_client(client)
    result = run_subagent_fanout(rag, "q", "DocumentChunk", _strategies(), purpose="research")
    assert result.winner.confidence == 0.0
    # All three strategies tie at 0.0 (below the floor) -> fallback keeps one
    # winner and excludes the other two, never crashes on an all-noise set.
    assert len(result.excluded) == 2


def test_empty_hits_scores_zero_confidence():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"hits": []})

    client = _mock_client(handler)
    rag = RagClient.from_client(client)
    result = run_subagent_fanout(rag, "q", "DocumentChunk", _strategies(), purpose="research")
    assert result.winner.confidence == 0.0


# ── strategy-count validation ───────────────────────────────────────────────


def test_rejects_fewer_than_two_strategies():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"hits": []})

    client = _mock_client(handler)
    rag = RagClient.from_client(client)
    with pytest.raises(ValueError):
        run_subagent_fanout(rag, "q", "DocumentChunk", [_strategies()[0]], purpose="research")


def test_rejects_more_than_five_strategies():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"hits": []})

    client = _mock_client(handler)
    rag = RagClient.from_client(client)
    six = [SubAgentStrategy(f"s{i}", {"search_mode": "hybrid"}) for i in range(6)]
    with pytest.raises(ValueError):
        run_subagent_fanout(rag, "q", "DocumentChunk", six, purpose="research")


# ── every strategy issues exactly one /rag/query call, nothing more ────────


def test_each_strategy_issues_exactly_one_call():
    call_count = 0

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(200, json={"hits": [_hit("c1", relevance_confidence=0.9)]})

    client = _mock_client(handler)
    rag = RagClient.from_client(client)
    run_subagent_fanout(rag, "q", "DocumentChunk", _strategies(), purpose="research")
    assert call_count == len(_strategies())


# ── async variant ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_arun_subagent_fanout_matches_sync_semantics():
    confidences = {"lexical": 0.9, "dense": 0.6, "hybrid": 0.05}

    def handler(req: httpx.Request) -> httpx.Response:
        mode = json.loads(req.content)["search_mode"]
        return httpx.Response(
            200, json={"hits": [_hit(f"{mode}-1", relevance_confidence=confidences[mode])]}
        )

    client = _mock_client(handler)
    rag = AsyncRagClient.from_client(client)
    result = await arun_subagent_fanout(
        rag, "q", "DocumentChunk", _strategies(), purpose="research"
    )
    assert result.winner.strategy.name == "lexical"
    assert {r.strategy.name for r in result.included} == {"lexical", "dense"}
