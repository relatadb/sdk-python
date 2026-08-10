"""Tests for the typed ``/rag/query`` client — RAG epic foundational SDK
ticket (#4523, wraps #4514's ADR-0299 contract).

Built and tested against the documented request/response contract with
``httpx.MockTransport`` — no live server required, since #4514 need not have
merged yet (the contract is frozen; #4523's own scope is a typed client
against that frozen shape, not a live integration test).
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from relata import RelataClient
from relata._http import AsyncHttpTransport, HttpTransport
from relata.exceptions import PurposeError
from relata.models import Clarification, EntityCandidate, RagHit, RagQueryResponse
from relata.rag import (
    CANONICAL_ENTITY_ID_FIELD,
    AsyncRagClient,
    RagClient,
    _build_rag_payload,
    filters_for_entity_ids,
)

BASE = "http://localhost:9090"

# A clarification object per #4514/#4534's contract — two candidates within
# margin of each other (see relata-identity::active_learning::disambiguate).
_CLARIFICATION = {
    "type": "entity_disambiguation",
    "question": 'Multiple entities match "Ahmad". Which one?',
    "candidates": [
        {
            "entity_id": "ent-ahmad-akhtar",
            "label": "Ahmad Akhtar",
            "document_count": 14,
            "top_aliases": ["A. Akhtar"],
        },
        {
            "entity_id": "ent-ahmad-nawaz",
            "label": "Ahmad Nawaz",
            "document_count": 6,
            "top_aliases": [],
        },
    ],
}
_AMBIGUOUS_PAYLOAD = {"hits": [], "clarification": _CLARIFICATION}

# A full response per #4514's per-hit contract table — every field present,
# bm25_score/vector_score kept separate (never fused).
_RAG_HIT = {
    "bm25_score": 4.2,
    "vector_score": 0.83,
    "rerank_score": None,
    "chunk_id": "chunk-1",
    "report_id": "doc-1",
    "text": "RelataDB fuses BM25 and vector retrieval natively.",
    "section_path": ["3", "3.2"],
    "page_start": 5,
    "page_end": 6,
    "prev_chunk_id": None,
    "next_chunk_id": "chunk-2",
    "entity_ids": ["ent-1", "ent-2"],
}
_RAG_PAYLOAD = {"hits": [_RAG_HIT]}


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


# ── payload assembly — #4514's exact request table ─────────────────────────


def test_payload_carries_every_required_field():
    payload = _build_rag_payload(
        query="graph databases",
        type="DocumentChunk",
        top_k=8,
        rerank=False,
        search_mode="hybrid",
        embedding_slot="text",
        filters=None,
        as_of=None,
        purpose="research",
        expand_window=False,
        graph_hops=0,
    )
    assert payload == {
        "query": "graph databases",
        "type": "DocumentChunk",
        "top_k": 8,
        "rerank": False,
        "search_mode": "hybrid",
        "embedding_slot": "text",
        "filters": [],
        "as_of": None,
        "purpose": "research",
        "expand_window": False,
        "graph_hops": 0,
    }


def test_payload_defaults_match_4514_contract():
    payload = _build_rag_payload(
        query="q", type="DocumentChunk", top_k=8, rerank=False, search_mode="hybrid",
        embedding_slot="text", filters=None, as_of=None, purpose="research",
        expand_window=False, graph_hops=0,
    )
    assert payload["top_k"] == 8  # RAG_RETRIEVE_DEFAULT_TOP_K
    assert payload["search_mode"] == "hybrid"
    assert payload["embedding_slot"] == "text"
    assert payload["rerank"] is False
    assert payload["expand_window"] is False
    assert payload["graph_hops"] == 0


def test_payload_requires_query():
    with pytest.raises(ValueError):
        _build_rag_payload(
            query="", type="DocumentChunk", top_k=8, rerank=False, search_mode="hybrid",
            embedding_slot="text", filters=None, as_of=None, purpose="research",
            expand_window=False, graph_hops=0,
        )


def test_payload_requires_type():
    with pytest.raises(ValueError):
        _build_rag_payload(
            query="q", type="", top_k=8, rerank=False, search_mode="hybrid",
            embedding_slot="text", filters=None, as_of=None, purpose="research",
            expand_window=False, graph_hops=0,
        )


def test_payload_requires_purpose():
    with pytest.raises(PurposeError):
        _build_rag_payload(
            query="q", type="DocumentChunk", top_k=8, rerank=False, search_mode="hybrid",
            embedding_slot="text", filters=None, as_of=None, purpose="",
            expand_window=False, graph_hops=0,
        )


def test_payload_rejects_bad_search_mode():
    with pytest.raises(ValueError):
        _build_rag_payload(
            query="q", type="DocumentChunk", top_k=8, rerank=False, search_mode="bogus",
            embedding_slot="text", filters=None, as_of=None, purpose="research",
            expand_window=False, graph_hops=0,
        )


def test_payload_rejects_bad_embedding_slot():
    with pytest.raises(ValueError):
        _build_rag_payload(
            query="q", type="DocumentChunk", top_k=8, rerank=False, search_mode="hybrid",
            embedding_slot="bogus", filters=None, as_of=None, purpose="research",
            expand_window=False, graph_hops=0,
        )


def test_payload_normalises_filters_dict_shorthand():
    payload = _build_rag_payload(
        query="q", type="DocumentChunk", top_k=8, rerank=False, search_mode="hybrid",
        embedding_slot="text", filters={"status": "published"}, as_of=None, purpose="research",
        expand_window=False, graph_hops=0,
    )
    assert payload["filters"] == [{"field": "status", "op": "eq", "value": "published"}]


def test_payload_carries_as_of():
    payload = _build_rag_payload(
        query="q", type="DocumentChunk", top_k=8, rerank=False, search_mode="hybrid",
        embedding_slot="text", filters=None, as_of="2026-01-01T00:00:00Z", purpose="research",
        expand_window=False, graph_hops=0,
    )
    assert payload["as_of"] == "2026-01-01T00:00:00Z"


# ── RagHit / RagQueryResponse — no field dropped or fused ──────────────────


def test_rag_hit_keeps_bm25_and_vector_scores_separate():
    hit = RagHit.model_validate(_RAG_HIT)
    assert hit.bm25_score == 4.2
    assert hit.vector_score == 0.83
    assert hit.rerank_score is None
    assert hit.chunk_id == "chunk-1"
    assert hit.report_id == "doc-1"
    assert hit.text.startswith("RelataDB")
    assert hit.section_path == ["3", "3.2"]
    assert hit.page_start == 5
    assert hit.page_end == 6
    assert hit.prev_chunk_id is None
    assert hit.next_chunk_id == "chunk-2"
    assert hit.entity_ids == ["ent-1", "ent-2"]


def test_rag_query_response_total_mirrors_hit_count():
    resp = RagQueryResponse.model_validate(_RAG_PAYLOAD)
    assert resp.total == 1
    assert len(resp.hits) == 1


# ── RagClient.query → POST /rag/query ───────────────────────────────────────


def test_rag_client_posts_full_contract_and_returns_typed_response():
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/rag/query"
        captured.update(json.loads(req.content))
        return httpx.Response(200, json=_RAG_PAYLOAD)

    client = _mock_client(handler)
    res = RagClient.from_client(client).query(
        "graph retrieval",
        "DocumentChunk",
        top_k=5,
        rerank=True,
        search_mode="lexical",
        embedding_slot="summary",
        filters=[{"field": "status", "op": "eq", "value": "published"}],
        as_of="2026-01-01T00:00:00Z",
        purpose="research",
        expand_window=True,
        graph_hops=2,
    )
    assert captured["query"] == "graph retrieval"
    assert captured["type"] == "DocumentChunk"
    assert captured["top_k"] == 5
    assert captured["rerank"] is True
    assert captured["search_mode"] == "lexical"
    assert captured["embedding_slot"] == "summary"
    assert captured["filters"] == [{"field": "status", "op": "eq", "value": "published"}]
    assert captured["as_of"] == "2026-01-01T00:00:00Z"
    assert captured["purpose"] == "research"
    assert captured["expand_window"] is True
    assert captured["graph_hops"] == 2
    assert isinstance(res, RagQueryResponse)
    assert res.hits[0].bm25_score == 4.2
    assert res.hits[0].vector_score == 0.83


def test_rag_client_falls_back_to_client_default_purpose():
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured.update(json.loads(req.content))
        return httpx.Response(200, json=_RAG_PAYLOAD)

    client = _mock_client(handler)
    client._default_purpose = "default-purpose"  # type: ignore[attr-defined]
    RagClient.from_client(client).query("q", "DocumentChunk")
    assert captured["purpose"] == "default-purpose"


def test_rag_client_raises_purpose_error_when_none_available():
    client = _mock_client(lambda req: httpx.Response(200, json=_RAG_PAYLOAD))
    with pytest.raises(PurposeError):
        RagClient.from_client(client).query("q", "DocumentChunk")


@pytest.mark.asyncio
async def test_async_rag_client_query():
    client = _mock_client(lambda req: httpx.Response(200, json=_RAG_PAYLOAD))
    res = await AsyncRagClient.from_client(client).query(
        "q", "DocumentChunk", purpose="research"
    )
    assert isinstance(res, RagQueryResponse)
    assert res.hits[0].chunk_id == "chunk-1"


# ── Clarification / EntityCandidate — #4534 ─────────────────────────────────


def test_clarification_round_trips_every_candidate_field():
    c = Clarification.model_validate(_CLARIFICATION)
    assert c.type == "entity_disambiguation"
    assert c.question == 'Multiple entities match "Ahmad". Which one?'
    assert len(c.candidates) == 2
    first = c.candidates[0]
    assert isinstance(first, EntityCandidate)
    assert first.entity_id == "ent-ahmad-akhtar"
    assert first.label == "Ahmad Akhtar"
    assert first.document_count == 14
    assert first.top_aliases == ["A. Akhtar"]


def test_entity_candidate_defaults_are_safe():
    c = EntityCandidate(entity_id="e1", label="Someone")
    assert c.document_count == 0
    assert c.top_aliases == []


def test_rag_query_response_carries_clarification_and_is_ambiguous():
    resp = RagQueryResponse.model_validate(_AMBIGUOUS_PAYLOAD)
    assert resp.is_ambiguous
    assert resp.clarification is not None
    assert len(resp.clarification.candidates) == 2
    assert resp.total == 0  # no hits when ambiguous


def test_rag_query_response_without_clarification_is_not_ambiguous():
    resp = RagQueryResponse.model_validate(_RAG_PAYLOAD)
    assert not resp.is_ambiguous
    assert resp.clarification is None


def test_rag_client_query_surfaces_clarification():
    client = _mock_client(lambda req: httpx.Response(200, json=_AMBIGUOUS_PAYLOAD))
    res = RagClient.from_client(client).query("Ahmad", "Person", purpose="research")
    assert res.is_ambiguous
    assert res.clarification.candidates[0].entity_id == "ent-ahmad-akhtar"


# ── filters_for_entity_ids / resume_with_selection — #4534 ─────────────────


def test_filters_for_entity_ids_builds_in_filter():
    filters = filters_for_entity_ids(["e1", "e2"])
    assert filters == [{"field": CANONICAL_ENTITY_ID_FIELD, "op": "in", "value": ["e1", "e2"]}]


def test_filters_for_entity_ids_rejects_empty_selection():
    with pytest.raises(ValueError):
        filters_for_entity_ids([])


def test_resume_with_selection_scopes_follow_up_to_chosen_entity():
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured.update(json.loads(req.content))
        return httpx.Response(200, json=_RAG_PAYLOAD)

    client = _mock_client(handler)
    res = RagClient.from_client(client).resume_with_selection(
        "Ahmad", "Person", ["ent-ahmad-akhtar"], purpose="research"
    )
    assert captured["filters"] == [
        {"field": "canonical_entity_id", "op": "in", "value": ["ent-ahmad-akhtar"]}
    ]
    assert isinstance(res, RagQueryResponse)


@pytest.mark.asyncio
async def test_async_resume_with_selection_scopes_follow_up_to_chosen_entity():
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured.update(json.loads(req.content))
        return httpx.Response(200, json=_RAG_PAYLOAD)

    client = _mock_client(handler)
    res = await AsyncRagClient.from_client(client).resume_with_selection(
        "Ahmad", "Person", ["ent-ahmad-akhtar", "ent-ahmad-other"], purpose="research"
    )
    assert captured["filters"] == [
        {
            "field": "canonical_entity_id",
            "op": "in",
            "value": ["ent-ahmad-akhtar", "ent-ahmad-other"],
        }
    ]
    assert isinstance(res, RagQueryResponse)
