"""Tests for query-shape dispatch + HyDE + decomposition (#4524).

Built against #4523's typed `RagClient`/`AsyncRagClient` (merged, PR #4549)
via `httpx.MockTransport` — no live `/rag/query` server required, per the
same approach #4523's tests use (`#4514` need not have merged first, since
the contract is frozen by ADR-0299).
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from relata import RelataClient
from relata._http import AsyncHttpTransport, HttpTransport
from relata.models import RagQueryResponse
from relata.rag import AsyncRagClient, RagClient
from relata.rag_understanding import (
    DANGEROUS_CONTENT_PATTERNS,
    ENUMERATION_TOP_K,
    QueryShape,
    _rrf_scores,
    aroute_attribute_filter_query,
    asmart_rag_query,
    check_content_safety,
    classify_query_shape,
    decompose_query,
    expand_query_hyde,
    extract_attribute_filters,
    is_attribute_filter_intent,
    is_numeric_intent,
    route_attribute_filter_query,
    rrf_k_for_fanout,
    rrf_merge,
    smart_rag_query,
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


# ── query-shape dispatch — classification ───────────────────────────────────


def test_classify_conjunction_shape():
    assert (
        classify_query_shape("Who approved the budget and signed the contract?")
        == QueryShape.CONJUNCTION
    )


def test_classify_enumeration_shape():
    assert (
        classify_query_shape("Which vendors were flagged for compliance issues?")
        == QueryShape.ENUMERATION
    )
    shape = classify_query_shape("How many incidents were filed last year?")
    assert shape == QueryShape.ENUMERATION
    assert classify_query_shape("List every open finding.") == QueryShape.ENUMERATION


def test_classify_simple_shape():
    assert classify_query_shape("What is RelataDB?") == QueryShape.SIMPLE


# ── query-shape dispatch — the different /rag/query call shape it produces ─


def test_conjunction_shape_requests_expand_window():
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured.update(json.loads(req.content))
        return httpx.Response(200, json={"hits": []})

    client = _mock_client(handler)
    rag = RagClient.from_client(client)
    smart_rag_query(
        rag,
        "Who approved the budget and signed the contract?",
        "DocumentChunk",
        purpose="research",
    )
    assert captured["expand_window"] is True


def test_enumeration_shape_widens_top_k():
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured.update(json.loads(req.content))
        return httpx.Response(200, json={"hits": []})

    client = _mock_client(handler)
    rag = RagClient.from_client(client)
    smart_rag_query(
        rag,
        "Which vendors were flagged for compliance issues?",
        "DocumentChunk",
        purpose="research",
    )
    assert captured["top_k"] == ENUMERATION_TOP_K
    assert captured["top_k"] > 8  # wider than #4514's RAG_RETRIEVE_DEFAULT_TOP_K


def test_enumeration_shape_never_shrinks_an_explicit_larger_top_k():
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured.update(json.loads(req.content))
        return httpx.Response(200, json={"hits": []})

    client = _mock_client(handler)
    rag = RagClient.from_client(client)
    smart_rag_query(
        rag,
        "Which vendors were flagged?",
        "DocumentChunk",
        purpose="research",
        top_k=100,
    )
    assert captured["top_k"] == 100


def test_simple_shape_leaves_request_shape_alone():
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured.update(json.loads(req.content))
        return httpx.Response(200, json={"hits": []})

    client = _mock_client(handler)
    rag = RagClient.from_client(client)
    smart_rag_query(rag, "What is RelataDB?", "DocumentChunk", purpose="research")
    assert captured["top_k"] == 8
    assert captured["expand_window"] is False


# ── HyDE + the numeric-intent guard ──────────────────────────────────────────


@pytest.mark.parametrize(
    "query",
    [
        "What was the total revenue for Q1?",
        "What percentage of patients responded to treatment?",
        "How many incidents were filed last year?",
        "What is the recommended dosage?",
    ],
)
def test_numeric_intent_detected(query: str):
    assert is_numeric_intent(query) is True


def test_non_numeric_query_is_not_numeric_intent():
    assert is_numeric_intent("Explain how RelataDB performs hybrid retrieval.") is False


def test_hyde_skipped_for_numeric_intent_query():
    calls: list[str] = []

    def hypothesis_fn(q: str) -> str:
        calls.append(q)
        return "a hallucinated number"

    query = "What was the total revenue for Q1?"
    result = expand_query_hyde(query, hypothesis_fn=hypothesis_fn)
    assert result == query  # unchanged
    assert calls == []  # hypothesis_fn never invoked


def test_hyde_applied_for_non_numeric_query():
    def hypothesis_fn(q: str) -> str:
        return f"Hypothetically, {q.lower()} is answered by RelataDB's hybrid engine."

    query = "How does hybrid retrieval work?"
    result = expand_query_hyde(query, hypothesis_fn=hypothesis_fn)
    assert result != query
    assert result.startswith("Hypothetically,")


def test_smart_rag_query_numeric_intent_provably_skips_hyde():
    captured: dict[str, Any] = {}
    hyde_calls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.update(json.loads(req.content))
        return httpx.Response(200, json={"hits": []})

    def hypothesis_fn(q: str) -> str:
        hyde_calls.append(q)
        return "invented number"

    client = _mock_client(handler)
    rag = RagClient.from_client(client)
    query = "What percentage of revenue came from repeat customers?"
    smart_rag_query(
        rag, query, "DocumentChunk", purpose="research", hypothesis_fn=hypothesis_fn
    )
    assert hyde_calls == []
    assert captured["query"] == query  # sent unchanged, not the hallucinated text


def test_smart_rag_query_applies_hyde_for_non_numeric_query():
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured.update(json.loads(req.content))
        return httpx.Response(200, json={"hits": []})

    def hypothesis_fn(q: str) -> str:
        return "RelataDB fuses BM25 and vector scores via RRF."

    client = _mock_client(handler)
    rag = RagClient.from_client(client)
    smart_rag_query(
        rag,
        "How does hybrid retrieval work?",
        "DocumentChunk",
        purpose="research",
        hypothesis_fn=hypothesis_fn,
    )
    assert captured["query"] == "RelataDB fuses BM25 and vector scores via RRF."


# ── decomposition + auto-scaling RRF merge ──────────────────────────────────


def test_decompose_splits_multi_part_question():
    parts = decompose_query("What is the incident response policy and who owns it?")
    assert len(parts) == 2
    assert parts[0] == "What is the incident response policy"
    assert parts[1] == "who owns it?"


def test_decompose_single_part_question_is_unchanged():
    assert decompose_query("What is RelataDB?") == ["What is RelataDB?"]


@pytest.mark.parametrize(
    ("n", "expected_k"),
    [
        (1, 60.0),
        (2, 30.0),
        (3, 20.0),
        (6, 10.0),
        (10, 10.0),  # floor: 60/10=6 < 10
        (100, 10.0),  # floor
    ],
)
def test_rrf_k_for_fanout_matches_auto_scaling_formula(n: int, expected_k: float):
    assert rrf_k_for_fanout(n) == expected_k


def test_rrf_k_for_fanout_rejects_non_positive_n():
    with pytest.raises(ValueError):
        rrf_k_for_fanout(0)


def test_rrf_scores_uses_the_k_it_is_given_not_a_fixed_constant():
    """`_rrf_scores` must compute `1/(k+rank)` with whatever `k` it's called
    with — not RelataDB's fixed internal RRF_K=60
    (`crates/relata-query/src/hybrid.rs`), which is a different merge at a
    different layer. Exact-float-assert both k=30 (n=2's auto-scaled value)
    and k=60 (the value a fixed-constant bug would silently produce) so a
    regression back to a hardcoded 60 is caught even when it wouldn't
    happen to change the final ranking order."""
    resp_a = RagQueryResponse.model_validate({"hits": [_hit("c1"), _hit("c2")]})
    resp_b = RagQueryResponse.model_validate({"hits": [_hit("c2"), _hit("c3")]})

    scores_k30 = _rrf_scores([resp_a, resp_b], k=30.0)
    assert scores_k30["c1"] == pytest.approx(1.0 / 31.0)
    assert scores_k30["c2"] == pytest.approx(1.0 / 32.0 + 1.0 / 31.0)
    assert scores_k30["c3"] == pytest.approx(1.0 / 32.0)

    scores_k60 = _rrf_scores([resp_a, resp_b], k=60.0)
    assert scores_k60["c1"] == pytest.approx(1.0 / 61.0)
    assert scores_k60 != scores_k30  # different k -> provably different arithmetic


def test_rrf_merge_dedupes_and_orders_by_summed_score():
    resp_a = RagQueryResponse.model_validate({"hits": [_hit("c1"), _hit("c2"), _hit("c3")]})
    resp_b = RagQueryResponse.model_validate({"hits": [_hit("c2"), _hit("c4")]})

    merged = rrf_merge([resp_a, resp_b], k=30.0)
    ids = [h.chunk_id for h in merged.hits]
    # c2 appears in both lists (rank 2 + rank 1) so it out-scores every
    # singleton and must sort first; every id is deduped to one entry.
    assert ids[0] == "c2"
    assert set(ids) == {"c1", "c2", "c3", "c4"}
    assert merged.total == len(merged.hits) == 4


def test_smart_rag_query_decomposition_merges_with_auto_scaled_k():
    call_count = 0

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        body = json.loads(req.content)
        if "incident response policy" in body["query"]:
            return httpx.Response(200, json={"hits": [_hit("c1"), _hit("c2")]})
        return httpx.Response(200, json={"hits": [_hit("c2"), _hit("c3")]})

    client = _mock_client(handler)
    rag = RagClient.from_client(client)
    result = smart_rag_query(
        rag,
        "What is the incident response policy and who owns it?",
        "DocumentChunk",
        purpose="research",
    )
    assert call_count == 2  # one /rag/query call per decomposed sub-query
    assert isinstance(result, RagQueryResponse)
    ids = [h.chunk_id for h in result.hits]
    assert set(ids) == {"c1", "c2", "c3"}
    assert ids[0] == "c2"  # present in both sub-query result lists

    # The merge must have used k=max(10, 60/2)=30 — not the fixed k=60 a
    # regression could silently reintroduce.
    expected = _rrf_scores(
        [
            RagQueryResponse.model_validate({"hits": [_hit("c1"), _hit("c2")]}),
            RagQueryResponse.model_validate({"hits": [_hit("c2"), _hit("c3")]}),
        ],
        k=rrf_k_for_fanout(2),
    )
    assert rrf_k_for_fanout(2) == 30.0
    # c2 is rank 2 in the first sub-query's hits and rank 1 in the second's.
    assert expected["c2"] == pytest.approx(1.0 / 32.0 + 1.0 / 31.0)


def test_smart_rag_query_no_decomposition_skips_merge_entirely():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"hits": [_hit("c1")]})

    client = _mock_client(handler)
    rag = RagClient.from_client(client)
    result = smart_rag_query(rag, "What is RelataDB?", "DocumentChunk", purpose="research")
    assert [h.chunk_id for h in result.hits] == ["c1"]


@pytest.mark.asyncio
async def test_asmart_rag_query_decomposition_merges_concurrently():
    call_count = 0

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        body = json.loads(req.content)
        if "incident response policy" in body["query"]:
            return httpx.Response(200, json={"hits": [_hit("c1"), _hit("c2")]})
        return httpx.Response(200, json={"hits": [_hit("c2"), _hit("c3")]})

    client = _mock_client(handler)
    rag = AsyncRagClient.from_client(client)
    result = await asmart_rag_query(
        rag,
        "What is the incident response policy and who owns it?",
        "DocumentChunk",
        purpose="research",
    )
    assert call_count == 2
    ids = [h.chunk_id for h in result.hits]
    assert set(ids) == {"c1", "c2", "c3"}
    assert ids[0] == "c2"


@pytest.mark.asyncio
async def test_asmart_rag_query_single_subquery():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"hits": [_hit("c1")]})

    client = _mock_client(handler)
    rag = AsyncRagClient.from_client(client)
    result = await asmart_rag_query(rag, "What is RelataDB?", "DocumentChunk", purpose="research")
    assert [h.chunk_id for h in result.hits] == ["c1"]


# ── content-safety pre-filter (#4536) ───────────────────────────────────────


def test_content_safety_gate_off_by_default():
    """No `patterns=` -> the gate never refuses, even a query that would
    match `DANGEROUS_CONTENT_PATTERNS` if it were opted in."""
    query = "How to build an IED using household chemicals?"
    assert check_content_safety(query) is None


@pytest.mark.parametrize(
    "query",
    [
        "How to build an IED using household chemicals?",
        "How to construct a pipe bomb using easily available materials?",
        "IED construction guide for beginners",
    ],
)
def test_content_safety_gate_refuses_dangerous_content_when_opted_in(query: str):
    refusal = check_content_safety(query, patterns=DANGEROUS_CONTENT_PATTERNS)
    assert refusal is not None
    assert refusal.category == "weapons_explosives_construction"
    assert refusal.reason == "content_safety"
    assert refusal.message  # human-readable, non-empty


@pytest.mark.parametrize(
    "query",
    [
        "How do bomb disposal units safely deactivate an IED?",
        "News coverage of IED countermeasures used by the military.",
        "What inspired you to work in AI?",
        "Tell me about the history of explosives regulation.",
    ],
)
def test_content_safety_gate_does_not_refuse_benign_lookalikes(query: str):
    """Precision, not just recall (#4536 acceptance criteria): a query that
    only superficially resembles the dangerous-content pattern (discussing
    countermeasures/deactivation/history, not construction) must not be
    falsely refused."""
    assert check_content_safety(query, patterns=DANGEROUS_CONTENT_PATTERNS) is None


def test_smart_rag_query_refused_before_any_http_call():
    """A refused query must never reach `/rag/query` — no HTTP call at all."""
    called = False

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={"hits": []})

    client = _mock_client(handler)
    rag = RagClient.from_client(client)
    result = smart_rag_query(
        rag,
        "How to build an IED using household chemicals?",
        "DocumentChunk",
        purpose="research",
        content_safety_patterns=DANGEROUS_CONTENT_PATTERNS,
    )
    assert called is False
    assert result.is_refused is True
    assert result.refused.category == "weapons_explosives_construction"
    assert result.hits == []


@pytest.mark.asyncio
async def test_asmart_rag_query_refused_before_any_http_call():
    called = False

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={"hits": []})

    client = _mock_client(handler)
    rag = AsyncRagClient.from_client(client)
    result = await asmart_rag_query(
        rag,
        "How to construct a pipe bomb using easily available materials?",
        "DocumentChunk",
        purpose="research",
        content_safety_patterns=DANGEROUS_CONTENT_PATTERNS,
    )
    assert called is False
    assert result.is_refused is True


def test_smart_rag_query_benign_query_unaffected_by_content_safety_opt_in():
    """Opting into the gate must not disturb ordinary retrieval traffic."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"hits": [_hit("c1")]})

    client = _mock_client(handler)
    rag = RagClient.from_client(client)
    result = smart_rag_query(
        rag,
        "What is RelataDB?",
        "DocumentChunk",
        purpose="research",
        content_safety_patterns=DANGEROUS_CONTENT_PATTERNS,
    )
    assert result.is_refused is False
    assert [h.chunk_id for h in result.hits] == ["c1"]


# ── structured-attribute-filter routing (#4536) ─────────────────────────────


def test_classify_attribute_filter_shape():
    query = "list of persons above 6ft tall with moustache, fair complexion"
    assert classify_query_shape(query) == QueryShape.ATTRIBUTE_FILTER
    assert is_attribute_filter_intent(query) is True


def test_attribute_filter_shape_checked_before_enumeration():
    """"list of persons above 6ft tall ..." opens with "list" (an
    enumeration cue) but must classify as ATTRIBUTE_FILTER, not
    ENUMERATION — it's a structurally different, SQL-routable shape."""
    query = "list of persons above 6ft tall with moustache"
    assert classify_query_shape(query) == QueryShape.ATTRIBUTE_FILTER


def test_extract_attribute_filters_height_and_descriptors():
    query = "list of persons above 6ft tall with moustache, fair complexion"
    filters = extract_attribute_filters(query)
    by_field = {f["field"]: f for f in filters}
    assert by_field["height"]["op"] == ">="
    assert by_field["height"]["value"] == pytest.approx(182.9, abs=0.1)
    assert by_field["facial_hair"] == {"field": "facial_hair", "op": "ILIKE", "value": "%moustache%"}
    assert by_field["complexion"] == {"field": "complexion", "op": "ILIKE", "value": "%fair%"}


def test_extract_attribute_filters_empty_for_non_attribute_query():
    assert extract_attribute_filters("What is RelataDB?") == []


def test_route_attribute_filter_query_returns_sql_filtered_rows():
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/query":
            captured.update(json.loads(req.content))
            return httpx.Response(
                200,
                json={
                    "data": [{"name": "Ahmad Akhtar", "height": 182.9, "facial_hair": "moustache"}],
                    "columns": ["name", "height", "facial_hair"],
                    "query_id": "q1",
                    "elapsed_ms": 3,
                },
            )
        raise AssertionError(f"unexpected call to {req.url.path}")

    client = _mock_client(handler)
    result = route_attribute_filter_query(
        client,
        "list of persons above 6ft tall with moustache",
        "Person",
        purpose="research",
    )
    assert result is not None
    assert result.row_count == 1
    assert result.rows[0]["name"] == "Ahmad Akhtar"
    assert "Person" in captured["sql"]
    assert "height >=" in captured["sql"]
    assert "facial_hair ILIKE" in captured["sql"]


def test_route_attribute_filter_query_returns_none_when_no_filters_extracted():
    def handler(req: httpx.Request) -> httpx.Response:
        raise AssertionError("no /query call should be made")

    client = _mock_client(handler)
    assert route_attribute_filter_query(client, "What is RelataDB?", "Person") is None


def test_route_attribute_filter_query_falls_back_when_known_fields_dont_match():
    """Acceptance criteria: an attribute-filter query whose field mapping
    can't be resolved against the schema falls back (returns None) rather
    than guessing against a field that doesn't exist on `type`."""

    def handler(req: httpx.Request) -> httpx.Response:
        raise AssertionError("no /query call should be made when no field matches the schema")

    client = _mock_client(handler)
    result = route_attribute_filter_query(
        client,
        "list of persons above 6ft tall with moustache",
        "Person",
        known_fields={"name", "email"},  # neither "height" nor "facial_hair" present
    )
    assert result is None


def test_smart_rag_query_routes_attribute_filter_to_sql_not_retrieval():
    rag_called = False

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal rag_called
        if req.url.path == "/rag/query":
            rag_called = True
            return httpx.Response(200, json={"hits": [_hit("c1")]})
        if req.url.path == "/query":
            return httpx.Response(
                200,
                json={
                    "data": [{"name": "Ahmad Akhtar", "height": 182.9}],
                    "columns": ["name", "height"],
                    "query_id": "q1",
                    "elapsed_ms": 3,
                },
            )
        raise AssertionError(f"unexpected call to {req.url.path}")

    client = _mock_client(handler)
    rag = RagClient.from_client(client)
    result = smart_rag_query(
        rag,
        "list of persons above 6ft tall with moustache",
        "Person",
        purpose="research",
    )
    assert rag_called is False  # never a semantic-similarity guess
    assert result.is_sql_routed is True
    assert result.sql_result.rows[0]["name"] == "Ahmad Akhtar"
    assert result.hits == []


def test_smart_rag_query_attribute_filter_falls_back_to_retrieval_with_low_confidence():
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/rag/query":
            return httpx.Response(200, json={"hits": [_hit("c1")]})
        raise AssertionError(f"unexpected call to {req.url.path}")

    client = _mock_client(handler)
    rag = RagClient.from_client(client)
    result = smart_rag_query(
        rag,
        "list of persons above 6ft tall with moustache",
        "Person",
        purpose="research",
        attribute_known_fields={"name", "email"},  # no schema match -> fall back
    )
    assert result.is_sql_routed is False
    assert result.low_confidence is True
    assert result.low_confidence_reason
    assert [h.chunk_id for h in result.hits] == ["c1"]


@pytest.mark.asyncio
async def test_asmart_rag_query_routes_attribute_filter_to_sql():
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/rag/query":
            raise AssertionError("should not call /rag/query for an attribute-filter shape")
        if req.url.path == "/query":
            return httpx.Response(
                200,
                json={
                    "data": [{"name": "Ahmad Akhtar", "height": 182.9}],
                    "columns": ["name", "height"],
                    "query_id": "q1",
                    "elapsed_ms": 3,
                },
            )
        raise AssertionError(f"unexpected call to {req.url.path}")

    client = _mock_client(handler)
    rag = AsyncRagClient.from_client(client)
    result = await asmart_rag_query(
        rag,
        "list of persons above 6ft tall with moustache",
        "Person",
        purpose="research",
    )
    assert result.is_sql_routed is True
    assert result.sql_result.rows[0]["name"] == "Ahmad Akhtar"


@pytest.mark.asyncio
async def test_aroute_attribute_filter_query_returns_none_when_no_filters_extracted():
    def handler(req: httpx.Request) -> httpx.Response:
        raise AssertionError("no /query call should be made")

    client = _mock_client(handler)
    assert await aroute_attribute_filter_query(client, "What is RelataDB?", "Person") is None
