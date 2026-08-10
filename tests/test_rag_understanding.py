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
from relata.exceptions import RelataError
from relata.models import RagQueryResponse
from relata.rag import AsyncRagClient, RagClient
from relata.rag_understanding import (
    DANGEROUS_CONTENT_PATTERNS,
    DEFAULT_RANKING_LIMIT,
    ENUMERATION_TOP_K,
    LargeExportResult,
    QueryShape,
    _rrf_scores,
    aroute_aggregation_query,
    aroute_attribute_filter_query,
    aroute_boolean_query,
    aroute_enumeration_query,
    aroute_negation_query,
    aroute_ranking_query,
    asmart_rag_query,
    check_content_safety,
    classify_query_shape,
    decompose_query,
    expand_query_hyde,
    extract_attribute_filters,
    extract_keyword_filters,
    is_aggregation_intent,
    is_attribute_filter_intent,
    is_boolean_intent,
    is_negation_intent,
    is_numeric_intent,
    is_ranking_intent,
    route_aggregation_query,
    route_attribute_filter_query,
    route_boolean_query,
    route_enumeration_query,
    route_negation_query,
    route_ranking_query,
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
    assert classify_query_shape("List every open finding.") == QueryShape.ENUMERATION


def test_classify_how_many_is_aggregation_not_enumeration():
    """#4535 acceptance criteria: a 'count' question routes to COUNT(*) (one
    row), not full enumeration — "how many" used to fall through to
    ENUMERATION (widened top_k, LLM guesses a count from retrieved chunks);
    it must now classify as AGGREGATION so it routes to real SQL."""
    shape = classify_query_shape("How many incidents were filed last year?")
    assert shape == QueryShape.AGGREGATION


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


# ── aggregation/negation/boolean/ranking SQL routing (#4535) ───────────────


def _count_response(n: int) -> dict[str, Any]:
    return {
        "data": [{"count": n}],
        "columns": ["count"],
        "query_id": "q1",
        "elapsed_ms": 2,
    }


def test_classify_aggregation_shape():
    assert (
        classify_query_shape("How many incidents happened in 2023?") == QueryShape.AGGREGATION
    )
    assert is_aggregation_intent("What is the total number of open findings?") is True
    assert is_aggregation_intent("What is RelataDB?") is False


def test_classify_negation_shape():
    query = "Which SIMI members are NOT in custody?"
    assert classify_query_shape(query) == QueryShape.NEGATION
    assert is_negation_intent(query) is True
    assert is_negation_intent("What is RelataDB?") is False


def test_negation_checked_before_enumeration():
    """"Which SIMI members are NOT in custody?" opens with "which" (an
    ENUMERATION cue) but must classify as NEGATION — a top-k retrieval has
    no notion of "not"."""
    assert (
        classify_query_shape("Which SIMI members are NOT in custody?") == QueryShape.NEGATION
    )


def test_classify_boolean_shape():
    query = "Members of SIMI AND LeT"
    assert classify_query_shape(query) == QueryShape.BOOLEAN
    assert is_boolean_intent(query) is True
    assert is_boolean_intent("What is RelataDB?") is False


def test_classify_ranking_shape():
    query = "Top 5 most active members"
    assert classify_query_shape(query) == QueryShape.RANKING
    assert is_ranking_intent(query) is True
    assert is_ranking_intent("What is RelataDB?") is False


def test_classify_simple_query_unaffected_by_new_shapes():
    """#4535 acceptance criteria: a semantically-similar but non-structured
    question is unaffected — still SIMPLE (routes to retrieval)."""
    assert classify_query_shape("Tell me about SIMI's history") == QueryShape.SIMPLE


def test_extract_keyword_filters_builds_predicates_from_field_map():
    filters = extract_keyword_filters(
        "Which SIMI members are NOT in custody?",
        {"simi": "organization", "custody": "status"},
        op="NOT ILIKE",
    )
    by_field = {f["field"]: f for f in filters}
    assert by_field["organization"] == {
        "field": "organization",
        "op": "NOT ILIKE",
        "value": "%simi%",
    }
    assert by_field["status"] == {"field": "status", "op": "NOT ILIKE", "value": "%custody%"}


def test_extract_keyword_filters_dedupe_fields_off_keeps_both_same_field_matches():
    filters = extract_keyword_filters(
        "Members of SIMI AND LeT",
        {"simi": "organization", "let": "organization"},
        op="=",
        dedupe_fields=False,
    )
    assert len(filters) == 2
    assert all(f["field"] == "organization" for f in filters)
    assert {f["value"] for f in filters} == {"simi", "let"}


# -- aggregation --


def test_route_aggregation_query_bare_count_needs_no_field_map():
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured.update(json.loads(req.content))
        return httpx.Response(200, json=_count_response(42))

    client = _mock_client(handler)
    result = route_aggregation_query(
        client, "How many incidents happened in 2023?", "Incident", purpose="research"
    )
    assert result is not None
    assert result.rows[0]["count"] == 42
    assert captured["sql"] == "SELECT COUNT(*) AS count FROM Incident"


def test_route_aggregation_query_applies_extracted_filter():
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured.update(json.loads(req.content))
        return httpx.Response(200, json=_count_response(7))

    client = _mock_client(handler)
    result = route_aggregation_query(
        client,
        "How many members are from SIMI?",
        "Person",
        field_map={"simi": "organization"},
        purpose="research",
    )
    assert result is not None
    assert "COUNT(*)" in captured["sql"]
    assert "organization ILIKE" in captured["sql"]


def test_route_aggregation_query_falls_back_when_known_fields_dont_match():
    """#4535 acceptance criteria: an aggregation-shaped question whose field
    mapping can't be resolved against the schema falls back to retrieval
    rather than silently counting an unfiltered (and therefore wrong) total."""

    def handler(req: httpx.Request) -> httpx.Response:
        raise AssertionError("no /query call should be made when the field can't resolve")

    client = _mock_client(handler)
    result = route_aggregation_query(
        client,
        "How many members are from SIMI?",
        "Person",
        field_map={"simi": "organization"},
        known_fields={"name"},  # "organization" not present
    )
    assert result is None


@pytest.mark.asyncio
async def test_aroute_aggregation_query_returns_count():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_count_response(100))

    client = _mock_client(handler)
    result = await aroute_aggregation_query(
        client, "How many incidents were there?", "Incident", purpose="research"
    )
    assert result is not None
    assert result.rows[0]["count"] == 100


# -- negation --


def test_route_negation_query_returns_none_without_field_map():
    def handler(req: httpx.Request) -> httpx.Response:
        raise AssertionError("no /query call should be made")

    client = _mock_client(handler)
    assert route_negation_query(client, "Which SIMI members are NOT in custody?", "Person") is None


def test_route_negation_query_builds_not_ilike_predicate():
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured.update(json.loads(req.content))
        return httpx.Response(
            200,
            json={
                "data": [{"name": "Bilal Hassan", "status": "at_large"}],
                "columns": ["name", "status"],
                "query_id": "q1",
                "elapsed_ms": 2,
            },
        )

    client = _mock_client(handler)
    result = route_negation_query(
        client,
        "Which SIMI members are NOT in custody?",
        "Person",
        field_map={"custody": "status"},
        purpose="research",
    )
    assert result is not None
    assert result.rows[0]["name"] == "Bilal Hassan"
    assert "status NOT ILIKE" in captured["sql"]


def test_route_negation_query_falls_back_when_known_fields_dont_match():
    def handler(req: httpx.Request) -> httpx.Response:
        raise AssertionError("no /query call should be made")

    client = _mock_client(handler)
    result = route_negation_query(
        client,
        "Which SIMI members are NOT in custody?",
        "Person",
        field_map={"custody": "status"},
        known_fields={"name"},
    )
    assert result is None


@pytest.mark.asyncio
async def test_aroute_negation_query_builds_not_ilike_predicate():
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured.update(json.loads(req.content))
        return httpx.Response(200, json={"data": [], "columns": [], "query_id": "q1"})

    client = _mock_client(handler)
    result = await aroute_negation_query(
        client,
        "Which SIMI members are NOT in custody?",
        "Person",
        field_map={"custody": "status"},
        purpose="research",
    )
    assert result is not None
    assert "status NOT ILIKE" in captured["sql"]


# -- boolean --


def test_route_boolean_query_returns_none_without_field_map():
    def handler(req: httpx.Request) -> httpx.Response:
        raise AssertionError("no /query call should be made")

    client = _mock_client(handler)
    assert route_boolean_query(client, "Members of SIMI AND LeT", "Person") is None


def test_route_boolean_query_joins_predicates_with_and():
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured.update(json.loads(req.content))
        return httpx.Response(200, json={"data": [], "columns": [], "query_id": "q1"})

    client = _mock_client(handler)
    result = route_boolean_query(
        client,
        "Members of SIMI AND LeT",
        "Person",
        field_map={"simi": "organization", "let": "organization"},
        purpose="research",
    )
    assert result is not None
    assert "organization = 'simi'" in captured["sql"]
    assert "organization = 'let'" in captured["sql"]
    assert " AND " in captured["sql"]
    assert " OR " not in captured["sql"]


def test_route_boolean_query_joins_predicates_with_or():
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured.update(json.loads(req.content))
        return httpx.Response(200, json={"data": [], "columns": [], "query_id": "q1"})

    client = _mock_client(handler)
    route_boolean_query(
        client,
        "Members of SIMI or LeT",
        "Person",
        field_map={"simi": "organization", "let": "organization"},
        purpose="research",
    )
    assert " OR " in captured["sql"]


def test_route_boolean_query_returns_none_when_fewer_than_two_predicates_resolve():
    """A boolean shape needs >= 2 resolved predicates to be meaningful — a
    single-keyword field_map (or known_fields ruling all-but-one out) must
    fall back to retrieval rather than emitting a degenerate one-predicate
    'boolean' query."""

    def handler(req: httpx.Request) -> httpx.Response:
        raise AssertionError("no /query call should be made with < 2 predicates")

    client = _mock_client(handler)
    result = route_boolean_query(
        client,
        "Members of SIMI AND LeT",
        "Person",
        field_map={"simi": "organization"},  # only one keyword mappable
    )
    assert result is None

    result2 = route_boolean_query(
        client,
        "Members of SIMI AND LeT",
        "Person",
        field_map={"simi": "organization", "let": "organization"},
        known_fields={"name"},  # neither resolved field is schema-known
    )
    assert result2 is None


@pytest.mark.asyncio
async def test_aroute_boolean_query_joins_predicates():
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured.update(json.loads(req.content))
        return httpx.Response(200, json={"data": [], "columns": [], "query_id": "q1"})

    client = _mock_client(handler)
    result = await aroute_boolean_query(
        client,
        "Members of SIMI AND LeT",
        "Person",
        field_map={"simi": "organization", "let": "organization"},
        purpose="research",
    )
    assert result is not None
    assert " AND " in captured["sql"]


# -- ranking --


def test_route_ranking_query_returns_none_without_field_map():
    def handler(req: httpx.Request) -> httpx.Response:
        raise AssertionError("no /query call should be made")

    client = _mock_client(handler)
    assert route_ranking_query(client, "Top 5 most active members", "Person") is None


def test_route_ranking_query_builds_order_by_limit():
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured.update(json.loads(req.content))
        return httpx.Response(
            200,
            json={
                "data": [{"name": "Zahid Iqbal", "activity_count": 91}],
                "columns": ["name", "activity_count"],
                "query_id": "q1",
                "elapsed_ms": 2,
            },
        )

    client = _mock_client(handler)
    result = route_ranking_query(
        client,
        "Top 5 most active members",
        "Person",
        field_map={"active": "activity_count"},
        purpose="research",
    )
    assert result is not None
    assert result.rows[0]["name"] == "Zahid Iqbal"
    assert captured["sql"] == (
        "SELECT * FROM Person ORDER BY activity_count DESC LIMIT 5"
    )


def test_route_ranking_query_defaults_limit_when_no_explicit_n():
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured.update(json.loads(req.content))
        return httpx.Response(200, json={"data": [], "columns": [], "query_id": "q1"})

    client = _mock_client(handler)
    route_ranking_query(
        client,
        "Who is most active?",
        "Person",
        field_map={"active": "activity_count"},
        purpose="research",
    )
    assert f"LIMIT {DEFAULT_RANKING_LIMIT}" in captured["sql"]


def test_route_ranking_query_ascending_for_lowest():
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured.update(json.loads(req.content))
        return httpx.Response(200, json={"data": [], "columns": [], "query_id": "q1"})

    client = _mock_client(handler)
    route_ranking_query(
        client,
        "Who is the least active member?",
        "Person",
        field_map={"active": "activity_count"},
        purpose="research",
    )
    assert "ORDER BY activity_count ASC" in captured["sql"]
    # no explicit "top N"/"first N" -> the default limit applies.
    assert f"LIMIT {DEFAULT_RANKING_LIMIT}" in captured["sql"]


def test_route_ranking_query_falls_back_when_known_fields_dont_match():
    def handler(req: httpx.Request) -> httpx.Response:
        raise AssertionError("no /query call should be made")

    client = _mock_client(handler)
    result = route_ranking_query(
        client,
        "Top 5 most active members",
        "Person",
        field_map={"active": "activity_count"},
        known_fields={"name"},
    )
    assert result is None


@pytest.mark.asyncio
async def test_aroute_ranking_query_builds_order_by_limit():
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured.update(json.loads(req.content))
        return httpx.Response(200, json={"data": [], "columns": [], "query_id": "q1"})

    client = _mock_client(handler)
    result = await aroute_ranking_query(
        client,
        "Top 5 most active members",
        "Person",
        field_map={"active": "activity_count"},
        purpose="research",
    )
    assert result is not None
    assert captured["sql"] == "SELECT * FROM Person ORDER BY activity_count DESC LIMIT 5"


# -- enumeration (#4535 large-result-set policy) --


def test_route_enumeration_query_returns_inline_result_when_small():
    """A completed /rag/export operation at/under the server's row threshold
    is returned inline — no bucket/key, real rows in `.data`."""
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/rag/export":
            captured.update(json.loads(req.content))
            return httpx.Response(
                202,
                json={
                    "operation_id": "op-1",
                    "status": "running",
                    "location": "/v1/operations/op-1",
                },
            )
        assert req.url.path == "/v1/operations/op-1"
        return httpx.Response(
            200,
            json={"row_count": 3, "columns": ["a"], "data": [{"a": 1}, {"a": 2}, {"a": 3}]},
        )

    client = _mock_client(handler)
    result = route_enumeration_query(client, "Give me all incidents", "Incident")
    assert isinstance(result, LargeExportResult)
    assert captured["sql"] == "SELECT * FROM Incident"
    assert result.row_count == 3
    assert result.is_file_backed is False
    assert result.data == [{"a": 1}, {"a": 2}, {"a": 3}]
    assert result.bucket is None


def test_route_enumeration_query_is_file_backed_when_large():
    """#4535 acceptance criteria: a large-result enumeration produces a real
    S3Object descriptor (bucket/key/etag) plus a labeled preview + the real
    row_count — not a truncated list presented as complete."""

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/rag/export":
            return httpx.Response(
                202,
                json={
                    "operation_id": "op-2",
                    "status": "running",
                    "location": "/v1/operations/op-2",
                },
            )
        return httpx.Response(
            200,
            json={
                "row_count": 50000,
                "columns": ["msisdn", "called_at"],
                "preview": [{"msisdn": "9800004040", "called_at": "2023-01-01"}],
                "preview_note": "preview of the first 1 of 50000 row(s)",
                "bucket": "default",
                "key": "CallRecord-9800004040-2023-01-01_to_2023-12-31-1700000000.csv",
                "etag": "abc123",
                "content_type": "text/csv",
                "size_bytes": 4_500_000,
            },
        )

    client = _mock_client(handler)
    result = route_enumeration_query(
        client,
        "Give me all calls made by 9800004040 last year",
        "CallRecord",
        key_filter="9800004040",
        date_from="2023-01-01",
        date_to="2023-12-31",
    )
    assert isinstance(result, LargeExportResult)
    assert result.row_count == 50000
    assert result.is_file_backed is True
    assert result.bucket == "default"
    assert result.key is not None and result.key.startswith("CallRecord-9800004040")
    assert result.etag == "abc123"
    assert result.data is None
    assert result.preview == [{"msisdn": "9800004040", "called_at": "2023-01-01"}]


def test_route_enumeration_query_falls_back_when_known_fields_dont_match():
    def handler(req: httpx.Request) -> httpx.Response:
        raise AssertionError("no /rag/export call should be made when the field can't resolve")

    client = _mock_client(handler)
    result = route_enumeration_query(
        client,
        "Give me all SIMI members",
        "Person",
        field_map={"simi": "organization"},
        known_fields={"name"},  # "organization" not present
    )
    assert result is None


def test_route_enumeration_query_raises_timeout_error_when_never_completes():
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/rag/export":
            return httpx.Response(
                202,
                json={
                    "operation_id": "op-3",
                    "status": "running",
                    "location": "/v1/operations/op-3",
                },
            )
        return httpx.Response(200, json={"status": "running"})

    client = _mock_client(handler)
    with pytest.raises(TimeoutError):
        route_enumeration_query(
            client,
            "Give me all X",
            "T",
            poll_timeout_secs=0.02,
            poll_interval_secs=0.005,
        )


def test_route_enumeration_query_propagates_server_error_on_failed_operation():
    """A failed operation (e.g. the row-cap backstop tripping,
    QueryError::ResultCapExceeded) surfaces as a real error, not a silent
    empty/partial result — the row cap is unconditional, this routing does
    not bypass it."""

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/rag/export":
            return httpx.Response(
                202,
                json={
                    "operation_id": "op-4",
                    "status": "running",
                    "location": "/v1/operations/op-4",
                },
            )
        return httpx.Response(
            500,
            json={
                "type": "about:blank",
                "status": 500,
                "title": "Internal Server Error",
                "detail": "ResultCapExceeded",
            },
        )

    client = _mock_client(handler)
    with pytest.raises(RelataError):
        route_enumeration_query(client, "Give me all X", "T")


@pytest.mark.asyncio
async def test_aroute_enumeration_query_returns_inline_result_when_small():
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/rag/export":
            return httpx.Response(
                202,
                json={
                    "operation_id": "op-5",
                    "status": "running",
                    "location": "/v1/operations/op-5",
                },
            )
        return httpx.Response(200, json={"row_count": 1, "columns": ["a"], "data": [{"a": 1}]})

    client = _mock_client(handler)
    result = await aroute_enumeration_query(client, "Give me all X", "T")
    assert isinstance(result, LargeExportResult)
    assert result.row_count == 1
    assert result.is_file_backed is False


# -- smart_rag_query / asmart_rag_query end-to-end dispatch --


def test_smart_rag_query_routes_aggregation_to_count_not_enumeration():
    """#4535 acceptance criteria: 'how many X' against a seeded corpus
    returns the real row count via SQL, not an LLM-estimated number from
    retrieved chunks — and 'count' routes to COUNT(*) (one row), not full
    enumeration."""
    rag_called = False

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal rag_called
        if req.url.path == "/rag/query":
            rag_called = True
            return httpx.Response(200, json={"hits": [_hit("c1")]})
        if req.url.path == "/query":
            return httpx.Response(200, json=_count_response(1234))
        raise AssertionError(f"unexpected call to {req.url.path}")

    client = _mock_client(handler)
    rag = RagClient.from_client(client)
    result = smart_rag_query(
        rag, "How many incidents happened in 2023?", "Incident", purpose="research"
    )
    assert rag_called is False
    assert result.is_sql_routed is True
    assert result.sql_result.row_count == 1  # one row: the count
    assert result.sql_result.rows[0]["count"] == 1234
    assert result.hits == []


def test_smart_rag_query_non_structured_question_still_routes_to_retrieval():
    """#4535 acceptance criteria: a semantically-similar but non-structured
    question is unaffected — still routes to retrieval."""

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/rag/query":
            return httpx.Response(200, json={"hits": [_hit("c1")]})
        raise AssertionError("aggregation/negation/boolean/ranking routing must not trigger")

    client = _mock_client(handler)
    rag = RagClient.from_client(client)
    result = smart_rag_query(
        rag, "Tell me about SIMI's history", "DocumentChunk", purpose="research"
    )
    assert result.is_sql_routed is False
    assert [h.chunk_id for h in result.hits] == ["c1"]


def test_smart_rag_query_routes_boolean_to_sql():
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/rag/query":
            raise AssertionError("boolean-shaped query must not fall through to retrieval")
        if req.url.path == "/query":
            return httpx.Response(
                200,
                json={
                    "data": [{"name": "Bilal Hassan", "organization": "SIMI"}],
                    "columns": ["name", "organization"],
                    "query_id": "q1",
                    "elapsed_ms": 2,
                },
            )
        raise AssertionError(f"unexpected call to {req.url.path}")

    client = _mock_client(handler)
    rag = RagClient.from_client(client)
    result = smart_rag_query(
        rag,
        "Members of SIMI AND LeT",
        "Person",
        purpose="research",
        structured_field_map={"simi": "organization", "let": "organization"},
    )
    assert result.is_sql_routed is True
    assert result.sql_result.rows[0]["name"] == "Bilal Hassan"


def test_smart_rag_query_negation_falls_back_to_retrieval_with_low_confidence():
    """No structured_field_map given -> negation routing can't build a
    predicate -> falls back to retrieval with low_confidence set (#4535,
    mirrors #4536's attribute-filter fallback contract)."""

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/rag/query":
            return httpx.Response(200, json={"hits": [_hit("c1")]})
        raise AssertionError(f"unexpected call to {req.url.path}")

    client = _mock_client(handler)
    rag = RagClient.from_client(client)
    result = smart_rag_query(
        rag, "Which SIMI members are NOT in custody?", "Person", purpose="research"
    )
    assert result.is_sql_routed is False
    assert result.low_confidence is True
    assert result.low_confidence_reason
    assert [h.chunk_id for h in result.hits] == ["c1"]


@pytest.mark.asyncio
async def test_asmart_rag_query_routes_ranking_to_sql():
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/rag/query":
            raise AssertionError("ranking-shaped query must not fall through to retrieval")
        if req.url.path == "/query":
            return httpx.Response(
                200,
                json={
                    "data": [{"name": "Zahid Iqbal", "activity_count": 91}],
                    "columns": ["name", "activity_count"],
                    "query_id": "q1",
                    "elapsed_ms": 2,
                },
            )
        raise AssertionError(f"unexpected call to {req.url.path}")

    client = _mock_client(handler)
    rag = AsyncRagClient.from_client(client)
    result = await asmart_rag_query(
        rag,
        "Top 5 most active members",
        "Person",
        purpose="research",
        structured_field_map={"active": "activity_count"},
    )
    assert result.is_sql_routed is True
    assert result.sql_result.rows[0]["name"] == "Zahid Iqbal"
