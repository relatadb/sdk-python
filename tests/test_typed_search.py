"""Tests for the typed search surface — T1/T9 (#1983/#1991, epic #1982).

Covers the rank_by/filters normalisation, the typed /v1/search payload shape (no
top-level `query` key — its presence would make the server treat the body as
the legacy Meilisearch shape), and the QueryResult return. Uses
httpx.MockTransport — no live server required.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from relata import RelataClient
from relata._http import AsyncHttpTransport, HttpTransport
from relata.models import QueryResult
from relata.search import _build_payload, _normalise_filters, _normalise_rank_by

BASE = "http://localhost:9090"

# Governed /v1/query envelope (what the server's typed /v1/search door dispatches to).
_QUERY_PAYLOAD = {
    "rows": [{"id": "d1", "title": "Knowledge graphs 101"}],
    "columns": ["id", "title"],
    "processing_time_ms": 3,
}


def _mock_client(handler: Callable[[httpx.Request], httpx.Response]) -> RelataClient:
    client = RelataClient(BASE, bearer_token="tok")
    mock = httpx.MockTransport(handler)
    extra = client._extra_headers
    client._RelataClient__sync_transport = HttpTransport(  # type: ignore[attr-defined]
        BASE, client._bearer_token, client._timeout, transport=mock, extra_headers=extra
    )
    client._RelataClient__async_transport = AsyncHttpTransport(  # type: ignore[attr-defined]
        BASE, client._bearer_token, client._timeout, transport=mock, extra_headers=extra
    )
    return client


# ── filters normalisation ────────────────────────────────────────────────────


def test_filters_dict_becomes_eq() -> None:
    assert _normalise_filters({"status": "x", "age": 30}) == [
        {"field": "status", "op": "eq", "value": "x"},
        {"field": "age", "op": "eq", "value": 30},
    ]


def test_filters_list_keeps_ops() -> None:
    assert _normalise_filters([{"field": "age", "op": "gte", "value": 18}]) == [
        {"field": "age", "op": "gte", "value": 18},
    ]


def test_filters_default_op_is_eq() -> None:
    assert _normalise_filters([{"field": "x", "value": 1}]) == [
        {"field": "x", "op": "eq", "value": 1},
    ]


def test_filters_unknown_op_rejected() -> None:
    with pytest.raises(ValueError):
        _normalise_filters([{"field": "x", "op": "bogus", "value": 1}])


def test_filters_none() -> None:
    assert _normalise_filters(None) is None


# ── rank_by normalisation ────────────────────────────────────────────────────


def test_rank_by_pass_through() -> None:
    assert _normalise_rank_by(["bm25", "title", "graph"]) == ["bm25", "title", "graph"]


def test_rank_by_vector_form() -> None:
    assert _normalise_rank_by(["vector", "ann", "graph"]) == ["vector", "ann", "graph"]


def test_rank_by_none() -> None:
    assert _normalise_rank_by(None) is None


def test_rank_by_empty_list_dropped() -> None:
    assert _normalise_rank_by([]) is None


def test_rank_by_non_string_mode_rejected() -> None:
    with pytest.raises(ValueError):
        _normalise_rank_by([1, "title", "x"])


def test_rank_by_bad_type_rejected() -> None:
    with pytest.raises(TypeError):
        _normalise_rank_by("bm25")  # type: ignore[arg-type]


# ── payload assembly ─────────────────────────────────────────────────────────


def test_payload_text_compiles_to_bm25_rank_by() -> None:
    payload = _build_payload(
        namespace="Document",
        text="graph retrieval",
        match_column="title",
        rank_by=None,
        filters=None,
        limit=10,
        include_attributes=None,
        consistency=None,
        purpose=None,
    )
    # Crucially NO top-level "query" key (would trigger legacy /v1/search shape).
    assert "query" not in payload
    assert payload["from"] == "Document"
    assert payload["rank_by"] == ["bm25", "title", "graph retrieval"]
    assert payload["limit"] == 10


def test_payload_text_default_column_is_star() -> None:
    payload = _build_payload(
        namespace="Doc", text="x", match_column="*", rank_by=None, filters=None,
        limit=5, include_attributes=None, consistency=None, purpose=None,
    )
    assert payload["rank_by"] == ["bm25", "*", "x"]


def test_payload_explicit_rank_by_passes_through() -> None:
    payload = _build_payload(
        namespace="Doc", text=None, match_column="*", rank_by=["vector", "ann", "q"],
        filters=None, limit=5, include_attributes=None, consistency=None, purpose=None,
    )
    assert payload["rank_by"] == ["vector", "ann", "q"]
    assert "query" not in payload


def test_payload_text_and_rank_by_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError):
        _build_payload(
            namespace="Doc", text="x", match_column="*", rank_by=["bm25", "t", "x"],
            filters=None, limit=5, include_attributes=None, consistency=None, purpose=None,
        )


def test_payload_carries_filters_include_attributes_consistency_purpose() -> None:
    payload = _build_payload(
        namespace="Doc", text="x", match_column="title", rank_by=None,
        filters=[{"field": "status", "op": "eq", "value": "published"}],
        limit=5, include_attributes=["id", "title"], consistency="eventual", purpose="rag",
    )
    assert payload["filters"] == [{"field": "status", "op": "eq", "value": "published"}]
    assert payload["include_attributes"] == ["id", "title"]
    assert payload["consistency"] == "eventual"
    assert payload["purpose"] == "rag"


def test_payload_carries_compute_attributes() -> None:
    payload = _build_payload(
        namespace="Doc", text="x", match_column="*", rank_by=None, filters=None,
        limit=5, include_attributes=None, consistency=None,
        compute_attributes={"bm25_score": "BM25()"}, purpose=None,
    )
    assert payload["compute_attributes"] == {"bm25_score": "BM25()"}


def test_payload_no_text_no_rank_by_is_filter_only() -> None:
    payload = _build_payload(
        namespace="Doc", text=None, match_column="*", rank_by=None,
        filters={"status": "published"}, limit=5, include_attributes=None,
        consistency=None, purpose=None,
    )
    assert "rank_by" not in payload
    assert payload["filters"] == [{"field": "status", "op": "eq", "value": "published"}]


# ── SearchClient.query → POST /v1/search ────────────────────────────────────────


def test_search_client_posts_typed_body_and_returns_query_result() -> None:
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/v1/search"
        captured.update(json.loads(req.content))
        return httpx.Response(200, json=_QUERY_PAYLOAD)

    from relata.search import SearchClient

    client = _mock_client(handler)
    res = SearchClient.from_client(client).query(
        "Document",
        text="graph retrieval",
        match_column="title",
        filters=[{"field": "status", "op": "eq", "value": "published"}],
        limit=10,
    )
    assert captured["from"] == "Document"
    assert captured["rank_by"] == ["bm25", "title", "graph retrieval"]
    assert captured["filters"] == [{"field": "status", "op": "eq", "value": "published"}]
    assert "query" not in captured
    assert isinstance(res, QueryResult)
    assert len(res) == 1
    assert res.rows[0]["title"] == "Knowledge graphs 101"


@pytest.mark.asyncio
async def test_async_search_client_query() -> None:
    from relata.search import AsyncSearchClient

    client = _mock_client(lambda req: httpx.Response(200, json=_QUERY_PAYLOAD))
    res = await AsyncSearchClient.from_client(client).query("Document", text="x")
    assert isinstance(res, QueryResult)


# ── compression off by default ───────────────────────────────────────────────


def test_compression_disabled_by_default() -> None:
    client = RelataClient(BASE, bearer_token="tok")
    _ = client._sync
    assert client._sync._client.headers.get("accept-encoding") == "identity"


def test_compression_enabled_opt_in() -> None:
    client = RelataClient(BASE, bearer_token="tok", compress=True)
    _ = client._sync
    assert client._sync._client.headers.get("accept-encoding") != "identity"
