"""Tests for the flagship namespace() handle — T9 (#1991, epic #1982).

Covers client.namespace("Doc").query/.write/.get/.delete_all and
namespace-name validation. Uses httpx.MockTransport — no live server required.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from relata import AsyncNamespace, Namespace, RelataClient
from relata._http import AsyncHttpTransport, HttpTransport

BASE = "http://localhost:9090"

_QUERY_PAYLOAD = {
    "rows": [{"id": "d1", "title": "Graphs 101"}],
    "columns": ["id", "title"],
    "processing_time_ms": 2,
}

_SCHEMALESS_RESP = {
    "object_type": "Document",
    "type_created": True,
    "inferred_schema": {"id": "text", "title": "text"},
    "inserted": 2,
}


def _mock_client(handler: Callable[[httpx.Request], httpx.Response]) -> RelataClient:
    client = RelataClient(BASE, bearer_token="tok", purpose="rag")
    mock = httpx.MockTransport(handler)
    extra = client._extra_headers
    client._RelataClient__sync_transport = HttpTransport(  # type: ignore[attr-defined]
        BASE, client._bearer_token, client._timeout, transport=mock, extra_headers=extra
    )
    client._RelataClient__async_transport = AsyncHttpTransport(  # type: ignore[attr-defined]
        BASE, client._bearer_token, client._timeout, transport=mock, extra_headers=extra
    )
    return client


# ── namespace() accessor + validation ────────────────────────────────────────


def test_namespace_returns_bound_handle() -> None:
    client = _mock_client(lambda req: httpx.Response(200, json={}))
    ns = client.namespace("Document")
    assert isinstance(ns, Namespace)
    assert ns.name == "Document"


def test_namespace_rejects_unsafe_name() -> None:
    client = _mock_client(lambda req: httpx.Response(200, json={}))
    with pytest.raises(ValueError):
        client.namespace("Doc; DROP TABLE")


# ── query() → /v1/search (typed) ────────────────────────────────────────────────


def test_namespace_query_posts_typed_search() -> None:
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured.update(json.loads(req.content))
        return httpx.Response(200, json=_QUERY_PAYLOAD)

    client = _mock_client(handler)
    res = client.namespace("Document").query(
        text="graph",
        match_column="title",
        filters=[{"field": "status", "op": "eq", "value": "published"}],
        limit=5,
    )
    assert captured["from"] == "Document"
    assert captured["rank_by"] == ["bm25", "title", "graph"]
    assert captured["filters"] == [{"field": "status", "op": "eq", "value": "published"}]
    assert "query" not in captured
    assert res.rows[0]["title"] == "Graphs 101"


# ── write() → /v1/ingest/auto (schemaless, T6) ──────────────────────────────────


def test_namespace_write_posts_to_ingest_auto() -> None:
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/v1/ingest/auto"
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json=_SCHEMALESS_RESP)

    client = _mock_client(handler)
    out = client.namespace("Document").write(
        [{"id": "d1", "title": "A"}, {"id": "d2", "title": "B"}],
    )
    assert seen["body"]["object_type"] == "Document"
    assert seen["body"]["rows"] == [{"id": "d1", "title": "A"}, {"id": "d2", "title": "B"}]
    assert seen["body"]["purpose"] == "rag"  # inherited from client default
    assert out["type_created"] is True
    assert out["inserted"] == 2
    assert out["inferred_schema"]["title"] == "text"


def test_namespace_write_passes_schema_overrides() -> None:
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json=_SCHEMALESS_RESP)

    client = _mock_client(handler)
    client.namespace("Doc").write([{"id": "1"}], schema={"embedding": "bge-m3:text:1.0:1024"})
    assert seen["body"]["schema"] == {"embedding": "bge-m3:text:1.0:1024"}


# ── get() → governed SELECT ──────────────────────────────────────────────────


def test_namespace_get_point_lookup() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/v1/query"
        body = json.loads(req.content)
        assert "SELECT * FROM Document WHERE id = 'd42'" in body["sql"]
        return httpx.Response(
            200, json={"rows": [{"id": "d42", "title": "Found"}], "columns": ["id", "title"]}
        )

    client = _mock_client(handler)
    row = client.namespace("Document").get("d42")
    assert row == {"id": "d42", "title": "Found"}


def test_namespace_get_missing_returns_none() -> None:
    client = _mock_client(lambda req: httpx.Response(200, json={"rows": [], "columns": []}))
    assert client.namespace("Document").get("nope") is None


# ── delete_all() → tombstone all rows ────────────────────────────────────────


def test_namespace_delete_all_tombstones_every_row() -> None:
    calls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req.url.path)
        if req.url.path == "/v1/query":
            return httpx.Response(
                200, json={"rows": [{"id": "d1"}, {"id": "d2"}], "columns": ["id"]}
            )
        rows = [json.loads(line) for line in req.content.decode().splitlines()]
        assert [r["id"] for r in rows] == ["d1", "d2"]
        assert all(r.get("_deleted") is True for r in rows)
        assert req.url.params["object_type"] == "Document"
        return httpx.Response(200, json={"inserted": 2})

    client = _mock_client(handler)
    out = client.namespace("Document").delete_all()
    assert out["deleted"] == 2
    # /v1/ingest (typed tombstone path), not /v1/ingest/auto which has no tombstone semantics
    assert calls == ["/v1/query", "/v1/ingest"]


def test_namespace_delete_all_empty_is_noop() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/v1/query"
        return httpx.Response(200, json={"rows": [], "columns": ["id"]})

    client = _mock_client(handler)
    assert client.namespace("Document").delete_all() == {"deleted": 0}


# ── async ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_async_namespace_query_and_write() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v1/search":
            return httpx.Response(200, json=_QUERY_PAYLOAD)
        if req.url.path == "/v1/ingest/auto":
            return httpx.Response(200, json=_SCHEMALESS_RESP)
        return httpx.Response(404)

    client = _mock_client(handler)
    ns = AsyncNamespace(client, "Document")
    res = await ns.query(text="graph")
    assert res.rows[0]["id"] == "d1"
    out = await ns.write([{"id": "d1", "title": "A"}])
    assert out["inserted"] == 1 or out["type_created"] is True
