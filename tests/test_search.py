"""Tests for search() / asearch() convenience methods (#670).

Uses httpx.MockTransport — no live server required.
"""

from __future__ import annotations

import json

import httpx
import pytest

from relata import RelataClient
from relata._http import AsyncHttpTransport, HttpTransport
from relata.models import SearchHit, SearchResponse

BASE = "http://localhost:8080"

_SEARCH_PAYLOAD = {
    "hits": [
        {
            "id": "p1",
            "object_type": "Person",
            "fields": {"name": "Alice Smith"},
            "score": 1.23,
            "highlights": {"name": "<em>Alice</em> Smith"},
        }
    ],
    "total": 1,
    "facets": {"tenant_id": {"EUROPOL": 1}},
    "processing_time_ms": 5,
}


def _mock_client(handler):
    client = RelataClient(BASE, bearer_token="tok")
    mock = httpx.MockTransport(handler)
    extra = client._extra_headers  # type: ignore[attr-defined]
    client._RelataClient__sync_transport = HttpTransport(  # type: ignore[attr-defined]
        BASE,
        client._bearer_token,  # type: ignore[attr-defined]
        client._timeout,  # type: ignore[attr-defined]
        transport=mock,
        extra_headers=extra,
    )
    client._RelataClient__async_transport = AsyncHttpTransport(  # type: ignore[attr-defined]
        BASE,
        client._bearer_token,  # type: ignore[attr-defined]
        client._timeout,  # type: ignore[attr-defined]
        transport=mock,
        extra_headers=extra,
    )
    return client


def _handler(request: httpx.Request) -> httpx.Response:
    assert request.url.path == "/search"
    body = json.loads(request.content)
    assert body["query"] == "alice smith"
    assert body["type"] == "Person"
    return httpx.Response(200, json=_SEARCH_PAYLOAD)


def test_search_returns_typed_response():
    client = _mock_client(_handler)
    result = client.search("alice smith", "Person")
    assert isinstance(result, SearchResponse)
    assert result.total == 1
    assert len(result.hits) == 1
    hit = result.hits[0]
    assert isinstance(hit, SearchHit)
    assert hit.id == "p1"
    assert hit.score == pytest.approx(1.23)
    assert hit.highlights["name"] == "<em>Alice</em> Smith"


def test_search_passes_optional_params():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={**_SEARCH_PAYLOAD, "hits": []})

    client = _mock_client(handler)
    client.search(
        "alice smith", "Person",
        limit=5, facets=["tenant_id"], highlight=True,
        filters={"tenant_id": "EUROPOL"},
    )
    assert captured["limit"] == 5
    assert captured["facets"] == ["tenant_id"]
    assert captured["highlight"] is True
    assert captured["filters"] == {"tenant_id": "EUROPOL"}


@pytest.mark.asyncio
async def test_asearch_returns_typed_response():
    client = _mock_client(_handler)
    result = await client.asearch("alice smith", "Person")
    assert isinstance(result, SearchResponse)
    assert result.total == 1
