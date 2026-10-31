"""``query_vector`` / ``query_vector_model`` pass-through on ``POST /v1/query`` (ADR-0309)."""

from __future__ import annotations

import json

import httpx
import pytest

from relata import RelataClient
from relata._http import AsyncHttpTransport, HttpTransport

BASE = "http://localhost:9090"
SQL = "SELECT * FROM HYBRID_SEARCH('Doc', 'alice')"


def _client(bodies: list[dict[str, object]]) -> RelataClient:
    def handler(req: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(req.content))
        return httpx.Response(200, json={"rows": [], "query_id": "q1", "elapsed_ms": 1})

    client = RelataClient(BASE, purpose="analytics")
    mock = httpx.MockTransport(handler)
    client._RelataClient__sync_transport = HttpTransport(  # type: ignore[attr-defined]
        BASE, None, 30.0, transport=mock, extra_headers=None
    )
    client._RelataClient__async_transport = AsyncHttpTransport(  # type: ignore[attr-defined]
        BASE, None, 30.0, transport=mock, extra_headers=None
    )
    return client


def test_query_sends_vector_and_model() -> None:
    bodies: list[dict[str, object]] = []
    _client(bodies).query(SQL, query_vector=[0.25, -1.0, 3.0], query_vector_model="m1")
    assert bodies[0]["query_vector"] == [0.25, -1.0, 3.0]
    assert bodies[0]["query_vector_model"] == "m1"
    assert bodies[0]["sql"] == SQL


def test_query_omits_vector_fields_by_default() -> None:
    bodies: list[dict[str, object]] = []
    _client(bodies).query(SQL)
    assert "query_vector" not in bodies[0]
    assert "query_vector_model" not in bodies[0]


@pytest.mark.asyncio
async def test_aquery_sends_vector() -> None:
    bodies: list[dict[str, object]] = []
    await _client(bodies).aquery(SQL, query_vector=[1.0, 2.0])
    assert bodies[0]["query_vector"] == [1.0, 2.0]
    assert "query_vector_model" not in bodies[0]
