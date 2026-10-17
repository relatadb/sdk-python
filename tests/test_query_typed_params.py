"""Typed /query wire contract; engine binding is qualified by server fixtures."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from relata import RelataClient
from relata._http import AsyncHttpTransport, HttpTransport
from relata.exceptions import RelataError

SQL = "SELECT id FROM Record WHERE active = $1 AND id IN ($2) AND payload = $3"
PARAMS = [True, ["safe", "x'); DROP TABLE Record; --", "$1"], {
    "enabled": False, "nested": [None, {"label": "'quoted'", "count": 7}],
}]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("status", [200, 400])
def test_typed_query_wire_and_indexed_error(asynchronous: bool, status: int) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.method == "POST"
        assert request.url.path == "/query"
        assert json.loads(request.content) == {"purpose": "analytics", "sql": SQL, "params": PARAMS}
        if status == 400:
            return httpx.Response(400, headers={"content-type": "application/problem+json"}, json={
                "type": "about:blank", "title": "Invalid query parameter", "status": 400,
                "detail": "parameter $2 is unsupported in this context",
            })
        return httpx.Response(200, json={"rows": [], "query_id": "typed", "elapsed_ms": 1})

    transport = httpx.MockTransport(handler)
    client = RelataClient("http://data", purpose="analytics")
    client._RelataClient__sync_transport = HttpTransport(  # type: ignore[attr-defined]
        "http://data", None, 30.0, transport=transport, extra_headers=None,
    )
    client._RelataClient__async_transport = AsyncHttpTransport(  # type: ignore[attr-defined]
        "http://data", None, 30.0, transport=transport, extra_headers=None,
    )

    def execute() -> None:
        if asynchronous:
            asyncio.run(client.aquery_params(SQL, PARAMS))
        else:
            client.query_params(SQL, PARAMS)

    if status == 400:
        with pytest.raises(RelataError, match=r"\$2"):
            execute()
    else:
        execute()
    assert len(requests) == 1, "typed binding errors must not be retried"
