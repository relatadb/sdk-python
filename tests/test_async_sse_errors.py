"""Async SSE consumers must raise the typed error on a non-2xx response, like the sync ones."""

from __future__ import annotations

import httpx
import pytest

from relata import AuthError, RelataClient
from relata._http import AsyncHttpTransport
from relata.streaming import AsyncStreamingClient

BASE = "http://localhost:9090"


def _client() -> RelataClient:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"title": "Unauthorized", "status": 401})

    client = RelataClient(BASE, purpose="analytics")
    client._RelataClient__async_transport = AsyncHttpTransport(  # type: ignore[attr-defined]
        BASE, None, 30.0, transport=httpx.MockTransport(handler), extra_headers=None,
    )
    return client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "open_stream",
    [
        lambda s: s.watch("SELECT 1", purpose="analytics"),
        lambda s: s.watch_stream("T"),
        lambda s: s.alerts(),
        lambda s: s.graph_changes(type="T"),
        lambda s: s.a2a_task_stream("t1"),
    ],
    ids=["watch", "watch_stream", "alerts", "graph_changes", "a2a_task_stream"],
)
async def test_async_sse_raises_on_unauthorized(open_stream) -> None:
    stream = AsyncStreamingClient(_client())
    with pytest.raises(AuthError):
        _ = [event async for event in await open_stream(stream)]
