"""#5716: object change feed reads."""

from __future__ import annotations

import httpx
import pytest

from relata.objects import AsyncObjectClient, ObjectClient

BASE = "http://localhost:9090"
PAGE = {"feedId": "f", "events": [], "nextCursor": "f:3", "hasMore": False, "head": 3}


def test_changes_sends_cursor_limit_type_and_start() -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json=PAGE)

    client = ObjectClient(BASE, purpose="audit", transport=httpx.MockTransport(handler))
    out = client.changes(cursor="f:2", limit=50, object_type="Doc")
    assert out["nextCursor"] == "f:3"
    (req,) = seen
    assert req.method == "GET"
    assert req.url.path == "/object-changes"
    assert dict(req.url.params) == {"cursor": "f:2", "limit": "50", "type": "Doc"}


def test_changes_without_arguments_has_no_query_string() -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json=PAGE)

    ObjectClient(BASE, transport=httpx.MockTransport(handler)).changes(start="latest")
    assert seen[0].url.query == b"start=latest"


def test_changes_rejects_an_unsafe_type_name() -> None:
    client = ObjectClient(BASE, transport=httpx.MockTransport(lambda r: httpx.Response(200, json=PAGE)))
    with pytest.raises(ValueError):
        client.changes(object_type="bad name!")


@pytest.mark.asyncio
async def test_async_changes() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.params["cursor"] == "f:9"
        return httpx.Response(200, json=PAGE)

    client = AsyncObjectClient(BASE, transport=httpx.MockTransport(handler))
    assert (await client.changes(cursor="f:9"))["head"] == 3
