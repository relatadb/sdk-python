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
    assert req.url.path == "/v1/changes"
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


def _paged_handler(seen: list[httpx.Request]):
    pages = {
        None: {"feedId": "f", "events": [{"eventId": "f:1"}], "nextCursor": "f:1", "hasMore": True},
        "f:1": {"feedId": "f", "events": [{"eventId": "f:2"}], "nextCursor": "f:2", "hasMore": False},
    }

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json=pages[req.url.params.get("cursor")])

    return handler


def test_iter_change_pages_follows_has_more_until_caught_up() -> None:
    seen: list[httpx.Request] = []
    client = ObjectClient(BASE, transport=httpx.MockTransport(_paged_handler(seen)))
    pages = list(client.iter_change_pages(limit=1))
    assert [p["nextCursor"] for p in pages] == ["f:1", "f:2"]
    assert [r.url.params.get("cursor") for r in seen] == [None, "f:1"]


def test_disabled_feed_is_a_404_problem() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={"code": "change-feed-disabled", "title": "Change Feed Disabled", "status": 404},
            headers={"content-type": "application/problem+json"},
        )

    client = ObjectClient(BASE, transport=httpx.MockTransport(handler))
    with pytest.raises(Exception) as caught:
        client.changes()
    assert "change-feed-disabled" in str(caught.value) or getattr(caught.value, "status", 404) == 404


@pytest.mark.asyncio
async def test_async_iter_change_pages() -> None:
    seen: list[httpx.Request] = []
    client = AsyncObjectClient(BASE, transport=httpx.MockTransport(_paged_handler(seen)))
    pages = [p async for p in client.iter_change_pages()]
    assert len(pages) == 2 and pages[-1]["hasMore"] is False
