"""#5716: object change feed reads."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from relata.exceptions import RelataError
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
    client = ObjectClient(
        BASE, transport=httpx.MockTransport(lambda r: httpx.Response(200, json=PAGE))
    )
    with pytest.raises(ValueError):
        client.changes(object_type="bad name!")


@pytest.mark.asyncio
async def test_async_changes() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.params["cursor"] == "f:9"
        return httpx.Response(200, json=PAGE)

    client = AsyncObjectClient(BASE, transport=httpx.MockTransport(handler))
    assert (await client.changes(cursor="f:9"))["head"] == 3


def _paged_handler(
    seen: list[httpx.Request],
) -> Callable[[httpx.Request], httpx.Response]:
    pages = {
        None: {"feedId": "f", "events": [{"eventId": "f:1"}], "nextCursor": "f:1", "hasMore": True},
        "f:1": {
            "feedId": "f",
            "events": [{"eventId": "f:2"}],
            "nextCursor": "f:2",
            "hasMore": False,
        },
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
    assert (
        "change-feed-disabled" in str(caught.value) or getattr(caught.value, "status", 404) == 404
    )


@pytest.mark.asyncio
async def test_async_iter_change_pages() -> None:
    seen: list[httpx.Request] = []
    client = AsyncObjectClient(BASE, transport=httpx.MockTransport(_paged_handler(seen)))
    pages = [p async for p in client.iter_change_pages()]
    assert len(pages) == 2 and pages[-1]["hasMore"] is False


def _flaky_handler(
    seen: list[httpx.Request], failures: list[object]
) -> Callable[[httpx.Request], httpx.Response]:
    """Serve f:1 then f:2; before the second page fail with each of ``failures``."""
    pending = list(failures)

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        cursor = req.url.params.get("cursor")
        if cursor == "f:1" and pending:
            item = pending.pop(0)
            if isinstance(item, Exception):
                raise item
            return httpx.Response(int(item), json={"detail": "x"})  # type: ignore[call-overload]
        pages = {
            None: {
                "feedId": "f",
                "events": [{"eventId": "f:1"}],
                "nextCursor": "f:1",
                "hasMore": True,
            },
            "f:1": {
                "feedId": "f",
                "events": [{"eventId": "f:2"}],
                "nextCursor": "f:2",
                "hasMore": False,
            },
        }
        return httpx.Response(200, json=pages[cursor])

    return handler


def test_follow_changes_reconnects_and_resumes_from_the_last_cursor() -> None:
    seen: list[httpx.Request] = []
    handler = _flaky_handler(seen, [httpx.ConnectError("down"), 503])
    # The transport does not retry by default, so the outage reaches the follower.
    client = ObjectClient(BASE, transport=httpx.MockTransport(handler))
    follower = client.follow_changes(poll_interval=0, reconnect_backoff=0, limit=1)
    got = [next(follower)["events"][0]["eventId"] for _ in range(2)]
    follower.close()
    assert got == ["f:1", "f:2"], "no gap and no duplicate across the outage"
    # first read, two failed reads of cursor f:1, then the resumed read.
    assert [r.url.params.get("cursor") for r in seen] == [None, "f:1", "f:1", "f:1"]


def test_follow_changes_gives_up_after_max_reconnects() -> None:
    seen: list[httpx.Request] = []
    handler = _flaky_handler(seen, [503, 503, 503])
    client = ObjectClient(BASE, transport=httpx.MockTransport(handler))
    follower = client.follow_changes(poll_interval=0, reconnect_backoff=0, max_reconnects=2)
    next(follower)
    with pytest.raises(RelataError):
        next(follower)
    assert len(seen) == 4  # 1 ok + 3 failed attempts (2 reconnects allowed)


def test_follow_changes_raises_non_transient_errors_immediately() -> None:
    seen: list[httpx.Request] = []
    handler = _flaky_handler(seen, [410])
    client = ObjectClient(BASE, transport=httpx.MockTransport(handler))
    follower = client.follow_changes(poll_interval=0, reconnect_backoff=0)
    next(follower)
    with pytest.raises(RelataError) as info:
        next(follower)
    assert info.value.status_code == 410
    assert len(seen) == 2


@pytest.mark.asyncio
async def test_async_follow_changes_reconnects() -> None:
    seen: list[httpx.Request] = []
    handler = _flaky_handler(seen, [httpx.ConnectError("down")])
    client = AsyncObjectClient(BASE, transport=httpx.MockTransport(handler))
    follower = client.follow_changes(poll_interval=0, reconnect_backoff=0, limit=1)
    got = [(await follower.__anext__())["events"][0]["eventId"] for _ in range(2)]
    await follower.aclose()
    assert got == ["f:1", "f:2"]
