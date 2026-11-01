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


def _scripted(
    seen: list[httpx.Request], script: list[object]
) -> Callable[[httpx.Request], httpx.Response]:
    """Answer each request with the next script item (an Exception, a status, or a page)."""
    pending = list(script)

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        item = pending.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, tuple):
            status, headers, body = item
            return httpx.Response(status, headers=headers, json=body)
        return httpx.Response(200, json=item)

    return handler


def _page(event: str | None, nxt: str, has_more: bool = False) -> dict[str, object]:
    return {
        "feedId": "f",
        "events": [{"eventId": event}] if event else [],
        "nextCursor": nxt,
        "hasMore": has_more,
    }


def test_follow_changes_honours_retry_after_and_backs_off_exponentially(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("relata.objects.time.sleep", sleeps.append)
    seen: list[httpx.Request] = []
    script: list[object] = [
        (429, {"Retry-After": "7"}, {"detail": "slow down"}),
        (503, {}, {"detail": "down"}),
        (503, {}, {"detail": "down"}),
        _page("f:1", "f:1"),
    ]
    client = ObjectClient(BASE, transport=httpx.MockTransport(_scripted(seen, script)))
    follower = client.follow_changes(poll_interval=0, reconnect_backoff=1.0, max_backoff=30.0)
    assert next(follower)["events"][0]["eventId"] == "f:1"
    follower.close()
    # attempt 0 honours the server hint; attempts 1 and 2 double from the base.
    assert sleeps == [7.0, 2.0, 4.0]
    assert [r.url.params.get("cursor") for r in seen] == [None] * 4


def test_follow_changes_failure_budget_is_per_outage_not_lifetime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("relata.objects.time.sleep", lambda _s: None)
    seen: list[httpx.Request] = []
    fail = (503, {}, {"detail": "x"})
    # Three outages of two failures each with max_reconnects=2: never exhausted,
    # because a successful read in between resets the budget.
    script: list[object] = [
        fail, fail, _page("f:1", "f:1", True),
        fail, fail, _page("f:2", "f:2", True),
        fail, fail, _page("f:3", "f:3"),
    ]
    client = ObjectClient(BASE, transport=httpx.MockTransport(_scripted(seen, script)))
    follower = client.follow_changes(poll_interval=0, reconnect_backoff=0, max_reconnects=2)
    got = [next(follower)["events"][0]["eventId"] for _ in range(3)]
    follower.close()
    assert got == ["f:1", "f:2", "f:3"]
    assert [r.url.params.get("cursor") for r in seen] == [
        None, None, None, "f:1", "f:1", "f:1", "f:2", "f:2", "f:2",
    ]


def test_follow_changes_polls_from_the_same_cursor_when_caught_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("relata.objects.time.sleep", sleeps.append)
    seen: list[httpx.Request] = []
    script: list[object] = [
        _page("f:1", "f:1"),  # caught up after one event
        _page(None, "f:1"),  # still nothing: empty pages are not yielded
        _page("f:2", "f:2"),
    ]
    client = ObjectClient(BASE, transport=httpx.MockTransport(_scripted(seen, script)))
    follower = client.follow_changes(poll_interval=0.25)
    got = [next(follower)["events"][0]["eventId"] for _ in range(2)]
    follower.close()
    assert got == ["f:1", "f:2"]
    assert sleeps == [0.25, 0.25]
    assert [r.url.params.get("cursor") for r in seen] == [None, "f:1", "f:1"]


def test_follow_changes_surfaces_resync_details_after_a_reconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An outage that outlives the retention window ends in an explicit 410, never a gap."""
    monkeypatch.setattr("relata.objects.time.sleep", lambda _s: None)
    seen: list[httpx.Request] = []
    gone = {
        "code": "change-feed-resync-required",
        "detail": "cursor expired",
        "resync": {"reason": "cursor_expired", "oldest": 900, "head": 1000},
    }
    script: list[object] = [
        _page("f:1", "f:1", True),
        (503, {}, {"detail": "restarting"}),
        (410, {}, gone),
    ]
    client = ObjectClient(BASE, transport=httpx.MockTransport(_scripted(seen, script)))
    follower = client.follow_changes(poll_interval=0, reconnect_backoff=0)
    next(follower)
    with pytest.raises(RelataError) as info:
        next(follower)
    assert info.value.status_code == 410
    assert info.value.code == "change-feed-resync-required"
    assert len(seen) == 3, "the 410 is not retried"
