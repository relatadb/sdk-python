"""#6302: a maintenance-pause 503 is retried transparently, for every verb."""
from __future__ import annotations

import asyncio
import json
from collections.abc import Callable

import httpx
import pytest

from relata import _http
from relata._http import AsyncHttpTransport, HttpTransport, _maintenance_pause_delay
from relata.client import RelataClient
from relata.exceptions import ServerError


def _pause(reason: str = "rollover_commit", *, retryable: bool = True,
           retry_after: str | None = "2") -> httpx.Response:
    body = {
        "type": "about:blank", "title": "Service Unavailable", "status": 503,
        "detail": "a backup snapshot, whole-store restore or generation rollover is "
                  "briefly pausing this node; retry shortly",
        "retryable": retryable, "retry_after_secs": 5, "pause_reason": reason,
    }
    headers = {"Content-Type": "application/problem+json"}
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return httpx.Response(503, content=json.dumps(body).encode(), headers=headers)


def _ok() -> httpx.Response:
    return httpx.Response(200, json={"inserted": 1})


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []
    monkeypatch.setattr(_http.time, "sleep", recorded.append)

    async def fake_sleep(delay: float) -> None:
        recorded.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return recorded


Reply = Callable[[], httpx.Response]


def _sync(responses: list[Reply], seen: list[str], **kwargs: float) -> HttpTransport:
    """A transport answering with each factory in turn, repeating the last."""
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.method)
        return (responses.pop(0) if len(responses) > 1 else responses[0])()
    return HttpTransport("http://test", None, 1, transport=httpx.MockTransport(handler),
                         **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("reason", ["rollover_begin", "rollover_commit", "backup", "restore"])
def test_post_during_a_pause_is_retried_after_retry_after(
    reason: str, sleeps: list[float]
) -> None:
    seen: list[str] = []
    transport = _sync([lambda: _pause(reason), _ok], seen)
    try:
        assert transport.post("/v1/ingest/bulk", {"objects": []}) == {"inserted": 1}
    finally:
        transport.close()
    assert seen == ["POST", "POST"]
    assert sleeps == [2.0]


def test_pause_retries_stop_at_the_budget_and_raise_the_503(sleeps: list[float]) -> None:
    seen: list[str] = []
    transport = _sync([_pause], seen, pause_retry_budget=3.0)
    try:
        with pytest.raises(ServerError) as caught:
            transport.post("/v1/query", {"sql": "SELECT 1"})
    finally:
        transport.close()
    assert caught.value.status_code == 503
    assert sleeps == [2.0, 1.0]
    assert seen == ["POST", "POST", "POST"]


@pytest.mark.parametrize("response", [
    lambda: _pause("poisoned", retryable=False, retry_after=None),
    lambda: _pause("storage_recovering"),
    lambda: _pause("rollover_begin", retryable=False),
    lambda: httpx.Response(503, json={"code": "memory-pressure", "retryable": True},
                           headers={"Retry-After": "1"}),
    lambda: httpx.Response(503, content=b"not-json", headers={"Retry-After": "1"}),
])
def test_other_503s_are_not_replayed(response: Reply, sleeps: list[float]) -> None:
    seen: list[str] = []
    transport = _sync([response], seen)
    try:
        with pytest.raises(ServerError):
            transport.post("/v1/ingest/bulk", {"objects": []})
    finally:
        transport.close()
    assert seen == ["POST"]
    assert sleeps == []


def test_zero_budget_disables_pause_retries(sleeps: list[float]) -> None:
    seen: list[str] = []
    transport = _sync([_pause, _ok], seen, pause_retry_budget=0)
    try:
        with pytest.raises(ServerError):
            transport.get("/v1/objects/Person/1")
    finally:
        transport.close()
    assert seen == ["GET"]
    assert sleeps == []


async def test_async_post_during_a_pause_is_retried(sleeps: list[float]) -> None:
    seen: list[str] = []
    responses = [_pause("rollover_begin", retry_after="1"), _ok()]

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.method)
        return responses.pop(0)
    transport = AsyncHttpTransport("http://test", None, 1,
                                   transport=httpx.MockTransport(handler))
    try:
        assert await transport.post("/v1/ingest/bulk", {"objects": []}) == {"inserted": 1}
    finally:
        await transport.aclose()
    assert seen == ["POST", "POST"]
    assert sleeps == [1.0]


@pytest.mark.parametrize("headers,body,expected", [
    ({"retry-after": "60"}, {}, 5.0),
    ({"retry-after": "0"}, {}, 0.1),
    ({}, {"retry_after_secs": 3}, 3.0),
    ({}, {"retry_after_secs": True}, 1.0),
    ({}, {}, 1.0),
])
def test_pause_delay_honours_retry_after_and_is_capped(
    headers: dict[str, str], body: dict[str, object], expected: float
) -> None:
    raw = json.dumps({"pause_reason": "backup", "retryable": True, **body}).encode()
    assert _maintenance_pause_delay(503, headers, raw) == expected
    assert _maintenance_pause_delay(502, headers, raw) is None


def test_client_passes_its_pause_budget_to_both_transports() -> None:
    client = RelataClient("http://test", pause_retry_budget_secs=0)
    assert client._sync._pause_retry_budget == 0
    assert client._async._pause_retry_budget == 0
    assert RelataClient("http://test")._sync._pause_retry_budget == 30.0
