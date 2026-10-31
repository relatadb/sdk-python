"""#5706 freshness client: wire shapes and header gating."""

from __future__ import annotations

import json

import httpx
import pytest

from relata.freshness import AsyncFreshnessClient, FreshnessClient, freshness_headers

BASE = "http://localhost:9090"


def test_status_reads_the_status_route() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert (req.method, req.url.path) == ("GET", "/v1/freshness/status")
        return httpx.Response(
            200, json={"contract": "relata.freshness/v1", "token": "fv1.x", "indexes": []}
        )

    out = FreshnessClient(BASE, transport=httpx.MockTransport(handler)).status()
    assert out["token"] == "fv1.x"


def test_barrier_sends_exact_body_and_returns_report() -> None:
    seen: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        assert (req.method, req.url.path) == ("POST", "/v1/freshness/barrier")
        seen.update(json.loads(req.content))
        return httpx.Response(200, json={"freshness": {"status": "complete"}})

    c = FreshnessClient(BASE, transport=httpx.MockTransport(handler))
    out = c.barrier("fv1.t", indexes=["lexical", "vector"], wait_ms=250, require_durable=True)
    assert seen == {
        "token": "fv1.t",
        "indexes": ["lexical", "vector"],
        "wait_ms": 250,
        "require_durable": True,
    }
    assert out["freshness"]["status"] == "complete"


def test_barrier_omits_unset_options() -> None:
    seen: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(json.loads(req.content))
        return httpx.Response(200, json={})

    FreshnessClient(BASE, transport=httpx.MockTransport(handler)).barrier("fv1.t")
    assert seen == {"token": "fv1.t"}


def test_not_ready_is_an_error_not_a_silent_success() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            json={"code": "freshness-not-ready", "status": 503, "title": "x", "type": "t"},
            headers={"content-type": "application/problem+json"},
        )

    c = FreshnessClient(BASE, transport=httpx.MockTransport(handler))
    with pytest.raises(Exception):  # noqa: B017,PT011 - any SDK error type is acceptable
        c.barrier("fv1.t", wait_ms=1)


def test_freshness_headers() -> None:
    assert freshness_headers("tok") == {"X-Relata-Min-Version": "tok"}
    full = freshness_headers(
        "tok", indexes=["vector"], wait_ms=50, require_durable=True, partial=True
    )
    assert full == {
        "X-Relata-Min-Version": "tok",
        "X-Relata-Require-Index": "vector",
        "X-Relata-Wait-Ms": "50",
        "X-Relata-Require-Durable": "true",
        "X-Relata-Freshness-Mode": "partial",
    }


@pytest.mark.asyncio
async def test_async_barrier() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/v1/freshness/barrier"
        return httpx.Response(200, json={"freshness": {"status": "complete"}})

    c = AsyncFreshnessClient(BASE, transport=httpx.MockTransport(handler))
    assert (await c.barrier("fv1.t"))["freshness"]["status"] == "complete"
    await c.close()
