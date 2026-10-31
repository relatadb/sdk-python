"""#5938 media-work client: routes, bodies, claim fence header on bulk write-back."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from relata.ingest import IngestClient
from relata.media_work import AsyncMediaWorkClient, MediaWorkClient, media_work_claim_headers

BASE = "http://localhost:9090"


def _recorder(seen: list):
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content) if req.content else None
        seen.append((req.method, req.url.raw_path.decode().split("?")[0], req.url.query.decode(), body))
        return httpx.Response(200, json={"items": [{"work_id": "w/1", "claim_id": 7}]})

    return handler


def test_routes_and_bodies() -> None:
    seen: list = []
    c = MediaWorkClient(BASE, transport=httpx.MockTransport(_recorder(seen)))
    out = c.claim("etl-1", max_items=2, lease_ms=5000)
    assert out["items"][0]["claim_id"] == 7
    c.list(state="queued", limit=3)
    c.list()
    c.get("w/1")
    c.heartbeat("w/1", 7, lease_ms=9000)
    c.complete("w/1", 7, producer_version="v1", batches=2, children=5)
    c.fail("w/1", 7, failure_class="permanent", code="bad", detail="corrupt")
    assert seen == [
        ("POST", "/v1/media/work/claim", "", {"claimant": "etl-1", "max_items": 2, "lease_ms": 5000}),
        ("GET", "/v1/media/work", "state=queued&limit=3", None),
        ("GET", "/v1/media/work", "", None),
        ("GET", "/v1/media/work/w%2F1", "", None),
        ("POST", "/v1/media/work/w%2F1/heartbeat", "", {"claim_id": 7, "lease_ms": 9000}),
        (
            "POST",
            "/v1/media/work/w%2F1/complete",
            "",
            {"claim_id": 7, "producer_version": "v1", "batches": 2, "children": 5},
        ),
        (
            "POST",
            "/v1/media/work/w%2F1/fail",
            "",
            {"claim_id": 7, "class": "permanent", "code": "bad", "detail": "corrupt"},
        ),
    ]


def test_validation() -> None:
    c = MediaWorkClient(BASE, transport=httpx.MockTransport(_recorder([])))
    with pytest.raises(ValueError):
        c.list(state="bogus")
    with pytest.raises(ValueError):
        c.fail("w", 1, failure_class="sometimes")


def test_async_claim_and_complete() -> None:
    seen: list = []

    async def run() -> None:
        c = AsyncMediaWorkClient(BASE, transport=httpx.MockTransport(_recorder(seen)))
        await c.claim("etl")
        await c.complete("w", 1, producer_version="v", batches=0, children=0)
        await c.close()

    asyncio.run(run())
    assert [s[1] for s in seen] == ["/v1/media/work/claim", "/v1/media/work/w/complete"]


def test_bulk_write_back_sends_claim_header() -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json={"inserted": 1})

    c = IngestClient(BASE, transport=httpx.MockTransport(handler))
    c.bulk(
        "MediaSegment",
        [{"_pk": "a"}],
        durability="sync",
        idempotency_key="media/w/v1/1",
        source_checkpoint={"source": "openetl/media/w", "position": "1"},
        media_work_claim=7,
    )
    assert seen[0].url.path == "/v1/ingest/bulk"
    assert seen[0].headers["X-Relata-Claim"] == "7"
    assert media_work_claim_headers(7) == {"X-Relata-Claim": "7"}
