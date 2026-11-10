"""#6147: ``upsert(on_conflict="skip"|"error")`` uses the synchronous bulk door.

The queued ``POST /v1/ingest`` door refuses those policies because it cannot
decide them atomically, so the SDK must not send them there.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from relata.objects import AsyncObjectClient, ObjectClient

BASE = "http://localhost:9090"
Handler = Callable[[httpx.Request], httpx.Response]
RECEIPT = {"inserted": 1, "updated": 0, "skipped": 0, "errors": []}


def _recorder(seen: list[httpx.Request]) -> Handler:
    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json=RECEIPT)

    return handler


def _assert_bulk(req: httpx.Request, policy: str) -> None:
    assert req.method == "POST"
    assert req.url.path == "/v1/ingest/bulk"
    body = json.loads(req.content)
    assert body["on_conflict"] == policy
    assert body["purpose"] == "audit"
    assert body["objects"] == [{"n": "x", "id": "a", "_source": "s", "_type": "Doc"}]


@pytest.mark.parametrize("policy", ["skip", "error"])
def test_upsert_conflict_policy_routes_to_bulk(policy: str) -> None:
    seen: list[httpx.Request] = []
    c = ObjectClient(BASE, purpose="audit", transport=httpx.MockTransport(_recorder(seen)))
    assert c.upsert("Doc", "a", {"n": "x"}, source="s", on_conflict=policy) == RECEIPT
    (req,) = seen
    _assert_bulk(req, policy)


def test_default_upsert_stays_on_the_queued_door() -> None:
    seen: list[httpx.Request] = []
    c = ObjectClient(BASE, purpose="audit", transport=httpx.MockTransport(_recorder(seen)))
    c.upsert("Doc", "a", {"n": "x"})
    (req,) = seen
    assert req.url.path == "/v1/ingest"
    assert "on_conflict" not in req.url.params


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["skip", "error"])
async def test_async_upsert_conflict_policy_routes_to_bulk(policy: str) -> None:
    seen: list[httpx.Request] = []
    c = AsyncObjectClient(BASE, purpose="audit", transport=httpx.MockTransport(_recorder(seen)))
    assert await c.upsert("Doc", "a", {"n": "x"}, source="s", on_conflict=policy) == RECEIPT
    (req,) = seen
    _assert_bulk(req, policy)
