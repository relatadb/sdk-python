"""Automatic Idempotency-Key on bulk ingest (#5689, owner decision D06).

Every ``POST /v1/ingest/bulk`` gets a generated key when the caller sent none,
the same key is reused by the transport's retries (connection errors, 502,
504; never a 503 from Relata), and an explicit key is never replaced. The queued
``/v1/ingest`` door takes no key and is never retried.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable

import httpx
import pytest

from relata import AsyncIngestClient, IngestClient
from relata._http import HttpTransport
from relata.exceptions import RelataError

BASE = "http://localhost:9090"
_ROWS = [{"note": "keyless"}]


def _recorder(
    seen: list[httpx.Request], statuses: list[int] | None = None
) -> Callable[[httpx.Request], httpx.Response]:
    replies = list(statuses or [])

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        status = replies.pop(0) if replies else 200
        return httpx.Response(status, json={"inserted": 1, "errors": []})

    return handle


def _client(handler: Callable[[httpx.Request], httpx.Response], retries: int = 0) -> IngestClient:
    return IngestClient(
        BASE,
        transport=httpx.MockTransport(handler),
        max_retries=retries,
        retry_backoff_secs=0.0,
    )


def test_bulk_request_gets_a_fresh_uuid_key_per_call() -> None:
    seen: list[httpx.Request] = []
    client = _client(_recorder(seen))
    client.bulk("Note", _ROWS, on_conflict="upsert")
    client.bulk("Note", _ROWS, on_conflict="upsert")
    keys = [r.headers["idempotency-key"] for r in seen]
    assert all(r.url.path == "/v1/ingest/bulk" for r in seen)
    assert len(set(keys)) == 2
    for key in keys:
        uuid.UUID(key)


def test_explicit_key_is_never_replaced() -> None:
    seen: list[httpx.Request] = []
    _client(_recorder(seen)).bulk("Note", _ROWS, idempotency_key="batch-7")
    assert seen[0].headers.get_list("idempotency-key") == ["batch-7"]


def test_retry_reuses_the_same_key() -> None:
    seen: list[httpx.Request] = []
    client = _client(_recorder(seen, [502, 504]), retries=2)
    assert client.bulk("Note", _ROWS, on_conflict="upsert")["inserted"] == 1
    assert len(seen) == 3
    assert len({r.headers["idempotency-key"] for r in seen}) == 1
    assert seen[0].content == seen[2].content


def test_a_relata_503_is_not_retried_even_with_a_key() -> None:
    # A 503 from Relata carries its own retryable verdict (queue full, uncertain
    # receipt, checkpoint not persisted); the caller decides, with its own key.
    seen: list[httpx.Request] = []
    with pytest.raises(RelataError):
        _client(_recorder(seen, [503, 200]), retries=2).bulk("Note", _ROWS, on_conflict="upsert")
    assert len(seen) == 1


def test_connection_failure_is_retried_with_the_same_key() -> None:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) == 1:
            raise httpx.ConnectError("server restarted", request=request)
        return httpx.Response(200, json={"inserted": 1, "errors": []})

    _client(handle, retries=1).bulk("Note", _ROWS, on_conflict="upsert")
    assert len(seen) == 2
    assert seen[0].headers["idempotency-key"] == seen[1].headers["idempotency-key"]


def test_without_retries_a_failure_is_sent_once() -> None:
    seen: list[httpx.Request] = []
    with pytest.raises(RelataError):
        _client(_recorder(seen, [502])).bulk("Note", _ROWS, on_conflict="upsert")
    assert len(seen) == 1


def test_queued_ingest_door_gets_no_key_and_is_not_retried() -> None:
    seen: list[httpx.Request] = []
    with pytest.raises(RelataError):
        _client(_recorder(seen, [502, 502]), retries=1).bulk("Note", _ROWS)
    assert len(seen) == 1
    assert seen[0].url.path == "/v1/ingest"
    assert "idempotency-key" not in seen[0].headers


def test_every_bulk_post_through_the_transport_is_keyed() -> None:
    seen: list[httpx.Request] = []
    transport = HttpTransport(BASE, None, 5.0, transport=httpx.MockTransport(_recorder(seen)))
    transport.post("/v1/ingest/bulk", {"objects": [{"_type": "Note", "note": "x"}]})
    transport.post("/v1/query", {"sql": "SELECT 1"})
    assert "idempotency-key" in seen[0].headers
    assert "idempotency-key" not in seen[1].headers


async def test_async_retry_reuses_the_same_key() -> None:
    seen: list[httpx.Request] = []
    handler = _recorder(seen, [504])  # gateway timeout: no server verdict
    client = AsyncIngestClient(
        BASE,
        transport=httpx.MockTransport(handler),
        max_retries=1,
        retry_backoff_secs=0.0,
    )
    await client.bulk("Note", _ROWS, on_conflict="upsert")
    assert len(seen) == 2
    assert seen[0].headers["idempotency-key"] == seen[1].headers["idempotency-key"]
    uuid.UUID(seen[0].headers["idempotency-key"])
