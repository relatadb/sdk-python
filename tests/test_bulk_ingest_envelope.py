"""Wire contract for the /ingest/bulk envelope options (#5689, #5714)."""

from __future__ import annotations

import json
from typing import Any

import httpx

from relata import AsyncIngestClient, IngestClient

BASE = "http://localhost:9090"


def _handler(requests: list[httpx.Request]) -> httpx.MockTransport:
    def record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"inserted": 1, "errors": []})

    return httpx.MockTransport(record)


_ROWS: list[dict[str, Any]] = [{"_pk": "k1", "qty": 3}]
_CP = {"source": "kafka/orders-0", "position": "100", "expected_position": ""}
_CONTRACT = {"dataset": "orders", "version": 2}


def _check(request: httpx.Request) -> None:
    assert request.url.path == "/ingest/bulk"
    assert request.headers["idempotency-key"] == "batch-100"
    assert request.headers["x-relata-durability"] == "sync"
    body = json.loads(request.content)
    assert body["objects"] == [{"_pk": "k1", "qty": 3, "_type": "OrderEvent"}]
    assert body["source_checkpoint"] == _CP
    assert body["contract"] == _CONTRACT
    assert body["purpose"] == "analytics"
    assert "on_conflict" not in body


def test_bulk_envelope_options_route_to_bulk_door_sync() -> None:
    seen: list[httpx.Request] = []
    client = IngestClient(BASE, purpose="analytics", transport=_handler(seen))
    client.bulk(
        "OrderEvent", _ROWS, idempotency_key="batch-100", durability="sync",
        source_checkpoint=_CP, contract=_CONTRACT,
    )
    _check(seen[0])


async def test_bulk_envelope_options_route_to_bulk_door_async() -> None:
    seen: list[httpx.Request] = []
    client = AsyncIngestClient(BASE, purpose="analytics", transport=_handler(seen))
    await client.bulk(
        "OrderEvent", _ROWS, idempotency_key="batch-100", durability="sync",
        source_checkpoint=_CP, contract=_CONTRACT,
    )
    _check(seen[0])


def test_plain_bulk_keeps_the_ndjson_door() -> None:
    seen: list[httpx.Request] = []
    client = IngestClient(BASE, purpose="analytics", transport=_handler(seen))
    client.bulk("OrderEvent", _ROWS)
    assert seen[0].url.path == "/ingest"
    assert "idempotency-key" not in seen[0].headers
