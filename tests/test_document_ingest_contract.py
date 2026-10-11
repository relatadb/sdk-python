"""Wire contract for prepared document rows through generic bulk ingest."""

from __future__ import annotations

import json
from typing import Any

import httpx

from relata import EventAppendRequest, RelataClient
from relata._http import AsyncHttpTransport, HttpTransport

BASE = "http://localhost:9090"
_ROWS = [
    {"_type": "DocumentSource", "_pk": "doc-1", "uri": "file://report.pdf"},
    {"_type": "DocumentChunk", "_pk": "chunk-1", "source_id": "doc-1", "text": "hello"},
]


def _client(handler: Any) -> tuple[RelataClient, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    client = RelataClient(BASE, purpose="analytics")
    transport = httpx.MockTransport(record)
    extra = client._extra_headers  # type: ignore[attr-defined]
    client._RelataClient__sync_transport = HttpTransport(
        BASE, client._bearer_token, client._timeout, transport=transport,
        extra_headers=extra,
    )
    client._RelataClient__async_transport = AsyncHttpTransport(
        BASE, client._bearer_token, client._timeout, transport=transport,
        extra_headers=extra,
    )
    return client, requests


def test_document_ingest_uses_generic_queued_bulk_and_task_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/ingest/bulk":
            assert json.loads(request.content) == {
                "objects": _ROWS,
                "queued": True,
                "purpose": "analytics",
            }
            return httpx.Response(
                200,
                json={
                    "task_id": "task-1",
                    "rows_queued": 2,
                    "rows_rejected": 0,
                    "errors": [],
                    "queue_depth": 2,
                },
            )
        assert request.url.path == "/ingest/tasks/task-1"
        return httpx.Response(
            200,
            json={
                "task_id": "task-1",
                "status": "complete",
                "rows_total": 2,
                "rows_written": 2,
                "rows_dropped": 0,
                "drop_reasons": [],
            },
        )

    client, requests = _client(handler)
    receipt = client.ingest_document(_ROWS)
    status = client.ingest_document_status(receipt.task_id)
    assert receipt.task_id == "task-1"
    assert receipt.rows_queued == 2
    assert status.status == "complete"
    assert status.rows_written == 2
    assert [request.url.path for request in requests] == [
        "/ingest/bulk", "/ingest/tasks/task-1"
    ]
    client.close()


def test_append_event_uses_generic_contract_and_returns_receipt() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/events/append"
        assert json.loads(request.content) == {
            "object_type": "DocumentUsageEvent",
            "idempotency_key": "answer-1:citation",
            "target": {
                "object_type": "DocumentSource",
                "key": "report_id",
                "value": "rpt-1",
            },
            "attributes": {"cited": True},
            "purpose": "rag-usage",
        }
        return httpx.Response(
            201,
            json={
                "event_id": "evt-abc",
                "object_type": "DocumentUsageEvent",
                "status": "accepted",
                "durability": "local_wal_sync",
            },
        )

    client, requests = _client(handler)
    receipt = client.append_event(
        EventAppendRequest(
            object_type="DocumentUsageEvent",
            idempotency_key="answer-1:citation",
            target={
                "object_type": "DocumentSource",
                "key": "report_id",
                "value": "rpt-1",
            },
            attributes={"cited": True},
            purpose="rag-usage",
        )
    )
    assert receipt.event_id == "evt-abc"
    assert receipt.status == "accepted"
    assert receipt.durability == "local_wal_sync"
    assert [request.url.path for request in requests] == ["/events/append"]
    client.close()
