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
        if request.url.path == "/v1/ingest/bulk":
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
        assert request.url.path == "/v1/ingest/tasks/task-1"
        return httpx.Response(
            200,
            json={
                "task_id": "task-1",
                "status": "partial",
                "rows_total": 2,
                "rows_written": 1,
                "rows_dropped": 1,
                "drop_reasons": ["write rejected: invalid row"],
                "error": {"code": "partial_write", "retryable": False},
                "partial_result": {"accepted_ids": ["chunk-1"]},
            },
        )

    client, requests = _client(handler)
    receipt = client.ingest_document(_ROWS)
    status = client.ingest_document_status(receipt.task_id)
    assert receipt.task_id == "task-1"
    assert receipt.rows_queued == 2
    assert status.status == "partial"
    assert status.rows_written == 1
    assert status.rows_dropped == 1
    assert status.drop_reasons == ["write rejected: invalid row"]
    assert status.model_extra["error"] == {"code": "partial_write", "retryable": False}
    assert status.model_extra["partial_result"] == {"accepted_ids": ["chunk-1"]}
    assert [request.url.path for request in requests] == [
        "/v1/ingest/bulk", "/v1/ingest/tasks/task-1"
    ]
    client.close()


def test_append_event_uses_generic_contract_and_returns_receipt() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/ingest/events"
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
    assert [request.url.path for request in requests] == ["/v1/ingest/events"]
    client.close()


def test_document_task_status_preserves_complete_and_failed_outcomes() -> None:
    from relata.models import IngestDocumentTaskStatus

    complete = IngestDocumentTaskStatus.model_validate({
        "task_id": "task-complete", "status": "complete", "rows_total": 2,
        "rows_written": 2, "rows_dropped": 0, "drop_reasons": [],
    })
    failed = IngestDocumentTaskStatus.model_validate({
        "task_id": "task-failed", "status": "failed", "rows_total": 2,
        "rows_written": 0, "rows_dropped": 2,
        "drop_reasons": ["writer refused queued rows after WAL durability failure"],
    })
    uncertain = IngestDocumentTaskStatus.model_validate({
        "task_id": "task-uncertain", "status": "uncertain", "rows_total": 3,
        "rows_written": 1, "rows_dropped": 0,
        "drop_reasons": ["WAL acknowledgement outcome unknown"],
        "outcome_uncertain": True, "retryable": False,
    })
    incomplete = IngestDocumentTaskStatus.model_validate({
        "task_id": "task-incomplete", "status": "incomplete", "rows_total": 3,
        "rows_written": 2, "rows_dropped": 1, "drop_reasons": ["confirmed writer drop"],
        "code": "derived-write-incomplete", "derived_error": "vector projection failed",
        "retryable": False,
    })
    pending_derived = IngestDocumentTaskStatus.model_validate({
        "task_id": "task-pending-derived", "status": "pending", "rows_total": 3,
        "rows_written": 1, "rows_dropped": 0,
        "code": "derived-write-incomplete", "derived_error": "vector projection failed",
        "retryable": False,
    })
    assert complete.status == "complete" and complete.rows_written == 2
    assert failed.status == "failed" and failed.rows_dropped == 2
    assert uncertain.status == "uncertain" and uncertain.rows_written == 1
    assert uncertain.rows_dropped == 0 and uncertain.outcome_uncertain is True
    assert uncertain.retryable is False
    assert incomplete.status == "incomplete" and incomplete.rows_written == 2
    assert incomplete.rows_dropped == 1 and incomplete.code == "derived-write-incomplete"
    assert incomplete.derived_error == "vector projection failed" and incomplete.retryable is False
    assert pending_derived.status == "pending"
    assert pending_derived.code == "derived-write-incomplete"
    assert pending_derived.derived_error == "vector projection failed"
    assert pending_derived.retryable is False


def test_submit_alert_sync_and_async_preserve_contract() -> None:
    import asyncio
    from relata import AlertSubmitRequest

    payload = {"matched_key": "_pk", "idempotency_key": "run-1", "purpose": "detection",
               "alert_type": "app-signal", "severity": "high",
               "matched_type": "Event", "matched_row_id": "row-1", "context": {"count": 2}}
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/alerts"
        assert json.loads(request.content) == payload
        return httpx.Response(200, json={"alert_id": "alert-stable", "status": "duplicate",
                                         "durability": "local_wal_sync"})
    client, requests = _client(handler)
    request = AlertSubmitRequest(**payload)
    assert client.submit_alert(request).alert_id == "alert-stable"
    async def run() -> None:
        receipt = await client.asubmit_alert(request)
        assert receipt.status == "duplicate"
        assert receipt.durability == "local_wal_sync"
        await client.aclose()
    asyncio.run(run())
    client.close()
    assert len(requests) == 2
