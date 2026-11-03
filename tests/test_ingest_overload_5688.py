"""Document and CDR SDK batches under server overload (#5688).

The server sheds load with 429 (queue/byte budget) and 503 (memory pressure)
plus ``Retry-After``. The SDK contract these tests pin, for both adapters that
queue batches (``ingest_document`` and ``ingest_cdr``):

* the shed response surfaces as the typed error carrying the server delay,
  and the batch POST is sent exactly once (never replayed by the transport);
* the caller-driven retry of the *same* stable-key batch succeeds once the
  overload clears, and sends byte-identical rows, so server-side keyed upserts
  make the retry idempotent;
* a batch that was accepted-then-uncertain (503 carrying ``task_id``) hands the
  caller the task to poll instead of a reason to re-send;
* partial graph processing is reported by the task status, not by the queue
  receipt, and the SDK-side invalid-row report survives the retry.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from relata import RelataClient
from relata._http import AsyncHttpTransport, HttpTransport
from relata.exceptions import RateLimitedError, ServerError
from relata.ingest import IngestClient

BASE = "http://localhost:9090"
_DOC_ROWS = [
    {"_type": "DocumentSource", "_pk": "doc-1", "uri": "file://report.pdf"},
    {"_type": "DocumentChunk", "_pk": "chunk-1", "source_id": "doc-1", "text": "hello"},
]
# Rows carry an explicit call_start: a row without one is stamped with the
# SDK wall clock at normalization time, so a retry would not be byte-identical.
_CDR_ROWS = [
    {"caller": "911234567890", "callee": "919876543210", "duration": 120,
     "tower": "CELL-042", "type": "voice", "_pk": "cdr-1", "source_file_id": 42,
     "call_start": 1700000000},
    {"caller": "not a number", "callee": "922222222222"},
    {"caller": "911111111111", "callee": "922222222222", "duration": 5,
     "tower": "CELL-001", "type": "sms", "_pk": "cdr-2", "source_file_id": 43,
     "call_start": 1700000001},
]


def _problem(status: int, retry_after: str, **extra: object) -> httpx.Response:
    body = {"detail": "overloaded", "code": "ingest-overloaded", "retryable": True, **extra}
    return httpx.Response(
        status,
        content=json.dumps(body).encode(),
        headers={"Content-Type": "application/problem+json", "Retry-After": retry_after},
    )


def _document_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> RelataClient:
    client = RelataClient(BASE, purpose="analytics")
    transport = httpx.MockTransport(handler)
    extra = client._extra_headers
    client._RelataClient__sync_transport = HttpTransport(  # type: ignore[attr-defined]
        BASE, client._bearer_token, client._timeout, transport=transport, extra_headers=extra,
    )
    client._RelataClient__async_transport = AsyncHttpTransport(  # type: ignore[attr-defined]
        BASE, client._bearer_token, client._timeout, transport=transport, extra_headers=extra,
    )
    return client


def test_document_batch_overload_is_surfaced_once_and_retried_by_the_caller() -> None:
    bodies: list[bytes] = []
    script = iter([_problem(429, "3"), _problem(503, "5")])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/ingest/bulk":
            bodies.append(request.content)
            shed = next(script, None)
            if shed is not None:
                return shed
            return httpx.Response(200, json={
                "task_id": "task-9", "rows_queued": 2, "rows_rejected": 0,
                "errors": [], "queue_depth": 2,
            })
        assert request.url.path == "/v1/ingest/tasks/task-9"
        return httpx.Response(200, json={
            "task_id": "task-9", "status": "partial", "rows_total": 2,
            "rows_written": 1, "rows_dropped": 1,
            "drop_reasons": ["graph edge target missing"],
            "partial_result": {"accepted_ids": ["doc-1"]},
        })

    client = _document_client(handler)
    with pytest.raises(RateLimitedError) as shed429:
        client.ingest_document(_DOC_ROWS)
    assert shed429.value.status_code == 429 and shed429.value.retry_after == 3
    assert len(bodies) == 1, "a 429 must not be replayed by the transport"

    with pytest.raises(ServerError) as shed503:
        client.ingest_document(_DOC_ROWS)
    assert shed503.value.status_code == 503 and shed503.value.retry_after == 5
    assert len(bodies) == 2, "a 503 must not be replayed by the transport"

    receipt = client.ingest_document(_DOC_ROWS)
    assert receipt.task_id == "task-9" and receipt.rows_queued == 2
    assert len(bodies) == 3 and bodies[0] == bodies[1] == bodies[2], (
        "caller retries must resend the identical stable-key batch"
    )
    status = client.ingest_document_status(receipt.task_id)
    assert status.status == "partial"
    assert status.rows_written == 1 and status.rows_dropped == 1
    client.close()


def test_accepted_then_uncertain_503_hands_back_the_task_not_a_reason_to_resend() -> None:
    posts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        posts.append(request.url.path)
        return _problem(503, "2", task_id="task-uncertain", accepted=True,
                        receipt_status="uncertain")

    client = _document_client(handler)
    with pytest.raises(ServerError) as caught:
        client.ingest_document(_DOC_ROWS)
    error = caught.value
    assert error.task_id == "task-uncertain" and error.accepted is True
    assert error.receipt_status == "uncertain" and error.retry_after == 2
    assert posts == ["/v1/ingest/bulk"]
    client.close()


def test_cdr_batch_overload_keeps_row_report_and_replays_nothing() -> None:
    bodies: list[bytes] = []
    script = iter([_problem(429, "1"), _problem(503, "4")])

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/ingest"
        bodies.append(request.content)
        shed = next(script, None)
        if shed is not None:
            return shed
        return httpx.Response(200, json={
            "rows_queued": 2, "rows_rejected": 0, "errors": [], "task_id": "itsk_cdr",
        })

    client = IngestClient(BASE, purpose="finint", transport=httpx.MockTransport(handler))
    with pytest.raises(RateLimitedError) as shed429:
        client.ingest_cdr(_CDR_ROWS)
    assert shed429.value.retry_after == 1 and len(bodies) == 1
    with pytest.raises(ServerError) as shed503:
        client.ingest_cdr(_CDR_ROWS)
    assert shed503.value.retry_after == 4 and len(bodies) == 2

    out = client.ingest_cdr(_CDR_ROWS)
    assert len(bodies) == 3 and bodies[0] == bodies[1] == bodies[2]
    sent = json.loads(bodies[2])
    assert [row["_pk"] for row in sent] == ["cdr-1", "cdr-2"], "invalid row never sent"
    assert out["task_id"] == "itsk_cdr"
    assert out["rows_rejected"] == 1 and len(out["errors"]) == 1, (
        "the SDK-side invalid-row report survives the overload retry"
    )


@pytest.mark.asyncio
async def test_async_document_batch_overload_is_not_replayed() -> None:
    seen: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.method)
        return _problem(429, "7")

    client = _document_client(handler)  # type: ignore[arg-type]
    with pytest.raises(RateLimitedError) as caught:
        await client.aingest_document(_DOC_ROWS)
    assert caught.value.retry_after == 7 and seen == ["POST"]
    await client.aclose()
