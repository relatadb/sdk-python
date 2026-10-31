"""Row-stream wire and lifecycle regressions, shared by sync/async clients."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator

import httpx
import pytest

from relata import IncompleteQueryStreamError, RelataClient, ResponseTooLargeError
from relata._http import AsyncHttpTransport, HttpTransport
from relata.streaming import AsyncStreamingClient, StreamingClient

BASE = "http://localhost:9090"
SQL = "SELECT id FROM Person"


def _frame(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False).encode("utf-8") + b"\n"


START = _frame({"_meta": {"query_id": "q1", "started_at": 1, "columns": ["id"]}})
ROW = _frame({"id": 1})


def _end(count: int, **extra: object) -> bytes:
    return _frame({"_meta": {
        "processing_time_ms": 1, "row_count": count, "truncated": False, **extra,
    }})


class _SyncBody(httpx.SyncByteStream):
    def __init__(self, chunks: list[bytes], *, fail: bool = False) -> None:
        self.chunks = chunks
        self.fail = fail
        self.closes = 0

    def __iter__(self) -> Iterator[bytes]:
        yield from self.chunks
        if self.fail:
            raise httpx.ReadError("disconnected")

    def close(self) -> None:
        self.closes += 1


class _AsyncBody(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes], *, fail: bool = False) -> None:
        self.chunks = chunks
        self.fail = fail
        self.closes = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk
        if self.fail:
            raise httpx.ReadError("disconnected")

    async def aclose(self) -> None:
        self.closes += 1


def _sync_client(body: _SyncBody, *, cap: int = 64 * 1024 * 1024) -> RelataClient:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/query/stream"
        assert request.headers["accept"] == "application/x-ndjson"
        assert request.headers["x-tenant-id"] == "acme"
        assert request.headers["authorization"] == "Bearer token"
        assert json.loads(request.content) == {"sql": SQL, "purpose": "analytics"}
        return httpx.Response(200, stream=body)

    client = RelataClient(BASE, purpose="analytics")
    client._RelataClient__sync_transport = HttpTransport(  # type: ignore[attr-defined]
        BASE, "token", 30.0, transport=httpx.MockTransport(handler),
        extra_headers={"X-Tenant-ID": "acme", "accept": "application/json"},
        max_response_bytes=cap,
    )
    return client


def _async_client(body: _AsyncBody, *, cap: int = 64 * 1024 * 1024) -> RelataClient:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/query/stream"
        assert request.headers["accept"] == "application/x-ndjson"
        assert request.headers["x-tenant-id"] == "acme"
        assert json.loads(request.content) == {"sql": SQL, "purpose": "analytics"}
        return httpx.Response(200, stream=body)

    client = RelataClient(BASE, purpose="analytics")
    client._RelataClient__async_transport = AsyncHttpTransport(  # type: ignore[attr-defined]
        BASE, "token", 30.0, transport=httpx.MockTransport(handler),
        extra_headers={"X-Tenant-ID": "acme", "accept": "application/json"},
        max_response_bytes=cap,
    )
    return client


def test_rows_filter_metadata_allow_extensions_and_preserve_meta_columns() -> None:
    start = _frame({"_meta": {
        "query_id": "q1", "started_at": 1, "columns": ["_meta"], "future": True,
    }})
    rows = [{"_meta": None}, {"_meta": "value"}, {"_meta": {"custom": 1}}]
    body = _SyncBody([start + b"".join(_frame(row) for row in rows)
                      + _end(3, future="extension").rstrip(b"\n")])
    with _sync_client(body) as client:
        assert list(StreamingClient(client).query_rows(SQL)) == rows
    assert body.closes == 1


_INCOMPLETE = [
    b"",
    START + ROW,
    ROW + _end(1),
    START + START + _end(0),
    START + _end(0) + _end(0),
    START + _end(0) + ROW,
    START + ROW + _end(2),
    START + ROW + _end(1, truncated=True),
    START + ROW + _end(1, error="failed"),
    START + _end(0, row_count=True),
    START + _end(0, processing_time_ms=-1),
    START + b"[]\n",
    START + b"{invalid}\n",
    START + b'{"id":"\xff"}\n',
    START + '{"id":1}\n'.encode("utf-16-le"),
    START + b'{"id":NaN}\n',
]


@pytest.mark.parametrize("payload", _INCOMPLETE)
def test_sync_incomplete_frames_close_without_retry(payload: bytes) -> None:
    body = _SyncBody([payload])
    with _sync_client(body) as client, pytest.raises(IncompleteQueryStreamError) as caught:
        list(StreamingClient(client).query_rows(SQL))
    assert caught.value.status_code == 200
    assert caught.value.retryable is False
    assert body.closes == 1


@pytest.mark.parametrize("payload", _INCOMPLETE)
async def test_async_incomplete_frames_close_without_retry(payload: bytes) -> None:
    body = _AsyncBody([payload])
    client = _async_client(body)
    try:
        with pytest.raises(IncompleteQueryStreamError):
            iterator = await AsyncStreamingClient(client).query_rows(SQL)
            _ = [row async for row in iterator]
        assert body.closes == 1
    finally:
        await client.aclose()


def test_partial_transport_failure_preserves_progress_and_closes() -> None:
    body = _SyncBody([START, ROW], fail=True)
    with _sync_client(body) as client:
        iterator = StreamingClient(client).query_rows(SQL)
        assert next(iterator) == {"id": 1}
        with pytest.raises(IncompleteQueryStreamError) as caught:
            next(iterator)
    assert caught.value.query_id == "q1"
    assert caught.value.rows_received == 1
    assert isinstance(caught.value.__cause__, httpx.ReadError)
    assert body.closes == 1


async def test_async_partial_transport_failure_preserves_progress_and_closes() -> None:
    body = _AsyncBody([START, ROW], fail=True)
    client = _async_client(body)
    try:
        iterator = await AsyncStreamingClient(client).query_rows(SQL)
        assert await anext(iterator) == {"id": 1}
        with pytest.raises(IncompleteQueryStreamError) as caught:
            await anext(iterator)
        assert caught.value.query_id == "q1"
        assert caught.value.rows_received == 1
        assert body.closes == 1
    finally:
        await client.aclose()


def test_intentional_close_releases_partial_response() -> None:
    body = _SyncBody([START + ROW, _end(1)])
    with _sync_client(body) as client:
        iterator = StreamingClient(client).query_rows(SQL)
        assert next(iterator) == {"id": 1}
        iterator.close()
    assert body.closes == 1


async def test_async_intentional_close_releases_partial_response() -> None:
    body = _AsyncBody([START + ROW, _end(1)])
    client = _async_client(body)
    try:
        iterator = await AsyncStreamingClient(client).query_rows(SQL)
        assert await anext(iterator) == {"id": 1}
        await iterator.aclose()
        assert body.closes == 1
    finally:
        await client.aclose()


def test_per_frame_cap_allows_long_stream_and_split_multibyte_utf8() -> None:
    rows = [{"id": "é"}] * 100
    wire = START + b"".join(_frame(row) for row in rows) + _end(len(rows))
    body = _SyncBody([bytes([byte]) for byte in wire])
    with _sync_client(body, cap=128) as client:
        assert list(StreamingClient(client).query_rows(SQL)) == rows
    assert len(wire) > 128
    assert body.closes == 1


def test_oversized_split_frame_is_rejected_and_response_closed() -> None:
    body = _SyncBody([START, b'{"id":"', b"x" * 129, b'"}\n', _end(1)])
    with _sync_client(body, cap=128) as client, pytest.raises(ResponseTooLargeError) as caught:
        list(StreamingClient(client).query_rows(SQL))
    assert caught.value.max_bytes == 128
    assert body.closes == 1


async def test_async_long_stream_and_cap() -> None:
    rows = [{"id": "é"}] * 100
    wire = START + b"".join(_frame(row) for row in rows) + _end(len(rows))
    body = _AsyncBody([bytes([byte]) for byte in wire])
    client = _async_client(body, cap=128)
    try:
        iterator = await AsyncStreamingClient(client).query_rows(SQL)
        assert [row async for row in iterator] == rows
        assert body.closes == 1
    finally:
        await client.aclose()

    oversized = _AsyncBody([START, b"x" * 129])
    client = _async_client(oversized, cap=128)
    try:
        with pytest.raises(ResponseTooLargeError):
            iterator = await AsyncStreamingClient(client).query_rows(SQL)
            _ = [row async for row in iterator]
        assert oversized.closes == 1
    finally:
        await client.aclose()
