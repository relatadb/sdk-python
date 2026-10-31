"""Authored follow-up HTTP error contract regressions; execution deferred."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
import pytest

from relata import _http
from relata._http import AsyncHttpTransport, HttpTransport
from relata.exceptions import PurposeError, RateLimitedError, ResponseTooLargeError
from relata.streaming import _aclassify_capped_error, _classify_capped_error


class CountingStream(httpx.SyncByteStream):
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.iterations = 0
        self.closed = False

    def __iter__(self):
        self.iterations += 1
        yield self.payload

    def close(self) -> None:
        self.closed = True


class CountingAsyncStream(httpx.AsyncByteStream):
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.iterations = 0
        self.closed = False

    async def __aiter__(self):
        self.iterations += 1
        yield self.payload

    async def aclose(self) -> None:
        self.closed = True


def problem() -> dict:
    return {"detail": "quota exceeded", "code": "quota", "type": "urn:relata:quota",
            "query_id": "query-7", "retryable": True, "retry_after": 9}


def assert_metadata(error: RateLimitedError, delay: float) -> None:
    assert error.status_code == 429
    assert error.code == "quota"
    assert error.type_url == "urn:relata:quota"
    assert error.request_id == "request-8"
    assert error.query_id == "query-7"
    assert error.retryable is True
    assert error.retry_after == delay


@pytest.mark.parametrize("retry_header", ["not-a-date", "NaN", "Infinity"])
def test_sync_stream_error_invalid_retry_header_uses_body_once(retry_header: str) -> None:
    stream = CountingStream(json.dumps(problem()).encode())
    transport = HttpTransport("http://test", None, 1, max_retries=0,
        transport=httpx.MockTransport(lambda request: httpx.Response(429, stream=stream,
            headers={"Content-Type": "application/problem+json", "Retry-After": retry_header,
                     "X-Request-ID": "request-8"})))
    try:
        with transport._client.stream("POST", "/v1/query/stream") as response:
            with pytest.raises(RateLimitedError) as caught:
                _classify_capped_error(transport, response)
            assert_metadata(caught.value, 9)
    finally:
        transport.close()
    assert stream.iterations == 1
    assert stream.closed


@pytest.mark.asyncio
async def test_async_stream_error_preserves_http_date_and_metadata_once(monkeypatch) -> None:
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2030, 1, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(_http, "datetime", FixedDatetime)
    stream = CountingAsyncStream(json.dumps(problem()).encode())
    transport = AsyncHttpTransport("http://test", None, 1, max_retries=0,
        transport=httpx.MockTransport(lambda request: httpx.Response(429, stream=stream,
            headers={"Content-Type": "application/problem+json",
                     "Retry-After": "Tue, 01 Jan 2030 00:00:30 GMT",
                     "X-Request-ID": "request-8"})))
    try:
        async with transport._client.stream("POST", "/v1/query/arrow") as response:
            with pytest.raises(RateLimitedError) as caught:
                await _aclassify_capped_error(transport, response)
            assert_metadata(caught.value, 30)
    finally:
        await transport.aclose()
    assert stream.iterations == 1
    assert stream.closed


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_stream_error_body_cap_closes_response(asynchronous: bool) -> None:
    stream = CountingAsyncStream(b"x" * 65) if asynchronous else CountingStream(b"x" * 65)
    mock = httpx.MockTransport(lambda request: httpx.Response(503, stream=stream))
    if asynchronous:
        transport = AsyncHttpTransport("http://test", None, 1, transport=mock, max_response_bytes=64)
        try:
            async with transport._client.stream("POST", "/v1/query/stream") as response:
                with pytest.raises(ResponseTooLargeError):
                    await _aclassify_capped_error(transport, response)
        finally:
            await transport.aclose()
    else:
        sync = HttpTransport("http://test", None, 1, transport=mock, max_response_bytes=64)
        try:
            with sync._client.stream("POST", "/v1/query/stream") as response:
                with pytest.raises(ResponseTooLargeError):
                    _classify_capped_error(sync, response)
        finally:
            sync.close()
    assert stream.iterations == 1
    assert stream.closed


def test_purpose_error_retains_problem_metadata() -> None:
    payload = {"detail": "purpose not registered", "code": "purpose-required",
               "type": "urn:relata:purpose", "retryable": False, "query_id": "purpose-query"}
    transport = HttpTransport("http://test", None, 1, max_retries=0,
        transport=httpx.MockTransport(lambda request: httpx.Response(400, json=payload,
            headers={"X-Request-ID": "purpose-request"})))
    try:
        with transport._client.stream("POST", "/v1/query/stream") as response:
            with pytest.raises(PurposeError) as caught:
                _classify_capped_error(transport, response)
            error = caught.value
            assert error.status_code == 400
            assert error.code == "purpose-required"
            assert error.type_url == "urn:relata:purpose"
            assert error.query_id == "purpose-query"
            assert error.request_id == "purpose-request"
            assert error.retryable is False
    finally:
        transport.close()


@pytest.mark.parametrize("payload", [None, [], ["error"], "failure", 42, {"detail": ["wrong shape"], "code": 7}])
def test_nonobject_or_malformed_problem_keeps_http_status(payload) -> None:
    from relata.exceptions import ServerError
    transport = HttpTransport("http://test", None, 1, max_retries=0,
        transport=httpx.MockTransport(lambda request: httpx.Response(503, json=payload)))
    try:
        # The real buffered transport path, not only the streaming wrapper.
        response = transport._client.get("/v1/query")
        with pytest.raises(ServerError) as caught:
            transport._handle(response)
        assert caught.value.status_code == 503
        assert caught.value.code is None
    finally:
        transport.close()
