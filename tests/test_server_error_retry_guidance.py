"""Server-error metadata and no automatic mutation replay on retryable503."""
from datetime import datetime, timezone

import httpx
import pytest

from relata import _http
from relata._http import AsyncHttpTransport, HttpTransport
from relata.exceptions import AuthError, PurposeError, RelataError, ServerError


@pytest.mark.parametrize('payload', [
    b'{"detail":"busy","code":"fixture-busy","type":"urn:fixture:busy","query_id":"q7","retryable":true}',
    b'not-json', b'null', b'[]', b'"string"', b'123',
])
def test_sync_server_error_keeps_header_delay_and_never_replays_post(payload):
    seen = []
    def handler(request):
        seen.append(request.method)
        return httpx.Response(503, content=payload, headers={
            'Content-Type':'application/problem+json','Retry-After':'2','X-Request-ID':'req7'})
    transport = HttpTransport('http://test', None, 1, max_retries=2,
        transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(ServerError) as caught:
            transport.post('/ingest/bulk', {'objects': [], 'queued': True})
    finally:
        transport.close()
    error = caught.value
    assert error.status_code == 503
    assert error.request_id == 'req7' and error.retry_after == 2
    assert seen == ['POST']
    if payload.startswith(b'{'):
        assert error.code == 'fixture-busy' and error.type_url == 'urn:fixture:busy'
        assert error.query_id == 'q7' and error.retryable is True


@pytest.mark.asyncio
@pytest.mark.parametrize('payload', [b'{"retry_after":7,"retryable":true}', b'null', b'not-json'])
async def test_async_server_error_preserves_delay_without_post_replay(payload):
    seen = []
    async def handler(request):
        seen.append(request.method)
        return httpx.Response(503, content=payload, headers={'Retry-After':'2'})
    transport = AsyncHttpTransport('http://test', None, 1, max_retries=2,
        transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(ServerError) as caught:
            await transport.post('/ingest/bulk', {'objects': [], 'queued': True})
    finally:
        await transport.aclose()
    assert caught.value.status_code == 503 and caught.value.retry_after == 2
    assert seen == ['POST']


@pytest.mark.parametrize('header,body,expected', [
    ('invalid', 7, 7), ('-1', 7, 7), ('NaN', 7, 7), ('Infinity', 7, 7),
    ('9007199254740992', 7, 7), ('0', 7, 0), ('2', 7, 2),
    ('invalid', True, None), ('invalid', -1, None), ('invalid', float('inf'), None),
])
def test_server_delay_header_precedence_and_finite_body_fallback(header, body, expected):
    error = _http._classify_error(503, {'retry_after': body}, retry_after=_http._parse_retry_after(header))
    assert isinstance(error, ServerError) and error.retry_after == expected


def test_server_error_http_date_and_legacy_positional_constructors(monkeypatch):
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2030, 1, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(_http, 'datetime', FixedDatetime)
    error = _http._classify_error(503, {}, retry_after=_http._parse_retry_after('Tue, 01 Jan 2030 00:00:30 GMT'))
    assert error.retry_after == 30
    for cls in (ServerError, AuthError, PurposeError):
        legacy = cls('message', 503, 'code', 'urn:old', True, 'rid', 'qid')
        assert legacy.status_code == 503 and legacy.request_id == 'rid'
        assert legacy.query_id == 'qid' and legacy.retry_after is None
    assert RelataError('message').retry_after is None
