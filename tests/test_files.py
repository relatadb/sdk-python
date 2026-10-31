"""Files uses the normal capped authenticated transport, including byte bodies."""
import asyncio
import json

import httpx
import pytest

from relata import AsyncFilesClient, FilesClient, RelataClient, ResponseTooLargeError, ServerError


PAYLOAD = b'\x00\xff\x80binary\r\n'
KEY = 'folder/a ?#&é.bin'
ETAG = '"fixture-etag"'


def metadata():
    return dict(bucket='tenant-a', key=KEY, row_id='physical-row', size=len(PAYLOAD),
        etag=ETAG, content_type='application/x-fixture', modified_ns=123,
        is_folder=False, download_url='/files/content?key=encoded')


class SyncChunks(httpx.SyncByteStream):
    def __init__(self, chunks):
        self.chunks, self.closed, self.reads = chunks, False, 0

    def __iter__(self):
        for chunk in self.chunks:
            self.reads += 1
            yield chunk

    def close(self):
        self.closed = True


class AsyncChunks(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks, self.closed, self.reads = chunks, False, 0

    async def __aiter__(self):
        for chunk in self.chunks:
            self.reads += 1
            yield chunk

    async def aclose(self):
        self.closed = True


def test_sync_files_round_trip_preserves_exact_bytes_headers_and_query_encoding():
    seen, streams = [], []
    extra = {'x-relata-tenant-id':'wrong-tenant', 'X-Custom':'retained'}
    before = dict(extra)
    def handler(request):
        seen.append(request.method)
        assert request.url.path == '/files/content'
        assert request.url.params['key'] == KEY and request.url.params['bucket'] == 'tenant-a'
        assert request.headers['Authorization'] == 'Bearer secret'
        assert request.headers.get_list('X-Relata-Tenant-Id') == ['tenant-a']
        assert request.headers['X-Custom'] == 'retained'
        if request.method == 'PUT':
            assert request.content == PAYLOAD
            assert request.headers['Content-Type'] == 'application/x-fixture'
            return httpx.Response(201, json=metadata())
        if request.method == 'DELETE':
            return httpx.Response(204)
        stream = SyncChunks([PAYLOAD[:2], PAYLOAD[2:]])
        streams.append(stream)
        return httpx.Response(200, stream=stream, headers={'ETag':ETAG, 'Content-Type':'application/x-fixture'})
    client = FilesClient('http://test', bearer_token='secret', tenant='tenant-a',
        extra_headers=extra, transport=httpx.MockTransport(handler))
    with client:
        meta = client.put(KEY, PAYLOAD, bucket='tenant-a', content_type='application/x-fixture')
        assert meta.etag == ETAG and meta.key == KEY and meta.size == len(PAYLOAD)
        content = client.get(KEY, bucket='tenant-a')
        assert content.data == PAYLOAD and content.size == len(PAYLOAD)
        assert content.etag == ETAG and content.content_type == 'application/x-fixture'
        assert client.delete(KEY, bucket='tenant-a') is None
    client.close()
    assert seen == ['PUT','GET','DELETE'] and streams[0].closed
    assert extra == before and client._t._client.is_closed


@pytest.mark.asyncio
async def test_async_files_round_trip_keeps_binary_and_closes_transport():
    seen, streams = [], []
    async def handler(request):
        seen.append(request.method)
        assert request.headers['Authorization'] == 'Bearer secret'
        assert request.headers['X-Relata-Tenant-Id'] == 'tenant-a'
        assert request.url.params['key'] == KEY
        if request.method == 'PUT':
            assert request.content == PAYLOAD
            return httpx.Response(201, json=metadata())
        if request.method == 'DELETE': return httpx.Response(204)
        stream = AsyncChunks([PAYLOAD]); streams.append(stream)
        return httpx.Response(200, stream=stream, headers={'ETag':ETAG,'Content-Type':'application/x-fixture'})
    client = AsyncFilesClient('http://test', bearer_token='secret', tenant='tenant-a', transport=httpx.MockTransport(handler))
    async with client:
        assert (await client.put(KEY, PAYLOAD)).etag == ETAG
        content = await client.get(KEY)
        assert content.data == PAYLOAD and content.etag == ETAG
        assert await client.delete(KEY) is None
    await client.aclose()
    assert seen == ['PUT','GET','DELETE'] and streams[0].closed and client._t._client.is_closed


@pytest.mark.parametrize('method', ['put','delete'])
def test_sync_files_never_replay_mutations_and_keep_503_guidance(method):
    calls = []
    def handler(request):
        calls.append(request.method)
        return httpx.Response(503, json={'code':'fixture-busy','detail':'uncertain','retryable':True},
            headers={'Retry-After':'2','X-Request-ID':'request-files'})
    with FilesClient('http://test', max_retries=2, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ServerError) as caught:
            client.put(KEY, PAYLOAD) if method == 'put' else client.delete(KEY)
    assert calls == [method.upper()]
    assert caught.value.status_code == 503 and caught.value.retry_after == 2
    assert caught.value.request_id == 'request-files' and caught.value.code == 'fixture-busy'


@pytest.mark.asyncio
@pytest.mark.parametrize('method', ['put','delete'])
async def test_async_files_never_replay_mutations_and_keep_503_guidance(method):
    calls = []
    async def handler(request):
        calls.append(request.method)
        return httpx.Response(503, content=b'null', headers={'Retry-After':'2','X-Request-ID':'request-files'})
    async with AsyncFilesClient('http://test', max_retries=2, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ServerError) as caught:
            if method == 'put': await client.put(KEY, PAYLOAD)
            else: await client.delete(KEY)
    assert calls == [method.upper()]
    assert caught.value.status_code == 503 and caught.value.retry_after == 2
    assert caught.value.request_id == 'request-files'


@pytest.mark.parametrize('advertised', [False,True])
def test_sync_files_cap_closes_body_before_or_during_read(advertised):
    stream = SyncChunks([b'abc', b'de'])
    headers = {'Content-Length':'5'} if advertised else {}
    with FilesClient('http://test', max_response_bytes=4,
        transport=httpx.MockTransport(lambda _:httpx.Response(200, stream=stream, headers=headers))) as client:
        with pytest.raises(ResponseTooLargeError): client.get(KEY)
    assert stream.closed and stream.reads == (0 if advertised else 2)


@pytest.mark.asyncio
@pytest.mark.parametrize('advertised', [False,True])
async def test_async_files_cap_closes_body_before_or_during_read(advertised):
    stream = AsyncChunks([b'abc', b'de'])
    headers = {'Content-Length':'5'} if advertised else {}
    async with AsyncFilesClient('http://test', max_response_bytes=4,
        transport=httpx.MockTransport(lambda _:httpx.Response(200, stream=stream, headers=headers))) as client:
        with pytest.raises(ResponseTooLargeError): await client.get(KEY)
    assert stream.closed and stream.reads == (0 if advertised else 2)


@pytest.mark.asyncio
async def test_async_files_cancellation_closes_partial_response_and_next_get_still_works():
    blocked = asyncio.Event()
    class SlowBody(httpx.AsyncByteStream):
        closed = False
        async def __aiter__(self):
            yield b'partial'
            blocked.set()
            await asyncio.Event().wait()
        async def aclose(self): self.closed = True
    stream = SlowBody()
    calls = []
    async def handler(request):
        calls.append(request.method)
        return httpx.Response(200, stream=stream) if len(calls)==1 else httpx.Response(200, content=PAYLOAD)
    async with AsyncFilesClient('http://test', transport=httpx.MockTransport(handler)) as client:
        task = asyncio.create_task(client.get(KEY))
        await asyncio.wait_for(blocked.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError): await task
        assert stream.closed
        assert (await client.get(KEY)).data == PAYLOAD
    assert calls == ['GET','GET']


def test_borrowed_parent_keeps_exact_cap_and_lifetime(monkeypatch):
    real_client = httpx.Client
    def configured_client(**kwargs):
        kwargs['transport'] = httpx.MockTransport(lambda _: httpx.Response(200, content=b'abc'))
        return real_client(**kwargs)
    monkeypatch.setattr(httpx, 'Client', configured_client)
    with RelataClient('http://test', max_response_bytes=2, max_retries=2) as parent:
        view = FilesClient.from_client(parent)
        assert view._t is parent._sync and view._t._max_retries == 2
        with pytest.raises(ResponseTooLargeError): view.get(KEY)
        view.close()
        assert not parent._sync._client.is_closed
    assert view._t._client.is_closed


def test_invalid_upload_type_fails_before_http_mutation():
    calls = []
    with FilesClient('http://test', transport=httpx.MockTransport(lambda request:calls.append(request))) as client:
        with pytest.raises(TypeError): client.put(KEY, 'not-bytes')
    assert calls == []


def test_files_get_retains_existing_safe_retry_policy():
    calls = []
    def handler(request):
        calls.append(request.method)
        return httpx.Response(503, json={'detail':'busy'}) if len(calls)==1 else httpx.Response(200, content=PAYLOAD)
    with FilesClient('http://test', max_retries=2, retry_backoff=0, transport=httpx.MockTransport(handler)) as client:
        assert client.get(KEY).data == PAYLOAD
    assert calls == ['GET','GET']


@pytest.mark.asyncio
async def test_async_borrowed_parent_preserves_cap_without_closing_parent(monkeypatch):
    real_client = httpx.AsyncClient
    def configured_client(**kwargs):
        kwargs['transport'] = httpx.MockTransport(lambda _: httpx.Response(200, content=b'abc'))
        return real_client(**kwargs)
    monkeypatch.setattr(httpx, 'AsyncClient', configured_client)
    async with RelataClient('http://test', max_response_bytes=2, max_retries=2) as parent:
        view = AsyncFilesClient.from_client(parent)
        assert view._t is parent._async and view._t._max_retries == 2
        with pytest.raises(ResponseTooLargeError): await view.get(KEY)
        await view.aclose()
        assert not parent._async._client.is_closed
    assert view._t._client.is_closed


@pytest.mark.parametrize('method,wrong_status', [('put',200),('get',201),('delete',200)])
def test_files_refuses_unexpected_success_status_instead_of_acknowledging(method, wrong_status):
    from relata import RelataError
    calls = []
    def handler(request):
        calls.append(request.method)
        return httpx.Response(wrong_status, content=PAYLOAD)
    with FilesClient('http://test', max_retries=2, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RelataError) as caught:
            if method == 'put': client.put(KEY, PAYLOAD)
            elif method == 'get': client.get(KEY)
            else: client.delete(KEY)
    assert caught.value.status_code == wrong_status
    assert caught.value.code == 'unexpected-response-status'
    assert calls == [method.upper()]


class FakeUploads:
    """In-memory model of the resumable-upload protocol with one injected failure."""
    def __init__(self, fail_chunk_at=None):
        self.data, self.size, self.sha, self.fail_at = bytearray(), 0, None, fail_chunk_at
        self.calls, self.digests, self.completed = [], [], None

    def session(self):
        return dict(upload_id='u1', bucket='tenant-a', key=KEY, size=self.size,
            offset=len(self.data), complete=len(self.data) == self.size, sha256=self.sha,
            expires_ns=1, max_chunk_bytes=4)

    def __call__(self, request):
        self.calls.append((request.method, request.url.path))
        if request.method == 'POST' and request.url.path == '/files/uploads':
            body = json.loads(request.content)
            self.size, self.sha = body['size'], body.get('sha256')
            return httpx.Response(201, json=self.session())
        if request.method == 'PUT':
            if self.fail_at is not None and len(self.data) == self.fail_at:
                self.fail_at = None
                return httpx.Response(503, json={'code':'busy','detail':'retry','retryable':True})
            offset = int(request.url.params['offset'])
            if offset != len(self.data):
                return httpx.Response(409, json={'code':'offset','detail':'mismatch'})
            self.digests.append(request.headers['X-Relata-Chunk-Sha256'])
            self.data.extend(request.content)
            return httpx.Response(200, json=self.session())
        if request.method == 'GET':
            return httpx.Response(200, json=self.session())
        if request.url.path.endswith('/complete'):
            self.completed = bytes(self.data)
            return httpx.Response(201, json=metadata())
        return httpx.Response(204)


def test_sync_resumable_upload_resumes_after_transient_failure_with_exact_bytes():
    import hashlib, io
    payload = bytes(range(10))
    server = FakeUploads(fail_chunk_at=4)
    with FilesClient('http://test', transport=httpx.MockTransport(server)) as client:
        meta = client.upload(KEY, io.BytesIO(payload), bucket='tenant-a', chunk_size=4)
    assert meta.size == len(PAYLOAD)
    assert server.completed == payload and server.sha == hashlib.sha256(payload).hexdigest()
    assert server.digests == [hashlib.sha256(payload[i:i+4]).hexdigest() for i in (0, 4, 8)]
    assert ('GET', '/files/uploads/u1') in server.calls  # offset re-read after the 503


def test_sync_resumable_upload_gives_up_after_max_attempts_and_keeps_session():
    server = FakeUploads()
    def handler(request):
        if request.method == 'PUT':
            return httpx.Response(503, json={'code':'busy','detail':'retry','retryable':True})
        return server(request)
    with FilesClient('http://test', transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ServerError):
            client.upload(KEY, b'0123456789', chunk_size=4, max_attempts=2)
    assert ('DELETE', '/files/uploads/u1') not in server.calls and server.completed is None


@pytest.mark.asyncio
async def test_async_resumable_upload_round_trip():
    payload = b'abcdefghij'
    server = FakeUploads(fail_chunk_at=4)
    async def handler(request):
        return server(request)
    async with AsyncFilesClient('http://test', transport=httpx.MockTransport(handler)) as client:
        await client.upload(KEY, payload, chunk_size=4)
        await client.abort_upload('u1')
    assert server.completed == payload
