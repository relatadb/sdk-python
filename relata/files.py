"""Buffered binary Files content through the existing governed HTTP transport.

PUT/GET/HEAD/DELETE ``/v1/files/content``, ``GET /v1/files`` (list),
``GET /v1/media/{row_id}/content`` and the resumable ``/v1/files/uploads`` session
(chunked, disk-staged, digest-verified; ``upload`` resumes after a failed chunk)
are wrapped. Range requests, conditional writes, versions, provider-copy
completion and physical deletion are not promised. PUT and DELETE are never
automatically replayed, even with ``max_retries > 0``; ``complete_upload`` is a
POST and is never replayed either (reconcile with ``upload_status``/a file read).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, BinaryIO
from urllib.parse import quote, urlencode

import httpx
from pydantic import BaseModel, ConfigDict

from relata._http import (
    DEFAULT_MAX_RESPONSE_BYTES,
    AsyncHttpTransport,
    HttpTransport,
    _RawHttpResponse,
)
from relata.exceptions import NotFoundError, RelataError

if TYPE_CHECKING:
    from relata.client import RelataClient


class FileMetadata(BaseModel):
    """The acknowledged PUT metadata; not a global-copy durability certificate."""
    model_config = ConfigDict(extra="allow")
    bucket: str
    key: str
    row_id: str | None
    size: int
    etag: str
    content_type: str
    modified_ns: int
    is_folder: bool
    download_url: str


class UploadSession(BaseModel):
    """Server-side view of a resumable upload: ``offset`` is the next byte expected."""
    model_config = ConfigDict(extra="allow")
    upload_id: str
    bucket: str
    key: str
    size: int
    offset: int
    complete: bool
    sha256: str | None = None
    expires_ns: int
    max_chunk_bytes: int | None = None


@dataclass(frozen=True)
class FileContent:
    """Exact buffered bytes, response ETag/type and actual buffered size."""
    data: bytes
    etag: str | None
    content_type: str | None

    @property
    def size(self) -> int:
        return len(self.data)


@dataclass(frozen=True)
class FileHead:
    """``HEAD /v1/files/content`` view: headers only, no body bytes."""
    etag: str | None
    content_type: str | None
    size: int


def _head(response: _RawHttpResponse) -> FileHead:
    _status(response, 200)
    length = response.headers.get("content-length")
    return FileHead(response.headers.get("etag"), response.headers.get("content-type"),
        int(length) if length is not None and length.isdigit() else 0)


def _list_path(bucket: str | None, prefix: str | None, cursor: str | None,
    limit: int | None) -> str:
    params: dict[str, str] = {}
    if bucket is not None:
        params["bucket"] = bucket
    if prefix is not None:
        params["prefix"] = prefix
    if cursor is not None:
        params["cursor"] = cursor
    if limit is not None:
        params["limit"] = str(limit)
    return "/v1/files" + ("?" + urlencode(params) if params else "")


def _media_path(row_id: str) -> str:
    if not isinstance(row_id, str) or not row_id:
        raise TypeError("row_id must be a non-empty string")
    return f"/v1/media/{quote(row_id, safe='')}/content"


def _json_page(response: _RawHttpResponse) -> dict[str, Any]:
    _status(response, 200)
    page: dict[str, Any] = json.loads(response.content)
    return page


def _path(key: str, bucket: str | None) -> str:
    if not isinstance(key, str) or (bucket is not None and not isinstance(bucket, str)):
        raise TypeError("file key and optional bucket must be strings")
    params = {"key": key}
    if bucket is not None:
        params["bucket"] = bucket
    return "/v1/files/content?" + urlencode(params)


_DEFAULT_CHUNK = 8 * 1024 * 1024
_RESUMABLE = {409, 422, 429, 502, 503, 504}


def _folder_body(key: str, bucket: str | None) -> bytes:
    if not isinstance(key, str) or (bucket is not None and not isinstance(bucket, str)):
        raise TypeError("folder key and optional bucket must be strings")
    body: dict[str, str] = {"key": key}
    if bucket is not None:
        body["bucket"] = bucket
    return json.dumps(body).encode()


def _upload_path(upload_id: str, suffix: str = "", offset: int | None = None) -> str:
    if not isinstance(upload_id, str) or not upload_id:
        raise TypeError("upload_id must be a non-empty string")
    query = "" if offset is None else "?" + urlencode({"offset": offset})
    return f"/v1/files/uploads/{quote(upload_id, safe='')}{suffix}{query}"


def _start_body(key: str, size: int, bucket: str | None, content_type: str,
    sha256: str | None) -> bytes:
    if not isinstance(key, str) or not isinstance(size, int) or size < 0:
        raise TypeError("key must be a string and size a non-negative int")
    body: dict[str, Any] = {"key": key, "size": size, "content_type": content_type}
    if bucket is not None:
        body["bucket"] = bucket
    if sha256 is not None:
        body["sha256"] = sha256
    return json.dumps(body).encode()


def _session(response: _RawHttpResponse, expected: int) -> UploadSession:
    _status(response, expected)
    return UploadSession.model_validate_json(response.content)


class _Source:
    """Random-access view over bytes or a seekable binary file."""
    def __init__(self, source: bytes | BinaryIO, size: int | None) -> None:
        self._bytes = source if isinstance(source, (bytes, bytearray)) else None
        self._file = None if self._bytes is not None else source
        if self._bytes is not None:
            self.size = len(self._bytes)
        elif size is not None:
            self.size = size
        else:
            self._file.seek(0, 2)  # type: ignore[union-attr]
            self.size = self._file.tell()  # type: ignore[union-attr]
        if size is not None and self.size != size:
            raise ValueError("size does not match the source length")

    def read(self, offset: int, length: int) -> bytes:
        if self._bytes is not None:
            return bytes(self._bytes[offset:offset + length])
        self._file.seek(offset)  # type: ignore[union-attr]
        data = self._file.read(length)  # type: ignore[union-attr]
        if len(data) != length:
            raise ValueError("source ended before the declared size")
        return data

    def sha256(self) -> str:
        digest, offset = hashlib.sha256(), 0
        while offset < self.size:
            piece = self.read(offset, min(1024 * 1024, self.size - offset))
            digest.update(piece)
            offset += len(piece)
        return digest.hexdigest()


def _headers(tenant: str | None, extra: dict[str, str] | None) -> dict[str, str]:
    headers = dict(extra or {})
    if tenant is not None:
        headers = {k: v for k, v in headers.items() if k.lower() != "x-relata-tenant-id"}
        headers["X-Relata-Tenant-Id"] = tenant
    return headers


def _status(response: _RawHttpResponse, expected: int) -> None:
    if response.status_code != expected:
        raise RelataError(
            f"Unexpected Files response status: expected {expected}, got {response.status_code}",
            status_code=response.status_code,
            code="unexpected-response-status",
            request_id=response.headers.get("x-request-id"),
        )


def _meta201(response: _RawHttpResponse) -> FileMetadata:
    _status(response, 201)
    return FileMetadata.model_validate_json(response.content)


def _content(response: _RawHttpResponse) -> FileContent:
    _status(response, 200)
    return FileContent(
        response.content, response.headers.get("etag"), response.headers.get("content-type")
    )


class FilesClient:
    """Sync Files content client; ``get`` is bounded by ``max_response_bytes``.

    An explicit ``tenant`` overrides differently cased tenant extra headers.
    ``from_client`` borrows the parent's exact transport/configuration; closing
    this borrowed view leaves the parent open. Keep the parent alive while used.
    """
    def __init__(self, base_url: str, *, bearer_token: str | None = None,
        tenant: str | None = None, timeout: float = 30.0,
        extra_headers: dict[str, str] | None = None,
        max_retries: int = 0, retry_backoff: float = 0.5,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        transport: httpx.BaseTransport | None = None) -> None:
        self._t = HttpTransport(base_url, bearer_token, timeout, transport=transport,
            extra_headers=_headers(tenant, extra_headers), max_retries=max_retries,
            retry_backoff=retry_backoff, max_response_bytes=max_response_bytes)
        self._owns_transport = True

    @classmethod
    def from_client(cls, client: RelataClient) -> FilesClient:
        result = cls.__new__(cls)
        result._t = client._sync  # noqa: SLF001
        result._owns_transport = False
        return result

    def put(self, key: str, data: bytes, *, bucket: str | None = None,
        content_type: str = "application/octet-stream") -> FileMetadata:
        """Upload bytes once; return PUT201 metadata, or a typed HTTP failure."""
        if not isinstance(data, bytes):
            raise TypeError("file content must be bytes")
        response = self._t.request_bytes("PUT", _path(key, bucket), content=data,
            content_type=content_type, accept="application/json")
        _status(response, 201)
        return FileMetadata.model_validate_json(response.content)

    def get(self, key: str, *, bucket: str | None = None) -> FileContent:
        """Read exact bytes; missing/forbidden content raises the typed HTTP error."""
        return _content(self._t.request_bytes("GET", _path(key, bucket)))

    def head(self, key: str, *, bucket: str | None = None) -> FileHead | None:
        """Existence/metadata check (``HEAD /v1/files/content``); ``None`` on 404."""
        try:
            return _head(self._t.request_bytes("HEAD", _path(key, bucket)))
        except NotFoundError:
            return None

    def list(self, *, bucket: str | None = None, prefix: str | None = None,
        cursor: str | None = None, limit: int | None = None) -> dict[str, Any]:
        """List objects (``GET /v1/files``), optionally under ``prefix``; paginate with
        ``cursor``/``limit``. Returns the server's JSON page."""
        return _json_page(self._t.request_bytes("GET", _list_path(bucket, prefix, cursor, limit),
            accept="application/json"))

    def media_content(self, row_id: str) -> FileContent:
        """Offloaded bytes of a stored media row (``GET /v1/media/{row_id}/content``)."""
        return _content(self._t.request_bytes("GET", _media_path(row_id)))

    def delete(self, key: str, *, bucket: str | None = None) -> None:
        """Acknowledge DELETE204; no claim of physical/global copy destruction."""
        _status(self._t.request_bytes("DELETE", _path(key, bucket), accept="application/json"), 204)


    def create_folder(self, key: str, *, bucket: str | None = None) -> FileMetadata:
        """Create a folder marker (``POST /v1/files/folders``); returns 201 metadata
        with ``is_folder`` true."""
        return _meta201(self._t.request_bytes("POST", "/v1/files/folders",
            content=_folder_body(key, bucket), content_type="application/json",
            accept="application/json"))

    def start_upload(self, key: str, size: int, *, bucket: str | None = None,
        content_type: str = "application/octet-stream",
        sha256: str | None = None) -> UploadSession:
        """Open a resumable upload (``size`` required); nothing is visible until completed."""
        return _session(self._t.request_bytes("POST", "/v1/files/uploads",
            content=_start_body(key, size, bucket, content_type, sha256),
            content_type="application/json", accept="application/json"), 201)

    def upload_chunk(self, upload_id: str, offset: int, data: bytes, *,
        sha256: str | None = None) -> UploadSession:
        """Append ``data`` at ``offset``; a wrong offset is a typed 409 (read ``upload_status``)."""
        if not isinstance(data, bytes):
            raise TypeError("chunk must be bytes")
        digest = sha256 if sha256 is not None else hashlib.sha256(data).hexdigest()
        return _session(self._t.request_bytes("PUT", _upload_path(upload_id, offset=offset),
            content=data, accept="application/json",
            extra_headers={"X-Relata-Chunk-Sha256": digest}), 200)

    def upload_status(self, upload_id: str) -> UploadSession:
        """Read the durable next-expected offset of an upload (survives server restarts)."""
        return _session(self._t.request_bytes("GET", _upload_path(upload_id),
            accept="application/json"), 200)

    def complete_upload(self, upload_id: str) -> FileMetadata:
        """Verify and commit the staged bytes; never replayed automatically."""
        response = self._t.request_bytes("POST", _upload_path(upload_id, "/complete"),
            accept="application/json")
        _status(response, 201)
        return FileMetadata.model_validate_json(response.content)

    def abort_upload(self, upload_id: str) -> None:
        """Discard an upload and its staged data (idempotent)."""
        _status(self._t.request_bytes("DELETE", _upload_path(upload_id),
            accept="application/json"), 204)

    def upload(self, key: str, source: bytes | BinaryIO, *, size: int | None = None,
        bucket: str | None = None, content_type: str = "application/octet-stream",
        chunk_size: int = _DEFAULT_CHUNK, max_attempts: int = 5,
        upload_id: str | None = None, verify_digest: bool = True) -> FileMetadata:
        """Upload ``source`` (bytes or a seekable binary file) in resumable chunks.

        Memory is one chunk. Each chunk is digest-checked and, after a retryable
        failure (409/422/429/5xx), the loop re-reads the server offset and resumes
        from there, up to ``max_attempts`` consecutive failures; the session is then
        left in place (pass its ``upload_id`` back to continue). ``verify_digest``
        sends the whole-file SHA-256 so the server rejects corrupt content.
        """
        src = _Source(source, size)
        session = (self.upload_status(upload_id) if upload_id is not None else
            self.start_upload(key, src.size, bucket=bucket, content_type=content_type,
                sha256=src.sha256() if verify_digest else None))
        step = min(chunk_size, session.max_chunk_bytes or chunk_size)
        failures = 0
        while session.offset < src.size:
            piece = src.read(session.offset, min(step, src.size - session.offset))
            try:
                session = self.upload_chunk(session.upload_id, session.offset, piece)
                failures = 0
            except RelataError as exc:
                failures += 1
                if exc.status_code not in _RESUMABLE or failures >= max_attempts:
                    raise
                session = self.upload_status(session.upload_id)
        return self.complete_upload(session.upload_id)

    def close(self) -> None:
        if self._owns_transport:
            self._t.close()

    def __enter__(self) -> FilesClient:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class AsyncFilesClient:
    """Async twin of FilesClient, including borrowed-parent close semantics."""
    def __init__(self, base_url: str, *, bearer_token: str | None = None,
        tenant: str | None = None, timeout: float = 30.0,
        extra_headers: dict[str, str] | None = None,
        max_retries: int = 0, retry_backoff: float = 0.5,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._t = AsyncHttpTransport(base_url, bearer_token, timeout, transport=transport,
            extra_headers=_headers(tenant, extra_headers), max_retries=max_retries,
            retry_backoff=retry_backoff, max_response_bytes=max_response_bytes)
        self._owns_transport = True

    @classmethod
    def from_client(cls, client: RelataClient) -> AsyncFilesClient:
        result = cls.__new__(cls)
        result._t = client._async  # noqa: SLF001
        result._owns_transport = False
        return result

    async def put(self, key: str, data: bytes, *, bucket: str | None = None,
        content_type: str = "application/octet-stream") -> FileMetadata:
        if not isinstance(data, bytes):
            raise TypeError("file content must be bytes")
        response = await self._t.request_bytes("PUT", _path(key, bucket), content=data,
            content_type=content_type, accept="application/json")
        _status(response, 201)
        return FileMetadata.model_validate_json(response.content)

    async def get(self, key: str, *, bucket: str | None = None) -> FileContent:
        return _content(await self._t.request_bytes("GET", _path(key, bucket)))

    async def head(self, key: str, *, bucket: str | None = None) -> FileHead | None:
        """Async variant of :meth:`FilesClient.head`."""
        try:
            return _head(await self._t.request_bytes("HEAD", _path(key, bucket)))
        except NotFoundError:
            return None

    async def list(self, *, bucket: str | None = None, prefix: str | None = None,
        cursor: str | None = None, limit: int | None = None) -> dict[str, Any]:
        """Async variant of :meth:`FilesClient.list`."""
        return _json_page(await self._t.request_bytes("GET",
            _list_path(bucket, prefix, cursor, limit), accept="application/json"))

    async def media_content(self, row_id: str) -> FileContent:
        """Async variant of :meth:`FilesClient.media_content`."""
        return _content(await self._t.request_bytes("GET", _media_path(row_id)))

    async def delete(self, key: str, *, bucket: str | None = None) -> None:
        _status(
            await self._t.request_bytes("DELETE", _path(key, bucket), accept="application/json"),
            204,
        )

    async def create_folder(self, key: str, *, bucket: str | None = None) -> FileMetadata:
        """Create a folder marker (async) -- see :meth:`FilesClient.create_folder`."""
        return _meta201(await self._t.request_bytes("POST", "/v1/files/folders",
            content=_folder_body(key, bucket), content_type="application/json",
            accept="application/json"))

    async def start_upload(self, key: str, size: int, *, bucket: str | None = None,
        content_type: str = "application/octet-stream",
        sha256: str | None = None) -> UploadSession:
        """Open a resumable upload (``size`` required); nothing is visible until completed."""
        return _session(await self._t.request_bytes("POST", "/v1/files/uploads",
            content=_start_body(key, size, bucket, content_type, sha256),
            content_type="application/json", accept="application/json"), 201)

    async def upload_chunk(self, upload_id: str, offset: int, data: bytes, *,
        sha256: str | None = None) -> UploadSession:
        """Append ``data`` at ``offset``; a wrong offset is a typed 409 (read ``upload_status``)."""
        if not isinstance(data, bytes):
            raise TypeError("chunk must be bytes")
        digest = sha256 if sha256 is not None else hashlib.sha256(data).hexdigest()
        return _session(await self._t.request_bytes("PUT", _upload_path(upload_id, offset=offset),
            content=data, accept="application/json",
            extra_headers={"X-Relata-Chunk-Sha256": digest}), 200)

    async def upload_status(self, upload_id: str) -> UploadSession:
        """Read the durable next-expected offset of an upload (survives server restarts)."""
        return _session(await self._t.request_bytes("GET", _upload_path(upload_id),
            accept="application/json"), 200)

    async def complete_upload(self, upload_id: str) -> FileMetadata:
        """Verify and commit the staged bytes; never replayed automatically."""
        response = await self._t.request_bytes("POST", _upload_path(upload_id, "/complete"),
            accept="application/json")
        _status(response, 201)
        return FileMetadata.model_validate_json(response.content)

    async def abort_upload(self, upload_id: str) -> None:
        """Discard an upload and its staged data (idempotent)."""
        _status(await self._t.request_bytes("DELETE", _upload_path(upload_id),
            accept="application/json"), 204)

    async def upload(self, key: str, source: bytes | BinaryIO, *, size: int | None = None,
        bucket: str | None = None, content_type: str = "application/octet-stream",
        chunk_size: int = _DEFAULT_CHUNK, max_attempts: int = 5,
        upload_id: str | None = None, verify_digest: bool = True) -> FileMetadata:
        """Upload ``source`` (bytes or a seekable binary file) in resumable chunks.

        Memory is one chunk. Each chunk is digest-checked and, after a retryable
        failure (409/422/429/5xx), the loop re-reads the server offset and resumes
        from there, up to ``max_attempts`` consecutive failures; the session is then
        left in place (pass its ``upload_id`` back to continue). ``verify_digest``
        sends the whole-file SHA-256 so the server rejects corrupt content.
        """
        src = _Source(source, size)
        session = (await self.upload_status(upload_id) if upload_id is not None else
            await self.start_upload(key, src.size, bucket=bucket, content_type=content_type,
                sha256=src.sha256() if verify_digest else None))
        step = min(chunk_size, session.max_chunk_bytes or chunk_size)
        failures = 0
        while session.offset < src.size:
            piece = src.read(session.offset, min(step, src.size - session.offset))
            try:
                session = await self.upload_chunk(session.upload_id, session.offset, piece)
                failures = 0
            except RelataError as exc:
                failures += 1
                if exc.status_code not in _RESUMABLE or failures >= max_attempts:
                    raise
                session = await self.upload_status(session.upload_id)
        return await self.complete_upload(session.upload_id)

    async def aclose(self) -> None:
        if self._owns_transport:
            await self._t.aclose()

    async def __aenter__(self) -> AsyncFilesClient:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.aclose()
