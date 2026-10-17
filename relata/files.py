"""Buffered binary Files content through the existing governed HTTP transport.

Only PUT/GET/DELETE ``/files/content`` are wrapped. Range requests, conditional
writes, versions, provider-copy completion and physical deletion are not promised.
PUT and DELETE are never automatically replayed, even with ``max_retries > 0``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

import httpx
from pydantic import BaseModel, ConfigDict

from relata._http import DEFAULT_MAX_RESPONSE_BYTES, AsyncHttpTransport, HttpTransport, _RawHttpResponse
from relata.exceptions import RelataError

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


@dataclass(frozen=True)
class FileContent:
    """Exact buffered bytes, response ETag/type and actual buffered size."""
    data: bytes
    etag: str | None
    content_type: str | None

    @property
    def size(self) -> int:
        return len(self.data)


def _path(key: str, bucket: str | None) -> str:
    if not isinstance(key, str) or (bucket is not None and not isinstance(bucket, str)):
        raise TypeError("file key and optional bucket must be strings")
    params = {"key": key}
    if bucket is not None:
        params["bucket"] = bucket
    return "/files/content?" + urlencode(params)


def _headers(tenant: str | None, extra: dict[str, str] | None) -> dict[str, str]:
    headers = dict(extra or {})
    if tenant is not None:
        headers = {k: v for k, v in headers.items() if k.lower() != "x-relata-tenant-id"}
        headers["X-Relata-Tenant-Id"] = tenant
    return headers


def _status(response: _RawHttpResponse, expected: int) -> None:
    if response.status_code != expected:
        raise RelataError(f"Unexpected Files response status: expected {expected}, got {response.status_code}",
            status_code=response.status_code, code="unexpected-response-status",
            request_id=response.headers.get("x-request-id"))


def _content(response: _RawHttpResponse) -> FileContent:
    _status(response, 200)
    return FileContent(response.content, response.headers.get("etag"), response.headers.get("content-type"))


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

    def delete(self, key: str, *, bucket: str | None = None) -> None:
        """Acknowledge DELETE204; no claim of physical/global copy destruction."""
        _status(self._t.request_bytes("DELETE", _path(key, bucket), accept="application/json"), 204)

    def close(self) -> None:
        if self._owns_transport:
            self._t.close()

    def __enter__(self) -> FilesClient:
        return self

    def __exit__(self, *_args: Any) -> None:
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

    async def delete(self, key: str, *, bucket: str | None = None) -> None:
        _status(await self._t.request_bytes("DELETE", _path(key, bucket), accept="application/json"), 204)

    async def aclose(self) -> None:
        if self._owns_transport:
            await self._t.aclose()

    async def __aenter__(self) -> AsyncFilesClient:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        await self.aclose()
