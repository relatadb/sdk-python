"""
Internal HTTP transport layer.

This module is **not** part of the public API.  All internals are subject to
change without notice.  Callers should use :class:`relata.client.RelataClient`
instead.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import httpx

from relata.exceptions import (
    AuthError,
    ConflictError,
    ConnectionError,
    ForbiddenError,
    NotFoundError,
    PurposeError,
    RateLimitedError,
    RelataError,
    ServerError,
    ValidationError,
)

# Request / response content-type we always send and expect.
_CONTENT_TYPE = "application/json"

# Server-side error messages that signal a missing/invalid purpose.
_PURPOSE_HINTS = frozenset(
    [
        "purpose",
        "purposeregistry",
        "purpose_registry",
        "missing purpose",
        "invalid purpose",
        "unregistered purpose",
    ]
)

# Default retry config (override via RelataClient constructor).
_DEFAULT_RETRY_ON: frozenset[int] = frozenset({502, 503, 504})
_DEFAULT_MAX_RETRIES = 0  # off by default — caller must opt in
_DEFAULT_RETRY_BACKOFF = 0.5


def _opt_float_header(headers: Any, name: str) -> float | None:
    """Parse a numeric response header to float, returning ``None`` when absent/unparseable."""
    raw = headers.get(name)
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _classify_error(
    status_code: int,
    body: dict[str, Any],
    *,
    request_id: str | None = None,
    retry_after: float | None = None,
    rate_limit_limit: float | None = None,
    rate_limit_remaining: float | None = None,
    rate_limit_reset: float | None = None,
) -> RelataError:
    """Map an HTTP error response to the most-specific SDK exception.

    Parses both the RFC 7807 ``application/problem+json`` shape (``code``,
    ``type``, ``detail``, ``retryable``, ``request_id``) and the legacy
    ``{"error": "..."}`` shape.
    """
    # RFC 7807 fields — present when the server emits problem+json.
    code: str | None = body.get("code")
    type_url: str | None = body.get("type")
    retryable: bool = bool(body.get("retryable", False))
    rid = request_id or body.get("request_id")
    detail: str = (
        body.get("detail") or body.get("error")
        or body.get("message") or "unknown server error"
    )
    lower_detail = detail.lower()

    # Shared kwargs for every subclass.
    base_kwargs: dict[str, Any] = {
        "code": code,
        "type_url": type_url,
        "retryable": retryable,
        "request_id": rid,
    }

    # Purpose-specific detection — works for both legacy and problem+json.
    is_purpose = status_code == 400 and (
        (code and "purpose" in code.lower())
        or any(hint in lower_detail for hint in _PURPOSE_HINTS)
    )
    if is_purpose:
        return PurposeError(
            f"Purpose rejected: {detail}. "
            "Pass purpose= to RelataClient.query() or set a default on the client.",
            status_code=status_code,
        )

    if status_code == 401:
        return AuthError(
            f"Authentication failed: {detail}. ",
            status_code=status_code,
            **base_kwargs,
        )
    if status_code == 403:
        return ForbiddenError(
            f"Forbidden: {detail}",
            status_code=status_code,
            **base_kwargs,
        )
    if status_code == 404:
        return NotFoundError(
            f"Not found: {detail}",
            status_code=status_code,
            **base_kwargs,
        )
    if status_code == 409:
        return ConflictError(
            f"Conflict: {detail}",
            status_code=status_code,
            **base_kwargs,
        )
    if status_code == 422:
        return ValidationError(
            f"Validation error: {detail}",
            status_code=status_code,
            **base_kwargs,
        )
    if status_code == 429:
        return RateLimitedError(
            f"Rate limited: {detail}",
            status_code=status_code,
            retry_after=retry_after,
            rate_limit_limit=rate_limit_limit,
            rate_limit_remaining=rate_limit_remaining,
            rate_limit_reset=rate_limit_reset,
            **base_kwargs,
        )
    if status_code >= 500:
        return ServerError(
            f"Server error: {detail}",
            status_code=status_code,
            **base_kwargs,
        )
    return RelataError(detail, status_code=status_code, **base_kwargs)


def _build_headers(
    bearer_token: str | None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    headers: dict[str, str] = {
        "Content-Type": _CONTENT_TYPE,
        "Accept": _CONTENT_TYPE,
        "User-Agent": "relata-sdk-python/0.1.0",
    }
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    if extra:
        # Caller-supplied headers win over the SDK defaults so tenants,
        # acting-as, request-id, etc. can be pinned per-request.
        headers.update(extra)
    return headers


# ---------------------------------------------------------------------------
# Sync transport
# ---------------------------------------------------------------------------


class HttpTransport:
    """Synchronous HTTP transport backed by :mod:`httpx`."""

    def __init__(
        self,
        base_url: str,
        bearer_token: str | None,
        timeout: float,
        transport: httpx.BaseTransport | None = None,
        extra_headers: dict[str, str] | None = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        retry_backoff: float = _DEFAULT_RETRY_BACKOFF,
        retry_on: frozenset[int] = _DEFAULT_RETRY_ON,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=_build_headers(bearer_token, extra_headers),
            timeout=timeout,
            transport=transport,
        )
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._retry_on = retry_on

    def _request_id(self) -> str:
        return str(uuid.uuid4())

    def _send_with_retry(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
        content: str | bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Send an HTTP request with retry + request_id propagation."""
        # If the caller already set X-Request-ID in the default headers bag,
        # respect it — only generate a per-attempt UUID when no default exists.
        has_default_rid = "x-request-id" in {k.lower() for k in self._client.headers}
        last_exc: RelataError | None = None
        for attempt in range(self._max_retries + 1):
            req_headers: dict[str, str] = {}
            if not has_default_rid:
                req_headers["X-Request-ID"] = self._request_id()
            if headers:
                req_headers.update(headers)
            try:
                if method == "GET":
                    resp = self._client.get(path, headers=req_headers)
                elif method == "DELETE":
                    resp = self._client.delete(path, headers=req_headers)
                elif json_payload is not None:
                    resp = self._client.request(
                        method, path, json=json_payload, headers=req_headers
                    )
                elif content is not None:
                    resp = self._client.request(method, path, content=content, headers=req_headers)
                else:
                    resp = self._client.request(method, path, headers=req_headers)
            except httpx.ConnectError as exc:
                last_exc = ConnectionError(
                    f"Cannot connect to Relata server: {exc}. "
                    "Check that the server is running and the base_url is correct."
                )
                if attempt < self._max_retries:
                    time.sleep(self._retry_backoff * (2**attempt))
                    continue
                raise last_exc from exc
            except httpx.TimeoutException as exc:
                last_exc = ConnectionError(
                    f"Request timed out: {exc}. "
                    "Increase timeout= on RelataClient or check server health."
                )
                if attempt < self._max_retries:
                    time.sleep(self._retry_backoff * (2**attempt))
                    continue
                raise last_exc from exc

            # Success or non-retryable error — return / raise via _handle.
            return self._handle(resp)

        # Exhausted retries on a connection error.
        assert last_exc is not None
        raise last_exc

    def get(self, path: str) -> dict[str, Any]:
        """Perform a GET request and return the decoded JSON body."""
        return self._send_with_retry("GET", path)

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Perform a POST request with a JSON body and return the decoded JSON body."""
        return self._send_with_retry("POST", path, json_payload=payload)

    def delete(self, path: str) -> dict[str, Any]:
        """Perform a DELETE request and return the decoded JSON body."""
        return self._send_with_retry("DELETE", path)

    def patch(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Perform a PATCH request with an optional JSON body."""
        return self._send_with_retry("PATCH", path, json_payload=payload or {})

    def put(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Perform a PUT request with an optional JSON body."""
        return self._send_with_retry("PUT", path, json_payload=payload or {})

    def close(self) -> None:
        self._client.close()

    @staticmethod
    def _handle(resp: httpx.Response) -> dict[str, Any]:
        if resp.is_success:
            return resp.json()  # type: ignore[no-any-return]
        try:
            body: dict[str, Any] = resp.json()
        except Exception:
            body = {"error": resp.text or "empty response"}
        request_id = resp.headers.get("x-request-id")
        retry_after_hdr = resp.headers.get("retry-after")
        retry_after = float(retry_after_hdr) if retry_after_hdr else None
        rl_limit = _opt_float_header(resp.headers, "x-ratelimit-limit")
        rl_remaining = _opt_float_header(resp.headers, "x-ratelimit-remaining")
        rl_reset = _opt_float_header(resp.headers, "x-ratelimit-reset")
        raise _classify_error(
            resp.status_code,
            body,
            request_id=request_id,
            retry_after=retry_after,
            rate_limit_limit=rl_limit,
            rate_limit_remaining=rl_remaining,
            rate_limit_reset=rl_reset,
        )


# ---------------------------------------------------------------------------
# Async transport
# ---------------------------------------------------------------------------


class AsyncHttpTransport:
    """Asynchronous HTTP transport backed by :mod:`httpx`."""

    def __init__(
        self,
        base_url: str,
        bearer_token: str | None,
        timeout: float,
        transport: httpx.AsyncBaseTransport | None = None,
        extra_headers: dict[str, str] | None = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        retry_backoff: float = _DEFAULT_RETRY_BACKOFF,
        retry_on: frozenset[int] = _DEFAULT_RETRY_ON,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=_build_headers(bearer_token, extra_headers),
            timeout=timeout,
            transport=transport,
        )
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._retry_on = retry_on

    def _request_id(self) -> str:
        return str(uuid.uuid4())

    async def _send_with_retry(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send an async HTTP request with retry + request_id propagation."""
        import asyncio

        last_exc: RelataError | None = None
        for attempt in range(self._max_retries + 1):
            req_headers = {"X-Request-ID": self._request_id()}
            try:
                if method == "GET":
                    resp = await self._client.get(path, headers=req_headers)
                elif method == "DELETE":
                    resp = await self._client.delete(path, headers=req_headers)
                else:
                    resp = await self._client.request(
                        method, path, json=json_payload or {}, headers=req_headers
                    )
            except httpx.ConnectError as exc:
                last_exc = ConnectionError(
                    f"Cannot connect to Relata server: {exc}."
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(self._retry_backoff * (2**attempt))
                    continue
                raise last_exc from exc
            except httpx.TimeoutException as exc:
                last_exc = ConnectionError(
                    f"Request timed out: {exc}."
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(self._retry_backoff * (2**attempt))
                    continue
                raise last_exc from exc
            return self._handle(resp)
        assert last_exc is not None
        raise last_exc

    async def get(self, path: str) -> dict[str, Any]:
        """Perform an async GET request and return the decoded JSON body."""
        return await self._send_with_retry("GET", path)

    async def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Perform an async POST request with a JSON body."""
        return await self._send_with_retry("POST", path, json_payload=payload)

    async def delete(self, path: str) -> dict[str, Any]:
        """Perform an async DELETE request."""
        return await self._send_with_retry("DELETE", path)

    async def patch(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Perform an async PATCH request with an optional JSON body."""
        return await self._send_with_retry("PATCH", path, json_payload=payload or {})

    async def put(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Perform an async PUT request with an optional JSON body."""
        return await self._send_with_retry("PUT", path, json_payload=payload or {})

    async def aclose(self) -> None:
        await self._client.aclose()

    @staticmethod
    def _handle(resp: httpx.Response) -> dict[str, Any]:
        if resp.is_success:
            return resp.json()  # type: ignore[no-any-return]
        try:
            body: dict[str, Any] = resp.json()
        except Exception:
            body = {"error": resp.text or "empty response"}
        request_id = resp.headers.get("x-request-id")
        retry_after_hdr = resp.headers.get("retry-after")
        retry_after = float(retry_after_hdr) if retry_after_hdr else None
        rl_limit = _opt_float_header(resp.headers, "x-ratelimit-limit")
        rl_remaining = _opt_float_header(resp.headers, "x-ratelimit-remaining")
        rl_reset = _opt_float_header(resp.headers, "x-ratelimit-reset")
        raise _classify_error(
            resp.status_code,
            body,
            request_id=request_id,
            retry_after=retry_after,
            rate_limit_limit=rl_limit,
            rate_limit_remaining=rl_remaining,
            rate_limit_reset=rl_reset,
        )
