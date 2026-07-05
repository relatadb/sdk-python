"""
Internal HTTP transport layer.

This module is **not** part of the public API.  All internals are subject to
change without notice.  Callers should use :class:`relata.client.RelataClient`
instead.
"""

from __future__ import annotations

from typing import Any

import httpx

from relata.exceptions import (
    AuthError,
    ConnectionError,
    PurposeError,
    QuotaError,
    RelataError,
    ServerError,
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


def _classify_error(status_code: int, body: dict[str, Any]) -> RelataError:
    """Map an HTTP error response to the most-specific SDK exception."""
    raw_message: str = body.get("error") or body.get("message") or "unknown server error"
    lower = raw_message.lower()

    if status_code == 401:
        return AuthError(
            f"Authentication failed: {raw_message}. "
            "Pass bearer_token= to RelataClient or set RELATA_BEARER_TOKEN on the server.",
            status_code=status_code,
        )
    if status_code == 429:
        return QuotaError(
            f"Query cost quota exhausted: {raw_message}. "
            "Reduce query scope (LIMIT / WHERE) or request a quota increase.",
            status_code=status_code,
        )
    if status_code == 400 and any(hint in lower for hint in _PURPOSE_HINTS):
        return PurposeError(
            f"Purpose rejected: {raw_message}. "
            "Every Relata query must declare a purpose registered in the tenant's "
            "PurposeRegistry (e.g. 'analytics', 'audit', 'analysis'). "
            "Pass purpose= to RelataClient.query() or set a default on the client.",
            status_code=status_code,
        )
    if status_code >= 500:
        return ServerError(
            f"Server error: {raw_message}",
            status_code=status_code,
        )
    return RelataError(raw_message, status_code=status_code)


def _build_headers(bearer_token: str | None) -> dict[str, str]:
    headers: dict[str, str] = {
        "Content-Type": _CONTENT_TYPE,
        "Accept": _CONTENT_TYPE,
        "User-Agent": "relata-sdk-python/0.1.0",
    }
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
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
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=_build_headers(bearer_token),
            timeout=timeout,
            transport=transport,
        )

    def get(self, path: str) -> dict[str, Any]:
        """Perform a GET request and return the decoded JSON body."""
        try:
            resp = self._client.get(path)
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"Cannot connect to Relata server: {exc}. "
                "Check that the server is running and the base_url is correct."
            ) from exc
        except httpx.TimeoutException as exc:
            raise ConnectionError(
                f"Request timed out: {exc}. "
                "Increase timeout= on RelataClient or check server health."
            ) from exc
        return self._handle(resp)

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Perform a POST request with a JSON body and return the decoded JSON body."""
        try:
            resp = self._client.post(path, json=payload)
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"Cannot connect to Relata server: {exc}. "
                "Check that the server is running and the base_url is correct."
            ) from exc
        except httpx.TimeoutException as exc:
            raise ConnectionError(
                f"Request timed out: {exc}. "
                "Increase timeout= on RelataClient or check server health."
            ) from exc
        return self._handle(resp)

    def delete(self, path: str) -> dict[str, Any]:
        """Perform a DELETE request and return the decoded JSON body."""
        try:
            resp = self._client.delete(path)
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"Cannot connect to Relata server: {exc}. "
                "Check that the server is running and the base_url is correct."
            ) from exc
        except httpx.TimeoutException as exc:
            raise ConnectionError(
                f"Request timed out: {exc}. "
                "Increase timeout= on RelataClient or check server health."
            ) from exc
        return self._handle(resp)

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
        raise _classify_error(resp.status_code, body)


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
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=_build_headers(bearer_token),
            timeout=timeout,
            transport=transport,
        )

    async def get(self, path: str) -> dict[str, Any]:
        """Perform an async GET request and return the decoded JSON body."""
        try:
            resp = await self._client.get(path)
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"Cannot connect to Relata server: {exc}. "
                "Check that the server is running and the base_url is correct."
            ) from exc
        except httpx.TimeoutException as exc:
            raise ConnectionError(
                f"Request timed out: {exc}. "
                "Increase timeout= on RelataClient or check server health."
            ) from exc
        return self._handle(resp)

    async def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Perform an async POST request with a JSON body and return the decoded JSON body."""
        try:
            resp = await self._client.post(path, json=payload)
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"Cannot connect to Relata server: {exc}. "
                "Check that the server is running and the base_url is correct."
            ) from exc
        except httpx.TimeoutException as exc:
            raise ConnectionError(
                f"Request timed out: {exc}. "
                "Increase timeout= on RelataClient or check server health."
            ) from exc
        return self._handle(resp)

    async def delete(self, path: str) -> dict[str, Any]:
        """Perform an async DELETE request and return the decoded JSON body."""
        try:
            resp = await self._client.delete(path)
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"Cannot connect to Relata server: {exc}. "
                "Check that the server is running and the base_url is correct."
            ) from exc
        except httpx.TimeoutException as exc:
            raise ConnectionError(
                f"Request timed out: {exc}. "
                "Increase timeout= on RelataClient or check server health."
            ) from exc
        return self._handle(resp)

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
        raise _classify_error(resp.status_code, body)
