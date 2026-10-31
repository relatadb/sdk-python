"""Audit + provenance SDK (#78).

Wraps ``/audit/count``, ``/audit/entries``, ``/audit/log``.
The ``prov_export`` helper is a thin wrapper over the server's ``prov export``
CLI shape; until ADR-117 lands (#59) the server-side ``prov export`` is a stub
and the SDK call returns a clear error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

from relata._http import AsyncHttpTransport, HttpTransport

if TYPE_CHECKING:
    import httpx

    from relata.client import RelataClient


class AuditClient:
    """Synchronous audit + provenance client."""

    def __init__(
        self,
        base_url: str,
        *,
        bearer_token: str | None = None,
        timeout: float = 30.0,
        extra_headers: dict[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._t = HttpTransport(
            base_url, bearer_token, timeout, transport=transport, extra_headers=extra_headers
        )

    @classmethod
    def from_client(cls, client: RelataClient) -> AuditClient:
        return cls(
            client._base_url,
            bearer_token=client._bearer_token,
            timeout=client._timeout,
            extra_headers=client._extra_headers,
        )

    # ------------------------------------------------------------------
    # Audit entries (#78)
    # ------------------------------------------------------------------

    def count(self) -> dict[str, Any]:
        """Total audit-entry count + chain integrity flag."""
        return self._t.get("/audit/count")

    def entries(
        self,
        *,
        principal: str | None = None,
        purpose: str | None = None,
        decision: str | None = None,
        request_id: str | None = None,
        since_ns: int | None = None,
        until_ns: int | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """Paginated audit-entry listing with filters. Returns
        ``{"entries": [...], "next_cursor": str | None, "chain_valid": bool}``."""
        params: dict[str, str] = {"limit": str(limit)}
        if principal:
            params["principal"] = principal
        if purpose:
            params["purpose"] = purpose
        if decision:
            params["decision"] = decision
        if request_id:
            params["request_id"] = request_id
        if since_ns is not None:
            params["since"] = str(since_ns)
        if until_ns is not None:
            params["until"] = str(until_ns)
        if cursor:
            params["cursor"] = cursor
        return self._t.get("/audit/entries?" + urlencode(params))

    def log(
        self,
        *,
        limit: int = 100,
        from_: str | None = None,
        to: str | None = None,
        table: str | None = None,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """Filtered mutation-style audit projections (``GET /audit/log``).

        ``from_``/``to`` are inclusive timestamp bounds, ``table`` a substring of
        the derived resource. ``tenant_id`` only applies to an unscoped admin
        caller; an attributed caller's own tenant always wins. Returns
        ``{"entries": [...], "chain_valid": bool}``.
        """
        params: dict[str, str] = {"limit": str(limit)}
        if from_:
            params["from"] = from_
        if to:
            params["to"] = to
        if table:
            params["table"] = table
        if tenant_id:
            params["tenant_id"] = tenant_id
        return self._t.get("/audit/log?" + urlencode(params))

    def find_by_request_id(self, request_id: str) -> dict[str, Any] | None:
        """Look up a single audit entry by its ``request_id``. Returns ``None``
        when no entry matches (the server returns an empty page)."""
        page = self.entries(request_id=request_id, limit=1)
        entries = page.get("entries") if isinstance(page, dict) else None
        if entries and isinstance(entries, list) and entries:
            entry = dict(entries[0])
            # Never hand back a different request's entry (e.g. an older
            # server that ignores the ``request_id`` filter returns the newest).
            if entry.get("request_id", request_id) != request_id:
                return None
            return entry
        return None

    # ------------------------------------------------------------------
    # Reports (#78)
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._t.close()

    def __enter__(self) -> AuditClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class AsyncAuditClient:
    """Asynchronous audit + provenance client — see :class:`AuditClient`."""

    def __init__(
        self,
        base_url: str,
        *,
        bearer_token: str | None = None,
        timeout: float = 30.0,
        extra_headers: dict[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._t = AsyncHttpTransport(
            base_url, bearer_token, timeout, transport=transport, extra_headers=extra_headers
        )

    @classmethod
    def from_client(cls, client: RelataClient) -> AsyncAuditClient:
        return cls(
            client._base_url,
            bearer_token=client._bearer_token,
            timeout=client._timeout,
            extra_headers=client._extra_headers,
        )

    async def count(self) -> dict[str, Any]:
        return await self._t.get("/audit/count")

    async def entries(
        self,
        *,
        principal: str | None = None,
        purpose: str | None = None,
        decision: str | None = None,
        request_id: str | None = None,
        since_ns: int | None = None,
        until_ns: int | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, str] = {"limit": str(limit)}
        if principal:
            params["principal"] = principal
        if purpose:
            params["purpose"] = purpose
        if decision:
            params["decision"] = decision
        if request_id:
            params["request_id"] = request_id
        if since_ns is not None:
            params["since"] = str(since_ns)
        if until_ns is not None:
            params["until"] = str(until_ns)
        if cursor:
            params["cursor"] = cursor
        return await self._t.get("/audit/entries?" + urlencode(params))

    async def log(
        self,
        *,
        limit: int = 100,
        from_: str | None = None,
        to: str | None = None,
        table: str | None = None,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """Filtered mutation-style audit projections (``GET /audit/log``).

        ``from_``/``to`` are inclusive timestamp bounds, ``table`` a substring of
        the derived resource. ``tenant_id`` only applies to an unscoped admin
        caller; an attributed caller's own tenant always wins. Returns
        ``{"entries": [...], "chain_valid": bool}``.
        """
        params: dict[str, str] = {"limit": str(limit)}
        if from_:
            params["from"] = from_
        if to:
            params["to"] = to
        if table:
            params["table"] = table
        if tenant_id:
            params["tenant_id"] = tenant_id
        return await self._t.get("/audit/log?" + urlencode(params))

    async def find_by_request_id(self, request_id: str) -> dict[str, Any] | None:
        page = await self.entries(request_id=request_id, limit=1)
        entries = page.get("entries") if isinstance(page, dict) else None
        if entries and isinstance(entries, list) and entries:
            entry = dict(entries[0])
            # Never hand back a different request's entry (e.g. an older
            # server that ignores the ``request_id`` filter returns the newest).
            if entry.get("request_id", request_id) != request_id:
                return None
            return entry
        return None

    async def close(self) -> None:
        await self._t.aclose()

    async def __aenter__(self) -> AsyncAuditClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()
