"""Audit + provenance SDK (#78).

Wraps ``GET /v1/audit/count`` and ``GET /v1/audit/log``.
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
    # Audit count + log (#78)
    # ------------------------------------------------------------------

    def count(self) -> dict[str, Any]:
        """Total audit-entry count + chain integrity flag."""
        return self._t.get("/v1/audit/count")

    def log(
        self,
        *,
        limit: int = 100,
        from_: str | None = None,
        to: str | None = None,
        table: str | None = None,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """Filtered mutation-style audit projections (``GET /v1/audit/log``).

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
        return self._t.get("/v1/audit/log?" + urlencode(params))

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
        return await self._t.get("/v1/audit/count")

    async def log(
        self,
        *,
        limit: int = 100,
        from_: str | None = None,
        to: str | None = None,
        table: str | None = None,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """Filtered mutation-style audit projections (``GET /v1/audit/log``).

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
        return await self._t.get("/v1/audit/log?" + urlencode(params))

    async def close(self) -> None:
        await self._t.aclose()

    async def __aenter__(self) -> AsyncAuditClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()
