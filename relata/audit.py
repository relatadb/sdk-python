"""Audit + provenance SDK (#78).

Wraps ``/audit/count``, ``/audit/entries``, ``/report/sign``, ``/report/pdf``.
The ``prov_export`` helper is a thin wrapper over the server's ``prov export``
CLI shape; until ADR-117 lands (#59) the server-side ``prov export`` is a stub
and the SDK call returns a clear error.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

from relata._http import AsyncHttpTransport, HttpTransport
from relata.exceptions import RelataError

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
            params["since_ns"] = str(since_ns)
        if until_ns is not None:
            params["until_ns"] = str(until_ns)
        if cursor:
            params["cursor"] = cursor
        return self._t.get("/audit/entries?" + urlencode(params))

    def find_by_request_id(self, request_id: str) -> dict[str, Any] | None:
        """Look up a single audit entry by its ``request_id``. Returns ``None``
        when no entry matches (the server returns an empty page)."""
        page = self.entries(request_id=request_id, limit=1)
        entries = page.get("entries") if isinstance(page, dict) else None
        if entries and isinstance(entries, list) and entries:
            return dict(entries[0])
        return None

    # ------------------------------------------------------------------
    # Reports (#78)
    # ------------------------------------------------------------------

    def sign_receipt(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Sign an Art. 17 / evidence receipt. Returns the signed receipt
        with the server's HMAC signature. Pairs with #61 — until #61 ships
        receipt-key rotation safety, callers should treat the signature as
        transient."""
        return self._t.post("/report/sign", payload)

    def export_pdf(self, case_id: str, *, template: str = "default") -> bytes:
        """Export a court-grade PDF report for ``case_id``. Returns the raw
        PDF bytes; write to disk with ``Path("case.pdf").write_bytes(pdf)``.

        Note: the server returns styled HTML by default (printable); the PDF
        rendering is via the host browser's print-to-pdf. The SDK surfaces
        whatever bytes the server returns.
        """
        # #3214: stream the POST and read through the transport's cap so a
        # malicious/buggy PDF-sized response can't OOM the SDK — the previous
        # `self._t._client.post(...).content` buffered the whole body unbounded.
        with self._t._client.stream(  # noqa: SLF001 — intentional private access
            "POST", "/report/pdf", json={"case_id": case_id, "template": template}
        ) as resp:
            if not resp.is_success:
                raw = self._t._read_capped(resp)  # noqa: SLF001
                try:
                    body = json.loads(raw)
                    raise RelataError(
                        body.get("error") or body.get("message") or "report/pdf failed",
                        status_code=resp.status_code,
                    )
                except ValueError as exc:
                    raise RelataError(
                        f"report/pdf failed (HTTP {resp.status_code})",
                        status_code=resp.status_code,
                    ) from exc
            return self._t._read_capped(resp)  # noqa: SLF001

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
            params["since_ns"] = str(since_ns)
        if until_ns is not None:
            params["until_ns"] = str(until_ns)
        if cursor:
            params["cursor"] = cursor
        return await self._t.get("/audit/entries?" + urlencode(params))

    async def find_by_request_id(self, request_id: str) -> dict[str, Any] | None:
        page = await self.entries(request_id=request_id, limit=1)
        entries = page.get("entries") if isinstance(page, dict) else None
        if entries and isinstance(entries, list) and entries:
            return dict(entries[0])
        return None

    async def sign_receipt(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._t.post("/report/sign", payload)

    async def export_pdf(self, case_id: str, *, template: str = "default") -> bytes:
        # #3214: stream + cap (async twin of the sync export_pdf).
        async with self._t._client.stream(  # noqa: SLF001
            "POST", "/report/pdf", json={"case_id": case_id, "template": template}
        ) as resp:
            if not resp.is_success:
                raw = await self._t._read_capped(resp)  # noqa: SLF001
                try:
                    body = json.loads(raw)
                    raise RelataError(
                        body.get("error") or body.get("message") or "report/pdf failed",
                        status_code=resp.status_code,
                    )
                except ValueError as exc:
                    raise RelataError(
                        f"report/pdf failed (HTTP {resp.status_code})",
                        status_code=resp.status_code,
                    ) from exc
            return await self._t._read_capped(resp)  # noqa: SLF001

    async def close(self) -> None:
        await self._t.aclose()

    async def __aenter__(self) -> AsyncAuditClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()
