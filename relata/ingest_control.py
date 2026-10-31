"""Ingestion control plane: source checkpoints, data contracts, quarantine (#5689, #5714).

Wraps ``GET /ingest/checkpoints/{source}``, ``GET /ingest/contracts``,
``GET|PUT /ingest/contracts/{dataset}``, ``GET /ingest/contracts/{dataset}/quality``,
``GET /ingest/quarantine`` and ``DELETE /ingest/quarantine/{id}``. These routes use
the data-plane credential and are scoped to the caller's tenant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

from relata._http import AsyncHttpTransport, HttpTransport, path_segment
from relata.exceptions import RelataError

if TYPE_CHECKING:
    import httpx

    from relata.client import RelataClient


def _quarantine_path(dataset: str | None, after_id: int | None, limit: int | None) -> str:
    params: dict[str, str] = {}
    if dataset is not None:
        params["dataset"] = dataset
    if after_id is not None:
        params["after_id"] = str(int(after_id))
    if limit is not None:
        params["limit"] = str(int(limit))
    return "/ingest/quarantine" + (f"?{urlencode(params)}" if params else "")


def _quarantine_item(quarantine_id: int) -> str:
    return f"/ingest/quarantine/{int(quarantine_id)}"


def _expect_204(status_code: int, request_id: str | None) -> None:
    if status_code != 204:
        raise RelataError(
            f"Unexpected quarantine delete status: expected 204, got {status_code}",
            status_code=status_code,
            code="unexpected-response-status",
            request_id=request_id,
        )


class IngestControlClient:
    """Synchronous ingestion control-plane client."""

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
    def from_client(cls, client: RelataClient) -> IngestControlClient:
        return cls(
            client._base_url,  # noqa: SLF001
            bearer_token=client._bearer_token,  # noqa: SLF001
            timeout=client._timeout,  # noqa: SLF001
            extra_headers=client._extra_headers,  # noqa: SLF001
        )

    def checkpoint(self, source: str) -> dict[str, Any]:
        """``GET /ingest/checkpoints/{source}``: last durably committed checkpoint."""
        return dict(self._t.get(f"/ingest/checkpoints/{path_segment(source)}"))

    def list_contracts(self) -> dict[str, Any]:
        """``GET /ingest/contracts``: ``{"contracts": [{"dataset", "latest_version"}]}``."""
        return dict(self._t.get("/ingest/contracts"))

    def get_contract(self, dataset: str) -> dict[str, Any]:
        """``GET /ingest/contracts/{dataset}``: every retained contract version."""
        return dict(self._t.get(f"/ingest/contracts/{path_segment(dataset)}"))

    def put_contract(self, dataset: str, contract: dict[str, Any]) -> dict[str, Any]:
        """``PUT /ingest/contracts/{dataset}``: define the next contract version.

        The server checks backward compatibility; ``contract`` is sent verbatim.
        Returns ``{"dataset", "version", "status": "created"}``.
        """
        return dict(self._t.put(f"/ingest/contracts/{path_segment(dataset)}", contract))

    def contract_quality(self, dataset: str) -> dict[str, Any]:
        """``GET /ingest/contracts/{dataset}/quality``: validation counts, freshness, drift."""
        return dict(self._t.get(f"/ingest/contracts/{path_segment(dataset)}/quality"))

    def list_quarantine(
        self, *, dataset: str | None = None, after_id: int | None = None, limit: int | None = None
    ) -> dict[str, Any]:
        """``GET /ingest/quarantine``: a page of quarantined records.

        Pass the previous page's ``next_after_id`` as ``after_id`` to continue.
        """
        return dict(self._t.get(_quarantine_path(dataset, after_id, limit)))

    def delete_quarantine(self, quarantine_id: int) -> None:
        """``DELETE /ingest/quarantine/{id}``: discard one record (404 when unknown)."""
        resp = self._t.request_bytes(
            "DELETE", _quarantine_item(quarantine_id), accept="application/json"
        )
        _expect_204(resp.status_code, resp.headers.get("x-request-id"))

    def close(self) -> None:
        self._t.close()

    def __enter__(self) -> IngestControlClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class AsyncIngestControlClient:
    """Asynchronous ingestion control-plane client -- see :class:`IngestControlClient`."""

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
    def from_client(cls, client: RelataClient) -> AsyncIngestControlClient:
        return cls(
            client._base_url,  # noqa: SLF001
            bearer_token=client._bearer_token,  # noqa: SLF001
            timeout=client._timeout,  # noqa: SLF001
            extra_headers=client._extra_headers,  # noqa: SLF001
        )

    async def checkpoint(self, source: str) -> dict[str, Any]:
        return dict(await self._t.get(f"/ingest/checkpoints/{path_segment(source)}"))

    async def list_contracts(self) -> dict[str, Any]:
        return dict(await self._t.get("/ingest/contracts"))

    async def get_contract(self, dataset: str) -> dict[str, Any]:
        return dict(await self._t.get(f"/ingest/contracts/{path_segment(dataset)}"))

    async def put_contract(self, dataset: str, contract: dict[str, Any]) -> dict[str, Any]:
        return dict(await self._t.put(f"/ingest/contracts/{path_segment(dataset)}", contract))

    async def contract_quality(self, dataset: str) -> dict[str, Any]:
        return dict(await self._t.get(f"/ingest/contracts/{path_segment(dataset)}/quality"))

    async def list_quarantine(
        self, *, dataset: str | None = None, after_id: int | None = None, limit: int | None = None
    ) -> dict[str, Any]:
        return dict(await self._t.get(_quarantine_path(dataset, after_id, limit)))

    async def delete_quarantine(self, quarantine_id: int) -> None:
        resp = await self._t.request_bytes(
            "DELETE", _quarantine_item(quarantine_id), accept="application/json"
        )
        _expect_204(resp.status_code, resp.headers.get("x-request-id"))

    async def close(self) -> None:
        await self._t.aclose()

    async def __aenter__(self) -> AsyncIngestControlClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()
