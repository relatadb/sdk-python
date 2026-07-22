"""Bulk ingest SDK (#83).

Surfaces the server's ``POST /ingest?object_type=<Type>`` for NDJSON / CSV /
Arrow batches. Distinct from :class:`relata.client.RelataClient.ingest_document`
which is the datagrep-extractor envelope; this module is the general batch
path the partner contract (§4) calls for.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable
from urllib.parse import urlencode

from relata._http import AsyncHttpTransport, HttpTransport

if TYPE_CHECKING:
    import httpx

    from relata.client import RelataClient


def _handle_raw(resp: httpx.Response) -> dict[str, Any]:
    """Static-method equivalent of HttpTransport._handle for raw-response paths."""
    if resp.is_success:
        return resp.json()  # type: ignore[no-any-return]
    try:
        body: dict[str, Any] = resp.json()
    except Exception:
        body = {"error": resp.text or "empty response"}
    from relata._http import _classify_error

    raise _classify_error(resp.status_code, body)


class IngestClient:
    """Synchronous bulk-ingest client."""

    def __init__(
        self,
        base_url: str,
        *,
        bearer_token: str | None = None,
        purpose: str | None = None,
        tenant: str | None = None,
        timeout: float = 60.0,
        extra_headers: dict[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._purpose = purpose
        self._tenant = tenant
        extra: dict[str, str] = {}
        if tenant is not None:
            extra["X-Relata-Tenant-Id"] = tenant
        if extra_headers:
            extra.update(extra_headers)
        self._t = HttpTransport(
            base_url, bearer_token, timeout, transport=transport, extra_headers=extra or None
        )

    @classmethod
    def from_client(cls, client: RelataClient) -> IngestClient:
        return cls(
            client._base_url,
            bearer_token=client._bearer_token,
            purpose=client._default_purpose,
            tenant=client._tenant,
            timeout=client._timeout,
            extra_headers=client._extra_headers,
        )

    def _params(self, object_type: str, purpose: str | None) -> str:
        eff_purpose = purpose or self._purpose
        params: dict[str, str] = {"object_type": object_type}
        if eff_purpose:
            params["purpose"] = eff_purpose
        return urlencode(params)

    def bulk(
        self,
        object_type: str,
        rows: list[dict[str, Any]],
        *,
        purpose: str | None = None,
    ) -> dict[str, Any]:
        """Bulk-ingest ``rows`` as NDJSON. Returns the server receipt."""
        import json

        body = "\n".join(json.dumps(r) for r in rows)
        resp = self._t._client.post(  # noqa: SLF001 — raw NDJSON body
            f"/ingest?{self._params(object_type, purpose)}",
            content=body,
            headers={"Content-Type": "application/x-ndjson"},
        )
        return _handle_raw(resp)

    def bulk_csv(
        self,
        object_type: str,
        csv_text: str,
        *,
        purpose: str | None = None,
    ) -> dict[str, Any]:
        """Bulk-ingest a CSV string. The server parses it server-side."""
        resp = self._t._client.post(  # noqa: SLF001
            f"/ingest?{self._params(object_type, purpose)}",
            content=csv_text,
            headers={"Content-Type": "text/csv"},
        )
        return _handle_raw(resp)

    def ingest_iter(
        self,
        object_type: str,
        rows: Iterable[dict[str, Any]],
        *,
        purpose: str | None = None,
        batch_size: int = 500,
    ) -> int:
        """Stream rows from an iterator into batched ``POST /ingest`` calls.

        Memory is O(batch_size), not O(total_rows). Returns the total number
        of rows successfully ingested. Stops on the first error.

        >>> rows = [{"name": "Alice"}, {"name": "Bob"}]
        >>> total = client.ingest_iter("Person", rows, purpose="onboarding", batch_size=500)
        """
        batch: list[dict[str, Any]] = []
        total = 0
        for row in rows:
            batch.append(row)
            if len(batch) >= batch_size:
                self.bulk(object_type, batch, purpose=purpose)
                total += len(batch)
                batch = []
        if batch:
            self.bulk(object_type, batch, purpose=purpose)
            total += len(batch)
        return total

    def media_status(self, task_id: str) -> dict[str, Any]:
        """Poll the status of a multipart media upload (paired with #76)."""
        return self._t.get(f"/ingest/media/{task_id}")

    def close(self) -> None:
        self._t.close()

    def __enter__(self) -> IngestClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class AsyncIngestClient:
    """Asynchronous bulk-ingest client — see :class:`IngestClient`."""

    def __init__(
        self,
        base_url: str,
        *,
        bearer_token: str | None = None,
        purpose: str | None = None,
        tenant: str | None = None,
        timeout: float = 60.0,
        extra_headers: dict[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._purpose = purpose
        self._tenant = tenant
        extra: dict[str, str] = {}
        if tenant is not None:
            extra["X-Relata-Tenant-Id"] = tenant
        if extra_headers:
            extra.update(extra_headers)
        self._t = AsyncHttpTransport(
            base_url, bearer_token, timeout, transport=transport, extra_headers=extra or None
        )

    @classmethod
    def from_client(cls, client: RelataClient) -> AsyncIngestClient:
        return cls(
            client._base_url,
            bearer_token=client._bearer_token,
            purpose=client._default_purpose,
            tenant=client._tenant,
            timeout=client._timeout,
            extra_headers=client._extra_headers,
        )

    def _params(self, object_type: str, purpose: str | None) -> str:
        eff_purpose = purpose or self._purpose
        params: dict[str, str] = {"object_type": object_type}
        if eff_purpose:
            params["purpose"] = eff_purpose
        return urlencode(params)

    async def bulk(
        self,
        object_type: str,
        rows: list[dict[str, Any]],
        *,
        purpose: str | None = None,
    ) -> dict[str, Any]:
        import json

        body = "\n".join(json.dumps(r) for r in rows)
        resp = await self._t._client.post(  # noqa: SLF001
            f"/ingest?{self._params(object_type, purpose)}",
            content=body,
            headers={"Content-Type": "application/x-ndjson"},
        )
        return _handle_raw(resp)

    async def bulk_csv(
        self,
        object_type: str,
        csv_text: str,
        *,
        purpose: str | None = None,
    ) -> dict[str, Any]:
        resp = await self._t._client.post(  # noqa: SLF001
            f"/ingest?{self._params(object_type, purpose)}",
            content=csv_text,
            headers={"Content-Type": "text/csv"},
        )
        return _handle_raw(resp)

    async def media_status(self, task_id: str) -> dict[str, Any]:
        return await self._t.get(f"/ingest/media/{task_id}")

    async def close(self) -> None:
        await self._t.aclose()

    async def __aenter__(self) -> AsyncIngestClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()
