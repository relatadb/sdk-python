"""Object store SDK (#82) — general state-object upsert + load surface.

The server's universal upsert path is ``POST /ingest?object_type=<Type>``;
this module wraps it as a typed ``ObjectClient`` so Python callers don't
have to know the ingest envelope shape. A future server-side ``POST /objects``
route would be a strict improvement; this module targets ``/ingest`` today
because that's what ships.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

from relata._http import AsyncHttpTransport, HttpTransport

if TYPE_CHECKING:
    import httpx

    from relata.client import RelataClient


def _row_to_ndjson(rows: list[dict[str, Any]]) -> str:
    """Serialise ``rows`` as newline-delimited JSON — one row per line."""
    import json

    return "\n".join(json.dumps(r) for r in rows)


class ObjectClient:
    """Synchronous typed upsert/load client for state objects."""

    def __init__(
        self,
        base_url: str,
        *,
        bearer_token: str | None = None,
        purpose: str | None = None,
        tenant: str | None = None,
        timeout: float = 30.0,
        extra_headers: dict[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._purpose = purpose
        self._tenant = tenant
        extra: dict[str, str] = {}
        if tenant is not None:
            extra["X-Organization-Id"] = tenant
        if extra_headers:
            extra.update(extra_headers)
        self._t = HttpTransport(
            base_url, bearer_token, timeout, transport=transport, extra_headers=extra or None
        )

    @classmethod
    def from_client(cls, client: RelataClient) -> ObjectClient:
        return cls(
            client._base_url,
            bearer_token=client._bearer_token,
            purpose=client._default_purpose,
            tenant=client._tenant,
            timeout=client._timeout,
            extra_headers=client._extra_headers,
        )

    def upsert(
        self,
        object_type: str,
        object_id: str,
        fields: dict[str, Any],
        *,
        purpose: str | None = None,
        source: str | None = None,
    ) -> dict[str, Any]:
        """Upsert a single state object. The ``object_id`` becomes the row's
        primary key; re-upserting with the same id supersedes bi-temporally.

        Returns the server's upsert receipt (``object_id``, ``write_seq``,
        ``valid_from``).
        """
        row = dict(fields)
        row["id"] = object_id
        if source is not None:
            row["_source"] = source
        body = _row_to_ndjson([row])
        # The /ingest endpoint accepts NDJSON in the body and object_type in
        # the query string. ``purpose`` is required by governance.
        eff_purpose = purpose or self._purpose
        params: dict[str, str] = {"object_type": object_type}
        if eff_purpose:
            params["purpose"] = eff_purpose
        headers = {"Content-Type": "application/x-ndjson"}
        resp = self._t._client.post(  # noqa: SLF001 — needs raw httpx for non-JSON body
            "/ingest?" + urlencode(params),
            content=body,
            headers=headers,
        )
        return dict(HttpTransport._handle(resp))

    def batch_upsert(
        self,
        object_type: str,
        rows: list[dict[str, Any]],
        *,
        purpose: str | None = None,
        source: str | None = None,
    ) -> dict[str, Any]:
        """Bulk upsert. Each row must carry its own ``id`` key. Returns the
        bulk receipt (``accepted``, ``rejected``, ``write_seq``, ``queue_depth``).
        """
        if source is not None:
            rows = [{**r, "_source": source} for r in rows]
        body = _row_to_ndjson(rows)
        eff_purpose = purpose or self._purpose
        params: dict[str, str] = {"object_type": object_type}
        if eff_purpose:
            params["purpose"] = eff_purpose
        headers = {"Content-Type": "application/x-ndjson"}
        resp = self._t._client.post(  # noqa: SLF001
            "/ingest?" + urlencode(params),
            content=body,
            headers=headers,
        )
        return dict(HttpTransport._handle(resp))

    def close(self) -> None:
        self._t.close()

    def __enter__(self) -> ObjectClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class AsyncObjectClient:
    """Asynchronous typed upsert client — see :class:`ObjectClient`."""

    def __init__(
        self,
        base_url: str,
        *,
        bearer_token: str | None = None,
        purpose: str | None = None,
        tenant: str | None = None,
        timeout: float = 30.0,
        extra_headers: dict[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._purpose = purpose
        self._tenant = tenant
        extra: dict[str, str] = {}
        if tenant is not None:
            extra["X-Organization-Id"] = tenant
        if extra_headers:
            extra.update(extra_headers)
        self._t = AsyncHttpTransport(
            base_url, bearer_token, timeout, transport=transport, extra_headers=extra or None
        )

    @classmethod
    def from_client(cls, client: RelataClient) -> AsyncObjectClient:
        return cls(
            client._base_url,
            bearer_token=client._bearer_token,
            purpose=client._default_purpose,
            tenant=client._tenant,
            timeout=client._timeout,
            extra_headers=client._extra_headers,
        )

    async def upsert(
        self,
        object_type: str,
        object_id: str,
        fields: dict[str, Any],
        *,
        purpose: str | None = None,
        source: str | None = None,
    ) -> dict[str, Any]:
        row = dict(fields)
        row["id"] = object_id
        if source is not None:
            row["_source"] = source
        body = _row_to_ndjson([row])
        eff_purpose = purpose or self._purpose
        params: dict[str, str] = {"object_type": object_type}
        if eff_purpose:
            params["purpose"] = eff_purpose
        headers = {"Content-Type": "application/x-ndjson"}
        resp = await self._t._client.post(  # noqa: SLF001
            "/ingest?" + urlencode(params),
            content=body,
            headers=headers,
        )
        return dict(HttpTransport._handle(resp))

    async def batch_upsert(
        self,
        object_type: str,
        rows: list[dict[str, Any]],
        *,
        purpose: str | None = None,
        source: str | None = None,
    ) -> dict[str, Any]:
        if source is not None:
            rows = [{**r, "_source": source} for r in rows]
        body = _row_to_ndjson(rows)
        eff_purpose = purpose or self._purpose
        params: dict[str, str] = {"object_type": object_type}
        if eff_purpose:
            params["purpose"] = eff_purpose
        headers = {"Content-Type": "application/x-ndjson"}
        resp = await self._t._client.post(  # noqa: SLF001
            "/ingest?" + urlencode(params),
            content=body,
            headers=headers,
        )
        return dict(HttpTransport._handle(resp))

    async def close(self) -> None:
        await self._t.aclose()

    async def __aenter__(self) -> AsyncObjectClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()
