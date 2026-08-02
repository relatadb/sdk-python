"""Object store SDK (#82) — general state-object upsert / get / delete surface.

The server's universal upsert path is ``POST /ingest?object_type=<Type>``;
this module wraps it as a typed ``ObjectClient`` so Python callers don't
have to know the ingest envelope shape. A future server-side ``POST /objects``
route would be a strict improvement; this module targets ``/ingest`` today
because that's what ships.

:meth:`ObjectClient.get` uses ``POST /query`` with a safe point-lookup SQL
statement. :meth:`ObjectClient.delete` tombstones the row via the ingest
path (``_deleted: true``) — the same WAL mechanism the server uses internally.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

from relata._http import AsyncHttpTransport, HttpTransport
from relata.exceptions import NotFoundError

if TYPE_CHECKING:
    import httpx

    from relata.client import RelataClient

# Allowlist for object-type names embedded into query SQL.
_TYPE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_object_type(name: str) -> str:
    if not _TYPE_RE.match(name):
        raise ValueError(
            f"Invalid object_type {name!r}: must match [A-Za-z_][A-Za-z0-9_]*"
        )
    return name


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
            extra["X-Relata-Tenant-Id"] = tenant
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
        on_conflict: str = "upsert",
    ) -> dict[str, Any]:
        """Upsert a single state object. The ``object_id`` becomes the row's
        primary key; re-upserting with the same id supersedes bi-temporally.

        Args:
            on_conflict: ``"upsert"`` (default) overwrites; ``"skip"`` ignores
                existing rows; ``"error"`` raises on conflict (HTTP 409).

        Returns the server's upsert receipt (``object_id``, ``write_seq``,
        ``valid_from``).
        """
        row = dict(fields)
        row["id"] = object_id
        if source is not None:
            row["_source"] = source
        body = _row_to_ndjson([row])
        eff_purpose = purpose or self._purpose
        params: dict[str, str] = {"object_type": object_type}
        if eff_purpose:
            params["purpose"] = eff_purpose
        if on_conflict != "upsert":
            params["on_conflict"] = on_conflict
        headers = {"Content-Type": "application/x-ndjson"}
        resp = self._t._client.post(  # noqa: SLF001 — needs raw httpx for non-JSON body
            "/ingest?" + urlencode(params),
            content=body,
            headers=headers,
        )
        return dict(HttpTransport._handle(resp))

    def typed_upsert(self, obj: Any, *, purpose: str | None = None) -> dict[str, Any]:
        """Upsert a typed object (Pydantic model or dataclass). Field names
        and types are validated by the model's schema before the request
        reaches the server — a malformed field is a local ``ValidationError``,
        not a server 400 (#967 Tier 2a).

        The object type is derived from ``obj.__class__.__name__`` or the
        ``_type`` attribute if present.
        """
        # Support Pydantic v2 models.
        if hasattr(obj, "model_dump"):
            fields = obj.model_dump()
        # Support dataclasses.
        elif hasattr(obj, "__dataclass_fields__"):
            from dataclasses import asdict
            fields = asdict(obj)
        else:
            fields = dict(obj)
        object_type = fields.pop("_type", type(obj).__name__)
        object_id = str(fields.get("_pk", fields.get("id", "")))
        return self.upsert(object_type, object_id, fields, purpose=purpose)

    def batch_upsert(
        self,
        object_type: str,
        rows: list[dict[str, Any]],
        *,
        purpose: str | None = None,
        source: str | None = None,
        on_conflict: str = "upsert",
    ) -> dict[str, Any]:
        """Bulk upsert via ``POST /ingest/bulk``. Each row must carry its own
        ``id`` key. Returns the bulk receipt with per-row error detail (#1760):
        ``{"inserted", "updated", "skipped", "errors": [...], "by_type": {...}}``.

        Args:
            on_conflict: ``"upsert"`` (default) overwrites; ``"skip"`` silently
                ignores rows whose ``id`` already exists; ``"error"`` raises on
                the first conflict (HTTP 409).
        """
        # Tag each row with _type so /ingest/bulk can route them.
        objects = [
            {**({**r, "_source": source} if source is not None else r), "_type": object_type}
            for r in rows
        ]
        payload: dict[str, Any] = {"objects": objects, "on_conflict": on_conflict}
        if (eff_purpose := purpose or self._purpose):
            payload["purpose"] = eff_purpose
        return dict(self._t.post("/ingest/bulk", payload))

    def get(
        self,
        object_type: str,
        object_id: str,
        *,
        purpose: str | None = None,
    ) -> dict[str, Any] | None:
        """Fetch a single object by primary key.

        Uses ``POST /query`` with a safe point-lookup SQL statement.

        Args:
            object_type: Object type name (e.g. ``"HmSkill"``).
            object_id:   Primary key value. Single quotes are escaped.
            purpose:     Query purpose; falls back to the client default.

        Returns:
            Row dict when found, ``None`` when the object does not exist.

        Raises:
            :class:`~relata.exceptions.ForbiddenError`: Not authorised.
            :class:`~relata.exceptions.PurposeError`: No purpose set.
        """
        _validate_object_type(object_type)
        safe_id = object_id.replace("'", "''")
        sql = f"SELECT * FROM {object_type} WHERE id = '{safe_id}' LIMIT 1"
        eff_purpose = purpose or self._purpose
        payload: dict[str, Any] = {"sql": sql}
        if eff_purpose:
            payload["purpose"] = eff_purpose
        try:
            data = self._t.post("/query", payload)
        except NotFoundError:
            return None
        # Wire shape: server sends {"data": [...], "rows": <int count>}.
        # QueryResult normalises this; we work with the raw dict here.
        raw = data.get("data") or []
        if not raw and isinstance(data.get("rows"), list):
            raw = data["rows"]
        return dict(raw[0]) if raw else None

    def delete(
        self,
        object_type: str,
        object_id: str,
        *,
        purpose: str | None = None,
    ) -> None:
        """Bi-temporally delete an object by primary key.

        Sends ``_deleted: true`` via the ingest path — the same WAL tombstone
        mechanism the server uses internally. History is preserved; the row is
        excluded from live scans and queries.

        Args:
            object_type: Object type name.
            object_id:   Primary key of the object to delete.
            purpose:     Ingest purpose; falls back to the client default.

        Raises:
            :class:`~relata.exceptions.ForbiddenError`: Not authorised.
        """
        _validate_object_type(object_type)
        eff_purpose = purpose or self._purpose
        params: dict[str, str] = {"object_type": object_type}
        if eff_purpose:
            params["purpose"] = eff_purpose
        body = _row_to_ndjson([{"id": object_id, "_deleted": True}])
        headers = {"Content-Type": "application/x-ndjson"}
        resp = self._t._client.post(  # noqa: SLF001
            "/ingest?" + urlencode(params),
            content=body,
            headers=headers,
        )
        HttpTransport._handle(resp)

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
            extra["X-Relata-Tenant-Id"] = tenant
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
        on_conflict: str = "upsert",
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
        if on_conflict != "upsert":
            params["on_conflict"] = on_conflict
        headers = {"Content-Type": "application/x-ndjson"}
        resp = await self._t._client.post(  # noqa: SLF001
            "/ingest?" + urlencode(params),
            content=body,
            headers=headers,
        )
        return dict(HttpTransport._handle(resp))

    async def typed_upsert(self, obj: Any, *, purpose: str | None = None) -> dict[str, Any]:
        """Async equivalent of :meth:`ObjectClient.typed_upsert` — upsert a
        typed object (Pydantic model or dataclass), validated locally before
        the request reaches the server (#967 Tier 2a)."""
        if hasattr(obj, "model_dump"):
            fields = obj.model_dump()
        elif hasattr(obj, "__dataclass_fields__"):
            from dataclasses import asdict
            fields = asdict(obj)
        else:
            fields = dict(obj)
        object_type = fields.pop("_type", type(obj).__name__)
        object_id = str(fields.get("_pk", fields.get("id", "")))
        return await self.upsert(object_type, object_id, fields, purpose=purpose)

    async def batch_upsert(
        self,
        object_type: str,
        rows: list[dict[str, Any]],
        *,
        purpose: str | None = None,
        source: str | None = None,
        on_conflict: str = "upsert",
    ) -> dict[str, Any]:
        """Async equivalent of :meth:`ObjectClient.batch_upsert`."""
        objects = [
            {**({**r, "_source": source} if source is not None else r), "_type": object_type}
            for r in rows
        ]
        payload: dict[str, Any] = {"objects": objects, "on_conflict": on_conflict}
        if (eff_purpose := purpose or self._purpose):
            payload["purpose"] = eff_purpose
        return dict(await self._t.post("/ingest/bulk", payload))

    async def get(
        self,
        object_type: str,
        object_id: str,
        *,
        purpose: str | None = None,
    ) -> dict[str, Any] | None:
        """Async equivalent of :meth:`ObjectClient.get`."""
        _validate_object_type(object_type)
        safe_id = object_id.replace("'", "''")
        sql = f"SELECT * FROM {object_type} WHERE id = '{safe_id}' LIMIT 1"
        eff_purpose = purpose or self._purpose
        payload: dict[str, Any] = {"sql": sql}
        if eff_purpose:
            payload["purpose"] = eff_purpose
        try:
            data = await self._t.post("/query", payload)
        except NotFoundError:
            return None
        raw = data.get("data") or []
        if not raw and isinstance(data.get("rows"), list):
            raw = data["rows"]
        return dict(raw[0]) if raw else None

    async def delete(
        self,
        object_type: str,
        object_id: str,
        *,
        purpose: str | None = None,
    ) -> None:
        """Async equivalent of :meth:`ObjectClient.delete`."""
        _validate_object_type(object_type)
        eff_purpose = purpose or self._purpose
        params: dict[str, str] = {"object_type": object_type}
        if eff_purpose:
            params["purpose"] = eff_purpose
        body = _row_to_ndjson([{"id": object_id, "_deleted": True}])
        headers = {"Content-Type": "application/x-ndjson"}
        resp = await self._t._client.post(  # noqa: SLF001
            "/ingest?" + urlencode(params),
            content=body,
            headers=headers,
        )
        HttpTransport._handle(resp)

    async def close(self) -> None:
        await self._t.aclose()

    async def __aenter__(self) -> AsyncObjectClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()
