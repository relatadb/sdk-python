"""Object store SDK (#82) — general state-object upsert / get / delete surface.

The server's universal upsert path is ``POST /v1/ingest?object_type=<Type>``;
this module wraps it as a typed ``ObjectClient`` so Python callers don't
have to know the ingest envelope shape. A future server-side ``POST /v1/objects``
route would be a strict improvement; this module targets ``/v1/ingest`` today
because that's what ships.

:meth:`ObjectClient.get` uses ``POST /v1/query`` with a safe point-lookup SQL
statement. :meth:`ObjectClient.delete` tombstones the row via the ingest
path (``_deleted: true``) — the same WAL mechanism the server uses internally.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import AsyncGenerator, AsyncIterator, Generator, Iterator
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlencode

from relata._http import AsyncHttpTransport, HttpTransport, path_segment
from relata.exceptions import ConnectionError as RelataConnectionError
from relata.exceptions import NotFoundError, RelataError

if TYPE_CHECKING:
    import httpx

    from relata.client import RelataClient

# Allowlist for object-type names embedded into query SQL.
_TYPE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


_TRANSIENT_STATUS = frozenset({429, 500, 502, 503, 504})


def _is_transient(exc: BaseException) -> bool:
    """Whether a change-feed read failed in a way worth reconnecting for.

    Network loss and 429/5xx are transient; every other API error (404
    ``change-feed-disabled``, 410 ``change-feed-resync-required``, 4xx) is a
    decision for the caller and is raised unchanged.
    """
    if isinstance(exc, RelataConnectionError):
        return True
    if isinstance(exc, RelataError):
        return exc.status_code in _TRANSIENT_STATUS
    return type(exc).__module__.startswith("httpx") and type(exc).__name__.endswith("Error")


def _reconnect_delay(exc: BaseException, attempt: int, backoff: float, cap: float) -> float:
    hinted = getattr(exc, "retry_after", None)
    base = min(cap, backoff * (2**attempt))
    return float(max(base, min(cap, hinted))) if isinstance(hinted, (int, float)) else base


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


def _object_path(object_type: str, object_id: str, purpose: str | None = None) -> str:
    path = (
        f"/v1/objects/{path_segment(_validate_object_type(object_type))}/{path_segment(object_id)}"
    )
    return f"{path}?{urlencode({'purpose': purpose})}" if purpose else path


def _delete_where_request(object_type: str, where: str) -> tuple[str, dict[str, Any]]:
    """Path and body of a delete-by-predicate; a blank predicate is a ``ValueError``."""
    if not isinstance(where, str) or not where.strip():
        raise ValueError("where must be a non-empty SQL WHERE expression")
    path = f"/v1/objects/{path_segment(_validate_object_type(object_type))}"
    return path, {"where": where}


def _precondition_headers(expected_version: int | None, if_absent: bool) -> dict[str, str]:
    """``If-Match`` / ``If-None-Match`` for a conditional write (#5766)."""
    if expected_version is not None and if_absent:
        raise ValueError("expected_version and if_absent are mutually exclusive")
    if if_absent:
        return {"If-None-Match": "*"}
    if expected_version is not None:
        return {"If-Match": f'"{int(expected_version)}"'}
    return {}


def _commit_payload(
    operations: list[dict[str, Any]], idempotency_key: str | None, purpose: str | None
) -> dict[str, Any]:
    """Wire body of ``POST /v1/commit`` (#5767). Operations may use either
    ``expected_version`` or ``expectedVersion``."""
    wire: list[dict[str, Any]] = []
    for operation in operations:
        item = dict(operation)
        if "expected_version" in item:
            item["expectedVersion"] = item.pop("expected_version")
        wire.append(item)
    payload: dict[str, Any] = {"operations": wire}
    if idempotency_key is not None:
        payload["idempotencyKey"] = idempotency_key
    if purpose:
        payload["purpose"] = purpose
    return payload


def _changes_path(
    cursor: str | None, limit: int | None, object_type: str | None, start: str | None
) -> str:
    params: dict[str, str] = {}
    if cursor is not None:
        params["cursor"] = cursor
    if limit is not None:
        params["limit"] = str(limit)
    if object_type is not None:
        params["type"] = _validate_object_type(object_type)
    if start is not None:
        params["start"] = start
    return "/v1/changes" + ("?" + urlencode(params) if params else "")


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
        expected_version: int | None = None,
        if_absent: bool = False,
    ) -> dict[str, Any]:
        """Upsert a single state object via ``POST /v1/ingest``. The ``object_id``
        becomes the row's primary key; re-upserting with the same id
        supersedes bi-temporally.

        Args:
            on_conflict: ``"upsert"`` (default) overwrites; ``"skip"`` ignores
                existing rows; ``"error"`` raises on conflict (HTTP 409).
            expected_version: #5766 — replace only if the object is at this
                version (``If-Match``); a different version raises
                :class:`~relata.exceptions.PreconditionFailedError`. Switches
                this call to the atomic ``PUT /v1/objects/{type}/{id}`` door, which
                is durable before it returns and returns the new ``version``
                instead of a queue acknowledgement.
            if_absent: #5766 — create only if the object does not exist
                (``If-None-Match: *``); otherwise ``PreconditionFailedError``.

        Schema rejections do NOT raise: a rejected row resolves normally with
        ``rows_rejected > 0`` and a populated ``errors`` list (e.g.
        ``{"rows_queued": 0, "rows_rejected": 1, "errors": ["Type.field:
        minCount 1 not met"]}``). "No exception" does not mean "stored" —
        always check ``rows_rejected`` / ``errors`` on the returned ack.

        ``POST /v1/ingest`` is **asynchronous**: this method returns a queue
        acknowledgement, not a confirmed-write receipt. A successful HTTP
        response confirms admission only; it returns an async task-queue-ack shaped like
        ``{"rows_queued": 1, "rows_rejected": 0, "task_id": "itsk_...",
        "connector": "...", "errors": [...], "queue_depth": N,
        "embed_queue_depth": N}``. Poll ``GET /v1/ingest/tasks/{task_id}`` and
        inspect its terminal status and aggregate ``rows_written``,
        ``rows_dropped``, and error/drop-reason accounting. For a particular
        row needed by a dependent operation, read that row after task completion
        to confirm it is visible. Do not submit a dependent write until its
        predecessor's task has completed.

        Valid unconditional upserts of current assertions for the same identity
        are ordered by successful publication under the identity claim, not by
        queue acknowledgement; concurrent writers are last-published-wins.
        Schema, authorization, WORM, conflict, and other failures can still
        occur after admission. An uncertain outcome is not a safe basis for an
        automatic retry.
        """
        if expected_version is not None or if_absent:
            # #5766: atomic conditional write — PUT with If-Match / If-None-Match.
            # Returns ``{"type","id","version","commitId","replayed"}``; a lost
            # race raises PreconditionFailedError carrying ``current_version``.
            body_fields = {k: v for k, v in fields.items() if k != "id"}
            return dict(self._t.put(
                _object_path(object_type, object_id, purpose or self._purpose),
                body_fields,
                extra_headers=_precondition_headers(expected_version, if_absent),
            ))
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
        return self._t.post_raw(
            "/v1/ingest?" + urlencode(params),
            body,
            content_type="application/x-ndjson",
        )

    def typed_upsert(self, obj: object, *, purpose: str | None = None) -> dict[str, Any]:
        """Upsert a typed object (Pydantic model or dataclass). Field names
        and types are validated by the model's schema before the request
        reaches the server — a malformed field is a local ``ValidationError``,
        not a server 400 (#967 Tier 2a).

        The object type is derived from ``obj.__class__.__name__`` or the
        ``_type`` attribute if present.
        """
        # Support Pydantic v2 models.
        if hasattr(obj, "model_dump"):
            fields = cast("Any", obj).model_dump()
        # Support dataclasses.
        elif hasattr(obj, "__dataclass_fields__"):
            from dataclasses import asdict
            fields = asdict(cast("Any", obj))
        else:
            fields = dict(cast("Any", obj))
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
        """Bulk upsert via ``POST /v1/ingest/bulk``. Each row must carry its own
        ``id`` key. Returns the bulk receipt with per-row error detail (#1760):
        ``{"inserted", "updated", "skipped", "errors": [...], "by_type": {...}}``.

        Args:
            on_conflict: ``"upsert"`` (default) overwrites; ``"skip"`` silently
                ignores rows whose ``id`` already exists; ``"error"`` raises on
                the first conflict (HTTP 409).
        """
        # Tag each row with _type so /v1/ingest/bulk can route them.
        objects = [
            {**({**r, "_source": source} if source is not None else r), "_type": object_type}
            for r in rows
        ]
        payload: dict[str, Any] = {"objects": objects, "on_conflict": on_conflict}
        if (eff_purpose := purpose or self._purpose):
            payload["purpose"] = eff_purpose
        return dict(self._t.post("/v1/ingest/bulk", payload))

    def get(
        self,
        object_type: str,
        object_id: str,
        *,
        purpose: str | None = None,
    ) -> dict[str, Any] | None:
        """Fetch a single object by primary key.

        Uses ``POST /v1/query`` with a safe point-lookup SQL statement.

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
            data = self._t.post("/v1/query", payload)
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
        expected_version: int | None = None,
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
        if expected_version is not None:
            # #5766: conditional tombstone — DELETE with If-Match.
            self._t.delete(
                _object_path(object_type, object_id, eff_purpose),
                extra_headers=_precondition_headers(expected_version, False),
            )
            return
        params: dict[str, str] = {"object_type": object_type}
        if eff_purpose:
            params["purpose"] = eff_purpose
        body = _row_to_ndjson([{"id": object_id, "_deleted": True}])
        self._t.post_raw(
            "/v1/ingest?" + urlencode(params),
            body,
            content_type="application/x-ndjson",
        )

    def delete_where(self, object_type: str, where: str) -> dict[str, Any]:
        """Tombstone every live row of ``object_type`` matching ``where``.

        Wraps ``DELETE /v1/objects/{type}`` with ``{"where": ...}`` (the same
        door the Go and TypeScript ``delete_where`` use). ``where`` is a SQL
        WHERE expression such as ``"status = 'stale'"``, not a statement. An
        empty or blank ``where`` is refused client-side: this method can never
        delete every row of a type by accident.

        Returns:
            The server receipt, e.g. ``{"deleted": 3, "type": "Doc"}``.
        """
        return self._t.delete(*_delete_where_request(object_type, where))

    def get_versioned(
        self,
        object_type: str,
        object_id: str,
    ) -> dict[str, Any] | None:
        """Fetch one object with its monotonic ``_version`` (``GET /v1/objects/{type}/{id}``).

        The version is what ``expected_version`` in :meth:`upsert`,
        :meth:`delete` and :meth:`commit` names. Returns ``None`` when the
        object does not exist.
        """
        try:
            return dict(self._t.get(_object_path(object_type, object_id)))
        except NotFoundError:
            return None

    def commit(
        self,
        operations: list[dict[str, Any]],
        *,
        idempotency_key: str | None = None,
        purpose: str | None = None,
    ) -> dict[str, Any]:
        """Atomically apply up to 64 operations (``POST /v1/commit``, #5767).

        Each operation is ``{"op": "create"|"update"|"delete", "type": ...,
        "id": ..., "expected_version"?: int, "data"?: {...}}``. All of them
        become durable and visible, or none do. A failed precondition raises
        :class:`~relata.exceptions.PreconditionFailedError` (``current_version``,
        ``failed_operation``) and changes nothing. A retry with the same
        ``idempotency_key`` returns the original result (``replayed: true``).
        """
        return dict(self._t.post(
            "/v1/commit", _commit_payload(operations, idempotency_key, purpose or self._purpose)
        ))

    def changes(
        self,
        *,
        cursor: str | None = None,
        limit: int | None = None,
        object_type: str | None = None,
        start: str | None = None,
    ) -> dict[str, Any]:
        """Read one page of the object change feed (``GET /v1/changes``, #5716).

        Pass the previous page's ``nextCursor`` as ``cursor`` and persist the new
        one only after every event in the page is processed. Delivery is
        at-least-once: dedupe on ``event["eventId"]``. Events carry identity,
        ``version`` and ``kind`` (``create``/``update``/``delete``), never field
        values. ``start="latest"`` (with no cursor) skips history. A cursor the
        server can no longer serve raises the API error for ``410``
        (``change-feed-resync-required``); re-read state and restart with
        ``start="latest"``.
        """
        return dict(self._t.get(_changes_path(cursor, limit, object_type, start)))

    def iter_change_pages(
        self,
        *,
        cursor: str | None = None,
        limit: int | None = None,
        object_type: str | None = None,
        start: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield successive pages of the change feed until the consumer is caught up.

        Each yielded page is the dict :meth:`changes` returns; persist its
        ``nextCursor`` only after the page's events are processed, then resume
        with ``cursor=`` after a restart. The generator ends on the first page
        with ``hasMore == False`` (call it again to poll). The feed is
        change-data-capture for every write path (objects, ingest, SQL, links,
        deletes) and is off unless the server sets ``RELATA_CHANGE_FEED_ENABLE``;
        a disabled feed raises the API error for ``404 change-feed-disabled``.
        """
        while True:
            page = self.changes(cursor=cursor, limit=limit, object_type=object_type, start=start)
            yield page
            if not page.get("hasMore"):
                return
            cursor, start = page["nextCursor"], None

    def follow_changes(
        self,
        *,
        cursor: str | None = None,
        limit: int | None = None,
        object_type: str | None = None,
        start: str | None = None,
        poll_interval: float = 1.0,
        max_reconnects: int = 8,
        reconnect_backoff: float = 0.5,
        max_backoff: float = 30.0,
    ) -> Generator[dict[str, Any], None, None]:
        """Follow the change feed indefinitely, reconnecting and resuming by cursor.

        Yields each non-empty page. The cursor only advances after the consumer
        asks for the next page, so persist ``page["nextCursor"]`` once its events
        are processed; after a crash pass it back as ``cursor=`` for at-least-once
        delivery with no gap. When caught up the follower sleeps ``poll_interval``.
        A transient failure (network loss, 429, 5xx) is retried from the last
        delivered cursor with exponential backoff (``reconnect_backoff`` doubling
        up to ``max_backoff``, honouring a server ``Retry-After``); after
        ``max_reconnects`` consecutive failures the last error is raised. Any
        other API error, including ``410 change-feed-resync-required`` and
        ``404 change-feed-disabled``, is raised at once. Stop by closing the
        generator.
        """
        failures = 0
        while True:
            try:
                page = self.changes(
                    cursor=cursor, limit=limit, object_type=object_type, start=start
                )
            except Exception as exc:
                if not _is_transient(exc) or failures >= max_reconnects:
                    raise
                time.sleep(_reconnect_delay(exc, failures, reconnect_backoff, max_backoff))
                failures += 1
                continue
            failures = 0
            cursor, start = page["nextCursor"], None
            if page.get("events"):
                yield page
            if not page.get("hasMore"):
                time.sleep(poll_interval)

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
        expected_version: int | None = None,
        if_absent: bool = False,
    ) -> dict[str, Any]:
        """Async equivalent of :meth:`ObjectClient.upsert`; see that
        docstring for its queue-admission, task-status, and publication-order
        contract. ``POST /v1/ingest`` returns an acknowledgement, not a
        confirmed-write receipt. Poll task status and inspect aggregate outcome
        before depending on the write; verify a particular row with a read when
        necessary. Uncertain outcomes are not a safe basis for automatic retry.
        """
        if expected_version is not None or if_absent:
            # #5766: atomic conditional write — PUT with If-Match / If-None-Match.
            # Returns ``{"type","id","version","commitId","replayed"}``; a lost
            # race raises PreconditionFailedError carrying ``current_version``.
            body_fields = {k: v for k, v in fields.items() if k != "id"}
            return dict(await self._t.put(
                _object_path(object_type, object_id, purpose or self._purpose),
                body_fields,
                extra_headers=_precondition_headers(expected_version, if_absent),
            ))
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
        return await self._t.post_raw(
            "/v1/ingest?" + urlencode(params),
            body,
            content_type="application/x-ndjson",
        )

    async def typed_upsert(self, obj: object, *, purpose: str | None = None) -> dict[str, Any]:
        """Async equivalent of :meth:`ObjectClient.typed_upsert` — upsert a
        typed object (Pydantic model or dataclass), validated locally before
        the request reaches the server (#967 Tier 2a)."""
        if hasattr(obj, "model_dump"):
            fields = cast("Any", obj).model_dump()
        elif hasattr(obj, "__dataclass_fields__"):
            from dataclasses import asdict
            fields = asdict(cast("Any", obj))
        else:
            fields = dict(cast("Any", obj))
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
        return dict(await self._t.post("/v1/ingest/bulk", payload))

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
            data = await self._t.post("/v1/query", payload)
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
        expected_version: int | None = None,
    ) -> None:
        """Async equivalent of :meth:`ObjectClient.delete`."""
        _validate_object_type(object_type)
        eff_purpose = purpose or self._purpose
        if expected_version is not None:
            await self._t.delete(
                _object_path(object_type, object_id, eff_purpose),
                extra_headers=_precondition_headers(expected_version, False),
            )
            return
        params: dict[str, str] = {"object_type": object_type}
        if eff_purpose:
            params["purpose"] = eff_purpose
        body = _row_to_ndjson([{"id": object_id, "_deleted": True}])
        await self._t.post_raw(
            "/v1/ingest?" + urlencode(params),
            body,
            content_type="application/x-ndjson",
        )

    async def delete_where(self, object_type: str, where: str) -> dict[str, Any]:
        """Async equivalent of :meth:`ObjectClient.delete_where`.

        Wraps ``DELETE /v1/objects/{type}`` with ``{"where": ...}`` (the same
        door the Go and TypeScript ``delete_where`` use). ``where`` is a SQL
        WHERE expression such as ``"status = 'stale'"``, not a statement. An
        empty or blank ``where`` is refused client-side: this method can never
        delete every row of a type by accident.

        Returns:
            The server receipt, e.g. ``{"deleted": 3, "type": "Doc"}``.
        """
        return await self._t.delete(*_delete_where_request(object_type, where))

    async def get_versioned(
        self,
        object_type: str,
        object_id: str,
    ) -> dict[str, Any] | None:
        """Fetch one object with its monotonic ``_version`` (``GET /v1/objects/{type}/{id}``).

        The version is what ``expected_version`` in :meth:`upsert`,
        :meth:`delete` and :meth:`commit` names. Returns ``None`` when the
        object does not exist.
        """
        try:
            return dict(await self._t.get(_object_path(object_type, object_id)))
        except NotFoundError:
            return None

    async def commit(
        self,
        operations: list[dict[str, Any]],
        *,
        idempotency_key: str | None = None,
        purpose: str | None = None,
    ) -> dict[str, Any]:
        """Atomically apply up to 64 operations (``POST /v1/commit``, #5767).

        Each operation is ``{"op": "create"|"update"|"delete", "type": ...,
        "id": ..., "expected_version"?: int, "data"?: {...}}``. All of them
        become durable and visible, or none do. A failed precondition raises
        :class:`~relata.exceptions.PreconditionFailedError` (``current_version``,
        ``failed_operation``) and changes nothing. A retry with the same
        ``idempotency_key`` returns the original result (``replayed: true``).
        """
        return dict(await self._t.post(
            "/v1/commit", _commit_payload(operations, idempotency_key, purpose or self._purpose)
        ))

    async def changes(
        self,
        *,
        cursor: str | None = None,
        limit: int | None = None,
        object_type: str | None = None,
        start: str | None = None,
    ) -> dict[str, Any]:
        """Read one page of the object change feed (``GET /v1/changes``, #5716).

        Pass the previous page's ``nextCursor`` as ``cursor`` and persist the new
        one only after every event in the page is processed. Delivery is
        at-least-once: dedupe on ``event["eventId"]``. Events carry identity,
        ``version`` and ``kind`` (``create``/``update``/``delete``), never field
        values. ``start="latest"`` (with no cursor) skips history. A cursor the
        server can no longer serve raises the API error for ``410``
        (``change-feed-resync-required``); re-read state and restart with
        ``start="latest"``.
        """
        return dict(await self._t.get(_changes_path(cursor, limit, object_type, start)))

    async def iter_change_pages(
        self,
        *,
        cursor: str | None = None,
        limit: int | None = None,
        object_type: str | None = None,
        start: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Async twin of :meth:`ObjectClient.iter_change_pages`."""
        while True:
            page = await self.changes(
                cursor=cursor, limit=limit, object_type=object_type, start=start
            )
            yield page
            if not page.get("hasMore"):
                return
            cursor, start = page["nextCursor"], None

    async def follow_changes(
        self,
        *,
        cursor: str | None = None,
        limit: int | None = None,
        object_type: str | None = None,
        start: str | None = None,
        poll_interval: float = 1.0,
        max_reconnects: int = 8,
        reconnect_backoff: float = 0.5,
        max_backoff: float = 30.0,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Follow the change feed indefinitely, reconnecting and resuming by cursor.

        Yields each non-empty page. The cursor only advances after the consumer
        asks for the next page, so persist ``page["nextCursor"]`` once its events
        are processed; after a crash pass it back as ``cursor=`` for at-least-once
        delivery with no gap. When caught up the follower sleeps ``poll_interval``.
        A transient failure (network loss, 429, 5xx) is retried from the last
        delivered cursor with exponential backoff (``reconnect_backoff`` doubling
        up to ``max_backoff``, honouring a server ``Retry-After``); after
        ``max_reconnects`` consecutive failures the last error is raised. Any
        other API error, including ``410 change-feed-resync-required`` and
        ``404 change-feed-disabled``, is raised at once. Stop by closing the
        generator.
        """
        failures = 0
        while True:
            try:
                page = await self.changes(
                    cursor=cursor, limit=limit, object_type=object_type, start=start
                )
            except Exception as exc:
                if not _is_transient(exc) or failures >= max_reconnects:
                    raise
                await asyncio.sleep(_reconnect_delay(exc, failures, reconnect_backoff, max_backoff))
                failures += 1
                continue
            failures = 0
            cursor, start = page["nextCursor"], None
            if page.get("events"):
                yield page
            if not page.get("hasMore"):
                await asyncio.sleep(poll_interval)

    async def close(self) -> None:
        await self._t.aclose()

    async def __aenter__(self) -> AsyncObjectClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()
