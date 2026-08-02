"""Namespace handle — T9 flagship retrieval surface (#1991, epic #1982).

A :class:`Namespace` binds a parent :class:`~relata.client.RelataClient` to a
single object type (the retrieval-market word for which is "namespace" /
"index" / "collection"). It exposes the search-developer shape the market
expects, with no SQL on the client side::

    with RelataClient("http://localhost:9090", purpose="rag") as client:
        docs = client.namespace("Document")

        # Schemaless write — POST /ingest/auto auto-creates the type with
        # fields inferred from the rows (T6, #1988 — no DDL):
        docs.write([
            {"id": "d1", "title": "Knowledge graphs 101", "body": "..."},
            {"id": "d2", "title": "Vector search", "body": "..."},
        ])

        # Typed ranked search (T1, #1983) — text compiles to a BM25 rank_by
        # directive; filters narrow the set server-side.
        res = docs.query(
            text="graph retrieval",
            match_column="title",
            filters=[{"field": "status", "op": "eq", "value": "published"}],
            limit=10,
        )
        for row in res:
            print(row)

All requests run under the parent client's auth, tenant, and purpose context
(governance travels in the substrate). The connection pool is the parent
client's single shared transport — one pool, reused across every namespace.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

    from relata.client import RelataClient

from relata.models import QueryResult

# Safe-identifier gate before embedding the namespace name into SQL / paths.
# Mirrors relata/objects.py:_TYPE_RE and the server's validate rule
# (`^[A-Za-z_][A-Za-z0-9_]{0,127}$`).
_TYPE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_namespace(name: str) -> None:
    if not isinstance(name, str) or not _TYPE_RE.match(name):
        raise ValueError(
            f"invalid namespace/type name {name!r}: must match ^[A-Za-z_][A-Za-z0-9_]*$"
        )


class Namespace:
    """Synchronous handle bound to one object type (a "namespace").

    Construct via :meth:`RelataClient.namespace`; do not instantiate directly.
    """

    def __init__(self, client: RelataClient, name: str) -> None:
        _validate_namespace(name)
        self._client = client
        self.name = name

    # ------------------------------------------------------------------
    # Query (T1 typed search door, #1983)
    # ------------------------------------------------------------------

    def query(
        self,
        *,
        text: str | None = None,
        match_column: str = "*",
        rank_by: list[Any] | None = None,
        filters: list[dict[str, Any]] | dict[str, Any] | None = None,
        limit: int = 20,
        include_attributes: list[str] | None = None,
        consistency: str | None = None,
        compute_attributes: dict[str, str] | None = None,
        purpose: str | None = None,
    ) -> QueryResult:
        """Typed ranked search over this namespace — see
        :meth:`~relata.search.SearchClient.query` for the full argument
        reference. Returns an iterable :class:`~relata.models.QueryResult`.
        """
        from relata.search import SearchClient

        return SearchClient.from_client(self._client).query(
            self.name,
            text=text,
            match_column=match_column,
            rank_by=rank_by,
            filters=filters,
            limit=limit,
            include_attributes=include_attributes,
            consistency=consistency,
            compute_attributes=compute_attributes,
            purpose=purpose,
        )

    # ------------------------------------------------------------------
    # Write (T6 schemaless ingest, #1988 — POST /ingest/auto)
    # ------------------------------------------------------------------

    def write(
        self,
        rows: Iterable[dict[str, Any]],
        *,
        schema: dict[str, str] | None = None,
        purpose: str | None = None,
    ) -> dict[str, Any]:
        """Schemaless upsert — auto-create the type with inferred fields.

        POSTs ``{"object_type", "rows", "schema"?, "purpose"?}`` to
        ``/ingest/auto`` (T6, #1988). If the type does not yet exist the server
        creates it with field types inferred from the rows; the response
        carries ``type_created`` and ``inferred_schema``. Optional ``schema``
        overrides let you pin a field's type (``"text"``/``"int"``/``"float"``
        /``"bool"``/``"bytes"``/``"datetime"``/``"<model>:<mod>:<ver>:<dim>"``).
        """
        payload: dict[str, Any] = {"object_type": self.name, "rows": list(rows)}
        if schema:
            payload["schema"] = schema
        eff_purpose = purpose or self._client._default_purpose  # noqa: SLF001
        if eff_purpose:
            payload["purpose"] = eff_purpose
        return self._client._sync.post("/ingest/auto", payload)  # noqa: SLF001

    # ------------------------------------------------------------------
    # Point lookup
    # ------------------------------------------------------------------

    def get(self, object_id: str, *, purpose: str | None = None) -> dict[str, Any] | None:
        """Fetch a single row by id, or ``None`` if not found."""
        escaped = str(object_id).replace("'", "''")
        sql = f"SELECT * FROM {self.name} WHERE id = '{escaped}' LIMIT 1"
        result = self._client.query(sql, purpose=purpose)
        return result.rows[0] if result.rows else None

    # ------------------------------------------------------------------
    # Bulk retract
    # ------------------------------------------------------------------

    def delete_all(self, *, purpose: str | None = None) -> dict[str, Any]:
        """Governed bulk retract — bi-temporal tombstone every row in the namespace.

        Fetches all ids via a governed ``SELECT id`` then writes ``_deleted:
        true`` tombstones through the ingest path. NOT a physical delete — rows
        remain in the bi-temporal history for audit/provenance but are excluded
        from subsequent reads.
        """
        sql = f"SELECT id FROM {self.name}"
        result = self._client.query(sql, purpose=purpose)
        ids = [row["id"] for row in result.rows if row.get("id") is not None]
        if not ids:
            return {"deleted": 0}
        tombstones = [{"id": i, "_deleted": True} for i in ids]
        eff_purpose = purpose or self._client._default_purpose  # noqa: SLF001
        payload: dict[str, Any] = {"object_type": self.name, "rows": tombstones}
        if eff_purpose:
            payload["purpose"] = eff_purpose
        self._client._sync.post("/ingest/auto", payload)  # noqa: SLF001
        return {"deleted": len(ids)}

    # ------------------------------------------------------------------
    # Branching (T10 — client-side; server door pending #1992)
    # ------------------------------------------------------------------

    def branch_from(
        self,
        source_namespace: str,
        *,
        purpose: str | None = None,
    ) -> dict[str, Any]:
        """Create this namespace as a copy-on-write branch of ``source_namespace``.

        Targets ``POST /v1/namespaces/{name}/branch`` (T10, #1992). The
        server-side door is pending #1992; this client method is ready for it.
        """
        _validate_namespace(source_namespace)
        payload: dict[str, Any] = {"from": source_namespace}
        if purpose:
            payload["purpose"] = purpose
        return self._client._sync.post(  # noqa: SLF001
            f"/v1/namespaces/{self.name}/branch",
            payload,
        )


class AsyncNamespace:
    """Asynchronous namespace handle — see :class:`Namespace`."""

    def __init__(self, client: RelataClient, name: str) -> None:
        _validate_namespace(name)
        self._client = client
        self.name = name

    async def query(
        self,
        *,
        text: str | None = None,
        match_column: str = "*",
        rank_by: list[Any] | None = None,
        filters: list[dict[str, Any]] | dict[str, Any] | None = None,
        limit: int = 20,
        include_attributes: list[str] | None = None,
        consistency: str | None = None,
        compute_attributes: dict[str, str] | None = None,
        purpose: str | None = None,
    ) -> QueryResult:
        from relata.search import AsyncSearchClient

        return await AsyncSearchClient.from_client(self._client).query(
            self.name,
            text=text,
            match_column=match_column,
            rank_by=rank_by,
            filters=filters,
            limit=limit,
            include_attributes=include_attributes,
            consistency=consistency,
            compute_attributes=compute_attributes,
            purpose=purpose,
        )

    async def write(
        self,
        rows: Iterable[dict[str, Any]],
        *,
        schema: dict[str, str] | None = None,
        purpose: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"object_type": self.name, "rows": list(rows)}
        if schema:
            payload["schema"] = schema
        eff_purpose = purpose or self._client._default_purpose  # noqa: SLF001
        if eff_purpose:
            payload["purpose"] = eff_purpose
        return await self._client._async.post("/ingest/auto", payload)  # noqa: SLF001

    async def get(self, object_id: str, *, purpose: str | None = None) -> dict[str, Any] | None:
        escaped = str(object_id).replace("'", "''")
        sql = f"SELECT * FROM {self.name} WHERE id = '{escaped}' LIMIT 1"
        result = await self._client.aquery(sql, purpose=purpose)
        return result.rows[0] if result.rows else None

    async def delete_all(self, *, purpose: str | None = None) -> dict[str, Any]:
        sql = f"SELECT id FROM {self.name}"
        result = await self._client.aquery(sql, purpose=purpose)
        ids = [row["id"] for row in result.rows if row.get("id") is not None]
        if not ids:
            return {"deleted": 0}
        tombstones = [{"id": i, "_deleted": True} for i in ids]
        eff_purpose = purpose or self._client._default_purpose  # noqa: SLF001
        payload: dict[str, Any] = {"object_type": self.name, "rows": tombstones}
        if eff_purpose:
            payload["purpose"] = eff_purpose
        await self._client._async.post("/ingest/auto", payload)  # noqa: SLF001
        return {"deleted": len(ids)}

    async def branch_from(
        self,
        source_namespace: str,
        *,
        purpose: str | None = None,
    ) -> dict[str, Any]:
        _validate_namespace(source_namespace)
        payload: dict[str, Any] = {"from": source_namespace}
        if purpose:
            payload["purpose"] = purpose
        return await self._client._async.post(  # noqa: SLF001
            f"/v1/namespaces/{self.name}/branch",
            payload,
        )
