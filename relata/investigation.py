"""Investigation workflow: alert evidence, timeline, pivots and saved searches (#5702).

``alert_evidence`` returns the source events behind an alert with a
reproduction query, completeness report and SHA-256. ``alert_timeline`` orders
the same facts (window, events, firing, triage changes) and lists the entities
to pivot on. ``pivot`` follows one of those entities into the caller's other
readable records; the value must come from evidence the caller could read, and
unreadable types and rows are omitted without a count. ``create_saved_search``
and friends keep a named SQL query (versioned, per tenant, durable);
``run_saved_search`` runs it through the governed ``POST /query`` path with the
*caller's* read policy, so saving a query grants nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlencode

from relata._http import AsyncHttpTransport, HttpTransport

if TYPE_CHECKING:
    import httpx

    from relata.client import RelataClient


def _aid(alert_id: str, suffix: str) -> str:
    return f"/alerts/{quote(alert_id, safe='')}/{suffix}"


def _sid(search_id: str, suffix: str = "") -> str:
    path = f"/searches/saved/{quote(search_id, safe='')}"
    return f"{path}/{suffix}" if suffix else path


def _bounds(max_events: int | None, deadline_ms: int | None) -> str:
    pairs = (("max_events", max_events), ("deadline_ms", deadline_ms))
    query = {k: str(v) for k, v in pairs if v is not None}
    return f"?{urlencode(query)}" if query else ""


def _pivot_body(
    field: str | None,
    value: Any,
    target_field: str | None,
    target_types: Sequence[str] | None,
    limit: int | None,
    deadline_ms: int | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {}
    if field is not None:
        body["field"] = field
    if value is not None:
        body["value"] = value
    if target_field is not None:
        body["target_field"] = target_field
    if target_types is not None:
        body["target_types"] = list(target_types)
    if limit is not None:
        body["limit"] = limit
    if deadline_ms is not None:
        body["deadline_ms"] = deadline_ms
    return body


def _search_body(
    name: str,
    sql: str,
    purpose: str | None,
    description: str | None,
    params: Sequence[Any] | None,
    expected_version: int | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"name": name, "sql": sql}
    if purpose is not None:
        body["purpose"] = purpose
    if description is not None:
        body["description"] = description
    if params is not None:
        body["params"] = list(params)
    if expected_version is not None:
        body["expected_version"] = expected_version
    return body


def _list_path(q: str | None, limit: int | None) -> str:
    query = {k: str(v) for k, v in (("q", q), ("limit", limit)) if v is not None}
    return "/searches/saved" + (f"?{urlencode(query)}" if query else "")


class InvestigationClient:
    """Synchronous client for ``/alerts/{id}/{evidence,timeline,pivot}`` and ``/searches/saved``."""

    def __init__(
        self,
        base_url: str,
        *,
        bearer_token: str | None = None,
        timeout: float = 60.0,
        extra_headers: dict[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._t = HttpTransport(
            base_url, bearer_token, timeout, transport=transport, extra_headers=extra_headers
        )

    @classmethod
    def from_client(cls, client: RelataClient) -> InvestigationClient:
        return cls(
            client._base_url,  # noqa: SLF001
            bearer_token=client._bearer_token,  # noqa: SLF001
            timeout=client._timeout,  # noqa: SLF001
            extra_headers=client._extra_headers,  # noqa: SLF001
        )

    def alert_evidence(
        self, alert_id: str, *, max_events: int | None = None, deadline_ms: int | None = None
    ) -> dict[str, Any]:
        """``GET /alerts/{id}/evidence``: events, reproduction query, completeness, SHA-256."""
        return dict(self._t.get(_aid(alert_id, "evidence") + _bounds(max_events, deadline_ms)))

    def alert_timeline(
        self, alert_id: str, *, max_events: int | None = None, deadline_ms: int | None = None
    ) -> dict[str, Any]:
        """``GET /alerts/{id}/timeline``: ordered entries, unplaced ids and pivot entities."""
        return dict(self._t.get(_aid(alert_id, "timeline") + _bounds(max_events, deadline_ms)))

    def pivot(
        self,
        alert_id: str,
        *,
        field: str | None = None,
        value: Any = None,
        target_field: str | None = None,
        target_types: Sequence[str] | None = None,
        limit: int | None = None,
        deadline_ms: int | None = None,
    ) -> dict[str, Any]:
        """``POST /alerts/{id}/pivot``: other readable records for one evidence entity."""
        body = _pivot_body(field, value, target_field, target_types, limit, deadline_ms)
        return dict(self._t.post(_aid(alert_id, "pivot"), body))

    def create_saved_search(
        self,
        name: str,
        sql: str,
        *,
        purpose: str | None = None,
        description: str | None = None,
        params: Sequence[Any] | None = None,
    ) -> dict[str, Any]:
        """``POST /searches/saved``: ``409 saved-search-exists`` if the name is taken."""
        body = _search_body(name, sql, purpose, description, params, None)
        return dict(self._t.post("/searches/saved", body))

    def list_saved_searches(
        self, *, q: str | None = None, limit: int | None = None
    ) -> dict[str, Any]:
        """``GET /searches/saved``: ``{"items": [...], "total": n}`` ordered by name."""
        return dict(self._t.get(_list_path(q, limit)))

    def get_saved_search(self, search_id: str) -> dict[str, Any]:
        """``GET /searches/saved/{id}``."""
        return dict(self._t.get(_sid(search_id)))

    def update_saved_search(
        self,
        search_id: str,
        name: str,
        sql: str,
        *,
        purpose: str | None = None,
        description: str | None = None,
        params: Sequence[Any] | None = None,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        """``PUT /searches/saved/{id}``; ``expected_version`` makes it conditional."""
        body = _search_body(name, sql, purpose, description, params, expected_version)
        return dict(self._t.put(_sid(search_id), body))

    def delete_saved_search(self, search_id: str) -> dict[str, Any]:
        """``DELETE /searches/saved/{id}``: leaves listings; history is kept."""
        return dict(self._t.delete(_sid(search_id)))

    def run_saved_search(
        self, search_id: str, *, params: Sequence[Any] | None = None
    ) -> dict[str, Any]:
        """``POST /searches/saved/{id}/run``: the ``POST /query`` response for the stored SQL."""
        body: dict[str, Any] = {} if params is None else {"params": list(params)}
        return dict(self._t.post(_sid(search_id, "run"), body))

    def close(self) -> None:
        self._t.close()

    def __enter__(self) -> InvestigationClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class AsyncInvestigationClient:
    """Asynchronous client -- see :class:`InvestigationClient`."""

    def __init__(
        self,
        base_url: str,
        *,
        bearer_token: str | None = None,
        timeout: float = 60.0,
        extra_headers: dict[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._t = AsyncHttpTransport(
            base_url, bearer_token, timeout, transport=transport, extra_headers=extra_headers
        )

    @classmethod
    def from_client(cls, client: RelataClient) -> AsyncInvestigationClient:
        return cls(
            client._base_url,  # noqa: SLF001
            bearer_token=client._bearer_token,  # noqa: SLF001
            timeout=client._timeout,  # noqa: SLF001
            extra_headers=client._extra_headers,  # noqa: SLF001
        )

    async def alert_evidence(
        self, alert_id: str, *, max_events: int | None = None, deadline_ms: int | None = None
    ) -> dict[str, Any]:
        return dict(await self._t.get(_aid(alert_id, "evidence") + _bounds(max_events, deadline_ms)))

    async def alert_timeline(
        self, alert_id: str, *, max_events: int | None = None, deadline_ms: int | None = None
    ) -> dict[str, Any]:
        return dict(await self._t.get(_aid(alert_id, "timeline") + _bounds(max_events, deadline_ms)))

    async def pivot(
        self,
        alert_id: str,
        *,
        field: str | None = None,
        value: Any = None,
        target_field: str | None = None,
        target_types: Sequence[str] | None = None,
        limit: int | None = None,
        deadline_ms: int | None = None,
    ) -> dict[str, Any]:
        body = _pivot_body(field, value, target_field, target_types, limit, deadline_ms)
        return dict(await self._t.post(_aid(alert_id, "pivot"), body))

    async def create_saved_search(
        self,
        name: str,
        sql: str,
        *,
        purpose: str | None = None,
        description: str | None = None,
        params: Sequence[Any] | None = None,
    ) -> dict[str, Any]:
        body = _search_body(name, sql, purpose, description, params, None)
        return dict(await self._t.post("/searches/saved", body))

    async def list_saved_searches(
        self, *, q: str | None = None, limit: int | None = None
    ) -> dict[str, Any]:
        return dict(await self._t.get(_list_path(q, limit)))

    async def get_saved_search(self, search_id: str) -> dict[str, Any]:
        return dict(await self._t.get(_sid(search_id)))

    async def update_saved_search(
        self,
        search_id: str,
        name: str,
        sql: str,
        *,
        purpose: str | None = None,
        description: str | None = None,
        params: Sequence[Any] | None = None,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        body = _search_body(name, sql, purpose, description, params, expected_version)
        return dict(await self._t.put(_sid(search_id), body))

    async def delete_saved_search(self, search_id: str) -> dict[str, Any]:
        return dict(await self._t.delete(_sid(search_id)))

    async def run_saved_search(
        self, search_id: str, *, params: Sequence[Any] | None = None
    ) -> dict[str, Any]:
        body: dict[str, Any] = {} if params is None else {"params": list(params)}
        return dict(await self._t.post(_sid(search_id, "run"), body))

    async def close(self) -> None:
        await self._t.aclose()

    async def __aenter__(self) -> AsyncInvestigationClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()
