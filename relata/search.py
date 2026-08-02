"""Typed search client — T9 flagship surface (#1991, epic #1982).

Wraps the server's typed ``POST /search`` JSON query door (T1, #1983 — landed
on develop via #2045) with a Pythonic, retrieval-native surface. The door
compiles the typed body to a governed SQL plan via
``relata_query::json_query::compile`` and dispatches it through the existing
governed ``/query`` path, so PURPOSE / ACL / cell-masking / tenant isolation
apply identically to a hand-written query — **no SQL on the client side**.

The ergonomic entry point is :class:`~relata.namespace.Namespace`, reached via
``client.namespace("docs").query(text=..., filters=...)``. This module exposes
the lower-level :class:`SearchClient` for callers that want the typed search
surface without binding to a single namespace.

Server contract (``JsonSearchRequest``, #1983):

- ``from`` (alias ``type``) — object type.
- ``rank_by`` — ``["bm25"|"text", "<column>", "<query text>"]`` (BM25 full-text
  + ``ORDER BY _score DESC``), ``["field_weight", {"<field>": <weight>, ...},
  "<query text>"]`` (BM25F per-field weighting — ``MATCH(*, ...)`` +
  ``RANK BY FIELD_WEIGHT(...)``, #2676; e.g. ``rank_by=["field_weight",
  {"title": 3.0, "body": 1.0, "tags": 5.0}, "database"]``), or
  ``["vector", "ann", "<query text>"]`` (HYBRID_SEARCH). The typed shape is
  detected by the **absence** of a top-level ``query`` key — so this client
  never sends one for typed calls.
- ``filters`` — ``[{field, op, value}]`` with ops
  ``eq | ne | gt | gte | lt | lte | like | ilike | in | between``.
- ``include_attributes`` — column projection (default ``*``).
- ``consistency`` — ``"strong"`` (default) | ``"eventual"``.
- ``compute_attributes`` — side-output ranking signals per hit, e.g.
  ``{"bm25_score": "BM25()", "snippet": "HIGHLIGHT(bio)"}`` (#1985 T3, #2465).

The response is the governed ``/query`` envelope (``{rows, columns, ...}``),
returned here as a :class:`~relata.models.QueryResult`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from relata.client import RelataClient

from relata.models import QueryResult

# ops the server's json_query compiler accepts (#1983).
_SUPPORTED_OPS = frozenset(
    {"eq", "ne", "gt", "gte", "lt", "lte", "like", "ilike", "in", "between"}
)


def _normalise_filters(
    filters: list[dict[str, Any]] | dict[str, Any] | None,
) -> list[dict[str, Any]] | None:
    """Normalise ``filters`` to the server's ``[{field, op, value}]`` array form.

    A dict shorthand ``{"status": "x", "age": 30}`` becomes equality filters.
    ``op`` defaults to ``"eq"`` when omitted; unknown ops raise so a typo does
    not silently degrade to equality.
    """
    if filters is None:
        return None
    if isinstance(filters, dict):
        return [{"field": str(f), "op": "eq", "value": v} for f, v in filters.items()]
    if isinstance(filters, list):
        out: list[dict[str, Any]] = []
        for entry in filters:
            if not isinstance(entry, dict) or "field" not in entry:
                raise ValueError("each filter entry needs a 'field' key")
            op = str(entry.get("op", "eq"))
            if op not in _SUPPORTED_OPS:
                raise ValueError(
                    f"unsupported filter op {op!r}; expected one of "
                    f"{sorted(_SUPPORTED_OPS)}"
                )
            out.append({"field": str(entry["field"]), "op": op, "value": entry["value"]})
        return out
    raise TypeError(f"filters must be list | dict, got {type(filters).__name__}")


def _normalise_rank_by(rank_by: list[Any] | None) -> list[Any] | None:
    """Pass-through validation of a ``rank_by`` directive.

    The server accepts the array forms ``["bm25","col","text"]``,
    ``["text","col","text"]``, ``["field_weight",{"title":3.0,...},"text"]``
    (#2676), and ``["vector","ann","text"]``. Each element is sent verbatim;
    we only sanity-check the shape so a malformed directive fails fast on the
    client rather than as a 400 round-trip.
    """
    if rank_by is None:
        return None
    if not isinstance(rank_by, list):
        raise TypeError(f"rank_by must be a list, got {type(rank_by).__name__}")
    if not rank_by:
        return None
    mode = rank_by[0]
    if not isinstance(mode, str):
        raise ValueError(
            "rank_by[0] must be a string mode: 'bm25' | 'text' | 'field_weight' | 'vector'"
        )
    return rank_by


class SearchClient:
    """Synchronous typed search client — T1/T9 flagship surface (#1991).

    Construct with :meth:`from_client` so the query executes under the parent
    client's auth, tenant, and purpose context.
    """

    def __init__(self, client: RelataClient) -> None:
        self._client = client

    @classmethod
    def from_client(cls, client: RelataClient) -> SearchClient:
        return cls(client)

    def _purpose(self, purpose: str | None) -> str | None:
        return purpose or self._client._default_purpose  # noqa: SLF001

    def query(
        self,
        namespace: str,
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
        """Run a typed search against ``namespace`` (an object type).

        Args:
            namespace: The object type to search (e.g. ``"Document"``).
            text: Convenience for full-text BM25 — compiles to
                ``rank_by=["bm25", match_column, text]``. Mutually exclusive
                with an explicit ``rank_by``.
            match_column: Column for the ``text`` BM25 match (default ``"*"``
                = whole-row match, mirroring the legacy ``/search`` default).
            rank_by: Explicit ranking directive — ``["bm25","title","..."]``,
                ``["field_weight",{"title":3.0,...},"..."]`` (BM25F per-field
                weighting, #2676), ``["vector","ann","..."]``. See module
                docstring.
            filters: ``[{field, op, value}]`` (ops ``eq/ne/gt/gte/lt/lte/like/
                ilike/in/between``) or equality dict ``{field: value}``.
            limit: Max hits (server clamps to ``[1, 10000]``).
            include_attributes: Column projection (default ``*``).
            consistency: ``"strong"`` (default) | ``"eventual"``.
            compute_attributes: Side-output ranking signals per hit,
                ``{label: expr}`` e.g. ``{"bm25_score": "BM25()"}`` — compiles
                to a trailing ``COMPUTE`` clause (#1985 T3). Purely additive:
                never affects which rows match or their order.
            purpose: Optional purpose override.

        Returns:
            :class:`~relata.models.QueryResult` — the governed ``/query``
            response (rows in BM25-relevance order when ``text``/BM25 rank is
            used).
        """
        payload = _build_payload(
            namespace=namespace,
            text=text,
            match_column=match_column,
            rank_by=rank_by,
            filters=filters,
            limit=limit,
            include_attributes=include_attributes,
            consistency=consistency,
            compute_attributes=compute_attributes,
            purpose=self._purpose(purpose),
        )
        data = self._client._sync.post("/search", payload)  # noqa: SLF001
        return QueryResult.model_validate(data)

    async def aquery(
        self,
        namespace: str,
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
        """Async variant of :meth:`query`."""
        payload = _build_payload(
            namespace=namespace,
            text=text,
            match_column=match_column,
            rank_by=rank_by,
            filters=filters,
            limit=limit,
            include_attributes=include_attributes,
            consistency=consistency,
            compute_attributes=compute_attributes,
            purpose=self._purpose(purpose),
        )
        data = await self._client._async.post("/search", payload)  # noqa: SLF001
        return QueryResult.model_validate(data)


class AsyncSearchClient:
    """Asynchronous typed search client — see :class:`SearchClient`."""

    def __init__(self, client: RelataClient) -> None:
        self._inner = SearchClient(client)

    @classmethod
    def from_client(cls, client: RelataClient) -> AsyncSearchClient:
        return cls(client)

    async def query(
        self,
        namespace: str,
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
        return await self._inner.aquery(
            namespace,
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


def _build_payload(
    *,
    namespace: str,
    text: str | None,
    match_column: str,
    rank_by: list[Any] | None,
    filters: list[dict[str, Any]] | dict[str, Any] | None,
    limit: int,
    include_attributes: list[str] | None,
    consistency: str | None,
    compute_attributes: dict[str, str] | None = None,
    purpose: str | None,
) -> dict[str, object]:
    """Assemble the typed ``POST /search`` JSON body (no ``query`` key)."""
    if text is not None and rank_by is not None:
        raise ValueError("pass either `text` or an explicit `rank_by`, not both")

    rank_directive = _normalise_rank_by(rank_by)
    if rank_directive is None and text is not None:
        # Convenience: text + match_column → ["bm25", column, text].
        rank_directive = ["bm25", match_column, text]

    # NOTE: deliberately no top-level "query" key — its presence makes the
    # server treat the body as the legacy Meilisearch shape (#667) and ignore
    # the typed fields. The typed shape is detected by the ABSENCE of "query".
    payload: dict[str, object] = {"from": namespace}
    if rank_directive is not None:
        payload["rank_by"] = rank_directive
    norm_filters = _normalise_filters(filters)
    if norm_filters is not None:
        payload["filters"] = norm_filters
    payload["limit"] = limit
    if include_attributes:
        payload["include_attributes"] = include_attributes
    if consistency:
        payload["consistency"] = consistency
    if compute_attributes:
        payload["compute_attributes"] = compute_attributes
    if purpose:
        payload["purpose"] = purpose
    return payload
