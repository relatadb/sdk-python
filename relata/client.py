"""
Relata client — the primary entry point for the Python SDK.

Usage::

    from relata import RelataClient

    with RelataClient("http://localhost:9090", purpose="analytics") as client:
        result = client.query("SELECT * FROM Person LIMIT 10")
        for row in result:
            print(row)

Async usage::

    import asyncio
    from relata import RelataClient

    async def main():
        async with RelataClient("http://localhost:9090", purpose="analytics") as client:
            result = await client.aquery("SELECT * FROM Person LIMIT 10")
            for row in result:
                print(row)

    asyncio.run(main())
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, cast
from urllib.parse import urlencode

from relata._http import (
    DEFAULT_MAX_RESPONSE_BYTES,
    AsyncHttpTransport,
    HttpTransport,
    path_segment,
)
from relata.dns_tunnel import (
    DNS_TUNNEL_MAX_ROWS,
    DNS_TUNNEL_PAGE_SIZE,
    DNS_TUNNEL_TYPES,
    dns_tunnel_result,
)
from relata.exceptions import PurposeError, RelataError
from relata.models import (
    AlertSubmitReceipt,
    AlertSubmitRequest,
    AuditCountResponse,
    EventAppendReceipt,
    EventAppendRequest,
    HealthResponse,
    IngestDocumentResponse,
    IngestDocumentTaskStatus,
    QueryResult,
    ReadyReport,
    SearchResponse,
    Stats,
    StatusResponse,
    VersionInfo,
)

if TYPE_CHECKING:
    from relata.namespace import Namespace
    from relata.query import QueryBuilder




#: Valid ``x-query-dialect`` header values (#3265). ``None`` omits the header
#: and lets the server auto-detect.
_VALID_DIALECTS = ("sql", "cypher", "gql")

#: :meth:`RelataClient.schema_alter`'s documented short action verbs, mapped
#: to the server's long-form verbs (`crates/relata-cli/src/serve/types_routes.rs`
#: ``schema_alter_handler``). The server only ever understood the long form —
#: this SDK previously sent the short verbs through unchanged, so every call
#: following the method's own docstring 400'd (#4704).
_SCHEMA_ALTER_ACTIONS = {
    "add": "add_column",
    "drop": "remove_column",
    "rename": "rename_column",
    "retype": "change_type",
}


def _dialect_headers(dialect: str | None) -> dict[str, str] | None:
    """Build the ``x-query-dialect`` header for :meth:`RelataClient.query`.

    Returns ``None`` when ``dialect`` is ``None`` (server auto-detection).
    Raises :class:`RelataError` for an unknown dialect so a typo fails fast
    client-side instead of surfacing as an opaque server 4xx.
    """
    if dialect is None:
        return None
    d = dialect.strip().lower()
    if d not in _VALID_DIALECTS:
        raise RelataError(
            f"unknown query dialect '{dialect}'; expected one of {', '.join(_VALID_DIALECTS)}"
        )
    return {"x-query-dialect": d}


def _rewrite_question_mark_params(sql: str) -> str:
    """Rewrite ``?`` placeholders to ``$1``, ``$2``, … left-to-right.

    This lets callers use the more familiar ``?`` form; the server only
    understands ``$N`` positional parameters (#1162).
    """
    import re

    counter = 0

    def _replace(m: re.Match[str]) -> str:
        nonlocal counter
        counter += 1
        return f"${counter}"

    return re.sub(r"\?", _replace, sql)


_MATCHER_HINTS: dict[str, str] = {
    "vf3": "/*+ matcher=vf3 */",
    "bfs": "/*+ matcher=bfs */",
}


def _inject_matcher_hint(sql: str, matcher: str | None) -> str:
    """Prefix ``sql`` with a ``/*+ matcher=… */`` hint comment (#1189).

    ``matcher`` selects the subgraph matcher for multi-hop Cypher patterns.
    ``None`` and ``"auto"`` inject nothing, leaving the server's default
    selection in place. ``"vf3"`` / ``"bfs"`` force the corresponding engine.
    Any other value raises :class:`ValueError`.
    """
    if matcher is None or matcher == "auto":
        return sql
    hint = _MATCHER_HINTS.get(matcher)
    if hint is None:
        raise ValueError(
            f"matcher must be 'auto', 'vf3', or 'bfs', got {matcher!r}"
        )
    return f"{hint} {sql}"


def _sql_literal(s: str) -> str:
    """Quote ``s`` as a SQL string literal, doubling internal single quotes."""
    return "'" + s.replace("'", "''") + "'"


def _similar_image_sql(
    media_ref: str,
    *,
    threshold: float | None,
    index: str | None,
) -> str:
    """Build a bare ``SIMILAR_IMAGE(...)`` ticket (#2840, PR #2859, #5497).

    Mirrors the server operator registered in ``relata_query::parser``
    (``SIMILAR_IMAGE('<media_ref>', THRESHOLD => <f>, INDEX => '<corpus>')``
    — ``THRESHOLD`` defaults to 0.9 server-side when omitted, ``INDEX`` is
    optional with no server default and scopes the search to a named
    corpus/index when given). Bare top-level form only — see
    the #5497 bare-form note: the engine dispatches it as a bare top-level
    statement only.
    """
    sql = f"SIMILAR_IMAGE({_sql_literal(media_ref)}"
    if threshold is not None:
        sql += f", THRESHOLD => {float(threshold)}"
    if index is not None:
        sql += f", INDEX => {_sql_literal(index)}"
    sql += ")"
    return sql






def _graphql_data(resp: dict[str, object]) -> dict[str, object] | list[dict[str, object]] | None:
    """Return the ``data`` field of a GraphQL envelope, raising on ``errors``.

    The server wraps every ``/v1/graphql`` response in ``{"data": …, "errors": […]}``.
    On a non-empty ``errors`` array, raise :class:`RelataError` carrying the
    first error message; otherwise return ``data``.
    """
    errors = cast("list[object]", resp.get("errors") or [])
    if errors:
        first = errors[0]
        msg = first.get("message", "graphql error") if isinstance(first, dict) else str(first)
        raise RelataError(f"graphql: {msg}")
    return cast("dict[str, object] | list[dict[str, object]] | None", resp.get("data"))



def _with_object_type(resp: SearchResponse, object_type: str) -> SearchResponse:
    """The wire hit has no ``object_type`` (the caller named it in the request); fill it in."""
    for hit in resp.hits:
        if not hit.object_type:
            hit.object_type = object_type
    return resp


class RelataClient:
    """Synchronous and asynchronous client for the Relata HTTP API.

    ``RelataClient`` supports both synchronous calls (via :meth:`query`,
    :meth:`health`, etc.) and their async counterparts (via :meth:`aquery`,
    :meth:`ahealth`, etc.).  Use it as a context manager — sync or async — to
    guarantee connection cleanup.

    Args:
        base_url: Base URL of the Relata server, e.g.
            ``"http://localhost:9090"``. Trailing slashes are stripped.
        bearer_token: Optional Bearer token sent as
            ``Authorization: Bearer <token>``.  Required when the server is
            configured with ``RELATA_BEARER_TOKEN``.
        purpose: Default purpose string for all queries made through this
            client.  Can be overridden per-call.  If ``None`` you must pass
            ``purpose=`` to every :meth:`query` call.  Common values:
            ``"analytics"``, ``"audit"``, ``"analysis"``.
        timeout: HTTP request timeout in seconds (default ``30.0``).
        tenant: Optional tenant / organisation id sent as ``X-Relata-Tenant-Id``
            on every request.  Required for multi-tenant deployments.  Set once
            on the client; there is no per-method ``tenant=`` override (#5821).
            ``bulk(tenant_id=)`` is the only per-call tenant argument.
        acting_as: Optional delegation principal sent as ``X-Acting-As`` — the
            caller asserts membership and the server's ``wire_acting_as()``
            parses it (#55).  Pairs with ``delegated_by``.
        delegated_by: Optional delegation chain root sent as ``X-Delegated-By``.
        headers: Optional dict of arbitrary HTTP headers overlaid on every
            request (e.g. ``{"X-Request-ID": "..."}`` for correlation, or
            ``{"X-Verified-Principal": "..."}`` for proxy-trust deployments).
            Caller-supplied headers win over the SDK defaults.
        admin_base_url: Base URL of the loopbound admin control-plane
            listener (``RELATA_ADMIN_BIND``, default ``127.0.0.1:9091``).
            Per ADR-0261, ``/admin/*`` routes are mounted **only** there on a
            hardened deployment -- set this so
            :class:`~relata.backup.BackupClient` reaches them. Leave unset (the default) when the
            admin listener isn't split from the data plane (e.g. local/free-profile
            dev) -- every request then goes to ``base_url`` unchanged.
        max_response_bytes: Cap on how much of any single response body the
            client will buffer (default ~64 MiB, #3214). A larger body raises
            :class:`~relata.exceptions.ResponseTooLargeError` instead of being
            read — defending the client from OOM via a malicious/buggy server.

    Raises:
        :class:`~relata.exceptions.AuthError`: When the server rejects the
            bearer token (HTTP 401).
        :class:`~relata.exceptions.PurposeError`: When a query is submitted
            without a purpose and no default is set.
        :class:`~relata.exceptions.QuotaError`: When the per-principal cost
            quota is exhausted (HTTP 429).
        :class:`~relata.exceptions.ConnectionError`: When the server is
            unreachable.

    Examples::

        # Minimal usage
        client = RelataClient("http://localhost:9090", purpose="analytics")
        result = client.query("SELECT * FROM Person LIMIT 5")
        client.close()

        # Preferred: context manager
        with RelataClient("http://localhost:9090", purpose="analytics") as client:
            result = client.query("SELECT * FROM Person LIMIT 5")

        # With authentication
        with RelataClient(
            "https://relata.example.com",
            bearer_token="s3cr3t",
            purpose="analytics",
        ) as client:
            result = client.query("SELECT * FROM Warrant LIMIT 10")
    """

    def __init__(
        self,
        base_url: str,
        *,
        bearer_token: str | None = None,
        purpose: str | None = None,
        timeout: float = 30.0,
        tenant: str | None = None,
        acting_as: str | None = None,
        delegated_by: str | None = None,
        headers: dict[str, str] | None = None,
        max_retries: int = 0,
        retry_backoff_secs: float = 0.5,
        compress: bool = False,
        admin_base_url: str | None = None,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        # #2321 (ADR-0261): /admin/* is mounted only on the loopbound admin
        # control-plane listener (RELATA_ADMIN_BIND, default 127.0.0.1:9091)
        # -- never this client's main base_url on a hardened deployment.
        # Set admin_base_url to point BackupClient at that listener; leave
        # unset (the default) when the admin listener isn't split from the
        # data plane (e.g. local/free-profile dev) -- unchanged behaviour.
        self._admin_base_url = admin_base_url.rstrip("/") if admin_base_url else None
        self._bearer_token = bearer_token
        self._default_purpose = purpose
        self._timeout = timeout
        self._tenant = tenant
        self._acting_as = acting_as
        self._delegated_by = delegated_by
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff_secs
        self._compress = compress
        # #3214: per-response buffering cap — a malicious/buggy server cannot
        # OOM the client (default ~64 MiB; parity with Rust #3198).
        self._max_response_bytes = max_response_bytes

        # Compose the static extra-headers bag: tenant/delegation first, then
        # caller-supplied headers win. Per-call overrides (acting_as=, request_id=)
        # are merged in transport-property accessors below; tenant is client-level
        # only (#5821).
        extra: dict[str, str] = {}
        if tenant is not None:
            extra["X-Relata-Tenant-Id"] = tenant
        if acting_as is not None:
            extra["X-Acting-As"] = acting_as
        if delegated_by is not None:
            extra["X-Delegated-By"] = delegated_by
        if headers:
            extra.update(headers)
        self._extra_headers: dict[str, str] | None = extra or None

        # Lazily initialised transports — created on first use so the client
        # can be constructed without immediately opening a connection.
        self.__sync_transport: HttpTransport | None = None
        self.__async_transport: AsyncHttpTransport | None = None

    # ------------------------------------------------------------------
    # Internal transport accessors
    # ------------------------------------------------------------------

    @property
    def _sync(self) -> HttpTransport:
        if self.__sync_transport is None:
            self.__sync_transport = HttpTransport(
                self._base_url,
                self._bearer_token,
                self._timeout,
                extra_headers=self._extra_headers,
                max_retries=self._max_retries,
                retry_backoff=self._retry_backoff,
                compress=self._compress,
                admin_base_url=self._admin_base_url,
                max_response_bytes=self._max_response_bytes,
            )
        return self.__sync_transport

    @property
    def _async(self) -> AsyncHttpTransport:
        if self.__async_transport is None:
            self.__async_transport = AsyncHttpTransport(
                self._base_url,
                self._bearer_token,
                self._timeout,
                extra_headers=self._extra_headers,
                max_retries=self._max_retries,
                retry_backoff=self._retry_backoff,
                compress=self._compress,
                admin_base_url=self._admin_base_url,
                max_response_bytes=self._max_response_bytes,
            )
        return self.__async_transport

    # ------------------------------------------------------------------
    # Purpose resolution
    # ------------------------------------------------------------------

    def _resolve_purpose(self, purpose: str | None) -> str:
        """Return the effective purpose, raising :class:`PurposeError` if unset."""
        effective = purpose or self._default_purpose
        if not effective:
            raise PurposeError(
                "Every Relata query must declare a purpose "
                "(e.g. 'analytics', 'audit', 'analysis'). "
                "Pass purpose= to query() or set a default: "
                "RelataClient(base_url, purpose='analytics')."
            )
        return effective

    # ------------------------------------------------------------------
    # Context manager — sync
    # ------------------------------------------------------------------

    def __enter__(self) -> RelataClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Context manager — async
    # ------------------------------------------------------------------

    async def __aenter__(self) -> RelataClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # Resource cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying sync HTTP connection pool and drop the bearer
        credential (#3214). The client is not usable after this."""
        if self.__sync_transport is not None:
            self.__sync_transport.close()
            self.__sync_transport = None
        # The async transport may still be open (two-phase teardown) — strip
        # its Authorization header so the secret does not linger there either.
        if self.__async_transport is not None:
            self.__async_transport.clear_credentials()
        # Zeroize: drop our reference to the secret so it does not linger on
        # the (long-lived) client object after shutdown.
        self._bearer_token = None

    async def aclose(self) -> None:
        """Close the underlying async HTTP connection pool and drop the bearer
        credential (#3214). The client is not usable after this."""
        if self.__async_transport is not None:
            await self.__async_transport.aclose()
            self.__async_transport = None
        if self.__sync_transport is not None:
            self.__sync_transport.clear_credentials()
        self._bearer_token = None

    # ------------------------------------------------------------------
    # Sync API
    # ------------------------------------------------------------------

    def query(
        self,
        sql: str,
        *,
        purpose: str | None = None,
        dialect: str | None = None,
        matcher: Literal["auto", "vf3", "bfs"] | None = None,
        query_vector: list[float] | None = None,
        query_vector_model: str | None = None,
    ) -> QueryResult:
        """Execute a governed SQL query against the Relata engine (synchronous).

        This generic client method sends ``POST /v1/query``; the language-neutral
        HTTP contract is published at ``GET /openapi.json``. The TypeScript
        ``RelataClient.query`` and Go ``Client.Query`` methods expose the same
        generic query path. Use ordinary SELECT statements to retrieve source
        rows for application-owned analysis.

        Args:
            sql: SQL statement to execute.  Relata extends standard SQL with:

                - ``AS OF 'timestamp'`` — bi-temporal point-in-time queries
                - ``WITH PROVENANCE`` — include PROV-O columns in results
                - ``PATHS_BETWEEN(a, b, max_hops => 4)`` — graph traversal
                - ``MATCH_FACE(face_embedding, col, 0.92)`` — face similarity
                - ``LOOKUP_IDENTITY('value')`` — resolve any identifier
                - ``HYBRID_SCORE(query)`` — BM25 + vector hybrid search

                With ``dialect="cypher"`` or ``dialect="gql"`` this carries a
                Cypher / GQL (ISO/IEC 39075) statement instead of SQL.
            purpose: Purpose for this query.  Overrides the client-level
                default.  Must be a value registered in the tenant's
                ``PurposeRegistry``.  Common values: ``"analytics"``,
                ``"audit"``, ``"analysis"``, ``"monitoring"``.
            dialect: Optional query-language override sent as the
                ``x-query-dialect`` header — ``"sql"``, ``"cypher"``, or
                ``"gql"`` (#3265). ``None`` (default) lets the server
                auto-detect (a ``MATCH``-prefixed body is treated as Cypher).
                GQL is only selected when this header is set; GQL-status
                errors surface as :class:`~relata.exceptions.RelataError`
                (42G04 syntax → 400, 0A501 unsupported feature → 501).
            matcher: Subgraph-matcher hint for multi-hop Cypher patterns
                (#1189).  ``None`` / ``"auto"`` let the server choose;
                ``"vf3"`` forces the VF3 subgraph-isomorphism matcher and
                ``"bfs"`` the chained-BFS fallback.  Injected as a
                ``/*+ matcher=… */`` comment; BFS is also the automatic
                fallback for patterns VF3 cannot handle (e.g. unbounded
                ``*`` variable-length paths).
            query_vector: Ready query embedding (at most 8192 floats) for
                ``HYBRID_SEARCH`` / ``RAG_RETRIEVE``. When set the server makes
                no embedding call and needs no embedding endpoint (ADR-0309).
            query_vector_model: Model tag ``query_vector`` was produced with; a
                tag matching none of the type's indexed models is HTTP 409
                ``query-embedding-model-mismatch``.

        Returns:
            :class:`~relata.models.QueryResult` — iterable over result rows.

        Raises:
            :class:`~relata.exceptions.PurposeError`: No purpose set.
            :class:`~relata.exceptions.QuotaError`: Quota exhausted.
            :class:`~relata.exceptions.AuthError`: Invalid/missing token.
            :class:`~relata.exceptions.ConnectionError`: Server unreachable.
            :class:`~relata.exceptions.RelataError`: Any other server error.

        Examples::

            result = client.query(
                "SELECT * FROM Person WHERE name LIKE 'Ahmed%' LIMIT 10",
                purpose="analytics",
            )
            for row in result:
                print(row["name"], row["dob"])

            # Bi-temporal query
            result = client.query(
                "SELECT * FROM BankAccount AS OF '2024-06-01' WHERE balance > 1000000"
            )

            # Graph traversal
            result = client.query(
                "SELECT * FROM PATHS_BETWEEN('person-123', 'org-456', max_hops => 4)"
            )

            # Multi-hop Cypher with an explicit VF3 matcher hint
            result = client.query(
                "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) "
                "RETURN a, b, c",
                matcher="vf3",
            )
        """
        effective_purpose = self._resolve_purpose(purpose)
        payload: dict[str, object] = {
            "purpose": effective_purpose,
            "sql": _inject_matcher_hint(sql, matcher),
        }
        if query_vector is not None:
            payload["query_vector"] = query_vector
        if query_vector_model is not None:
            payload["query_vector_model"] = query_vector_model
        extra_headers = _dialect_headers(dialect)
        data = self._sync.post("/v1/query", payload, extra_headers=extra_headers)
        return QueryResult.model_validate(data)

    def query_params(
        self,
        sql: str,
        params: list[Any],
        *,
        purpose: str | None = None,
        matcher: Literal["auto", "vf3", "bfs"] | None = None,
    ) -> QueryResult:
        """Execute a parameterized SQL query (synchronous, #1162).

        Positional placeholders ``$1``, ``$2``, … in ``sql`` are substituted
        server-side with the corresponding values from ``params``.
        Pass ``?`` placeholders instead and they will be rewritten to ``$1``,
        ``$2``, … before the request is sent.

        Args:
            sql: SQL with ``$N`` or ``?`` positional placeholders.
            params: Values to bind in order. ``None`` maps to SQL ``NULL``.
            purpose: Purpose override for this call.
            matcher: Subgraph-matcher hint for multi-hop Cypher patterns
                (#1189).  See :meth:`query`.

        Returns:
            :class:`~relata.models.QueryResult`

        Examples::

            result = client.query_params(
                "SELECT * FROM Person WHERE age = $1 AND city = $2",
                [25, "Karachi"],
                purpose="analytics",
            )

            # ? form — rewritten to $1, $2, … automatically
            result = client.query_params("SELECT * FROM T WHERE id = ?", [42])
        """
        sql = _rewrite_question_mark_params(sql)
        effective_purpose = self._resolve_purpose(purpose)
        payload = {
            "purpose": effective_purpose,
            "sql": _inject_matcher_hint(sql, matcher),
            "params": params,
        }
        data = self._sync.post("/v1/query", payload)
        return QueryResult.model_validate(data)

    def query_arrow(
        self,
        sql: str,
        *,
        purpose: str | None = None,
    ) -> pyarrow.Table:  # type: ignore[name-defined]  # noqa: F821
        """Execute a SQL query and return results as a ``pyarrow.Table`` (#1744).

        Calls ``POST /v1/query/arrow`` which streams Arrow IPC binary.  Requires
        ``pyarrow`` to be installed (``pip install pyarrow``).

        Args:
            sql: SQL query string.
            purpose: Purpose override.

        Returns:
            A ``pyarrow.Table``.

        Example::

            tbl = client.query_arrow("SELECT * FROM Person LIMIT 1000", purpose="analytics")
            df = tbl.to_pandas()
        """
        import io

        import pyarrow.ipc as ipc  # type: ignore[import-not-found,unused-ignore]

        from relata.streaming import StreamingClient

        effective_purpose = self._resolve_purpose(purpose)
        chunks = StreamingClient.from_client(self).query_arrow_raw(sql, purpose=effective_purpose)
        buf = io.BytesIO(b"".join(chunks))
        return ipc.open_stream(buf).read_all()







    def search(
        self,
        query: str,
        type: str,  # noqa: A002 — mirrors server API field name
        *,
        limit: int | None = None,
        facets: list[str] | None = None,
        highlight: bool = False,
        filters: dict[str, str] | None = None,
        matching_strategy: str | None = None,
        typo_tolerance: dict[str, object] | None = None,
        metric: str | None = None,
        weights: list[float] | None = None,
    ) -> SearchResponse:
        """Full-text search over a governed object type (#670).

        By default this is **BM25-only** — it never touches the vector
        channel. Pass ``metric`` and/or ``weights`` to route the request
        through the server's real HYBRID_SEARCH fusion (BM25 + vector
        reciprocal-rank fusion, #1330/#1338) instead; either one alone is
        enough (`search_handler` in `crates/relata-cli/src/serve.rs` only
        builds the `HYBRID_SEARCH` statement when at least one is set — #2672).

        Args:
            query: Free-text search string.
            type: Object type to search (e.g. ``"Person"``).
            limit: Maximum number of hits to return (server default: 20).
            facets: Field names to aggregate counts for.
            highlight: Include field-level ``<em>`` snippets.
            filters: Equality filters applied server-side (``{"field": "val"}``).
            matching_strategy: ``"all"`` (AND), ``"last"``, ``"frequency"``,
                ``"boolean"``, or ``"any"`` (OR, default). Controls which query
                terms are required (#967). ``"boolean"`` (#3263) interprets
                uppercase ``AND``/``OR``/``NOT`` operators in the query text as
                posting-list set operations (left-associative; a bare space
                means OR): ``"alice AND bob NOT eve"``.
            typo_tolerance: Dict with ``enabled``, ``min_word_size``,
                ``disable_on_words``, ``disable_on_attributes`` (#967).
            metric: Vector distance metric for the HYBRID_SEARCH channel
                (e.g. ``"cosine"``, ``"euclidean"``, ``"dot"``). Setting this
                (or ``weights``) is what actually triggers hybrid fusion (#2672).
            weights: Three-element ``[graph, bm25, vector]`` fusion weights for
                HYBRID_SEARCH (#1338). Setting this (or ``metric``) is what
                actually triggers hybrid fusion (#2672).

        Returns:
            :class:`~relata.models.SearchResponse` with ``hits``, ``total``,
            ``facets``, ``estimated_total_hits``, and ``processing_time_ms``.

        Example::

            # Pure BM25 — never touches the vector channel.
            results = client.search(
                "alice smith", "Person",
                limit=10, facets=["tenant_id"], highlight=True,
                matching_strategy="all",
            )
            for hit in results.hits:
                print(hit.score, hit.fields.get("name"))

            # Real hybrid fusion (BM25 + vector RRF) — set metric/weights.
            hybrid = client.search(
                "alice smith", "Person",
                metric="cosine", weights=[0.0, 0.5, 0.5],
            )
        """
        payload: dict[str, object] = {"query": query, "type": type, "highlight": highlight}
        if limit is not None:
            payload["limit"] = limit
        if facets:
            payload["facets"] = facets
        if filters:
            payload["filters"] = filters
        if matching_strategy is not None:
            payload["matching_strategy"] = matching_strategy
        if typo_tolerance is not None:
            payload["typo_tolerance"] = typo_tolerance
        if metric is not None:
            payload["metric"] = metric
        if weights is not None:
            payload["weights"] = weights
        data = self._sync.post("/v1/search", payload)
        return _with_object_type(SearchResponse.model_validate(data), type)

    async def asearch(
        self,
        query: str,
        type: str,  # noqa: A002
        *,
        limit: int | None = None,
        facets: list[str] | None = None,
        highlight: bool = False,
        filters: dict[str, str] | None = None,
        matching_strategy: str | None = None,
        typo_tolerance: dict[str, object] | None = None,
        metric: str | None = None,
        weights: list[float] | None = None,
    ) -> SearchResponse:
        """Async variant of :meth:`search`.

        Like :meth:`search`, this stays BM25-only unless ``metric`` and/or
        ``weights`` are supplied, in which case it routes through the
        server's HYBRID_SEARCH fusion path (#2672).
        """
        payload: dict[str, object] = {"query": query, "type": type, "highlight": highlight}
        if limit is not None:
            payload["limit"] = limit
        if facets:
            payload["facets"] = facets
        if filters:
            payload["filters"] = filters
        if matching_strategy is not None:
            payload["matching_strategy"] = matching_strategy
        if typo_tolerance is not None:
            payload["typo_tolerance"] = typo_tolerance
        if metric is not None:
            payload["metric"] = metric
        if weights is not None:
            payload["weights"] = weights
        data = await self._async.post("/v1/search", payload)
        return _with_object_type(SearchResponse.model_validate(data), type)

    def multi_search(
        self,
        queries: list[dict[str, object]],
    ) -> dict[str, object]:
        """Federated multi-query search (#967).

        Args:
            queries: List of dicts, each with ``query``, ``type``, and optional
                ``limit``, ``matching_strategy``, ``typo_tolerance``.

        Returns:
            Dict with ``results`` (list of per-query responses) and
            ``processing_time_ms``.
        """
        return self._sync.post("/v1/search/multi", {"queries": queries})

    async def amulti_search(
        self,
        queries: list[dict[str, object]],
    ) -> dict[str, object]:
        """Async variant of :meth:`multi_search`."""
        return await self._async.post("/v1/search/multi", {"queries": queries})



    # ------------------------------------------------------------------
    # Type management & ontology (#967)
    # ------------------------------------------------------------------

    def list_types(self) -> dict[str, object]:
        """List all registered object types with row counts."""
        return self._sync.get("/v1/types")

    def query_schema(self, *, purpose: str | None = None) -> dict[str, object]:
        """Return names-only schema visible to this authenticated principal.

        Intended for application-owned natural-language query composition.
        Any generated statement must still go through :meth:`query` so the
        server applies its normal authorization and validation.
        """
        selected = purpose or self._default_purpose
        suffix = f"?{urlencode({'purpose': selected})}" if selected else ""
        return self._sync.get(f"/v1/query/schema{suffix}")

    def register_type(self, name: str, **kwargs: object) -> dict[str, object]:
        """Register a custom object type at runtime (``POST /v1/types``).

        ``kwargs`` (all optional) mirrors the server's full ``TypeDef`` body
        (``crates/relata-cli/src/serve/types_routes.rs``):

        - ``description`` (``str``) -- free-text note, not schema-enforced.
        - ``owner`` (``str``) -- tenant/agency owning this type.
        - ``properties`` (``list[dict]``) -- each ``{"name": str, "required":
          bool, "state_machine": {"initial_state": str, "transitions":
          [{"from": str, "to": str}, ...]}}``. ``state_machine`` constrains
          that one property's value transitions.
        - ``computed_columns`` (``list[dict]``) -- each ``{"name": str,
          "kind": "concat" | "static", "fields": list[str], "separator":
          str, "value": str}``.
        - ``graph_triggers`` (``list[dict]``) -- each ``{"link_type": str,
          "src_field": str, "dst_field": str}``: on every row of this type,
          materialise a ``LinkStore`` edge from the ``src_field`` value to
          the ``dst_field`` value, typed ``link_type`` -- the mechanism
          behind automatic graph wiring on ingest.
        - ``bm25_params`` (``dict``) -- ``{"k1": float, "b": float}``,
          overrides the process-wide full-text scoring preset for this type
          only; both fields are required together.
        - ``force`` (``bool``) -- required to redefine a ``computed_columns``
          formula or ``state_machine`` on a type that already has rows
          (otherwise the server returns 409); existing rows are never
          retroactively recomputed/revalidated.
        """
        payload: dict[str, object] = {"name": name, **kwargs}
        return self._sync.post("/v1/types", payload)

    def deregister_type(self, name: str) -> dict[str, object]:
        """Deregister an empty custom type. Admin token required; types with history or protected
        data are rejected.
        """
        return self._sync.delete(f"/v1/types/{path_segment(name)}")

    def type_detail(self, name: str) -> dict[str, object]:
        """Get type detail (properties, owner, row count)."""
        return self._sync.get(f"/v1/types/{path_segment(name)}")

    def type_lifecycle(self, name: str) -> dict[str, object]:
        """Lifecycle progress of a type (#5724): the durable journal of its
        register/alter/deregister outcomes (``applied`` / ``refused`` / ``failed`` /
        ``in_progress`` / ``interrupted``, surviving restart and restore) plus
        per-index derived-index progress and a freshness token."""
        return self._sync.get(f"/v1/types/{path_segment(name)}/lifecycle")

    def schema_alter(
        self,
        type_name: str,
        action: str,
        column: str,
        *,
        new_column: str | None = None,
        col_type: str | None = None,
        optional: bool | None = None,
        expected_schema_version: str | None = None,
    ) -> dict[str, object]:
        """Online schema evolution — ``PATCH /v1/types/:name/schema`` (#1307).

        ``action`` is ``"add"`` | ``"drop"`` | ``"rename"`` | ``"retype"``;
        ``column`` is the target column; ``new_column`` / ``col_type`` /
        ``optional`` apply to the relevant actions. The short verbs above are
        translated to the server's ``add_column`` / ``remove_column`` /
        ``rename_column`` / ``change_type`` on the wire (#4704) — the
        server's own long-form verbs are also accepted unchanged, for
        callers who prefer to pass them directly.

        ``expected_schema_version`` (from ``type_detail(name)["schema_version"]``
        or a prior alter) makes the call a compare-and-set: a stale value is
        refused with 412 ``schema-version-conflict`` and nothing changes (#5724).
        """
        payload: dict[str, object] = {
            "action": _SCHEMA_ALTER_ACTIONS.get(action, action),
            "column": column,
        }
        if new_column is not None:
            payload["new_column"] = new_column
        if col_type is not None:
            payload["col_type"] = col_type
        if optional is not None:
            payload["optional"] = optional
        if expected_schema_version is not None:
            payload["expected_schema_version"] = expected_schema_version
        return self._sync.patch(f"/v1/types/{path_segment(type_name)}/schema", payload)

    # ------------------------------------------------------------------
    # Schema-BRANCH CRUD + typed edge definitions (#2497, #2476 ALTER stays
    # distinct — these are the schema-BRANCH surface implemented by this
    # supported client.
    # ------------------------------------------------------------------

    def list_edge_types(self) -> dict[str, object]:
        """List registered cross-type edge definitions (ADR-007).

        Wraps ``GET /v1/types/edges``. Returns the server's edge-declaration
        table consumed by ``PATHS_BETWEEN`` traversal.
        """
        return self._sync.get("/v1/types/edges")

    def list_type_detect_configs(self) -> dict[str, object]:
        """List per-type SmartIngest detector-pack configs (``GET /v1/types/detect-config``)."""
        return self._sync.get("/v1/types/detect-config")

    def set_type_detect_config(self, type_name: str, packs: list[str]) -> dict[str, object]:
        """Configure SmartIngest detector packs for a type (admin token required).

        Wraps ``POST /v1/types/detect-config`` with ``{type, packs}``; ``packs``
        takes the ``RELATA_DETECT_PACKS`` grammar (``["all"]``, ``["none"]``,
        ``[]`` for none, or pack names).
        """
        return self._sync.post("/v1/types/detect-config", {"type": type_name, "packs": packs})

    def register_edge_type(
        self,
        from_type: str,
        to_type: str,
        label: str,
    ) -> dict[str, object]:
        """Register a typed named edge for graph traversal (ADR-007).

        Wraps ``POST /v1/types/edges`` with ``{from_type, to_type, label}``.
        """
        return self._sync.post(
            "/v1/types/edges",
            {"from_type": from_type, "to_type": to_type, "label": label},
        )

    def create_schema_branch(
        self,
        name: str,
        from_branch: str,
    ) -> dict[str, object]:
        """Create a named schema branch off ``from_branch`` (#207).

        Wraps ``POST /v1/schema/branches/:name`` with ``{"from": from_branch}``.
        The branch carries its own ontology / type set; mutations on it do not
        affect the parent until merged.
        """
        return self._sync.post(
            f"/v1/schema/branches/{path_segment(name)}",
            {"from": from_branch},
        )

    def delete_schema_branch(self, name: str) -> dict[str, object]:
        """Delete a named schema branch (#207).

        Wraps ``DELETE /v1/schema/branches/:name``.
        """
        return self._sync.delete(f"/v1/schema/branches/{path_segment(name)}")

    async def alist_edge_types(self) -> dict[str, object]:
        """Async variant of :meth:`list_edge_types`."""
        return await self._async.get("/v1/types/edges")

    async def alist_type_detect_configs(self) -> dict[str, object]:
        """Async variant of :meth:`list_type_detect_configs`."""
        return await self._async.get("/v1/types/detect-config")

    async def aset_type_detect_config(
        self, type_name: str, packs: list[str]
    ) -> dict[str, object]:
        """Async variant of :meth:`set_type_detect_config`."""
        return await self._async.post(
            "/v1/types/detect-config", {"type": type_name, "packs": packs}
        )

    async def aquery_schema(self, *, purpose: str | None = None) -> dict[str, object]:
        """Async variant of :meth:`query_schema`."""
        selected = purpose or self._default_purpose
        suffix = f"?{urlencode({'purpose': selected})}" if selected else ""
        return await self._async.get(f"/v1/query/schema{suffix}")

    async def aregister_edge_type(
        self,
        from_type: str,
        to_type: str,
        label: str,
    ) -> dict[str, object]:
        """Async variant of :meth:`register_edge_type`."""
        return await self._async.post(
            "/v1/types/edges",
            {"from_type": from_type, "to_type": to_type, "label": label},
        )

    async def acreate_schema_branch(
        self,
        name: str,
        from_branch: str,
    ) -> dict[str, object]:
        """Async variant of :meth:`create_schema_branch`."""
        return await self._async.post(
            f"/v1/schema/branches/{path_segment(name)}",
            {"from": from_branch},
        )

    async def adelete_schema_branch(self, name: str) -> dict[str, object]:
        """Async variant of :meth:`delete_schema_branch`."""
        return await self._async.delete(f"/v1/schema/branches/{path_segment(name)}")

    def ontology_migrate(self, schema: dict[str, object]) -> dict[str, object]:
        """SHACL schema migration — register type specs, link types, property
        constraints in one governed call."""
        return self._sync.post("/v1/ontology/migrate", schema)

    def enrichment_rules(self, rules: dict[str, object]) -> dict[str, object]:
        """Register identity enrichment rules for SmartIngest."""
        return self._sync.post("/v1/ontology/enrichment-rules", rules)

    def create_link(
        self,
        link_name: str,
        source_id: str,
        source_type: str,
        target_id: str,
        target_type: str,
    ) -> dict[str, object]:
        """Create a typed, governed link (edge) between two objects."""
        return self._sync.post("/v1/links", {
            "link_name": link_name,
            "source_id": source_id,
            "source_type": source_type,
            "target_id": target_id,
            "target_type": target_type,
        })

    def create_links(
        self,
        edges: list[dict[str, object]],
        *,
        idempotency_key: str | None = None,
        durability: str | None = None,
    ) -> dict[str, object]:
        """Bulk-create up to 5000 edges in one idempotent, durable request.

        Each edge is ``{"link_name", "source_id", "source_type", "target_id",
        "target_type", "weight"?}``. Edges are keyed by ``(link_name,
        source_id, target_id)`` within the tenant: an existing current edge is
        counted under ``already_present`` and never written twice, so a retry
        after a timeout or crash cannot duplicate. ``idempotency_key`` (sent as
        ``Idempotency-Key``) additionally replays the first response.
        ``durability`` is ``"sync"`` (server default when persistent) or
        ``"async"``. Sends ``POST /v1/links/batch``.
        """
        headers: dict[str, str] = {}
        if idempotency_key is not None:
            headers["Idempotency-Key"] = idempotency_key
        if durability is not None:
            headers["X-Relata-Durability"] = durability
        return self._sync.post("/v1/links/batch", {"edges": edges}, extra_headers=headers or None)

    # ------------------------------------------------------------------
    # Identity resolution & entity lifecycle (#967)
    # ------------------------------------------------------------------

    def delete_link(self, link_name: str, source_id: str, target_id: str) -> dict[str, object]:
        """Close matching current native edges in this tenant, retaining history.

        Sends ``DELETE /v1/links`` with the exact link name and endpoint IDs.
        All matching parallel edges owned by the authenticated tenant close;
        the response contains ``deleted``, which is zero on a repeated request
        when no current matches remain. Row-backed edge objects are unaffected.

        Use the client's ``X-Relata-Durability: sync`` header to require a
        durable journal acknowledgment (enabled persistent WAL or selected
        recovery generation required). A request exceeding 4,096 matching
        edges or an 8 MiB journal record returns HTTP 413 before mutation.
        Nodes with configured replication peers return HTTP 501 until native
        deletion replication is implemented. HTTP 503 with ``retryable: false``
        means the outcome needs reconciliation; mutations are not retried.
        """
        return self._sync.delete("/v1/links", {
            "link_name": link_name, "source_id": source_id, "target_id": target_id,
        })

    async def adelete_link(
        self, link_name: str, source_id: str, target_id: str
    ) -> dict[str, object]:
        """Async counterpart of delete_link; mutations are not automatically retried."""
        return await self._async.delete("/v1/links", {
            "link_name": link_name, "source_id": source_id, "target_id": target_id,
        })

    def resolve_identity(
        self,
        value: str,
        *,
        mode: str | None = None,
        purpose: str | None = None,
    ) -> dict[str, object]:
        """Resolve an identity value to all known objects and clusters.
        Executes ``RESOLVE_IDENTITY('<value>')`` via ``POST /v1/query``.

        ``mode`` selects the resolution strategy: ``"canonical"`` (single
        best match), ``"cluster"`` (server default — same as
        :meth:`identity_cluster`), or ``"fuse"`` (merge overlapping
        clusters). Omit to use the server default (``cluster``).

        ``value`` is bound as ``$1`` via the server-side parameterized path —
        never interpolated into the SQL text (#3211).
        """
        p = purpose or self._default_purpose or "analytics"
        if mode is not None and mode not in ("canonical", "cluster", "fuse"):
            raise ValueError(
                f"mode must be one of 'canonical', 'cluster', 'fuse'; got {mode!r}"
            )
        sql = "RESOLVE_IDENTITY($1)"
        if mode is not None:
            sql = f"RESOLVE_IDENTITY($1, MODE => '{mode}')"
        return self._sync.post("/v1/query", {"purpose": p, "sql": sql, "params": [value]})

    def detect_identities(self, text: str, *, purpose: str | None = None) -> dict[str, object]:
        """Detect identities in free text via SmartIngest.
        Executes ``DETECT_IDENTITIES('<text>')`` — ``text`` is bound as ``$1``
        via the server-side parameterized path (#3211)."""
        p = purpose or self._default_purpose or "analytics"
        return self._sync.post(
            "/v1/query", {"purpose": p, "sql": "DETECT_IDENTITIES($1)", "params": [text]}
        )

    def erase_subject(
        self, subject: str, *, reason: str = "gdpr-art17-request", purpose: str | None = None,
    ) -> dict[str, object]:
        """Request GDPR Art. 17 erasure for a subject. Destructive steps are
        irreversible; controller-local access revocation is not certified
        cryptographic erasure. Inspect receipt_scope and incomplete copy states.
        An operation_id identifies the authorized deletion inspection route.

        ``subject``/``reason`` are bound as ``$1``/``$2`` via the server-side
        parameterized path (#3211).
        """
        p = purpose or self._default_purpose or "gdpr-erasure"
        return self._sync.post(
            "/v1/query",
            {
                "purpose": p,
                "sql": "ERASE SUBJECT $1 REASON $2 CERTIFY",
                "params": [subject, reason],
            },
        )

    # ------------------------------------------------------------------
    # Sessions (#967 Tier 2d)
    # ------------------------------------------------------------------

    def export_data(self, object_type: str, *, format: str = "json") -> dict[str, object]:
        """Bulk export all rows of a type (#967 Tier 5c)."""
        qs = urlencode({"type": object_type, "format": format, "purpose": "export"})
        return self._sync.get(f"/v1/export?{qs}")

    def session_diff(self, session_id: str) -> dict[str, object]:
        """View uncommitted session changes."""
        return self._sync.get(f"/v1/sessions/{path_segment(session_id)}/diff")

    def session_commit(self, session_id: str) -> dict[str, object]:
        """Commit a session's draft writes."""
        return self._sync.post(f"/v1/sessions/{path_segment(session_id)}/commit", {})

    def session_discard(self, session_id: str) -> dict[str, object]:
        """Discard uncommitted session changes."""
        return self._sync.delete(f"/v1/sessions/{path_segment(session_id)}/draft")

    # ------------------------------------------------------------------
    # Entity merge, dedup & identity resolution (#967)
    # ------------------------------------------------------------------

    def identity_cluster(self, value: str, *, purpose: str | None = None) -> dict[str, object]:
        """Resolve an identity to its full cluster of linked identifiers.

        ``value`` is bound as ``$1`` via the server-side parameterized path
        (#3211).
        """
        p = purpose or self._default_purpose or "analytics"
        return self._sync.post(
            "/v1/query",
            {"purpose": p, "sql": "RESOLVE_IDENTITY($1, MODE => 'cluster')", "params": [value]},
        )

    def same_identity(self, a: str, b: str, *, purpose: str | None = None) -> bool:
        """Predicate: do two identifiers resolve to the same entity?

        Executes ``SAME_IDENTITY('<a>', '<b>')`` via ``POST /v1/query`` and
        returns the ``match`` verdict (#2246). Both values are bound as
        ``$1``/``$2`` via the server-side parameterized path (#3211).
        """
        p = purpose or self._default_purpose or "analytics"
        data = self._sync.post(
            "/v1/query", {"purpose": p, "sql": "SAME_IDENTITY($1, $2)", "params": [a, b]}
        )
        rows = data.get("data") or []
        return bool(rows and rows[0].get("match"))

    def fuse_identities(
        self, id_a: str, id_b: str, *, purpose: str | None = None
    ) -> dict[str, object]:
        """Ontological merge of two identities — writes an IdentityLink with
        link_type='fused' and returns the merged cluster (#967). Values are
        bound as ``$1``/``$2`` (#3211)."""
        p = purpose or self._default_purpose or "analytics"
        return self._sync.post(
            "/v1/query", {"purpose": p, "sql": "FUSE_IDENTITIES($1, $2)", "params": [id_a, id_b]}
        )

    def split_identities(
        self, id_a: str, id_b: str, *, purpose: str | None = None
    ) -> dict[str, object]:
        """Ontological unmerge — inverse of fuse_identities (#967). Values are
        bound as ``$1``/``$2`` (#3211)."""
        p = purpose or self._default_purpose or "analytics"
        return self._sync.post(
            "/v1/query", {"purpose": p, "sql": "SPLIT_IDENTITIES($1, $2)", "params": [id_a, id_b]}
        )


    # ------------------------------------------------------------------
    # Graph algorithm operators (#967)
    # ------------------------------------------------------------------

    def graph_dijkstra(
        self, object_type: str, from_id: str, to_id: str, *, purpose: str | None = None
    ) -> dict[str, object]:
        """Shortest path between two entities."""
        p = purpose or self._default_purpose or "analytics"
        sql = (
            f"GRAPH_DIJKSTRA({_sql_literal(object_type)}, "
            f"FROM => {_sql_literal(from_id)}, TO => {_sql_literal(to_id)})"
        )
        return self._sync.post("/v1/query", {"purpose": p, "sql": sql})

    def graph_shortest_path(
        self,
        from_id: str,
        to_id: str,
        *,
        max_hops: int | None = None,
    ) -> dict[str, object]:
        """Shortest path via the governed ``GET /v1/graph/shortest-path`` door.

        Unlike :meth:`graph_dijkstra` (a SQL operator), this exposes the HTTP
        door's ``max_hops`` ceiling (#2344).
        """
        params: dict[str, str | int] = {"from": from_id, "to": to_id}
        if max_hops is not None:
            params["max_hops"] = max_hops
        return self._sync.get(f"/v1/graph/shortest-path?{urlencode(params)}")

    def graph_traverse(
        self,
        from_id: str,
        *,
        direction: str | None = None,
        max_depth: int | None = None,
        limit: int | None = None,
    ) -> dict[str, object]:
        """Breadth-first traversal via the governed ``GET /v1/graph/traverse`` door.

        ``direction`` is ``"out"`` | ``"in"`` | ``"both"`` (server default out).
        """
        params: dict[str, str | int] = {"from": from_id}
        if direction is not None:
            params["direction"] = direction
        if max_depth is not None:
            params["max_depth"] = max_depth
        if limit is not None:
            params["limit"] = limit
        return self._sync.get(f"/v1/graph/traverse?{urlencode(params)}")

    def graph_pagerank(
        self,
        object_type: str,
        *,
        damping: float | None = None,
        max_iter: int | None = None,
        purpose: str | None = None,
    ) -> dict[str, object]:
        """PageRank centrality."""
        p = purpose or self._default_purpose or "analytics"
        parts = [f"GRAPH_PAGERANK({_sql_literal(object_type)}"]
        if damping is not None:
            parts.append(f", DAMPING => {damping}")
        if max_iter is not None:
            parts.append(f", MAX_ITER => {max_iter}")
        parts.append(")")
        return self._sync.post("/v1/query", {"purpose": p, "sql": "".join(parts)})

    def graph_scc(self, object_type: str, *, purpose: str | None = None) -> dict[str, object]:
        """Strongly connected components (fraud-ring detection)."""
        p = purpose or self._default_purpose or "analytics"
        return self._sync.post(
            "/v1/query", {"purpose": p, "sql": f"GRAPH_SCC({_sql_literal(object_type)})"}
        )

    def graph_cycles(self, object_type: str, *, purpose: str | None = None) -> dict[str, object]:
        """Cycle detection in the graph."""
        p = purpose or self._default_purpose or "analytics"
        return self._sync.post(
            "/v1/query", {"purpose": p, "sql": f"GRAPH_CYCLES({_sql_literal(object_type)})"}
        )

    def graph_community(self, object_type: str, *, purpose: str | None = None) -> dict[str, object]:
        """Community detection via label propagation."""
        p = purpose or self._default_purpose or "analytics"
        return self._sync.post(
            "/v1/query", {"purpose": p, "sql": f"GRAPH_COMMUNITY({_sql_literal(object_type)})"}
        )

    def graph_node_similarity(
        self, object_type: str, node: str, *, purpose: str | None = None
    ) -> dict[str, object]:
        """Node similarity — find entities similar to a seed node."""
        p = purpose or self._default_purpose or "analytics"
        sql = f"GRAPH_NODE_SIMILARITY({_sql_literal(object_type)}, NODE => {_sql_literal(node)})"
        return self._sync.post("/v1/query", {"purpose": p, "sql": sql})

    def graph_link_predict(
        self, object_type: str, from_id: str, to_id: str, *, purpose: str | None = None
    ) -> dict[str, object]:
        """Link prediction between two nodes (#4633).

        The engine requires both ``FROM =>`` and ``TO =>`` — mirrors
        :meth:`graph_dijkstra`'s FROM/TO shape.
        """
        p = purpose or self._default_purpose or "analytics"
        sql = (
            f"GRAPH_LINK_PREDICT({_sql_literal(object_type)}, "
            f"FROM => {_sql_literal(from_id)}, TO => {_sql_literal(to_id)})"
        )
        return self._sync.post("/v1/query", {"purpose": p, "sql": sql})

    def graph_triangle_count(
        self, object_type: str, *, purpose: str | None = None
    ) -> dict[str, object]:
        """Triangle count — measures graph density / cohesion."""
        p = purpose or self._default_purpose or "analytics"
        return self._sync.post(
            "/v1/query", {"purpose": p, "sql": f"TRIANGLE_COUNT({_sql_literal(object_type)})"}
        )

    # ------------------------------------------------------------------
    # Intelligence operators (#967)
    # ------------------------------------------------------------------

    def convoy_detect(
        self,
        *,
        radius_m: float | None = None,
        time_tol_secs: int | None = None,
        min_points: int | None = None,
        purpose: str | None = None,
    ) -> dict[str, object]:
        """Convoy detection — find entities traveling together."""
        p = purpose or self._default_purpose or "analytics"
        parts = []
        if radius_m is not None:
            parts.append(f"RADIUS => {radius_m}")
        if time_tol_secs is not None:
            parts.append(f"TIME_TOL => {time_tol_secs * 1_000_000_000}")
        if min_points is not None:
            parts.append(f"MIN_POINTS => {min_points}")
        sql = f"CONVOY({', '.join(parts)})" if parts else "CONVOY()"
        return self._sync.post("/v1/query", {"purpose": p, "sql": sql})

    def dns_tunnel_detect(
        self,
        *,
        min_entropy: float = 3.0,
        min_queries: int = 5,
        purpose: str | None = None,
        page_size: int = DNS_TUNNEL_PAGE_SIZE,
        max_rows: int = DNS_TUNNEL_MAX_ROWS,
    ) -> dict[str, object]:
        """Triage DNS tunnel candidates over governed, paginated DNS rows.

        Uses the authenticated names-only schema endpoint, then normal `/v1/query`
        requests for registered `DnsLog`, `DnsQuery`, and `DnsEvent` types. A
        bounded scan returns `complete=False` and a per-type continuation cursor
        when its row budget is reached. The server continues to enforce tenant,
        purpose, and row authorization on every page.
        """
        p = purpose or self._default_purpose or "security"
        if (
            isinstance(page_size, bool)
            or not isinstance(page_size, int)
            or isinstance(max_rows, bool)
            or not isinstance(max_rows, int)
            or page_size < 1
            or max_rows < 1
        ):
            raise ValueError("page_size and max_rows must be positive")
        page_size = min(page_size, 10_000)
        schema = self.query_schema(purpose=p)
        visible_types = schema.get("object_types")
        if not isinstance(visible_types, list) or not all(
            isinstance(name, str) for name in visible_types
        ):
            raise TypeError("query_schema result must contain an object_types array")
        types = [name for name in DNS_TUNNEL_TYPES if name in visible_types]
        rows: list[dict[str, Any]] = []
        continuation: dict[str, str] = {}
        rows_scanned = 0
        complete = True
        incomplete_reason: str | None = None

        for type_index, object_type in enumerate(types):
            cursor: str | None = None
            seen_cursors: set[str] = set()
            while True:
                remaining = max_rows - rows_scanned
                if remaining <= 0:
                    complete = False
                    incomplete_reason = "max_rows_reached"
                    break
                limit = min(page_size, remaining)
                after = "" if cursor is None else f" AFTER {_sql_literal(cursor)}"
                sql = (
                    f"SELECT * FROM {object_type} LIMIT {limit}{after} WITH CACHE BYPASS"
                )
                result = self.query(sql, purpose=p)
                page_rows = result.rows
                rows.extend(page_rows)
                rows_scanned += len(page_rows)
                if result.partial:
                    complete = False
                    incomplete_reason = "server_returned_partial_shard_result"
                    break
                if result.truncated:
                    complete = False
                    incomplete_reason = "server_truncated_page_without_safe_resume_cursor"
                    break
                next_cursor = result.next_cursor
                if next_cursor is not None:
                    if next_cursor in seen_cursors or next_cursor == cursor:
                        complete = False
                        incomplete_reason = "pagination_cursor_did_not_advance"
                        continuation[object_type] = next_cursor
                        break
                    seen_cursors.add(next_cursor)
                elif len(page_rows) == limit and not result.has_next_cursor_metadata:
                    complete = False
                    incomplete_reason = "server_did_not_return_pagination_metadata"
                    break
                if next_cursor is None:
                    break
                if rows_scanned >= max_rows:
                    complete = False
                    incomplete_reason = "max_rows_reached"
                    continuation[object_type] = next_cursor
                    break
                cursor = next_cursor
            if not complete:
                continuation.update({"remaining_types": ",".join(types[type_index:])})
                break

        return dns_tunnel_result(
            rows,
            min_entropy=min_entropy,
            min_queries=min_queries,
            rows_scanned=rows_scanned,
            types_scanned=types,
            complete=complete,
            continuation=continuation,
            incomplete_reason=incomplete_reason,
        )

    async def adns_tunnel_detect(
        self,
        *,
        min_entropy: float = 3.0,
        min_queries: int = 5,
        purpose: str | None = None,
        page_size: int = DNS_TUNNEL_PAGE_SIZE,
        max_rows: int = DNS_TUNNEL_MAX_ROWS,
    ) -> dict[str, object]:
        """Async variant of :meth:`dns_tunnel_detect`."""
        p = purpose or self._default_purpose or "security"
        if (
            isinstance(page_size, bool)
            or not isinstance(page_size, int)
            or isinstance(max_rows, bool)
            or not isinstance(max_rows, int)
            or page_size < 1
            or max_rows < 1
        ):
            raise ValueError("page_size and max_rows must be positive")
        page_size = min(page_size, 10_000)
        schema = await self.aquery_schema(purpose=p)
        visible_types = schema.get("object_types")
        if not isinstance(visible_types, list) or not all(
            isinstance(name, str) for name in visible_types
        ):
            raise TypeError("query_schema result must contain an object_types array")
        types = [name for name in DNS_TUNNEL_TYPES if name in visible_types]
        rows: list[dict[str, Any]] = []
        continuation: dict[str, str] = {}
        rows_scanned = 0
        complete = True
        incomplete_reason: str | None = None
        for type_index, object_type in enumerate(types):
            cursor: str | None = None
            seen_cursors: set[str] = set()
            while True:
                remaining = max_rows - rows_scanned
                if remaining <= 0:
                    complete = False
                    incomplete_reason = "max_rows_reached"
                    break
                limit = min(page_size, remaining)
                after = "" if cursor is None else f" AFTER {_sql_literal(cursor)}"
                sql = f"SELECT * FROM {object_type} LIMIT {limit}{after} WITH CACHE BYPASS"
                result = await self.aquery(sql, purpose=p)
                page_rows = result.rows
                rows.extend(page_rows)
                rows_scanned += len(page_rows)
                if result.partial:
                    complete = False
                    incomplete_reason = "server_returned_partial_shard_result"
                    break
                if result.truncated:
                    complete = False
                    incomplete_reason = "server_truncated_page_without_safe_resume_cursor"
                    break
                next_cursor = result.next_cursor
                if next_cursor is not None:
                    if next_cursor in seen_cursors or next_cursor == cursor:
                        complete = False
                        incomplete_reason = "pagination_cursor_did_not_advance"
                        continuation[object_type] = next_cursor
                        break
                    seen_cursors.add(next_cursor)
                elif len(page_rows) == limit and not result.has_next_cursor_metadata:
                    complete = False
                    incomplete_reason = "server_did_not_return_pagination_metadata"
                    break
                if next_cursor is None:
                    break
                if rows_scanned >= max_rows:
                    complete = False
                    incomplete_reason = "max_rows_reached"
                    continuation[object_type] = next_cursor
                    break
                cursor = next_cursor
            if not complete:
                continuation.update({"remaining_types": ",".join(types[type_index:])})
                break
        return dns_tunnel_result(
            rows,
            min_entropy=min_entropy,
            min_queries=min_queries,
            rows_scanned=rows_scanned,
            types_scanned=types,
            complete=complete,
            continuation=continuation,
            incomplete_reason=incomplete_reason,
        )

    def geofence(
        self, fence: str, *, target_type: str | None = None, purpose: str | None = None
    ) -> dict[str, object]:
        """Geo-fence query — find entities within a geographic fence."""
        p = purpose or self._default_purpose or "analytics"
        parts = [f"GEOFENCE({_sql_literal(fence)}"]
        if target_type is not None:
            parts.append(f", TARGET_TYPE => {_sql_literal(target_type)}")
        parts.append(")")
        return self._sync.post("/v1/query", {"purpose": p, "sql": "".join(parts)})

    # ------------------------------------------------------------------
    # Multimedia operators (#2251, epic #655 — ADR-030)
    # ------------------------------------------------------------------
    #
    # Biometric / perceptual-hash search via the governed SQL operator door.
    # Both route through ``POST /v1/query`` so PURPOSE / ACL / cell-masking /
    # tenant isolation apply identically to a hand-written query. The server
    # operators are registered in ``relata_query::parser``:
    #
    # - ``SIMILAR_IMAGE('<media_ref>', THRESHOLD => f, INDEX => 'corpus')``
    #   (#2840, PR #2859 — near-duplicate/similar-image search scoped to an
    #   optional named index/corpus; THRESHOLD defaults to 0.9 server-side).
    #
    # ``FACE_SEARCH`` / ``MATCH_PDQ`` (and the MCP ``face_match`` tool) were
    # retired from core (#5935); embeddings and fingerprints are delivered by
    # OpenETL and compared with generic vector search or ``SIMILAR_IMAGE``.

    def similar_image(
        self,
        media_ref: str,
        *,
        threshold: float | None = None,
        index: str | None = None,
        purpose: str | None = None,
    ) -> QueryResult:
        """Near-duplicate / similar-image search (#2840, PR #2859).

        Executes ``SIMILAR_IMAGE(...)`` through the governed ``/v1/query`` door.

        Args:
            media_ref: Reference media identifier to search from.
            threshold: Minimum similarity in ``[0.0, 1.0]``; omitted means
                the server default (0.9) applies.
            index: Optional named index/corpus to scope the search to; when
                omitted, the server searches its default scope.
            purpose: Purpose override for this query.

        Returns:
            :class:`~relata.models.QueryResult` ranked by score desc.

        Example::

            result = client.similar_image(
                "media-42", threshold=0.6, index="ncmec", purpose="investigation",
            )
            for row in result:
                print(row["entity_id"], row["score"])
        """
        sql = _similar_image_sql(media_ref, threshold=threshold, index=index)
        return self.query(sql, purpose=purpose)

    async def asimilar_image(
        self,
        media_ref: str,
        *,
        threshold: float | None = None,
        index: str | None = None,
        purpose: str | None = None,
    ) -> QueryResult:
        """Async variant of :meth:`similar_image`."""
        sql = _similar_image_sql(media_ref, threshold=threshold, index=index)
        return await self.aquery(sql, purpose=purpose)

    def health(self) -> HealthResponse:
        """Check node health (synchronous).

        Returns:
            :class:`~relata.models.HealthResponse` with ``status``,
            ``profile``, and ``node_id``.

        Raises:
            :class:`~relata.exceptions.ConnectionError`: Server unreachable.
        """
        data = self._sync.get("/health")
        return HealthResponse.model_validate(data)

    def status(self) -> StatusResponse:
        """Retrieve node and quota status (synchronous).

        Returns:
            :class:`~relata.models.StatusResponse` with ``profile``,
            ``role``, and ``query_quota``.

        Raises:
            :class:`~relata.exceptions.AuthError`: Invalid/missing token.
            :class:`~relata.exceptions.ConnectionError`: Server unreachable.
        """
        data = self._sync.get("/v1/system/status")
        return StatusResponse.model_validate(data)

    def audit_count(self) -> AuditCountResponse:
        """Retrieve the audit log entry count and chain integrity status (synchronous).

        The Relata audit chain is hash-chained and tamper-evident.  A
        ``chain_valid=False`` response indicates potential tampering and must
        be escalated immediately.

        Returns:
            :class:`~relata.models.AuditCountResponse`.

        Raises:
            :class:`~relata.exceptions.AuthError`: Invalid/missing token.
            :class:`~relata.exceptions.ConnectionError`: Server unreachable.
        """
        data = self._sync.get("/v1/audit/count")
        return AuditCountResponse.model_validate(data)

    # ------------------------------------------------------------------
    # Introspection (stats / version / ready) — pairs with #86
    # ------------------------------------------------------------------

    def stats(self) -> Stats:
        """Return engine-wide counts for health dashboards (synchronous).

        Wraps ``GET /v1/system/stats``. The shape mirrors the
        ``storage-backend-requirements.md`` §9 contract — ``records``,
        ``states``, ``snapshot_rows``, ``log_leaves``, ``tokens`` — to the
        extent the server currently distinguishes them.
        """
        data = self._sync.get("/v1/system/stats")
        return Stats.model_validate(data)

    def version(self) -> VersionInfo:
        """Return runtime build-info (synchronous).

        Wraps ``GET /version``. Useful for migration checks and capability
        negotiation.
        """
        data = self._sync.get("/version")
        return VersionInfo.model_validate(data)

    def ready(self) -> ReadyReport:
        """Return the 9-condition readiness report (synchronous).

        Wraps ``GET /health/ready``. Returns ``ReadyReport.is_ready == True``
        on HTTP 200; on HTTP 503 the SDK raises
        :class:`~relata.exceptions.ServerError` carrying the shed reason —
        callers who want the typed model even on 503 should catch and inspect
        ``ServerError.message``.
        """
        data = self._sync.get("/health/ready")
        return ReadyReport.model_validate(data)

    # ------------------------------------------------------------------
    # Observability stream (#3822, #3828)
    # ------------------------------------------------------------------

    def observe_stream(self) -> Iterator[dict[str, object]]:
        """Stream live observability events from ``GET /v1/observe/stream``.

        The endpoint is opt-in server-side (``RELATA_OBSERVE_STREAM=on``)
        and emits one structured ``ObservabilityEvent`` JSON object per SSE
        ``data:`` line. It requires bearer auth and is tenant-scoped
        server-side.

        Each yielded dict carries ``ts_ns``, ``level``, ``target``,
        ``kind``, ``door``, ``purpose``, ``tenant_id``, ``trace_id``,
        ``stage``, ``sql_template``, ``rows``, ``latency_ms``, ``error``,
        and ``message``. Lines that are not ``data:`` payloads (comments,
        heartbeats) are ignored.

        A connection drop — the server shutting down or the client
        disconnecting — ends the generator cleanly rather than raising.

        Yields:
            Per-event dict (parsed JSON from each SSE ``data:`` line).

        Example::

            for event in client.observe_stream():
                print(event["kind"], event["level"], event["message"])
        """
        from relata.streaming import _classify_capped_error

        sync = self._sync
        try:
            with sync._client.stream("GET", "/v1/observe/stream") as resp:  # noqa: SLF001
                if not resp.is_success:
                    _classify_capped_error(sync, resp)
                for line in resp.iter_lines():
                    if line.startswith("data: "):
                        payload = line[len("data: "):]
                        if payload.startswith("{"):
                            yield json.loads(payload)
        except (ConnectionError, OSError):
            # The stream may close when the server shuts down or the client
            # disconnects — end the generator cleanly instead of propagating.
            return

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    async def aquery(
        self,
        sql: str,
        *,
        purpose: str | None = None,
        dialect: str | None = None,
        matcher: Literal["auto", "vf3", "bfs"] | None = None,
        query_vector: list[float] | None = None,
        query_vector_model: str | None = None,
    ) -> QueryResult:
        """Execute a SQL query against the Relata engine (asynchronous).

        Identical to :meth:`query` but non-blocking.  Suitable for use in
        ``asyncio`` event loops, ``FastAPI`` route handlers, Jupyter notebooks
        with ``%autoawait``, etc.

        Args:
            sql: SQL statement to execute (or Cypher/GQL when ``dialect`` is set).
            purpose: Purpose override for this query.
            dialect: Optional ``x-query-dialect`` override — ``"sql"``,
                ``"cypher"``, or ``"gql"`` (#3265). See :meth:`query`.
            matcher: Subgraph-matcher hint for multi-hop Cypher patterns
                (#1189).  See :meth:`query`.
            query_vector: Ready query embedding; see :meth:`query`.
            query_vector_model: Model tag for ``query_vector``; see :meth:`query`.

        Returns:
            :class:`~relata.models.QueryResult`.

        Raises:
            :class:`~relata.exceptions.PurposeError`: No purpose set.
            :class:`~relata.exceptions.QuotaError`: Quota exhausted.
            :class:`~relata.exceptions.AuthError`: Invalid/missing token.
            :class:`~relata.exceptions.ConnectionError`: Server unreachable.

        Examples::

            async with RelataClient("http://localhost:9090", purpose="audit") as client:
                result = await client.aquery("SELECT COUNT(*) FROM AuditLog")
                print(result.rows)
        """
        effective_purpose = self._resolve_purpose(purpose)
        payload: dict[str, object] = {
            "purpose": effective_purpose,
            "sql": _inject_matcher_hint(sql, matcher),
        }
        if query_vector is not None:
            payload["query_vector"] = query_vector
        if query_vector_model is not None:
            payload["query_vector_model"] = query_vector_model
        extra_headers = _dialect_headers(dialect)
        data = await self._async.post("/v1/query", payload, extra_headers=extra_headers)
        return QueryResult.model_validate(data)

    async def aquery_params(
        self,
        sql: str,
        params: list[Any],
        *,
        purpose: str | None = None,
        matcher: Literal["auto", "vf3", "bfs"] | None = None,
    ) -> QueryResult:
        """Async parameterized query — see :meth:`query_params` (#1162)."""
        sql = _rewrite_question_mark_params(sql)
        effective_purpose = self._resolve_purpose(purpose)
        payload = {
            "purpose": effective_purpose,
            "sql": _inject_matcher_hint(sql, matcher),
            "params": params,
        }
        data = await self._async.post("/v1/query", payload)
        return QueryResult.model_validate(data)

    async def ahealth(self) -> HealthResponse:
        """Check node health (asynchronous).

        Returns:
            :class:`~relata.models.HealthResponse`.
        """
        data = await self._async.get("/health")
        return HealthResponse.model_validate(data)

    async def astatus(self) -> StatusResponse:
        """Retrieve node and quota status (asynchronous).

        Returns:
            :class:`~relata.models.StatusResponse`.
        """
        data = await self._async.get("/v1/system/status")
        return StatusResponse.model_validate(data)

    async def aaudit_count(self) -> AuditCountResponse:
        """Retrieve audit log entry count and chain integrity (asynchronous).

        Returns:
            :class:`~relata.models.AuditCountResponse`.
        """
        data = await self._async.get("/v1/audit/count")
        return AuditCountResponse.model_validate(data)

    # ------------------------------------------------------------------
    # Introspection — async mirrors
    # ------------------------------------------------------------------

    async def astats(self) -> Stats:
        """Return engine-wide counts for health dashboards (asynchronous)."""
        data = await self._async.get("/v1/system/stats")
        return Stats.model_validate(data)

    async def aversion(self) -> VersionInfo:
        """Return runtime build-info (asynchronous)."""
        data = await self._async.get("/version")
        return VersionInfo.model_validate(data)

    async def aready(self) -> ReadyReport:
        """Return the 9-condition readiness report (asynchronous)."""
        data = await self._async.get("/health/ready")
        return ReadyReport.model_validate(data)

    # ------------------------------------------------------------------
    # Fluent query builder entry point
    # ------------------------------------------------------------------

    def select(self, *columns_or_table: str) -> QueryBuilder:
        """Start a fluent :class:`~relata.query.QueryBuilder` chain.

        This is a convenience entry point.  The returned builder is bound to
        this client and will call :meth:`query` when :meth:`~relata.query.QueryBuilder.execute`
        is called.

        Args:
            *columns_or_table: Either a single table name (``"Person"``) for
                ``SELECT * FROM Person``, or specific columns followed by
                ``from_("Person")`` in the chain.

        Returns:
            :class:`~relata.query.QueryBuilder` bound to this client.

        Examples::

            result = (
                client.select("Person")
                .purpose("analytics")
                .where("age > 30")
                .limit(50)
                .execute()
            )
        """
        from relata.query import QueryBuilder

        return QueryBuilder(client=self).select(*columns_or_table)

    # ------------------------------------------------------------------
    # T9 flagship namespace handle (#1991, epic #1982)
    # ------------------------------------------------------------------

    def namespace(self, name: str) -> Namespace:
        """Return a typed :class:`~relata.namespace.Namespace` handle bound to
        one object type — the flagship retrieval surface.

        The handle reuses this client's connection pool, auth, tenant, and
        purpose context::

            docs = client.namespace("Document")
            docs.write([{"id": "d1", "title": "..."}])      # schemaless (T6, /v1/ingest/auto)
            res = docs.query(                                # typed search (T1, /v1/search)
                text="retrieval",
                match_column="title",
                filters=[{"field": "status", "op": "eq", "value": "published"}],
            )
            for row in res:
                print(row)
        """
        from relata.namespace import Namespace

        return Namespace(self, name)

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest_document(
        self,
        objects: list[dict[str, Any]],
        *,
        purpose: str | None = None,
    ) -> IngestDocumentResponse:
        """Queue prepared rows through the generic mixed-type bulk contract.

        Document extraction and mapping belong in application adapters. Pass
        ``DocumentSource``/``DocumentChunk`` rows with ``_type`` and stable
        ``_pk`` fields. All types share one task identifier. Queued source
        payloads remain volatile; configured local authority records durable
        intent and confirmed progress, while explicit volatile profiles retain
        RAM-only status. Poll the originating node with the authenticated
        tenant. Recovered unfinished work is uncertain and never automatically
        reingested; local receipts do not certify indexing or replication.

        Args:
            objects: Prepared object rows, each carrying ``_type`` and ``_pk``.
            purpose: Optional purpose override for the request.

        Returns:
            A queue response with ``task_id``. Poll it with
            :meth:`ingest_document_status` to observe local evidence, uncertainty,
            writes and drops.

        Raises:
            :class:`~relata.exceptions.AuthError`: Invalid/missing token.
            :class:`~relata.exceptions.QuotaError`: Ingest queue full (HTTP 429).
            :class:`~relata.exceptions.ConnectionError`: Server unreachable.
            :class:`~relata.exceptions.RelataError`: Validation or server error.

        """
        effective_purpose = purpose or self._default_purpose
        payload: dict[str, Any] = {"objects": objects, "queued": True}
        if effective_purpose:
            payload["purpose"] = effective_purpose
        data = self._sync.post("/v1/ingest/bulk", payload)
        return IngestDocumentResponse.model_validate(data)

    async def aingest_document(
        self,
        objects: list[dict[str, Any]],
        *,
        purpose: str | None = None,
    ) -> IngestDocumentResponse:
        """Queue prepared rows asynchronously through generic bulk ingest.

        Identical to :meth:`ingest_document` but non-blocking.
        """
        effective_purpose = purpose or self._default_purpose
        payload: dict[str, Any] = {"objects": objects, "queued": True}
        if effective_purpose:
            payload["purpose"] = effective_purpose
        data = await self._async.post("/v1/ingest/bulk", payload)
        return IngestDocumentResponse.model_validate(data)

    def commit(
        self,
        operations: list[dict[str, Any]],
        *,
        idempotency_key: str | None = None,
        purpose: str | None = None,
    ) -> dict[str, Any]:
        """Atomically apply up to 64 create/update/delete operations (#5767).

        All-or-nothing: every ``expected_version`` precondition is checked
        before anything is written, and the bundle is one durable record. A
        failed precondition raises
        :class:`~relata.exceptions.PreconditionFailedError` and changes nothing.
        Retrying with the same ``idempotency_key`` returns the original result.
        Returns ``{"commitId", "operations": [{"op","type","id","version"}],
        "replayed"}``.
        """
        from relata.objects import _commit_payload

        return dict(self._sync.post(
            "/v1/commit",
            _commit_payload(operations, idempotency_key, purpose or self._default_purpose),
        ))

    async def acommit(
        self,
        operations: list[dict[str, Any]],
        *,
        idempotency_key: str | None = None,
        purpose: str | None = None,
    ) -> dict[str, Any]:
        """Async equivalent of :meth:`commit`."""
        from relata.objects import _commit_payload

        return dict(await self._async.post(
            "/v1/commit",
            _commit_payload(operations, idempotency_key, purpose or self._default_purpose),
        ))

    def ingest_document_status(self, task_id: str) -> IngestDocumentTaskStatus:
        """Read the current status of a document-ingest task.

        Persistent local receipts survive restart; interrupted work is uncertain
        and is never automatically reingested. Volatile profiles remain
        process-local. Inspect receipt_durability/recovered in retained metadata.
        A task belongs to its originating node and authenticated tenant.
        Confirmed counters do not certify global index or replica completion.
        """
        data = self._sync.get(f"/v1/ingest/tasks/{path_segment(task_id)}")
        return IngestDocumentTaskStatus.model_validate(data)

    async def aingest_document_status(self, task_id: str) -> IngestDocumentTaskStatus:
        """Async counterpart to :meth:`ingest_document_status`."""
        data = await self._async.get(f"/v1/ingest/tasks/{path_segment(task_id)}")
        return IngestDocumentTaskStatus.model_validate(data)

    def submit_alert(self, request: AlertSubmitRequest) -> AlertSubmitReceipt:
        """Submit an alert with restart-safe identity on a single persistent node.

        Exact retries return the original identity without replaying notifications.
        Webhook and SSE delivery is best-effort, not a durable delivery receipt.
        """
        data = self._sync.post("/v1/alerts", request.model_dump(exclude_none=True))
        return AlertSubmitReceipt.model_validate(data)

    async def asubmit_alert(self, request: AlertSubmitRequest) -> AlertSubmitReceipt:
        """Async counterpart to :meth:`submit_alert`."""
        data = await self._async.post("/v1/alerts", request.model_dump(exclude_none=True))
        return AlertSubmitReceipt.model_validate(data)

    def append_event(self, request: EventAppendRequest) -> EventAppendReceipt:
        """Append a schema-governed event using its required idempotency key.

        Exact retries return the original event receipt. A key reused with a
        different payload raises the server's conflict error. The endpoint
        requires a local persistent WAL and currently rejects cluster mode.
        """
        data = self._sync.post("/v1/ingest/events", request.model_dump(exclude_none=True))
        return EventAppendReceipt.model_validate(data)

    async def aappend_event(self, request: EventAppendRequest) -> EventAppendReceipt:
        """Async counterpart to :meth:`append_event`."""
        data = await self._async.post("/v1/ingest/events", request.model_dump(exclude_none=True))
        return EventAppendReceipt.model_validate(data)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        auth = "authenticated" if self._bearer_token else "unauthenticated"
        return (
            f"RelataClient("
            f"base_url={self._base_url!r}, "
            f"purpose={self._default_purpose!r}, "
            f"auth={auth!r})"
        )

    def graphql(
        self,
        query: str,
        variables: dict[str, object] | None = None,
        operation_name: str | None = None,
    ) -> dict[str, object] | list[dict[str, object]] | None:
        """Execute a GraphQL query against the governed query path (ADR-220).

        Args:
            query: The GraphQL query string.
            variables: Optional variables map. ``$var`` references in ``limit``
                and ``where`` are bound server-side (#3260). A referenced
                variable that is missing, null, or wrong-typed is a hard server
                error (surfaced as :class:`RelataError`) — never silently
                dropped. String values are bound as quoted literals, so
                variable content cannot inject SQL.
            operation_name: Optional operation name (multi-operation documents).

        Returns:
            The ``data`` field — a list of row objects for a query, or the
            ``__schema`` object for an introspection request.

        Raises:
            RelataError: if the server returns a non-empty ``errors`` array.
        """
        body: dict[str, object] = {"query": query}
        if variables is not None:
            body["variables"] = variables
        if operation_name is not None:
            body["operationName"] = operation_name
        return _graphql_data(self._sync.post("/v1/graphql", body))


    async def agraphql(
        self,
        query: str,
        variables: dict[str, object] | None = None,
        operation_name: str | None = None,
    ) -> dict[str, object] | list[dict[str, object]] | None:
        """Async variant of :meth:`graphql`."""
        body: dict[str, object] = {"query": query}
        if variables is not None:
            body["variables"] = variables
        if operation_name is not None:
            body["operationName"] = operation_name
        return _graphql_data(await self._async.post("/v1/graphql", body))
