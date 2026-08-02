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

from types import TracebackType
from typing import TYPE_CHECKING

from relata._http import AsyncHttpTransport, HttpTransport
from relata.exceptions import PurposeError, RelataError
from relata.models import (
    AuditCountResponse,
    ClusterNode,
    HealthResponse,
    IngestDocumentResponse,
    QueryResult,
    ReadyReport,
    SearchResponse,
    Stats,
    StatusResponse,
    VersionInfo,
)

if TYPE_CHECKING:
    from relata.query import QueryBuilder


def _graphql_data(resp: dict[str, object]) -> dict[str, object] | list[dict[str, object]] | None:
    """Return the ``data`` field of a GraphQL envelope, raising on ``errors``.

    The server wraps every ``/graphql`` response in ``{"data": …, "errors": […]}``.
    On a non-empty ``errors`` array, raise :class:`RelataError` carrying the
    first error message; otherwise return ``data``.
    """
    errors = resp.get("errors") or []
    if errors:
        first = errors[0]
        msg = first.get("message", "graphql error") if isinstance(first, dict) else str(first)
        raise RelataError(f"graphql: {msg}")
    return resp.get("data")


def _rewrite_question_mark_params(sql: str) -> str:
    """Rewrite ``?`` placeholders to ``$1``, ``$2``, … left-to-right.

    This lets callers use the more familiar ``?`` form; the server only
    understands ``$N`` positional parameters (#1162).
    """
    import re

    counter = 0

    def _replace(m: "re.Match[str]") -> str:  # type: ignore[name-defined]
        nonlocal counter
        counter += 1
        return f"${counter}"

    return re.sub(r"\?", _replace, sql)


def _sql_literal(s: str) -> str:
    """Quote ``s`` as a SQL string literal, doubling internal single quotes."""
    return "'" + s.replace("'", "''") + "'"


def _face_search_sql(
    gallery_id: str,
    embedding: list[float] | str,
    *,
    k: int,
    threshold: float,
) -> str:
    """Build a ``SELECT * FROM FACE_SEARCH(...)`` ticket (#2251).

    Mirrors the server operator registered in
    ``relata_query::parser`` — ``FACE_SEARCH('<csv floats>', '<gallery>',
    K => <n>, THRESHOLD => <f>)`` (defaults K=10, THRESHOLD=0.7).
    """
    if isinstance(embedding, str):
        emb_csv = embedding
    else:
        emb_csv = ",".join(repr(float(x)) for x in embedding)
    return (
        "SELECT * FROM FACE_SEARCH("
        f"{_sql_literal(emb_csv)}, {_sql_literal(gallery_id)}, "
        f"K => {int(k)}, THRESHOLD => {float(threshold)})"
    )


def _match_pdq_sql(corpus_id: str, query_hash: str, *, threshold: float) -> str:
    """Build a ``SELECT * FROM MATCH_PDQ(...)`` ticket (#2251).

    Mirrors ``MATCH_PDQ('<hash>', '<corpus>', THRESHOLD => <f>)``
    (default THRESHOLD=0.9).
    """
    return (
        "SELECT * FROM MATCH_PDQ("
        f"{_sql_literal(query_hash)}, {_sql_literal(corpus_id)}, "
        f"THRESHOLD => {float(threshold)})"
    )


def _flight_ticket(sql: str, purpose: str | None) -> str:
    """Assemble an Arrow Flight ``do_get`` ticket from SQL + optional PURPOSE.

    The Flight door (``crates/relata-cli/src/flight_server.rs::do_get``) reads
    the ticket as raw UTF-8 SQL and extracts ``/* PURPOSE '<id>' */`` from a
    leading SQL comment, defaulting to ``"flight_query"`` when absent (#958).
    """
    if not purpose:
        return sql
    return f"/* PURPOSE '{purpose.replace("'", "''")}' */ {sql}"


def _flight_endpoint_from(base_url: str, flight_endpoint: str | None) -> str:
    """Resolve the Arrow Flight gRPC endpoint.

    Defaults to ``grpc://<base_url host>:8815`` (the server's default Flight
    port, ``RELATA_FLIGHT_PORT``). Pass ``flight_endpoint`` to override.
    """
    if flight_endpoint:
        return flight_endpoint
    from urllib.parse import urlparse, urlencode

    host = urlparse(base_url).hostname or "localhost"
    return f"grpc://{host}:8815"


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
            on every request.  Required for multi-tenant deployments; overrides
            per-call via the ``tenant=`` argument on individual methods.
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
            Per ADR-0261, ``/admin/*`` and ``/platform/*`` routes are mounted
            **only** there on a hardened server/cluster deployment -- set
            this so :class:`~relata.backup.BackupClient` and
            :class:`~relata.tenants.TenantAdminClient`'s platform-tenant
            methods reach them. Leave unset (the default) when the admin
            listener isn't split from the data plane (e.g. local/free-profile
            dev) -- every request then goes to ``base_url`` unchanged.

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
    ) -> None:
        self._base_url = base_url.rstrip("/")
        # #2321 (ADR-0261): /admin/* and /platform/* are mounted only on the
        # loopbound admin control-plane listener (RELATA_ADMIN_BIND, default
        # 127.0.0.1:9091) -- never this client's main base_url on a hardened
        # server/cluster deployment. Set admin_base_url to point BackupClient/
        # TenantAdminClient's platform-tenant methods at that listener; leave
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

        # Compose the static extra-headers bag: tenant/delegation first, then
        # caller-supplied headers win. Per-call overrides (tenant=, acting_as=,
        # request_id=) are merged in transport-property accessors below.
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
        """Close the underlying sync HTTP connection pool."""
        if self.__sync_transport is not None:
            self.__sync_transport.close()
            self.__sync_transport = None

    async def aclose(self) -> None:
        """Close the underlying async HTTP connection pool."""
        if self.__async_transport is not None:
            await self.__async_transport.aclose()
            self.__async_transport = None

    # ------------------------------------------------------------------
    # Sync API
    # ------------------------------------------------------------------

    def query(self, sql: str, *, purpose: str | None = None) -> QueryResult:
        """Execute a SQL query against the Relata engine (synchronous).

        Args:
            sql: SQL statement to execute.  Relata extends standard SQL with:

                - ``AS OF 'timestamp'`` — bi-temporal point-in-time queries
                - ``WITH PROVENANCE`` — include PROV-O columns in results
                - ``PATHS_BETWEEN(a, b, max_hops => 4)`` — graph traversal
                - ``MATCH_FACE(face_embedding, col, 0.92)`` — face similarity
                - ``LOOKUP_IDENTITY('value')`` — resolve any identifier
                - ``HYBRID_SCORE(query)`` — BM25 + vector hybrid search

            purpose: Purpose for this query.  Overrides the client-level
                default.  Must be a value registered in the tenant's
                ``PurposeRegistry``.  Common values: ``"analytics"``,
                ``"audit"``, ``"analysis"``, ``"monitoring"``.

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
        """
        effective_purpose = self._resolve_purpose(purpose)
        payload = {"purpose": effective_purpose, "sql": sql}
        data = self._sync.post("/query", payload)
        return QueryResult.model_validate(data)

    def query_params(
        self,
        sql: str,
        params: list,
        *,
        purpose: str | None = None,
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
        payload = {"purpose": effective_purpose, "sql": sql, "params": params}
        data = self._sync.post("/query", payload)
        return QueryResult.model_validate(data)

    def query_arrow(
        self,
        sql: str,
        *,
        purpose: str | None = None,
    ) -> "pyarrow.Table":  # type: ignore[name-defined]  # noqa: F821
        """Execute a SQL query and return results as a ``pyarrow.Table`` (#1744).

        Calls ``POST /query/arrow`` which streams Arrow IPC binary.  Requires
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
        import pyarrow.ipc as ipc  # type: ignore[import]

        from relata.streaming import StreamingClient

        effective_purpose = self._resolve_purpose(purpose)
        chunks = StreamingClient.from_client(self).query_arrow_raw(sql, purpose=effective_purpose)
        buf = io.BytesIO(b"".join(chunks))
        return ipc.open_stream(buf).read_all()

    def query_flight(
        self,
        sql: str,
        *,
        flight_endpoint: str | None = None,
        purpose: str | None = None,
        bearer_token: str | None = None,
    ) -> "pyarrow.Table":  # type: ignore[name-defined]  # noqa: F821
        """Execute a SQL query over Arrow Flight ``do_get`` and return the
        result as a zero-copy ``pyarrow.Table`` (#2253, #958).

        Unlike :meth:`query_arrow` (which decodes the HTTP Arrow-IPC stream
        from ``POST /query/arrow``), this opens a real Arrow Flight client to
        the server's gRPC Flight door and runs ``do_get`` with the SQL as the
        ticket bytes — no JSON intermediate.

        The Flight door is enabled server-side with ``RELATA_FLIGHT_ENABLE=true``
        (default port 8815). The ticket is the raw UTF-8 SQL, optionally
        prefixed with ``/* PURPOSE '<id>' */`` which the server extracts as the
        query purpose (defaulting to ``"flight_query"``).

        Requires ``pyarrow`` to be installed (``pip install pyarrow``).

        Args:
            sql: SQL query string used verbatim as the Flight ticket.
            flight_endpoint: Flight gRPC endpoint (e.g.
                ``"grpc://localhost:8815"``). Defaults to
                ``grpc://<base_url host>:8815``.
            purpose: Optional purpose injected as a ``/* PURPOSE '...' */``
                SQL comment. Note this is **not** resolved from the client
                default — Flight has no purposeless-query gate, so callers
                pass it explicitly when they want one.
            bearer_token: Bearer token for Flight metadata. Defaults to the
                client's token.

        Returns:
            A ``pyarrow.Table`` with the governed result set.

        Example::

            tbl = client.query_flight(
                "SELECT * FROM Person LIMIT 1000", purpose="analytics",
            )
            df = tbl.to_pandas()
        """
        import pyarrow.flight as flight  # type: ignore[import]

        endpoint = _flight_endpoint_from(self._base_url, flight_endpoint)
        ticket_sql = _flight_ticket(sql, purpose)
        bearer = bearer_token if bearer_token is not None else self._bearer_token

        client = flight.FlightClient(endpoint)
        options = flight.FlightCallOptions()
        if bearer:
            options = flight.FlightCallOptions(
                headers=[(b"authorization", f"Bearer {bearer}".encode())]
            )
        reader = client.do_get(flight.Ticket(ticket_sql.encode("utf-8")), options)
        return reader.read_all()

    async def aquery_flight(
        self,
        sql: str,
        *,
        flight_endpoint: str | None = None,
        purpose: str | None = None,
        bearer_token: str | None = None,
    ) -> "pyarrow.Table":  # type: ignore[name-defined]  # noqa: F821
        """Async variant of :meth:`query_flight`.

        ``pyarrow.flight`` is synchronous, so the call is dispatched to a
        worker thread via :func:`asyncio.to_thread`.
        """
        import asyncio

        return await asyncio.to_thread(
            self.query_flight,
            sql,
            flight_endpoint=flight_endpoint,
            purpose=purpose,
            bearer_token=bearer_token,
        )

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
            matching_strategy: ``"all"`` (AND), ``"last"``, ``"frequency"``, or
                ``"any"`` (OR, default). Controls which query terms are required (#967).
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
        data = self._sync.post("/search", payload)
        return SearchResponse.model_validate(data)

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
        data = await self._async.post("/search", payload)
        return SearchResponse.model_validate(data)

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
        return self._sync.post("/multi-search", {"queries": queries})

    async def amulti_search(
        self,
        queries: list[dict[str, object]],
    ) -> dict[str, object]:
        """Async variant of :meth:`multi_search`."""
        return await self._async.post("/multi-search", {"queries": queries})

    def graphql(
        self,
        query: str,
        variables: dict[str, object] | None = None,
        operation_name: str | None = None,
    ) -> dict[str, object] | list[dict[str, object]] | None:
        """Execute a GraphQL query against the governed query path (ADR-220).

        Args:
            query: The GraphQL query string.
            variables: Optional variables map.
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
        return _graphql_data(self._sync.post("/graphql", body))

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
        return _graphql_data(await self._async.post("/graphql", body))

    # ------------------------------------------------------------------
    # Type management & ontology (#967)
    # ------------------------------------------------------------------

    def list_types(self) -> dict[str, object]:
        """List all registered object types with row counts."""
        return self._sync.get("/types")

    def register_type(self, name: str, **kwargs: object) -> dict[str, object]:
        """Register a custom object type at runtime. ``kwargs`` may include
        ``description``, ``owner``, ``properties``, ``computed_columns``."""
        payload: dict[str, object] = {"name": name, **kwargs}
        return self._sync.post("/types", payload)

    def deregister_type(self, name: str) -> dict[str, object]:
        """Deregister a custom type. Admin token required."""
        return self._sync.delete(f"/types/{name}")

    def type_detail(self, name: str) -> dict[str, object]:
        """Get type detail (properties, owner, row count)."""
        return self._sync.get(f"/types/{name}")

    def schema_alter(
        self,
        type_name: str,
        action: str,
        column: str,
        *,
        new_column: str | None = None,
        col_type: str | None = None,
        optional: bool | None = None,
    ) -> dict[str, object]:
        """Online schema evolution — ``PATCH /types/:name/schema`` (#1307).

        ``action`` is ``"add"`` | ``"drop"`` | ``"rename"`` | ``"retype"``;
        ``column`` is the target column; ``new_column`` / ``col_type`` /
        ``optional`` apply to the relevant actions.
        """
        payload: dict[str, object] = {"action": action, "column": column}
        if new_column is not None:
            payload["new_column"] = new_column
        if col_type is not None:
            payload["col_type"] = col_type
        if optional is not None:
            payload["optional"] = optional
        return self._sync.patch(f"/types/{type_name}/schema", payload)

    # ------------------------------------------------------------------
    # Schema-BRANCH CRUD + typed edge definitions (#2497, #2476 ALTER stays
    # distinct — these are the schema-BRANCH surface mirrored from the Rust
    # flat client at crates/relata-sdk-rust/src/http.rs:2535-2570).
    # ------------------------------------------------------------------

    def list_edge_types(self) -> dict[str, object]:
        """List registered cross-type edge definitions (ADR-007).

        Wraps ``GET /types/edges``. Returns the server's edge-declaration
        table consumed by ``PATHS_BETWEEN`` traversal.
        """
        return self._sync.get("/types/edges")

    def register_edge_type(
        self,
        from_type: str,
        to_type: str,
        label: str,
    ) -> dict[str, object]:
        """Register a typed named edge for graph traversal (ADR-007).

        Wraps ``POST /types/edges`` with ``{from_type, to_type, label}``.
        """
        return self._sync.post(
            "/types/edges",
            {"from_type": from_type, "to_type": to_type, "label": label},
        )

    def create_schema_branch(
        self,
        name: str,
        from_branch: str,
    ) -> dict[str, object]:
        """Create a named schema branch off ``from_branch`` (#207).

        Wraps ``POST /schema/branches/:name`` with ``{"from": from_branch}``.
        The branch carries its own ontology / type set; mutations on it do not
        affect the parent until merged.
        """
        return self._sync.post(
            f"/schema/branches/{name}",
            {"from": from_branch},
        )

    def delete_schema_branch(self, name: str) -> dict[str, object]:
        """Delete a named schema branch (#207).

        Wraps ``DELETE /schema/branches/:name``.
        """
        return self._sync.delete(f"/schema/branches/{name}")

    async def alist_edge_types(self) -> dict[str, object]:
        """Async variant of :meth:`list_edge_types`."""
        return await self._async.get("/types/edges")

    async def aregister_edge_type(
        self,
        from_type: str,
        to_type: str,
        label: str,
    ) -> dict[str, object]:
        """Async variant of :meth:`register_edge_type`."""
        return await self._async.post(
            "/types/edges",
            {"from_type": from_type, "to_type": to_type, "label": label},
        )

    async def acreate_schema_branch(
        self,
        name: str,
        from_branch: str,
    ) -> dict[str, object]:
        """Async variant of :meth:`create_schema_branch`."""
        return await self._async.post(
            f"/schema/branches/{name}",
            {"from": from_branch},
        )

    async def adelete_schema_branch(self, name: str) -> dict[str, object]:
        """Async variant of :meth:`delete_schema_branch`."""
        return await self._async.delete(f"/schema/branches/{name}")

    def ontology_migrate(self, schema: dict[str, object]) -> dict[str, object]:
        """SHACL schema migration — register type specs, link types, property
        constraints in one governed call."""
        return self._sync.post("/ontology/migrate", schema)

    def enrichment_rules(self, rules: dict[str, object]) -> dict[str, object]:
        """Register identity enrichment rules for SmartIngest."""
        return self._sync.post("/ontology/enrichment-rules", rules)

    def list_modules(self) -> dict[str, object]:
        """List installed modules / extensions."""
        return self._sync.get("/modules")

    def create_link(
        self,
        link_name: str,
        source_id: str,
        source_type: str,
        target_id: str,
        target_type: str,
    ) -> dict[str, object]:
        """Create a typed, governed link (edge) between two objects."""
        return self._sync.post("/links", {
            "link_name": link_name,
            "source_id": source_id,
            "source_type": source_type,
            "target_id": target_id,
            "target_type": target_type,
        })

    # ------------------------------------------------------------------
    # Identity resolution & entity lifecycle (#967)
    # ------------------------------------------------------------------

    def resolve_identity(
        self,
        value: str,
        *,
        mode: str | None = None,
        purpose: str | None = None,
    ) -> dict[str, object]:
        """Resolve an identity value to all known objects and clusters.
        Executes ``RESOLVE_IDENTITY('<value>')`` via ``POST /query``.

        ``mode`` selects the resolution strategy: ``"canonical"`` (single
        best match), ``"cluster"`` (server default — same as
        :meth:`identity_cluster`), or ``"fuse"`` (merge overlapping
        clusters). Omit to use the server default (``cluster``).
        """
        p = purpose or self._default_purpose or "analytics"
        escaped = value.replace(chr(39), chr(39) + chr(39))
        sql = f"RESOLVE_IDENTITY('{escaped}')"
        if mode is not None:
            if mode not in ("canonical", "cluster", "fuse"):
                raise ValueError(
                    f"mode must be one of 'canonical', 'cluster', 'fuse'; got {mode!r}"
                )
            sql = f"RESOLVE_IDENTITY('{escaped}', MODE => '{mode}')"
        return self._sync.post("/query", {"purpose": p, "sql": sql})

    def detect_identities(self, text: str, *, purpose: str | None = None) -> dict[str, object]:
        """Detect identities in free text via SmartIngest.
        Executes ``DETECT_IDENTITIES('<text>')``."""
        p = purpose or self._default_purpose or "analytics"
        sql = f"DETECT_IDENTITIES('{text.replace(chr(39), chr(39)+chr(39))}')"
        return self._sync.post("/query", {"purpose": p, "sql": sql})

    def erase_subject(
        self, subject: str, *, reason: str = "gdpr-art17-request", purpose: str | None = None,
    ) -> dict[str, object]:
        """GDPR Art. 17 erasure: crypto-shred every row, vector, and blob
        linked to a subject. **Irreversible.**"""
        p = purpose or self._default_purpose or "gdpr-erasure"
        s = subject.replace(chr(39), chr(39) + chr(39))
        r = reason.replace(chr(39), chr(39) + chr(39))
        sql = f"ERASE SUBJECT '{s}' REASON '{r}' CERTIFY"
        return self._sync.post("/query", {"purpose": p, "sql": sql})

    # ------------------------------------------------------------------
    # SPARQL, sessions & cluster (#967 Tier 2d)
    # ------------------------------------------------------------------

    def export_data(self, object_type: str, *, format: str = "json") -> dict[str, object]:
        """Bulk export all rows of a type (#967 Tier 5c)."""
        return self._sync.get(f"/export?type={object_type}&format={format}&purpose=export")

    def register_webhook(self, url: str, event_types: list[str] | None = None) -> dict[str, object]:
        """Register a webhook for push notifications (#967 Tier 5b)."""
        return self._sync.post("/webhooks", {"url": url, "event_types": event_types or []})

    def list_webhooks(self) -> dict[str, object]:
        """List registered webhooks."""
        return self._sync.get("/webhooks")

    def delete_webhook(self, webhook_id: str) -> dict[str, object]:
        """Delete a webhook."""
        return self._sync.delete(f"/webhooks/{webhook_id}")

    def sparql(self, query: str) -> dict[str, object]:
        """Execute a SPARQL query."""
        return self._sync.post("/sparql", {"query": query})

    def cluster_topology(self) -> dict[str, object]:
        """Get cluster topology (nodes, partitions, roles)."""
        return self._sync.get("/cluster/topology")

    def cluster_rebalance(self) -> dict[str, object]:
        """Trigger a cluster rebalance."""
        return self._sync.post("/cluster/rebalance", {})

    def cluster_drain(self, node_id: str) -> dict[str, object]:
        """Drain a node for maintenance."""
        return self._sync.post(f"/cluster/drain/{node_id}", {})

    def session_diff(self, session_id: str) -> dict[str, object]:
        """View uncommitted session changes."""
        return self._sync.get(f"/session/{session_id}/diff")

    def session_commit(self, session_id: str) -> dict[str, object]:
        """Commit a session's draft writes."""
        return self._sync.post(f"/session/{session_id}/commit", {})

    def session_discard(self, session_id: str) -> dict[str, object]:
        """Discard uncommitted session changes."""
        return self._sync.delete(f"/session/{session_id}/draft")

    # ------------------------------------------------------------------
    # Entity merge, dedup & identity resolution (#967)
    # ------------------------------------------------------------------

    def identity_cluster(self, value: str, *, purpose: str | None = None) -> dict[str, object]:
        """Resolve an identity to its full cluster of linked identifiers."""
        p = purpose or self._default_purpose or "analytics"
        v = value.replace("'", "''")
        return self._sync.post("/query", {"purpose": p, "sql": f"RESOLVE_IDENTITY('{v}', MODE => 'cluster')"})

    def same_identity(self, a: str, b: str, *, purpose: str | None = None) -> bool:
        """Predicate: do two identifiers resolve to the same entity?

        Executes ``SAME_IDENTITY('<a>', '<b>')`` via ``POST /query`` and
        returns the ``match`` verdict (#2246).
        """
        p = purpose or self._default_purpose or "analytics"
        ca = a.replace("'", "''")
        cb = b.replace("'", "''")
        sql = f"SAME_IDENTITY('{ca}', '{cb}')"
        data = self._sync.post("/query", {"purpose": p, "sql": sql})
        rows = data.get("data") or []
        return bool(rows and rows[0].get("match"))

    def fuse_identities(self, id_a: str, id_b: str, *, purpose: str | None = None) -> dict[str, object]:
        """Ontological merge of two identities — writes an IdentityLink with
        link_type='fused' and returns the merged cluster (#967)."""
        p = purpose or self._default_purpose or "analytics"
        a = id_a.replace("'", "''")
        b = id_b.replace("'", "''")
        return self._sync.post("/query", {"purpose": p, "sql": f"FUSE_IDENTITIES('{a}', '{b}')"})
    def split_identities(self, id_a: str, id_b: str, *, purpose: str | None = None) -> dict[str, object]:
        """Ontological unmerge — inverse of fuse_identities (#967)."""
        p = purpose or self._default_purpose or "analytics"
        a = id_a.replace("'", "''"); b = id_b.replace("'", "''")
        return self._sync.post("/query", {"purpose": p, "sql": f"SPLIT_IDENTITIES('{a}', '{b}')"})


    # ------------------------------------------------------------------
    # Graph algorithm operators (#967)
    # ------------------------------------------------------------------

    def graph_dijkstra(self, object_type: str, from_id: str, to_id: str, *, purpose: str | None = None) -> dict[str, object]:
        """Shortest path between two entities."""
        p = purpose or self._default_purpose or "analytics"
        sql = f"GRAPH_DIJKSTRA('{object_type}', FROM => '{from_id}', TO => '{to_id}')"
        return self._sync.post("/query", {"purpose": p, "sql": sql})

    def graph_shortest_path(
        self,
        from_id: str,
        to_id: str,
        *,
        max_hops: int | None = None,
    ) -> dict[str, object]:
        """Shortest path via the governed ``GET /graph/shortest-path`` door.

        Unlike :meth:`graph_dijkstra` (a SQL operator), this exposes the HTTP
        door's ``max_hops`` ceiling (#2344).
        """
        params = {"from": from_id, "to": to_id}
        if max_hops is not None:
            params["max_hops"] = max_hops
        return self._sync.get(f"/graph/shortest-path?{urlencode(params)}")

    def graph_traverse(
        self,
        from_id: str,
        *,
        direction: str | None = None,
        max_depth: int | None = None,
        limit: int | None = None,
    ) -> dict[str, object]:
        """Breadth-first traversal via the governed ``GET /graph/traverse`` door.

        ``direction`` is ``"out"`` | ``"in"`` | ``"both"`` (server default out).
        """
        params = {"from": from_id}
        if direction is not None:
            params["direction"] = direction
        if max_depth is not None:
            params["max_depth"] = max_depth
        if limit is not None:
            params["limit"] = limit
        return self._sync.get(f"/graph/traverse?{urlencode(params)}")

    def graph_pagerank(self, object_type: str, *, damping: float | None = None, max_iter: int | None = None, purpose: str | None = None) -> dict[str, object]:
        """PageRank centrality."""
        p = purpose or self._default_purpose or "analytics"
        parts = [f"GRAPH_PAGERANK('{object_type}'"]
        if damping is not None: parts.append(f", DAMPING => {damping}")
        if max_iter is not None: parts.append(f", MAX_ITER => {max_iter}")
        parts.append(")")
        return self._sync.post("/query", {"purpose": p, "sql": "".join(parts)})

    def graph_scc(self, object_type: str, *, purpose: str | None = None) -> dict[str, object]:
        """Strongly connected components (fraud-ring detection)."""
        p = purpose or self._default_purpose or "analytics"
        return self._sync.post("/query", {"purpose": p, "sql": f"GRAPH_SCC('{object_type}')"})

    def graph_cycles(self, object_type: str, *, purpose: str | None = None) -> dict[str, object]:
        """Cycle detection in the graph."""
        p = purpose or self._default_purpose or "analytics"
        return self._sync.post("/query", {"purpose": p, "sql": f"GRAPH_CYCLES('{object_type}')"})

    def graph_community(self, object_type: str, *, purpose: str | None = None) -> dict[str, object]:
        """Community detection via label propagation."""
        p = purpose or self._default_purpose or "analytics"
        return self._sync.post("/query", {"purpose": p, "sql": f"GRAPH_COMMUNITY('{object_type}')"})

    def graph_node_similarity(self, object_type: str, node: str, *, purpose: str | None = None) -> dict[str, object]:
        """Node similarity — find entities similar to a seed node."""
        p = purpose or self._default_purpose or "analytics"
        return self._sync.post("/query", {"purpose": p, "sql": f"GRAPH_NODE_SIMILARITY('{object_type}', NODE => '{node}')"})

    def graph_link_predict(self, object_type: str, *, purpose: str | None = None) -> dict[str, object]:
        """Link prediction — predict missing relationships."""
        p = purpose or self._default_purpose or "analytics"
        return self._sync.post("/query", {"purpose": p, "sql": f"GRAPH_LINK_PREDICT('{object_type}')"})

    def graph_triangle_count(self, object_type: str, *, purpose: str | None = None) -> dict[str, object]:
        """Triangle count — measures graph density / cohesion."""
        p = purpose or self._default_purpose or "analytics"
        return self._sync.post("/query", {"purpose": p, "sql": f"TRIANGLE_COUNT('{object_type}')"})

    # ------------------------------------------------------------------
    # Intelligence operators (#967)
    # ------------------------------------------------------------------

    def beneficial_ownership_chain(self, party: str, *, max_depth: int | None = None, purpose: str | None = None) -> dict[str, object]:
        """Trace ownership to ultimate beneficial owner."""
        p = purpose or self._default_purpose or "compliance"
        parts = [f"BENEFICIAL_OWNERSHIP_CHAIN('{party}'"]
        if max_depth is not None: parts.append(f", MAX_DEPTH => {max_depth}")
        parts.append(")")
        return self._sync.post("/query", {"purpose": p, "sql": "".join(parts)})

    def sanctions_screen(self, party: str, *, threshold: float | None = None, purpose: str | None = None) -> dict[str, object]:
        """Sanctions screening with fuzzy threshold."""
        p = purpose or self._default_purpose or "compliance"
        parts = [f"SANCTIONS_SCREEN('{party}'"]
        if threshold is not None: parts.append(f", THRESHOLD => {threshold}")
        parts.append(")")
        return self._sync.post("/query", {"purpose": p, "sql": "".join(parts)})

    def convoy_detect(self, *, radius_m: float | None = None, time_tol_secs: int | None = None, min_points: int | None = None, purpose: str | None = None) -> dict[str, object]:
        """Convoy detection — find entities traveling together."""
        p = purpose or self._default_purpose or "analytics"
        parts = []
        if radius_m is not None: parts.append(f"RADIUS => {radius_m}")
        if time_tol_secs is not None: parts.append(f"TIME_TOL => {time_tol_secs * 1_000_000_000}")
        if min_points is not None: parts.append(f"MIN_POINTS => {min_points}")
        sql = f"CONVOY({', '.join(parts)})" if parts else "CONVOY()"
        return self._sync.post("/query", {"purpose": p, "sql": sql})

    def burner_detect(self, *, max_age_days: int | None = None, max_calls: int | None = None, purpose: str | None = None) -> dict[str, object]:
        """Burner phone detection."""
        p = purpose or self._default_purpose or "analytics"
        parts = []
        if max_age_days is not None: parts.append(f"MAX_AGE => {max_age_days * 86_400_000_000_000}")
        if max_calls is not None: parts.append(f"MAX_CALLS => {max_calls}")
        sql = f"BURNER_DETECT({', '.join(parts)})" if parts else "BURNER_DETECT()"
        return self._sync.post("/query", {"purpose": p, "sql": sql})

    def crypto_trace(self, entity: str, *, purpose: str | None = None) -> dict[str, object]:
        """Follow cryptocurrency fund flow."""
        p = purpose or self._default_purpose or "analytics"
        return self._sync.post("/query", {"purpose": p, "sql": f"CRYPTO_TRACE('{entity}')"})

    def wire_reconstruction(
        self, account: str, *, tolerance_pct: float | None = None, purpose: str | None = None
    ) -> dict[str, object]:
        """Reconstruct a wire-transfer chain (FinINT, #2249)."""
        p = purpose or self._default_purpose or "analytics"
        parts = [f"WIRE_RECONSTRUCTION('{account}'"]
        if tolerance_pct is not None:
            parts.append(f", TOLERANCE_PCT => {tolerance_pct}")
        parts.append(")")
        return self._sync.post("/query", {"purpose": p, "sql": "".join(parts)})

    def hawala_trace(
        self, seed: str, *, max_hops: int | None = None, purpose: str | None = None
    ) -> dict[str, object]:
        """Trace an informal hawala value-transfer network (FinINT, #2249)."""
        p = purpose or self._default_purpose or "analytics"
        hops = 5 if max_hops is None else max(1, min(10, max_hops))
        return self._sync.post(
            "/query", {"purpose": p, "sql": f"HAWALA_TRACE('{seed}', MAX_HOPS => {hops})"}
        )

    def dns_tunnel_detect(self, entity: str, *, purpose: str | None = None) -> dict[str, object]:
        """DNS tunnel detection."""
        p = purpose or self._default_purpose or "security"
        return self._sync.post("/query", {"purpose": p, "sql": f"DNS_TUNNEL_DETECT('{entity}')"})

    def crime_pattern_cluster(self, area: str, *, purpose: str | None = None) -> dict[str, object]:
        """Spatial crime pattern analysis."""
        p = purpose or self._default_purpose or "analytics"
        return self._sync.post("/query", {"purpose": p, "sql": f"CRIME_PATTERN_CLUSTER('{area}')"})

    def geofence(self, fence: str, *, target_type: str | None = None, purpose: str | None = None) -> dict[str, object]:
        """Geo-fence query — find entities within a geographic fence."""
        p = purpose or self._default_purpose or "analytics"
        parts = [f"GEOFENCE('{fence}'"]
        if target_type is not None: parts.append(f", TARGET_TYPE => '{target_type}'")
        parts.append(")")
        return self._sync.post("/query", {"purpose": p, "sql": "".join(parts)})

    # ------------------------------------------------------------------
    # Maritime operators (#2247)
    # ------------------------------------------------------------------

    def vessel_track(self, mmsi: int, *, window_secs: int | None = None, purpose: str | None = None) -> dict[str, object]:
        """Track a vessel's AIS position reports over a time window."""
        p = purpose or self._default_purpose or "analytics"
        parts = [f"VESSEL_TRACK({mmsi}"]
        if window_secs is not None: parts.append(f", WINDOW => {window_secs * 1_000_000_000}")
        parts.append(")")
        return self._sync.post("/query", {"purpose": p, "sql": "".join(parts)})

    def dark_fleet_detect(self, *, max_gap_hours: float | None = None, purpose: str | None = None) -> dict[str, object]:
        """Detect vessels with AIS gaps (transponder-off 'going dark' periods)."""
        p = purpose or self._default_purpose or "analytics"
        if max_gap_hours is not None:
            sql = f"DARK_FLEET_DETECT(MAX_GAP_HOURS => {max_gap_hours})"
        else:
            sql = "DARK_FLEET_DETECT()"
        return self._sync.post("/query", {"purpose": p, "sql": sql})

    def vessel_to_vessel_transfer(self, *, proximity_nm: float | None = None, time_window_minutes: int | None = None, purpose: str | None = None) -> dict[str, object]:
        """Detect ship-to-ship transfers (close-proximity vessel encounters)."""
        p = purpose or self._default_purpose or "analytics"
        parts = []
        if proximity_nm is not None: parts.append(f"PROXIMITY_NM => {proximity_nm}")
        if time_window_minutes is not None: parts.append(f"TIME_WINDOW_MINUTES => {time_window_minutes}")
        sql = f"VESSEL_TO_VESSEL_TRANSFER({', '.join(parts)})" if parts else "VESSEL_TO_VESSEL_TRANSFER()"
        return self._sync.post("/query", {"purpose": p, "sql": sql})

    # ------------------------------------------------------------------
    # Multimedia operators (#2251, epic #655 — ADR-030)
    # ------------------------------------------------------------------
    #
    # Biometric / perceptual-hash search via the governed SQL operator door.
    # Both route through ``POST /query`` so PURPOSE / ACL / cell-masking /
    # tenant isolation apply identically to a hand-written query. The server
    # operators are registered in ``relata_query::parser``:
    #
    # - ``FACE_SEARCH('<csv floats>', '<gallery>', K => n, THRESHOLD => f)``
    #   (cosine k-NN over ``MediaEmbedding`` rows; columns ``entity_id``,
    #   ``gallery_id``, ``score``, ``modality``).
    # - ``MATCH_PDQ('<hash>', '<corpus>', THRESHOLD => f)``
    #   (Hamming similarity over ``MediaHash`` rows; columns ``entity_id``,
    #   ``media_type``, ``corpus_id``, ``score``, ``matcher``).
    #
    # The MCP ``face_match`` tool is gated behind ADR-155 (#409) and returns
    # 403 — these SQL-reachable operators are the unblocked path the SDK uses.

    def face_search(
        self,
        gallery_id: str,
        embedding: list[float] | str,
        *,
        k: int = 10,
        threshold: float = 0.7,
        purpose: str | None = None,
    ) -> QueryResult:
        """Biometric face k-NN search against a gallery (#2251, ADR-030).

        Executes ``SELECT * FROM FACE_SEARCH(...)`` through the governed
        ``/query`` door.

        Args:
            gallery_id: Gallery to search (matches ``MediaEmbedding.gallery_id``).
            embedding: Probe face embedding as a list of floats, or a
                pre-formatted comma-separated string (the form the SQL
                operator expects).
            k: Max candidates (server default 10).
            threshold: Minimum cosine similarity in ``[0.0, 1.0]`` (default 0.7).
            purpose: Purpose override for this query.

        Returns:
            :class:`~relata.models.QueryResult` with columns ``entity_id``,
            ``gallery_id``, ``score``, ``modality``, ranked by score desc.

        Example::

            result = client.face_search(
                "gallery-1", [0.1, 0.2, 0.3], k=5, threshold=0.6,
                purpose="investigation",
            )
            for row in result:
                print(row["entity_id"], row["score"])
        """
        sql = _face_search_sql(gallery_id, embedding, k=k, threshold=threshold)
        return self.query(sql, purpose=purpose)

    async def aface_search(
        self,
        gallery_id: str,
        embedding: list[float] | str,
        *,
        k: int = 10,
        threshold: float = 0.7,
        purpose: str | None = None,
    ) -> QueryResult:
        """Async variant of :meth:`face_search`."""
        sql = _face_search_sql(gallery_id, embedding, k=k, threshold=threshold)
        return await self.aquery(sql, purpose=purpose)

    def match_pdq(
        self,
        corpus_id: str,
        query_hash: str,
        *,
        threshold: float = 0.9,
        purpose: str | None = None,
    ) -> QueryResult:
        """Perceptual-hash (PDQ) near-duplicate search over a corpus (#2251).

        Executes ``SELECT * FROM MATCH_PDQ(...)`` through the governed
        ``/query`` door. PDQ near-duplicates differ by ≤ 31 bits (ADR-187).

        Args:
            corpus_id: Hash corpus to search (matches ``MediaHash.corpus_id``).
            query_hash: PDQ hash as a hex string.
            threshold: Minimum Hamming similarity in ``[0.0, 1.0]`` (default 0.9).
            purpose: Purpose override for this query.

        Returns:
            :class:`~relata.models.QueryResult` with columns ``entity_id``,
            ``media_type``, ``corpus_id``, ``score``, ``matcher``, ranked by
            score desc.
        """
        sql = _match_pdq_sql(corpus_id, query_hash, threshold=threshold)
        return self.query(sql, purpose=purpose)

    async def amatch_pdq(
        self,
        corpus_id: str,
        query_hash: str,
        *,
        threshold: float = 0.9,
        purpose: str | None = None,
    ) -> QueryResult:
        """Async variant of :meth:`match_pdq`."""
        sql = _match_pdq_sql(corpus_id, query_hash, threshold=threshold)
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
        data = self._sync.get("/status")
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
        data = self._sync.get("/audit/count")
        return AuditCountResponse.model_validate(data)

    def cluster_nodes(self) -> list[ClusterNode]:
        """List all nodes in the cluster (synchronous).

        Returns:
            List of :class:`~relata.models.ClusterNode`.

        Raises:
            :class:`~relata.exceptions.ConnectionError`: Server unreachable.
        """
        data = self._sync.get("/cluster/nodes")
        nodes_raw = data.get("nodes", [])
        return [ClusterNode.model_validate(n) for n in nodes_raw]

    # ------------------------------------------------------------------
    # Introspection (stats / version / ready) — pairs with #86
    # ------------------------------------------------------------------

    def stats(self) -> Stats:
        """Return engine-wide counts for health dashboards (synchronous).

        Wraps ``GET /debug/stats``. The shape mirrors the
        ``storage-backend-requirements.md`` §9 contract — ``records``,
        ``states``, ``snapshot_rows``, ``log_leaves``, ``tokens`` — to the
        extent the server currently distinguishes them.
        """
        data = self._sync.get("/debug/stats")
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
    # Async API
    # ------------------------------------------------------------------

    async def aquery(self, sql: str, *, purpose: str | None = None) -> QueryResult:
        """Execute a SQL query against the Relata engine (asynchronous).

        Identical to :meth:`query` but non-blocking.  Suitable for use in
        ``asyncio`` event loops, ``FastAPI`` route handlers, Jupyter notebooks
        with ``%autoawait``, etc.

        Args:
            sql: SQL statement to execute.
            purpose: Purpose override for this query.

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
        payload = {"purpose": effective_purpose, "sql": sql}
        data = await self._async.post("/query", payload)
        return QueryResult.model_validate(data)

    async def aquery_params(
        self,
        sql: str,
        params: list,
        *,
        purpose: str | None = None,
    ) -> QueryResult:
        """Async parameterized query — see :meth:`query_params` (#1162)."""
        sql = _rewrite_question_mark_params(sql)
        effective_purpose = self._resolve_purpose(purpose)
        payload = {"purpose": effective_purpose, "sql": sql, "params": params}
        data = await self._async.post("/query", payload)
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
        data = await self._async.get("/status")
        return StatusResponse.model_validate(data)

    async def aaudit_count(self) -> AuditCountResponse:
        """Retrieve audit log entry count and chain integrity (asynchronous).

        Returns:
            :class:`~relata.models.AuditCountResponse`.
        """
        data = await self._async.get("/audit/count")
        return AuditCountResponse.model_validate(data)

    async def acluster_nodes(self) -> list[ClusterNode]:
        """List all nodes in the cluster (asynchronous).

        Returns:
            List of :class:`~relata.models.ClusterNode`.
        """
        data = await self._async.get("/cluster/nodes")
        nodes_raw = data.get("nodes", [])
        return [ClusterNode.model_validate(n) for n in nodes_raw]

    # ------------------------------------------------------------------
    # Introspection — async mirrors
    # ------------------------------------------------------------------

    async def astats(self) -> Stats:
        """Return engine-wide counts for health dashboards (asynchronous)."""
        data = await self._async.get("/debug/stats")
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
            docs.write([{"id": "d1", "title": "..."}])      # schemaless (T6, /ingest/auto)
            res = docs.query(                                # typed search (T1, /search)
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
        chunks_jsonl: str,
        manifest_json: str,
    ) -> IngestDocumentResponse:
        """Ingest a datagrep extractor document into Relata (synchronous).

        Submits the ``_chunks.jsonl`` and ``_manifest.json`` outputs from a
        ``dgrep`` extraction run to ``POST /ingest/document``.  The server
        parses and version-checks the protocol envelope, then queues the chunks
        for storage.

        Args:
            chunks_jsonl: Newline-delimited JSON string — one chunk per line,
                as produced by ``dgrep run --mode chunks``.
            manifest_json: JSON string of the extraction manifest, as produced
                by ``dgrep run``.

        Returns:
            :class:`~relata.models.IngestDocumentResponse` with ``report_id``,
            ``chunks_ingested``, ``warnings``, and ``queue_depth``.

        Raises:
            :class:`~relata.exceptions.AuthError`: Invalid/missing token.
            :class:`~relata.exceptions.QuotaError`: Ingest queue full (HTTP 429).
            :class:`~relata.exceptions.ConnectionError`: Server unreachable.
            :class:`~relata.exceptions.RelataError`: Protocol error (HTTP 422)
                or any other server-side failure.

        Examples::

            with open("report_chunks.jsonl") as f:
                chunks = f.read()
            with open("report_manifest.json") as f:
                manifest = f.read()

            result = client.ingest_document(chunks, manifest)
            print(f"Ingested {result.chunks_ingested} chunks as {result.report_id}")
        """
        payload = {"chunks_jsonl": chunks_jsonl, "manifest_json": manifest_json}
        data = self._sync.post("/ingest/document", payload)
        return IngestDocumentResponse.model_validate(data)

    async def aingest_document(
        self,
        chunks_jsonl: str,
        manifest_json: str,
    ) -> IngestDocumentResponse:
        """Ingest a datagrep extractor document into Relata (asynchronous).

        Identical to :meth:`ingest_document` but non-blocking.

        Args:
            chunks_jsonl: Newline-delimited JSON chunk data.
            manifest_json: JSON extraction manifest.

        Returns:
            :class:`~relata.models.IngestDocumentResponse`.
        """
        payload = {"chunks_jsonl": chunks_jsonl, "manifest_json": manifest_json}
        data = await self._async.post("/ingest/document", payload)
        return IngestDocumentResponse.model_validate(data)

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
