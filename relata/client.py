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
    ) -> None:
        self._base_url = base_url.rstrip("/")
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

        effective_purpose = self._resolve_purpose(purpose)
        chunks = self.streaming.query_arrow_raw(sql, purpose=effective_purpose)
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
    ) -> SearchResponse:
        """Full-text search over a governed object type (#670).

        Args:
            query: Free-text search string (BM25 + vector hybrid).
            type: Object type to search (e.g. ``"Person"``).
            limit: Maximum number of hits to return (server default: 20).
            facets: Field names to aggregate counts for.
            highlight: Include field-level ``<em>`` snippets.
            filters: Equality filters applied server-side (``{"field": "val"}``).
            matching_strategy: ``"all"`` (AND), ``"last"``, ``"frequency"``, or
                ``"any"`` (OR, default). Controls which query terms are required (#967).
            typo_tolerance: Dict with ``enabled``, ``min_word_size``,
                ``disable_on_words``, ``disable_on_attributes`` (#967).

        Returns:
            :class:`~relata.models.SearchResponse` with ``hits``, ``total``,
            ``facets``, ``estimated_total_hits``, and ``processing_time_ms``.

        Example::

            results = client.search(
                "alice smith", "Person",
                limit=10, facets=["tenant_id"], highlight=True,
                matching_strategy="all",
            )
            for hit in results.hits:
                print(hit.score, hit.fields.get("name"))
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
    ) -> SearchResponse:
        """Async variant of :meth:`search`."""
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

    def resolve_identity(self, value: str, *, purpose: str | None = None) -> dict[str, object]:
        """Resolve an identity value to all known objects and clusters.
        Executes ``RESOLVE_IDENTITY('<value>')`` via ``POST /query``."""
        p = purpose or self._default_purpose or "analytics"
        sql = f"RESOLVE_IDENTITY('{value.replace(chr(39), chr(39)+chr(39))}')"
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
