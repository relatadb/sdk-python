"""
Relata client — the primary entry point for the Python SDK.

Usage::

    from relata import RelataClient

    with RelataClient("http://localhost:8080", purpose="analytics") as client:
        result = client.query("SELECT * FROM Person LIMIT 10")
        for row in result:
            print(row)

Async usage::

    import asyncio
    from relata import RelataClient

    async def main():
        async with RelataClient("http://localhost:8080", purpose="analytics") as client:
            result = await client.aquery("SELECT * FROM Person LIMIT 10")
            for row in result:
                print(row)

    asyncio.run(main())
"""

from __future__ import annotations

from types import TracebackType
from typing import TYPE_CHECKING

from relata._http import AsyncHttpTransport, HttpTransport
from relata.exceptions import PurposeError
from relata.models import (
    AuditCountResponse,
    ClusterNode,
    HealthResponse,
    IngestDocumentResponse,
    QueryResult,
    StatusResponse,
)

if TYPE_CHECKING:
    from relata.query import QueryBuilder


class RelataClient:
    """Synchronous and asynchronous client for the Relata HTTP API.

    ``RelataClient`` supports both synchronous calls (via :meth:`query`,
    :meth:`health`, etc.) and their async counterparts (via :meth:`aquery`,
    :meth:`ahealth`, etc.).  Use it as a context manager — sync or async — to
    guarantee connection cleanup.

    Args:
        base_url: Base URL of the Relata server, e.g.
            ``"http://localhost:8080"``.  Trailing slashes are stripped.
        bearer_token: Optional Bearer token sent as
            ``Authorization: Bearer <token>``.  Required when the server is
            configured with ``RELATA_BEARER_TOKEN``.
        purpose: Default purpose string for all queries made through this
            client.  Can be overridden per-call.  If ``None`` you must pass
            ``purpose=`` to every :meth:`query` call.  Common values:
            ``"analytics"``, ``"audit"``, ``"analysis"``.
        timeout: HTTP request timeout in seconds (default ``30.0``).

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
        client = RelataClient("http://localhost:8080", purpose="analytics")
        result = client.query("SELECT * FROM Person LIMIT 5")
        client.close()

        # Preferred: context manager
        with RelataClient("http://localhost:8080", purpose="analytics") as client:
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
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._bearer_token = bearer_token
        self._default_purpose = purpose
        self._timeout = timeout

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
                self._base_url, self._bearer_token, self._timeout
            )
        return self.__sync_transport

    @property
    def _async(self) -> AsyncHttpTransport:
        if self.__async_transport is None:
            self.__async_transport = AsyncHttpTransport(
                self._base_url, self._bearer_token, self._timeout
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

    def __enter__(self) -> "RelataClient":
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

    async def __aenter__(self) -> "RelataClient":
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

            async with RelataClient("http://localhost:8080", purpose="audit") as client:
                result = await client.aquery("SELECT COUNT(*) FROM AuditLog")
                print(result.rows)
        """
        effective_purpose = self._resolve_purpose(purpose)
        payload = {"purpose": effective_purpose, "sql": sql}
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
    # Fluent query builder entry point
    # ------------------------------------------------------------------

    def select(self, *columns_or_table: str) -> "QueryBuilder":
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
