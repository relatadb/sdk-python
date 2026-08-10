"""
Pydantic v2 response models for the Relata HTTP API.

All fields mirror the exact JSON keys returned by the server.  Optional fields
use ``None`` as the default so that older server versions that omit a field
remain compatible.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


class QueryResult(BaseModel):
    """Result of a ``POST /query`` call.

    The server returns ``{"rows": <int count>, "columns": [...], "data": [...],
    "query_id": "...", "elapsed_ms": N}``. The SDK normalises this so callers
    always see ``rows`` as a list of dicts, regardless of the wire shape.

    Attributes:
        rows: List of result rows, each row is a ``dict`` mapping column name
            to value.  Values use Python-native types (int, float, str, bool,
            ``None``).
        query_id: Opaque server-assigned identifier for this query execution.
            Include this in bug reports or support tickets.
        elapsed_ms: Server-side execution time in milliseconds.  Does **not**
            include network round-trip time.
        row_count: Number of rows returned.  Always equals ``len(rows)``; the
            field exists as a convenience to avoid ``len()`` calls.
        columns: Column names in projection order (when the server provides them).
    """

    rows: list[dict[str, Any]] = Field(default_factory=list)
    query_id: str = Field("", description="Server-assigned query execution ID")
    elapsed_ms: int = Field(0, description="Server-side execution time in ms (legacy)", ge=0)
    processing_time_ms: int | None = Field(
        None, description="Server-side processing time in ms (#1252)", ge=0
    )
    row_count: int = Field(0, description="Number of rows returned")
    columns: list[str] = Field(default_factory=list, description="Column names in order")

    @model_validator(mode="before")
    @classmethod
    def _normalise_wire_shape(cls, data: Any) -> Any:  # noqa: ANN401
        """The server sends ``rows`` as an int count and the actual row data
        in ``data``. Normalise so ``rows`` is always a list."""
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                data["rows"] = data["data"]
            elif "rows" in data and isinstance(data["rows"], int):
                data["rows"] = []
            # Populate columns if the server sent them
            if "columns" not in data:
                data["columns"] = []
            # Back-fill processing_time_ms from elapsed_ms when the new field
            # is absent (older server versions) (#1252).
            if data.get("processing_time_ms") is None and data.get("elapsed_ms") is not None:
                data["processing_time_ms"] = data["elapsed_ms"]
        return data

    @model_validator(mode="after")
    def _sync_row_count(self) -> QueryResult:
        self.row_count = len(self.rows)
        return self

    # Make QueryResult iterable so callers can do `for row in result:`.
    def __iter__(self) -> Iterator[dict[str, Any]]:  # type: ignore[override]
        return iter(self.rows)

    def __len__(self) -> int:
        return self.row_count

    def __repr__(self) -> str:
        return (
            f"QueryResult("
            f"query_id={self.query_id!r}, "
            f"row_count={self.row_count}, "
            f"elapsed_ms={self.elapsed_ms})"
        )

    def __bool__(self) -> bool:
        return self.row_count > 0


# ---------------------------------------------------------------------------
# Health / Status
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response from ``GET /health``.

    Attributes:
        status: ``"ok"`` when the node is healthy.
        profile: Deployment profile — one of ``"free"``, ``"server"``,
            ``"cluster"``.
        node_id: Stable node identifier (e.g. ``"coordinator"``,
            ``"reader-1"``).
    """

    status: str = Field(..., description="'ok' when the node is healthy")
    profile: str = Field(..., description="Deployment profile: free | server | cluster")
    node_id: str = Field(..., description="Stable node identifier")

    @property
    def is_healthy(self) -> bool:
        """Return ``True`` when ``status == 'ok'``."""
        return self.status == "ok"

    def __repr__(self) -> str:
        return (
            f"HealthResponse("
            f"status={self.status!r}, "
            f"profile={self.profile!r}, "
            f"node_id={self.node_id!r})"
        )


class StatusResponse(BaseModel):
    """Response from ``GET /status``.

    Attributes:
        profile: Deployment profile — one of ``"free"``, ``"server"``,
            ``"cluster"``.
        role: Node role — one of ``"coordinator"``, ``"reader"``,
            ``"writer"``, ``"indexer"``.
        query_quota: Remaining query cost units for the current principal.
            Hard limit enforced by the server; queries that exceed this raise
            :class:`~relata.exceptions.QuotaError`.
    """

    profile: str = Field(..., description="Deployment profile")
    role: str = Field(..., description="Node role: coordinator | reader | writer | indexer")
    query_quota: int = Field(..., description="Remaining cost units for this principal", ge=0)

    def __repr__(self) -> str:
        return (
            f"StatusResponse("
            f"profile={self.profile!r}, "
            f"role={self.role!r}, "
            f"query_quota={self.query_quota})"
        )


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


class AuditCountResponse(BaseModel):
    """Response from ``GET /audit/count``.

    Attributes:
        entries: Total number of audit log entries recorded.
        chain_valid: ``True`` when the hash chain covering all entries is
            intact.  A ``False`` value indicates potential tampering and should
            be escalated immediately.
    """

    entries: int = Field(..., description="Total audit log entries", ge=0)
    chain_valid: bool = Field(..., description="True when the hash chain is intact")

    @property
    def is_tampered(self) -> bool:
        """Return ``True`` when the audit chain is broken (potential tampering)."""
        return not self.chain_valid

    def __repr__(self) -> str:
        return (
            f"AuditCountResponse("
            f"entries={self.entries}, "
            f"chain_valid={self.chain_valid})"
        )


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


class IngestDocumentResponse(BaseModel):
    """Response from ``POST /rag/ingest`` (renamed by #4499).

    Attributes:
        report_id: Server-assigned manifest ID for the ingested document.
        chunks_ingested: Number of chunks accepted into the ingest queue.
        warnings: Non-fatal protocol warnings (e.g. newer-minor-version fields).
        schema_version: Protocol version the server parsed the document as.
        queue_depth: Current ingest queue depth after this submission.
    """

    report_id: str = Field(..., description="Manifest ID assigned by the server")
    chunks_ingested: int = Field(..., description="Chunks accepted into the ingest queue", ge=0)
    warnings: list[str] = Field(default_factory=list, description="Non-fatal server warnings")
    schema_version: str = Field(..., description="Protocol version used during parsing")
    queue_depth: int = Field(0, description="Ingest queue depth after submission", ge=0)

    def __repr__(self) -> str:
        return (
            f"IngestDocumentResponse("
            f"report_id={self.report_id!r}, "
            f"chunks_ingested={self.chunks_ingested}, "
            f"schema_version={self.schema_version!r})"
        )


class DocumentUsageResponse(BaseModel):
    """Response from ``POST /rag/documents/{report_id}/usage`` (#4498).

    Reports the ``DocumentSource`` row's usage counters *after* applying this
    call's increments — ``citation_count``/``retrieval_count``/
    ``last_cited_at``/``feedback_avg`` are write-BACK signals maintained by
    repeated calls to this endpoint, not ingest-time constants.

    Attributes:
        report_id: The document this usage event was recorded against.
        citation_count: Total citations recorded so far.
        retrieval_count: Total retrievals recorded so far.
        last_cited_at: Nanoseconds since epoch of the most recent citation,
            or ``None`` if the document has never been cited.
        feedback_avg: Running mean of every ``feedback_score`` recorded so
            far, or ``None`` if none has been recorded yet.
    """

    report_id: str = Field(..., description="The DocumentSource this usage event targeted")
    citation_count: int | None = Field(None, description="Total citations recorded so far")
    retrieval_count: int | None = Field(None, description="Total retrievals recorded so far")
    last_cited_at: int | None = Field(
        None, description="Nanoseconds since epoch of the most recent citation"
    )
    feedback_avg: float | None = Field(
        None, description="Running mean of every feedback_score recorded so far"
    )

    def __repr__(self) -> str:
        return (
            f"DocumentUsageResponse("
            f"report_id={self.report_id!r}, "
            f"citation_count={self.citation_count}, "
            f"retrieval_count={self.retrieval_count})"
        )


# ---------------------------------------------------------------------------
# Cluster
# ---------------------------------------------------------------------------


class ClusterNode(BaseModel):
    """Metadata for a single node in a Relata cluster.

    Attributes:
        node_id: Stable identifier for this node (e.g. ``"coordinator"``,
            ``"reader-1"``).
        role: Node role — one of ``"coordinator"``, ``"reader"``,
            ``"writer"``, ``"indexer"``.
        url: Base HTTP URL for this node (e.g. ``"http://reader-1:8081"``).
    """

    node_id: str = Field(..., description="Stable node identifier")
    role: str = Field(..., description="Node role")
    url: str = Field(..., description="Base HTTP URL for this node")

    def __repr__(self) -> str:
        return (
            f"ClusterNode("
            f"node_id={self.node_id!r}, "
            f"role={self.role!r}, "
            f"url={self.url!r})"
        )


# ---------------------------------------------------------------------------
# Introspection — Version / Stats / ReadyReport (pairs with #86)
# ---------------------------------------------------------------------------


class VersionInfo(BaseModel):
    """Response from ``GET /version``.

    Attributes:
        version: Relata server version (e.g. ``"1.1.0"``).
        commit: Git commit hash the binary was built from.
        profile: Deployment profile — ``free`` / ``server`` / ``cluster``.
        schema_version: Ontology / row-model schema version, useful for
            migration gating.
        features: Optional list of compiled-in feature flags.
    """

    version: str = Field(..., description="Relata server version")
    commit: str | None = Field(None, description="Git commit hash")
    profile: str | None = Field(None, description="Deployment profile")
    schema_version: str | None = Field(None, description="Ontology schema version")
    features: list[str] = Field(default_factory=list, description="Compiled-in feature flags")

    def __repr__(self) -> str:
        return (
            f"VersionInfo("
            f"version={self.version!r}, "
            f"profile={self.profile!r}, "
            f"schema_version={self.schema_version!r})"
        )


class Stats(BaseModel):
    """Response from ``GET /debug/stats``.

    The shape mirrors the partner storage-backend contract §9 — every field
    the server populates is exposed; fields the server does not yet emit
    (e.g. ``log_leaves`` pending #85, ``tokens`` pending #84) default to
    ``None`` so the model is forward-compatible.

    Attributes:
        records: Total content-addressed blobs (partner §2).
        states: Total live rows across all types (partner §3).
        snapshot_rows: Total rows in incrementally-refreshed MVs (partner §4).
        log_leaves: Current WAL write_seq (partner §5; pending #85).
        tokens: Current dedup-token count (partner §7; pending #84).
        raw: The full server response, in case the caller wants a field the
            typed model does not surface.
    """

    records: int | None = Field(None, description="Total content-addressed blobs")
    states: int | None = Field(None, description="Total live rows")
    snapshot_rows: int | None = Field(None, description="Rows in MVs")
    log_leaves: int | None = Field(None, description="WAL write_seq (pending #85)")
    tokens: int | None = Field(None, description="Dedup token count (pending #84)")
    raw: dict[str, Any] = Field(
        default_factory=dict,
        description="Full server response for fields the typed model does not surface",
    )

    @model_validator(mode="after")
    def _capture_raw(self) -> Stats:
        # Pydantic v2 does not expose the input dict after validation; we use a
        # separate constructor on the client side if the caller wants the raw
        # payload. This validator is a no-op placeholder for forward-compat.
        return self

    def __repr__(self) -> str:
        return (
            f"Stats("
            f"records={self.records}, "
            f"states={self.states}, "
            f"snapshot_rows={self.snapshot_rows}, "
            f"log_leaves={self.log_leaves}, "
            f"tokens={self.tokens})"
        )


class ReadyReport(BaseModel):
    """Response from ``GET /health/ready``.

    Attributes:
        is_ready: ``True`` when the node is ready to serve (HTTP 200).
            ``False`` when any of the 9 downstream conditions trips (HTTP 503).
            Derived from the HTTP status / ``status`` field when the server
            omits it.
        status: Server-side status string (e.g. ``"ok"``, ``"shedding"``).
        reason: Machine-friendly tag identifying which condition tripped
            (``queue_backpressure`` / ``wal_failures`` / ``audit_drops`` /
            ``dead_worker`` / ``remote_backend_unreachable`` /
            ``kms_unreachable`` / ``replication_lag`` /
            ``embedder_circuit_open`` / ``lease_renewal_failures``).
        detail: Optional human-friendly explanation.
    """

    is_ready: bool | None = Field(None, description="True when the node is ready to serve")
    status: str = Field(..., description="Server-side status string")
    reason: str | None = Field(None, description="Machine-friendly shed reason")
    detail: str | None = Field(None, description="Human-friendly explanation")

    @model_validator(mode="after")
    def _derive_is_ready(self) -> ReadyReport:
        # The server returns 200 with status="ok" on the happy path and 503
        # with status="shedding" on a trip. The SDK's _classify_error raises
        # ServerError on 503, so if we got here we're on 200 — but be
        # defensive: derive is_ready from the status string when the server
        # omits the field.
        if self.is_ready is None:
            self.is_ready = self.status.lower() == "ok"
        return self

    def __repr__(self) -> str:
        return (
            f"ReadyReport("
            f"is_ready={self.is_ready}, "
            f"status={self.status!r}, "
            f"reason={self.reason!r})"
        )


# ---------------------------------------------------------------------------
# Search (#670)
# ---------------------------------------------------------------------------


class SearchHit(BaseModel):
    """A single document returned by ``POST /search``."""

    id: str = Field(..., description="Object ID")
    object_type: str = Field(..., description="Object type name")
    fields: dict[str, Any] = Field(default_factory=dict, description="Object fields")
    score: float = Field(..., description="BM25 relevance score")
    highlights: dict[str, str] = Field(
        default_factory=dict,
        description="Field-level snippets with <em> tags (present when highlight=True)",
    )


class SearchResponse(BaseModel):
    """Response from ``POST /search`` (#670)."""

    hits: list[SearchHit] = Field(default_factory=list)
    total: int = Field(0, description="Total matching documents")
    estimated_total_hits: int = Field(0, description="Full matching-set size (#967)")
    facets: dict[str, dict[str, int]] = Field(
        default_factory=dict, description="Facet counts keyed by field then value"
    )
    facet_stats: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="Numeric facet stats (min/max/sum/avg) (#967)"
    )
    processing_time_ms: int = Field(0, description="Server-side processing time")
