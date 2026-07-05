"""
Pydantic v2 response models for the Relata HTTP API.

All fields mirror the exact JSON keys returned by the server.  Optional fields
use ``None`` as the default so that older server versions that omit a field
remain compatible.
"""

from __future__ import annotations

from typing import Any, Iterator

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


class QueryResult(BaseModel):
    """Result of a ``POST /query`` call.

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
    """

    rows: list[dict[str, Any]] = Field(default_factory=list)
    query_id: str = Field(..., description="Server-assigned query execution ID")
    elapsed_ms: int = Field(..., description="Server-side execution time in ms", ge=0)
    row_count: int = Field(0, description="Number of rows returned")

    @model_validator(mode="after")
    def _sync_row_count(self) -> "QueryResult":
        # Always keep row_count consistent with the actual rows list.
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
        profile: Deployment profile — one of ``"lite"``, ``"server"``,
            ``"cluster"``.
        node_id: Stable node identifier (e.g. ``"coordinator"``,
            ``"reader-1"``).
    """

    status: str = Field(..., description="'ok' when the node is healthy")
    profile: str = Field(..., description="Deployment profile: lite | server | cluster")
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
        profile: Deployment profile — one of ``"lite"``, ``"server"``,
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
    """Response from ``POST /ingest/document``.

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
