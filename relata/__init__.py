"""
Relata Python SDK
=================

Python SDK for the `Relata <https://relata.io>`_ enterprise-grade data
engine.  Relata is a Rust-native ontology-driven engine for bi-temporal,
governed knowledge workloads: link analysis, identity resolution, full-text
and vector search, access-scoped restricted data, and provenance-tracked
analytics.

Quick start::

    from relata import RelataClient

    with RelataClient("http://localhost:9090", purpose="analytics") as client:
        # Simple query
        result = client.query("SELECT * FROM Person WHERE name LIKE 'Ahmed%' LIMIT 10")
        for row in result:
            print(row)

        # Fluent query builder
        result = (
            client.select("Person")
            .where("nationality = 'IN'")
            .as_of("2025-01-01")
            .with_provenance()
            .limit(20)
            .execute()
        )

Public API
----------
- :class:`~relata.client.RelataClient` — main client (sync + async)
- :class:`~relata.query.QueryBuilder` — fluent SQL builder
- :mod:`~relata.models` — Pydantic response models
- :mod:`~relata.exceptions` — SDK exception hierarchy
"""

from relata.a2a import A2AClient, AsyncA2AClient
from relata.audit import AsyncAuditClient, AuditClient
from relata.client import RelataClient
from relata import aml
from relata import canonical
from relata.exceptions import (
    AuthError,
    ConnectionError,
    PurposeError,
    QuotaError,
    RelataError,
    ServerError,
)
from relata.governance import AsyncGovernanceClient, GovernanceClient
from relata.identity import AsyncIdentityClient, IdentityClient
from relata.ingest import AsyncIngestClient, IngestClient
from relata.mcp import AsyncMcpClient, McpClient
from relata.memory import AsyncMemory, Memory
from relata.models import (
    AuditCountResponse,
    ClusterNode,
    HealthResponse,
    IngestDocumentResponse,
    QueryResult,
    ReadyReport,
    Stats,
    StatusResponse,
    VersionInfo,
)
from relata.namespace import AsyncNamespace, Namespace
from relata.objects import AsyncObjectClient, ObjectClient
from relata.query import QueryBuilder, select
from relata.s3 import AsyncS3Client, S3Client
from relata.search import AsyncSearchClient, SearchClient
from relata.streaming import AsyncStreamingClient, StreamingClient
from relata.system import AsyncSystemClient, SystemClient
from relata.tenants import AsyncTenantAdminClient, TenantAdminClient
from relata.vectors import AsyncVectorClient, VectorClient

__version__ = "0.2.0"

__all__ = [
    # Client
    "RelataClient",
    # High-level memory client
    "Memory",
    "AsyncMemory",
    # v1.1 SDK modules
    "GovernanceClient",
    "AsyncGovernanceClient",
    "McpClient",
    "AsyncMcpClient",
    "A2AClient",
    "AsyncA2AClient",
    "AuditClient",
    "AsyncAuditClient",
    "IdentityClient",
    "AsyncIdentityClient",
    "ObjectClient",
    "AsyncObjectClient",
    "IngestClient",
    "AsyncIngestClient",
    "VectorClient",
    "AsyncVectorClient",
    "S3Client",
    "AsyncS3Client",
    "SystemClient",
    "AsyncSystemClient",
    "StreamingClient",
    "AsyncStreamingClient",
    "TenantAdminClient",
    "AsyncTenantAdminClient",
    # T9 flagship retrieval surface (#1991)
    "Namespace",
    "AsyncNamespace",
    "SearchClient",
    "AsyncSearchClient",
    # Query builder
    "QueryBuilder",
    "select",
    # Models
    "QueryResult",
    "HealthResponse",
    "StatusResponse",
    "AuditCountResponse",
    "ClusterNode",
    "IngestDocumentResponse",
    "VersionInfo",
    "Stats",
    "ReadyReport",
    # Exceptions
    "RelataError",
    "PurposeError",
    "QuotaError",
    "AuthError",
    "ConnectionError",
    "ServerError",
    # Foundational typed surfaces (#2248 client canonical validation, #2255 AML decoders)
    "aml",
    "canonical",
]
