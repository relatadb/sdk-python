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

    with RelataClient("http://localhost:8080", purpose="analytics") as client:
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

from relata.client import RelataClient
from relata.exceptions import (
    AuthError,
    ConnectionError,
    PurposeError,
    QuotaError,
    RelataError,
    ServerError,
)
from relata.memory import AsyncMemory, Memory
from relata.models import (
    AuditCountResponse,
    ClusterNode,
    HealthResponse,
    IngestDocumentResponse,
    QueryResult,
    StatusResponse,
)
from relata.query import QueryBuilder, select

__version__ = "0.1.0"
__all__ = [
    # Client
    "RelataClient",
    # High-level memory client
    "Memory",
    "AsyncMemory",
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
    # Exceptions
    "RelataError",
    "PurposeError",
    "QuotaError",
    "AuthError",
    "ConnectionError",
    "ServerError",
]
