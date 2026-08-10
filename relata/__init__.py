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
from relata.backup import AsyncBackupClient, BackupClient
from relata.flight import AsyncFlightClient, FlightClient
from relata.client import RelataClient
from relata import aml
from relata import canonical
from relata.coref import AsyncCorefResolver, CorefResolver, subject_from_hit
from relata.exceptions import (
    AuthError,
    ConflictError,
    ConnectionError,
    ForbiddenError,
    NotFoundError,
    PurposeError,
    QuotaError,
    RateLimitedError,
    RelataError,
    ResponseTooLargeError,
    ServerError,
    ValidationError,
)
from relata.governance import AsyncGovernanceClient, GovernanceClient
from relata.identity import AsyncIdentityClient, IdentityClient
from relata.ingest import AsyncIngestClient, IngestClient
from relata.log import AsyncLogClient, LogClient
from relata.mcp import AsyncMcpClient, McpClient
from relata.memory import AsyncMemory, Memory
from relata.models import (
    AuditCountResponse,
    Clarification,
    ClusterNode,
    DocumentUsageResponse,
    EntityCandidate,
    HealthResponse,
    IngestDocumentResponse,
    QueryResult,
    RagHit,
    RagQueryResponse,
    ReadyReport,
    SearchHit,
    SearchResponse,
    Stats,
    StatusResponse,
    VersionInfo,
)
from relata.namespace import AsyncNamespace, Namespace
from relata.objects import AsyncObjectClient, ObjectClient
from relata.query import QueryBuilder, select
from relata.rag import AsyncRagClient, RagClient, filters_for_entity_ids
from relata.rag_loop import (
    CORRECTIVE_FRACTION_CORRECT_FLOOR,
    HEURISTIC_PASS_THRESHOLD,
    HEURISTIC_RETRY_THRESHOLD,
    LOOP_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_FLOOR,
    MAX_FANOUT_STRATEGIES,
    MAX_ITERATIONS,
    MERGE_THRESHOLD,
    MIN_FANOUT_STRATEGIES,
    CorrectiveGradingResult,
    FanoutResult,
    GateDecision,
    HeuristicGateResult,
    HitGrade,
    LoopIteration,
    LoopResult,
    SubAgentResult,
    SubAgentStrategy,
    arun_agentic_loop,
    arun_subagent_fanout,
    grade_hits,
    heuristic_gate,
    run_agentic_loop,
    run_subagent_fanout,
)
from relata.rag_rank import (
    DEFAULT_MMR_LAMBDA,
    MMR_LAMBDA_BY_PURPOSE,
    default_relevance,
    default_text_similarity,
    mmr_lambda_for_purpose,
    mmr_select,
    mmr_select_for_purpose,
)
from relata.rag_understanding import (
    ENUMERATION_TOP_K,
    NUMERIC_INTENT_WORDS,
    QueryShape,
    asmart_rag_query,
    classify_query_shape,
    decompose_query,
    expand_query_hyde,
    is_numeric_intent,
    rrf_k_for_fanout,
    rrf_merge,
    smart_rag_query,
)
from relata.s3 import AsyncS3Client, S3Client
from relata.search import AsyncSearchClient, SearchClient
from relata.structural_navigation import (
    ChildSelector,
    StructureNode,
    fetch_child_nodes,
    fetch_root_node,
    lexical_child_selector,
    navigate_structural_tree,
)
from relata.streaming import AsyncStreamingClient, StreamingClient
from relata.synthesis import (
    Citation,
    EntailmentFn,
    LlmFn,
    SynthesisResult,
    SynthesizedSentence,
    synthesize,
)
from relata.system import AsyncSystemClient, SystemClient
from relata.tenants import AsyncTenantAdminClient, TenantAdminClient
from relata.tokens import AsyncTokenClient, TokenClient
from relata.vectors import AsyncVectorClient, VectorClient
from relata._version import __version__

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
    "FlightClient",
    "AsyncFlightClient",
    "TenantAdminClient",
    "AsyncTenantAdminClient",
    # #2757: previously import-only (required an explicit submodule import)
    "BackupClient",
    "AsyncBackupClient",
    "TokenClient",
    "AsyncTokenClient",
    "LogClient",
    "AsyncLogClient",
    # T9 flagship retrieval surface (#1991)
    "Namespace",
    "AsyncNamespace",
    "SearchClient",
    "AsyncSearchClient",
    # RAG epic — typed /rag/query client (#4523, foundational; #4514/ADR-0299)
    "RagClient",
    "AsyncRagClient",
    # RAG epic — citation injection + post-synthesis faithfulness scoring
    # (#4527, highest-priority SDK ticket)
    "synthesize",
    "SynthesisResult",
    "SynthesizedSentence",
    "Citation",
    "LlmFn",
    "EntailmentFn",
    # RAG epic — entity disambiguation (#4534)
    "filters_for_entity_ids",
    # RAG epic — session-scoped coreference resolution (#4530)
    "CorefResolver",
    "AsyncCorefResolver",
    "subject_from_hit",
    # RAG epic — query-shape dispatch + HyDE + decomposition (#4524, Python-only
    # SDK-side orchestration per ADR-0298)
    "QueryShape",
    "classify_query_shape",
    "is_numeric_intent",
    "NUMERIC_INTENT_WORDS",
    "expand_query_hyde",
    "decompose_query",
    "rrf_k_for_fanout",
    "rrf_merge",
    "ENUMERATION_TOP_K",
    "smart_rag_query",
    "asmart_rag_query",
    # RAG epic — heuristic gate + corrective retrieval grading + loop
    # confidence, the three-tier cost ladder (#4525)
    "GateDecision",
    "HeuristicGateResult",
    "heuristic_gate",
    "HEURISTIC_PASS_THRESHOLD",
    "HEURISTIC_RETRY_THRESHOLD",
    "HitGrade",
    "CorrectiveGradingResult",
    "grade_hits",
    "CORRECTIVE_FRACTION_CORRECT_FLOOR",
    "LoopIteration",
    "LoopResult",
    "run_agentic_loop",
    "arun_agentic_loop",
    "LOOP_CONFIDENCE_THRESHOLD",
    "MAX_ITERATIONS",
    # RAG epic — sub-agent fan-out + deterministic merge + MMR diversity
    # (#4526)
    "SubAgentStrategy",
    "SubAgentResult",
    "FanoutResult",
    "run_subagent_fanout",
    "arun_subagent_fanout",
    "LOW_CONFIDENCE_FLOOR",
    "MERGE_THRESHOLD",
    "MIN_FANOUT_STRATEGIES",
    "MAX_FANOUT_STRATEGIES",
    "mmr_select",
    "mmr_select_for_purpose",
    "mmr_lambda_for_purpose",
    "default_relevance",
    "default_text_similarity",
    "MMR_LAMBDA_BY_PURPOSE",
    "DEFAULT_MMR_LAMBDA",
    # RAG epic — structural table-of-contents navigation (#4542, Python-only
    # SDK-side agentic tree descent per ADR-0298)
    "StructureNode",
    "ChildSelector",
    "fetch_root_node",
    "fetch_child_nodes",
    "lexical_child_selector",
    "navigate_structural_tree",
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
    "DocumentUsageResponse",
    "VersionInfo",
    "Stats",
    "ReadyReport",
    "SearchHit",
    "SearchResponse",
    "RagHit",
    "RagQueryResponse",
    "Clarification",
    "EntityCandidate",
    # Exceptions
    "RelataError",
    "PurposeError",
    "QuotaError",
    "AuthError",
    "ConnectionError",
    "ServerError",
    "ForbiddenError",
    "NotFoundError",
    "ConflictError",
    "ValidationError",
    "RateLimitedError",
    "ResponseTooLargeError",
    # Foundational typed surfaces (#2248 client canonical validation, #2255 AML decoders)
    "aml",
    "canonical",
]
