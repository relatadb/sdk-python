"""Tests for the v1.1 SDK-module wave (#74, #77, #78, #79, #82, #83, #88, #81).

Uses httpx.MockTransport — no live server required. Every test asserts both
the request shape (path / method / body / query-string) and the response shape
so the SDK is locked against server contract drift.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import httpx
import pytest

if TYPE_CHECKING:
    from relata import RelataClient

BASE = "http://localhost:9090"

Handler = Callable[[httpx.Request], httpx.Response]


def _wrap(handler: Handler, **_client_kwargs: str) -> httpx.MockTransport:
    """Build a MockTransport; ``_client_kwargs`` are accepted for API symmetry
    with the other test modules but unused."""
    return httpx.MockTransport(handler)


# ---------------------------------------------------------------------------
# #74 — Governance
# ---------------------------------------------------------------------------


def test_governance_create_rule_posts_to_rules() -> None:
    from relata.governance import GovernanceClient

    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        assert req.method == "POST"
        assert req.url.path == "/rules"
        assert dict(req.url.params) == {"purpose": "audit"}
        body = json.loads(req.content)
        assert body == {"name": "r1", "object_type": "Person", "condition": "age > 18"}
        return httpx.Response(200, json={"id": "rule-1", "name": "r1"})

    g = GovernanceClient(BASE, purpose="audit", transport=_wrap(handler))
    out = g.create_rule({"name": "r1", "object_type": "Person", "condition": "age > 18"})
    assert out["id"] == "rule-1"


def test_governance_create_rule_requires_purpose() -> None:
    from relata.exceptions import PurposeError
    from relata.governance import GovernanceClient

    g = GovernanceClient(BASE, transport=_wrap(lambda req: httpx.Response(200, json={})))
    with pytest.raises(PurposeError):
        g.create_rule({"name": "r1"})


def test_governance_place_legal_hold_posts_payload() -> None:
    from relata.governance import GovernanceClient

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/retention/holds"
        body = json.loads(req.content)
        assert body == {
            "case_id": "case-42",
            "object_type": "Person",
            "field": "env",
            "value": "prod",
        }
        return httpx.Response(200, json={"case_id": "case-42", "status": "active"})

    g = GovernanceClient(BASE, transport=_wrap(handler))
    out = g.place_legal_hold("case-42", "Person", field="env", value="prod")
    assert out["status"] == "active"


def test_governance_request_and_approve_breakglass() -> None:
    from relata.governance import GovernanceClient

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/humint/breakglass/request":
            body = json.loads(req.content)
            assert body["source_id"] == "SRC-0042"
            assert body["justification"] == "urgent triage"
            return httpx.Response(200, json={"request_id": "bg-1", "status": "pending"})
        if req.url.path == "/humint/breakglass/approve":
            body = json.loads(req.content)
            assert body["request_id"] == "bg-1"
            return httpx.Response(200, json={"request_id": "bg-1", "approvals": 1})
        return httpx.Response(404)

    g = GovernanceClient(BASE, transport=_wrap(handler))
    req = g.request_breakglass("SRC-0042", justification="urgent triage")
    assert req["request_id"] == "bg-1"
    app = g.approve_breakglass("bg-1")
    assert app["approvals"] == 1


def test_governance_submit_dsar_uses_query_params() -> None:
    from relata.governance import GovernanceClient

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "GET"
        assert req.url.path == "/gdpr/dsar"
        # The query string carries subject_identity + reason.
        qs = str(req.url)
        assert "subject_identity=alice" in qs
        assert "reason=gdpr-art-15" in qs
        return httpx.Response(200, json={"dsar_id": "dsar-1"})

    g = GovernanceClient(BASE, transport=_wrap(handler))
    out = g.submit_dsar("alice", reason="gdpr-art-15")
    assert out["dsar_id"] == "dsar-1"


def test_governance_import_sigma_sends_yaml() -> None:
    from relata.governance import GovernanceClient

    yaml_text = "title: Suspicious\nlogsource: ...\n"

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/rules/sigma"
        assert dict(req.url.params) == {"purpose": "audit"}
        assert req.headers["content-type"] == "application/x-yaml"
        assert req.content.decode() == yaml_text
        return httpx.Response(200, json={"rules_imported": 1, "rules_skipped": 0})

    g = GovernanceClient(BASE, purpose="audit", transport=_wrap(handler))
    out = g.import_sigma(yaml_text)
    assert out["rules_imported"] == 1


def test_governance_suppress_rule_sends_entity_id() -> None:
    from relata.governance import GovernanceClient

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/rules/r1/suppress"
        body = json.loads(req.content)
        assert body == {"entity_id": "entity-42", "condition": "known FP"}
        return httpx.Response(200, json={"rule_id": "r1"})

    g = GovernanceClient(BASE, transport=_wrap(handler))
    out = g.suppress_rule("r1", "entity-42", condition="known FP")
    assert out["rule_id"] == "r1"


# ---------------------------------------------------------------------------
# #77 — MCP client
# ---------------------------------------------------------------------------


def test_mcp_list_tools_unwraps_envelope() -> None:
    from relata.mcp import McpClient

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/mcp/tools"
        return httpx.Response(
            200,
            json={
                "content": [
                    {"type": "text", "text": json.dumps({"tools": [{"name": "query_knowledge"}]})}
                ],
                "isError": False,
            },
        )

    c = McpClient(BASE, transport=_wrap(handler))
    tools = c.list_tools()
    assert tools == [{"name": "query_knowledge"}]


def test_mcp_call_tool_sends_arguments() -> None:
    from relata.mcp import McpClient

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/mcp/tools/call"
        body = json.loads(req.content)
        assert body == {"name": "query_knowledge", "arguments": {"sql": "SELECT 1", "purpose": "x"}}
        return httpx.Response(
            200,
            json={
                "content": [
                    {"type": "text", "text": json.dumps({"rows": [{"x": 1}]})}
                ],
                "isError": False,
            },
        )

    c = McpClient(BASE, transport=_wrap(handler))
    out = c.query_knowledge("SELECT 1", purpose="x")
    assert out["rows"] == [{"x": 1}]


# ---------------------------------------------------------------------------
# #77 — A2A client
# ---------------------------------------------------------------------------


def test_a2a_submit_and_get_task() -> None:
    from relata.a2a import A2AClient

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "POST" and req.url.path == "/a2a/tasks":
            return httpx.Response(200, json={"id": "task-1", "status": "pending"})
        if req.method == "GET" and req.url.path == "/a2a/tasks/task-1":
            return httpx.Response(200, json={"id": "task-1", "status": "completed"})
        return httpx.Response(404)

    c = A2AClient(BASE, transport=_wrap(handler))
    submit = c.submit_task({"name": "analyse", "input": {}})
    assert submit["id"] == "task-1"
    status = c.get_task("task-1")
    assert status["status"] == "completed"


# ---------------------------------------------------------------------------
# #78 — Audit entries + reports
# ---------------------------------------------------------------------------


def test_audit_entries_filters_in_query_string() -> None:
    from relata.audit import AuditClient

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/audit/entries"
        qs = str(req.url)
        assert "principal=alice" in qs
        assert "decision=Deny" in qs
        assert "limit=50" in qs
        return httpx.Response(
            200,
            json={"entries": [{"request_id": "r1"}], "next_cursor": None, "chain_valid": True},
        )

    c = AuditClient(BASE, transport=_wrap(handler))
    page = c.entries(principal="alice", decision="Deny", limit=50)
    assert page["entries"][0]["request_id"] == "r1"
    assert page["chain_valid"] is True


def test_audit_find_by_request_id_returns_first_match() -> None:
    from relata.audit import AuditClient

    def handler(req: httpx.Request) -> httpx.Response:
        assert "request_id=req-7" in str(req.url)
        return httpx.Response(
            200,
            json={"entries": [{"request_id": "req-7", "principal": "alice"}], "next_cursor": None},
        )

    c = AuditClient(BASE, transport=_wrap(handler))
    entry = c.find_by_request_id("req-7")
    assert entry is not None
    assert entry["principal"] == "alice"


def test_audit_sign_receipt_posts_payload() -> None:
    from relata.audit import AuditClient

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/report/sign"
        body = json.loads(req.content)
        assert body == {"case_id": "c-1"}
        return httpx.Response(200, json={"signature": "abc", "signed_at_ns": 1760})

    c = AuditClient(BASE, transport=_wrap(handler))
    out = c.sign_receipt({"case_id": "c-1"})
    assert out["signature"] == "abc"


# ---------------------------------------------------------------------------
# #79 — Identity / lookup / ERASE
# ---------------------------------------------------------------------------


def test_identity_label_posts_payload() -> None:
    from relata.identity import IdentityClient

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/identity/label"
        body = json.loads(req.content)
        assert body == {"identity": "alice@example.com", "label": "Alice", "confidence": 0.9}
        return httpx.Response(200, json={"identity": "alice@example.com", "label": "Alice"})

    c = IdentityClient(BASE, transport=_wrap(handler))
    out = c.label("alice@example.com", "Alice", confidence=0.9)
    assert out["label"] == "Alice"


def test_identity_register_lookup_posts_csv() -> None:
    from relata.identity import IdentityClient

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/lookup/register"
        body = json.loads(req.content)
        assert body["name"] == "country_codes"
        assert "KE,Kenya" in body["csv_data"]
        return httpx.Response(200, json={"name": "country_codes", "rows": 1})

    c = IdentityClient(BASE, transport=_wrap(handler))
    out = c.register_lookup("country_codes", "code,name\nKE,Kenya", key_column="code")
    assert out["rows"] == 1


def test_identity_erase_subject_builds_sql() -> None:
    from relata.identity import IdentityClient

    seen: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/query"
        body = json.loads(req.content)
        seen.append(body)
        return httpx.Response(200, json={"rows": [], "query_id": "q1", "elapsed_ms": 1})

    c = IdentityClient(BASE, transport=_wrap(handler))
    c.erase_subject("alice@example.com", "gdpr-art-17", certify=True, purpose="gdpr")
    assert "ERASE SUBJECT $1 REASON $2" in seen[0]["sql"]
    assert "CERTIFY" in seen[0]["sql"]
    assert seen[0]["params"] == ["alice@example.com", "gdpr-art-17"]
    assert seen[0]["purpose"] == "gdpr"


# ---------------------------------------------------------------------------
# #82 — ObjectClient upsert via /ingest
# ---------------------------------------------------------------------------


def test_object_upsert_sends_ndjson_to_ingest() -> None:
    from relata.objects import ObjectClient

    seen: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/ingest"
        assert "object_type=Person" in str(req.url)
        # The body is NDJSON, one row per line.
        body_text = req.content.decode()
        rows = [json.loads(line) for line in body_text.strip().split("\n")]
        seen.extend(rows)
        return httpx.Response(200, json={"accepted": 1, "rejected": 0, "write_seq": 42})

    c = ObjectClient(BASE, purpose="analytics", transport=_wrap(handler))
    out = c.upsert("Person", "p-1", {"name": "Alice", "age": 30})
    assert out["accepted"] == 1
    assert seen[0] == {"name": "Alice", "age": 30, "id": "p-1"}


# ---------------------------------------------------------------------------
# #83 — Bulk ingest (NDJSON + CSV)
# ---------------------------------------------------------------------------


def test_ingest_bulk_ndjson() -> None:
    from relata.ingest import IngestClient

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.headers["content-type"] == "application/x-ndjson"
        body_text = req.content.decode()
        rows = [json.loads(line) for line in body_text.strip().split("\n")]
        assert len(rows) == 3
        return httpx.Response(200, json={"accepted": 3, "write_seq": 100})

    c = IngestClient(BASE, purpose="ingest", transport=_wrap(handler))
    out = c.bulk("Person", [{"id": f"p-{i}", "n": i} for i in range(3)])
    assert out["accepted"] == 3


def test_ingest_bulk_csv() -> None:
    from relata.ingest import IngestClient

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.headers["content-type"] == "text/csv"
        assert b"name,age" in req.content
        return httpx.Response(200, json={"accepted": 2})

    c = IngestClient(BASE, purpose="ingest", transport=_wrap(handler))
    out = c.bulk_csv("Person", "name,age\nAlice,30\nBob,40")
    assert out["accepted"] == 2


# ---------------------------------------------------------------------------
# #2252 — CDR + OTLP ingest helpers
# ---------------------------------------------------------------------------


def test_ingest_cdr_serialises_csv_with_aliases() -> None:
    from relata.ingest import IngestClient

    seen: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/ingest/cdr"
        assert "purpose=finint" in str(req.url)
        assert req.headers["content-type"] == "text/csv"
        body_text = req.content.decode()
        lines = body_text.split("\n")
        # Header + 2 rows.
        assert lines[0] == "caller,callee,start,duration,tower,type"
        seen.append({"body": body_text})
        return httpx.Response(200, json={"rows_queued": 2, "errors": [], "queue_depth": 0})

    c = IngestClient(BASE, purpose="finint", transport=_wrap(handler))
    out = c.ingest_cdr(
        [
            {
                "caller": "+911234567890", "callee": "919876543210",
                "duration": 120, "tower": "CELL-042", "type": "voice",
            },
            {
                "a_number": "911111111111", "b_number": "922222222222",
                "tower_id": "CELL-001", "call_type": "sms",
            },
        ],
        purpose="finint",
    )
    assert out["rows_queued"] == 2
    body = seen[0]["body"]
    # Aliases resolve to canonical columns; digits kept (server strips non-digits).
    assert "911234567890" in body
    assert "911111111111" in body
    assert "CELL-042" in body


def test_ingest_otlp_helpers_post_json_with_purpose() -> None:
    from relata.ingest import IngestClient

    seen: list[tuple[str, dict[str, Any]]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.url.path, json.loads(req.content)))
        return httpx.Response(200, json={"ok": True})

    c = IngestClient(BASE, purpose="otel", transport=_wrap(handler))
    payload = {"resourceSpans": []}
    assert c.otlp_traces(payload, purpose="otel")["ok"] is True
    assert c.otlp_logs({"resourceLogs": []}, purpose="otel")["ok"] is True
    assert c.otlp_metrics({"resourceMetrics": []}, purpose="otel")["ok"] is True

    assert seen[0][0] == "/v1/traces"
    assert seen[1][0] == "/v1/logs"
    assert seen[2][0] == "/v1/metrics"
    assert seen[0][1] == payload


# ---------------------------------------------------------------------------
# #2249 — FinINT trace ops (wire_reconstruction, hawala_trace)
# ---------------------------------------------------------------------------


def _client_with_mock(handler: Handler) -> RelataClient:
    """Build a RelataClient whose sync transport is mocked."""
    from relata import RelataClient
    from relata._http import HttpTransport

    client = RelataClient(BASE, purpose="analytics")
    client._RelataClient__sync_transport = (  # type: ignore[attr-defined]
        HttpTransport(BASE, None, 30.0, transport=_wrap(handler), extra_headers=None)
    )
    return client


def test_resolve_identity_omits_mode_by_default() -> None:
    seen: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        return httpx.Response(200, json={"rows": [], "query_id": "q1", "elapsed_ms": 1})

    client = _client_with_mock(handler)
    client.resolve_identity("alice@example.com")
    assert seen[0]["sql"] == "RESOLVE_IDENTITY($1)"
    assert seen[0]["params"] == ["alice@example.com"]


def test_resolve_identity_supports_canonical_and_fuse_modes() -> None:
    seen: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        return httpx.Response(200, json={"rows": [], "query_id": "q1", "elapsed_ms": 1})

    client = _client_with_mock(handler)
    client.resolve_identity("alice@example.com", mode="canonical")
    assert seen[0]["sql"] == "RESOLVE_IDENTITY($1, MODE => 'canonical')"
    assert seen[0]["params"] == ["alice@example.com"]

    client.resolve_identity("alice@example.com", mode="fuse")
    assert seen[1]["sql"] == "RESOLVE_IDENTITY($1, MODE => 'fuse')"
    assert seen[1]["params"] == ["alice@example.com"]


def test_resolve_identity_rejects_invalid_mode() -> None:
    client = _client_with_mock(lambda req: httpx.Response(200, json={}))
    with pytest.raises(ValueError, match="mode must be one of"):
        client.resolve_identity("alice@example.com", mode="bogus")


def test_wire_reconstruction_builds_sql() -> None:
    seen: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/query"
        seen.append(json.loads(req.content))
        return httpx.Response(200, json={"rows": [], "query_id": "q1", "elapsed_ms": 1})

    client = _client_with_mock(handler)
    out = client.wire_reconstruction("ACC-007", tolerance_pct=2.5, purpose="finint")
    assert out["query_id"] == "q1"
    assert seen[0]["sql"] == "WIRE_RECONSTRUCTION('ACC-007', TOLERANCE_PCT => 2.5)"
    assert seen[0]["purpose"] == "finint"


def test_hawala_trace_clamps_and_defaults_purpose() -> None:
    seen: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        return httpx.Response(200, json={"rows": [], "query_id": "q2", "elapsed_ms": 1})

    client = _client_with_mock(handler)
    # max_hops above the server clamp of 10 is clamped client-side to 10.
    client.hawala_trace("SEED-1", max_hops=99)
    assert seen[0]["sql"] == "HAWALA_TRACE('SEED-1', MAX_HOPS => 10)"
    # No explicit purpose → client default ("analytics").
    assert seen[0]["purpose"] == "analytics"


# ---------------------------------------------------------------------------
# #88 — Vector client (SQL-backed)
# ---------------------------------------------------------------------------


def test_vector_knn_search_builds_pgvector_sql() -> None:
    """The KNN helper uses the cosine-distance form ``ORDER BY <slot> <=> '[...]'``."""
    from relata import RelataClient
    from relata.vectors import VectorClient

    seen_sql: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/query"
        body = json.loads(req.content)
        seen_sql.append(body["sql"])
        return httpx.Response(
            200,
            json={"rows": [{"id": "p-1"}], "query_id": "q1", "elapsed_ms": 1},
        )

    # Build a RelataClient whose transport is mocked.
    client = RelataClient(BASE, purpose="analytics")
    client._RelataClient__sync_transport = (  # type: ignore[attr-defined]
        __import__("relata._http", fromlist=["HttpTransport"]).HttpTransport(
            BASE, None, 30.0, transport=_wrap(handler), extra_headers=None,
        )
    )

    v = VectorClient.from_client(client)
    rows = v.knn_search("Person", "_embedding", [0.1, 0.2, 0.3], k=5)
    assert rows == [{"id": "p-1"}]
    assert "ORDER BY _embedding <=> '[0.1, 0.2, 0.3]'" in seen_sql[0]
    assert "LIMIT 5" in seen_sql[0]


def test_vector_hybrid_search_requires_text_or_embedding() -> None:
    from relata.vectors import VectorClient

    client = _client_with_mock(lambda req: httpx.Response(200, json={"rows": []}))
    v = VectorClient.from_client(client)
    with pytest.raises(ValueError, match="requires query_text"):
        v.hybrid_search("Person")


def test_vector_hybrid_search_builds_tvf_sql() -> None:
    from relata.vectors import VectorClient

    seen: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        return httpx.Response(200, json={"rows": [], "query_id": "q1", "elapsed_ms": 1})

    v = VectorClient.from_client(_client_with_mock(handler))
    v.hybrid_search("Person", query_text="alice", k=3)
    assert seen[0]["sql"] == "HYBRID_SEARCH FROM Person QUERY $1 LIMIT 3"
    assert seen[0]["params"] == ["alice"]


def test_vector_hybrid_search_emits_tuning_clauses() -> None:
    from relata.vectors import VectorClient

    seen: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        return httpx.Response(200, json={"rows": [], "query_id": "q1", "elapsed_ms": 1})

    v = VectorClient.from_client(_client_with_mock(handler))
    v.hybrid_search(
        "Document",
        query_text="o'reilly",
        k=5,
        rerank=True,
        metric="cosine",
        weights=[0.3, 0.5, 0.2],
    )
    assert seen[0]["sql"] == (
        "HYBRID_SEARCH FROM Document QUERY $1 LIMIT 5 "
        "RERANK METRIC cosine WEIGHTS 0.3 0.5 0.2"
    )
    # query_text is bound, never interpolated — the quote needs no escaping (#3211).
    assert seen[0]["params"] == ["o'reilly"]


def test_vector_similar_to_builds_sql() -> None:
    from relata.vectors import VectorClient

    seen: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        return httpx.Response(200, json={"rows": [], "query_id": "q1", "elapsed_ms": 1})

    v = VectorClient.from_client(_client_with_mock(handler))
    v.similar_to("Person", "p-1", k=2)
    assert seen[0]["sql"] == "SELECT * FROM SIMILAR TO Person WHERE id = $1 LIMIT 2"
    assert seen[0]["params"] == ["p-1"]


# ---------------------------------------------------------------------------
# #1172 — VectorClient.embed / embed_batch (direct /embed HTTP door)
# ---------------------------------------------------------------------------


def test_vector_embed_posts_to_embed_endpoint() -> None:
    from relata.vectors import VectorClient

    seen: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/embed"
        seen.append(json.loads(req.content))
        return httpx.Response(200, json={"embedding": [0.1, 0.2], "model": "lexical", "dim": 2})

    v = VectorClient.from_client(_client_with_mock(handler))
    result = v.embed("Alice Smith")
    assert seen[0] == {"text": "Alice Smith"}
    assert result == {"embedding": [0.1, 0.2], "model": "lexical", "dim": 2}


def test_vector_embed_batch_posts_to_embed_batch_endpoint() -> None:
    from relata.vectors import VectorClient

    seen: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/embed/batch"
        seen.append(json.loads(req.content))
        return httpx.Response(
            200,
            json={"embeddings": [[0.1], [0.2]], "model": "lexical", "dim": 1, "count": 2},
        )

    v = VectorClient.from_client(_client_with_mock(handler))
    result = v.embed_batch(["Alice", "Bob"], model="lexical")
    assert seen[0] == {"texts": ["Alice", "Bob"], "model": "lexical"}
    assert result["count"] == 2


# ---------------------------------------------------------------------------
# #81 — S3Client returns a configured boto3 client (skip if boto3 not installed)
# ---------------------------------------------------------------------------


def test_s3_client_boto3_configured() -> None:
    pytest.importorskip("boto3")
    from relata.s3 import S3Client

    s3 = S3Client("http://localhost:9090", bearer_token="tok", tenant="acme").boto3()
    assert s3.meta.region_name == "us-east-1"
    # endpoint_url is stored on the meta endpoint URL.
    assert "localhost:9090" in s3.meta.endpoint_url


def test_query_result_processing_time_ms_from_new_field() -> None:
    """#1252: processing_time_ms populated from new field when present."""
    from relata.models import QueryResult

    r = QueryResult.model_validate(
        {"rows": [], "query_id": "q1", "elapsed_ms": 3, "processing_time_ms": 9}
    )
    assert r.processing_time_ms == 9


def test_query_result_processing_time_ms_fallback() -> None:
    """#1252: processing_time_ms falls back to elapsed_ms on older servers."""
    from relata.models import QueryResult

    r = QueryResult.model_validate(
        {"rows": [], "query_id": "q1", "elapsed_ms": 5}
    )
    assert r.processing_time_ms == 5
