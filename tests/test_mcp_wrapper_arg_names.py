"""#253: the MCP convenience wrappers must send the arg keys the registered
``input_schema`` declares, or ``validate_args_against_schema`` 400s before
dispatch. These tests capture the wire ``arguments`` map each wrapper emits and
assert the corrected field names (source of truth: the tool schemas in
``crates/relata-cli/src/serve.rs``).

Uses ``httpx.MockTransport`` — no live server required (mirrors the memory-verb
fix pattern).
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from relata import McpClient

BASE = "http://localhost:9090"


def _capture() -> tuple[McpClient, dict[str, Any]]:
    """An McpClient whose transport records the last /mcp/tools/call payload."""
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/mcp/tools/call":
            seen.update(json.loads(request.content))
        result = {"ok": True}
        body = {"content": [{"type": "text", "text": json.dumps(result)}], "isError": False}
        return httpx.Response(200, json=body)

    client = McpClient(BASE, transport=httpx.MockTransport(handler))
    return client, seen


def test_get_entities_sends_entity_type_and_filters() -> None:
    client, seen = _capture()
    client.get_entities("Person", filters={"city": "Dublin"}, limit=25)
    args = seen["arguments"]
    assert args["entity_type"] == "Person"
    assert args["filters"] == {"city": "Dublin"}
    assert "object_type" not in args and "filter" not in args


def test_search_entities_sends_entity_types_array() -> None:
    client, seen = _capture()
    client.search_entities("acme", entity_types=["Organization", "Person"])
    args = seen["arguments"]
    assert args["entity_types"] == ["Organization", "Person"]
    assert "object_type" not in args


def test_get_domain_summary_sends_domain() -> None:
    client, seen = _capture()
    client.get_domain_summary("financial")
    args = seen["arguments"]
    assert args["domain"] == "financial"
    assert "object_type" not in args


def test_find_in_social_corpus_sends_object_type_and_text_query() -> None:
    client, seen = _capture()
    client.find_in_social_corpus("Post", text_query="protest", user="alice")
    args = seen["arguments"]
    assert args["object_type"] == "Post"
    assert args["text_query"] == "protest"
    assert args["user"] == "alice"
    assert "query" not in args and "corpus" not in args


def test_lookup_identity_sends_raw() -> None:
    client, seen = _capture()
    client.lookup_identity("+353871234567")
    args = seen["arguments"]
    assert args["raw"] == "+353871234567"
    assert "value" not in args


def test_get_entity_profile_sends_name() -> None:
    client, seen = _capture()
    client.get_entity_profile("Jane Roe", purpose="analytics")
    args = seen["arguments"]
    assert args["name"] == "Jane Roe"
    assert "entity_id" not in args


def test_get_timeline_sends_entity() -> None:
    client, seen = _capture()
    client.get_timeline("Jane Roe", purpose="analytics")
    args = seen["arguments"]
    assert args["entity"] == "Jane Roe"
    assert "entity_id" not in args


def test_get_case_summary_drops_case_id_sends_toggles() -> None:
    # #4658: get_case_summary is tenant-wide, not case-scoped — case_id was
    # never read server-side and has been dropped.
    client, seen = _capture()
    client.get_case_summary(purpose="analytics", include_graph=False)
    assert seen["name"] == "get_case_summary"
    args = seen["arguments"]
    assert args["purpose"] == "analytics"
    assert args["include_graph"] is False
    assert "case_id" not in args


def test_rag_store_answer_sends_sources_not_source_ids() -> None:
    # #4659: the server reads a `sources` array of objects; `source_ids` was
    # never read and silently dropped.
    client, seen = _capture()
    client.rag_store_answer(
        "q", "a", purpose="analytics", sources=[{"id": "src-1", "score": 0.9}]
    )
    args = seen["arguments"]
    assert args["sources"] == [{"id": "src-1", "score": 0.9}]
    assert "source_ids" not in args


def test_rag_store_elements_sends_required_source_filename() -> None:
    # #4660: source_filename is required server-side (no default) — every
    # call 400ed without it.
    client, seen = _capture()
    client.rag_store_elements([{"text": "e1"}], "report.pdf", purpose="analytics")
    args = seen["arguments"]
    assert args["elements"] == [{"text": "e1"}]
    assert args["source_filename"] == "report.pdf"


def test_ingest_document_sends_source_and_text() -> None:
    # #4661: the server's flat schema is source/text/label/confidence/
    # entities/relations — the previous chunks_jsonl/manifest_json keys were
    # never read and every call silently no-op'd.
    client, seen = _capture()
    client.ingest_document("report.pdf", "full document body", purpose="analytics")
    args = seen["arguments"]
    assert args["source"] == "report.pdf"
    assert args["text"] == "full document body"
    assert "chunks_jsonl" not in args and "manifest_json" not in args


def test_get_timeline_has_no_since_until_kwargs() -> None:
    # #4664: since_ns/until_ns had no server-side effect (mcp_tool_get_timeline
    # has no time-range concept) and were dropped from the signature entirely.
    client, seen = _capture()
    client.get_timeline("Jane Roe", purpose="analytics")
    args = seen["arguments"]
    assert "since_ns" not in args and "until_ns" not in args


def test_add_case_note_has_no_author_kwarg() -> None:
    # #4664: author had no server-side effect (mcp_tool_add_case_note_with_gate
    # never reads it) and was dropped from the signature entirely.
    client, seen = _capture()
    client.add_case_note("c-1", "note text")
    args = seen["arguments"]
    assert args["case_id"] == "c-1"
    assert args["note"] == "note text"
    assert "author" not in args


def test_get_relationships_sends_subject_not_entity_id() -> None:
    client, seen = _capture()
    client.get_relationships("Acme Corp", purpose="analytics", predicate="controls")
    args = seen["arguments"]
    assert args["subject"] == "Acme Corp"
    assert args["predicate"] == "controls"
    assert "entity_id" not in args and "depth" not in args


def test_get_audit_trail_sends_schema_fields_only() -> None:
    client, seen = _capture()
    client.get_audit_trail(purpose="audit", principal_filter="api-user")
    args = seen["arguments"]
    assert args["purpose"] == "audit"
    assert args["principal_filter"] == "api-user"
    assert "case_id" not in args and "entity_id" not in args


def test_search_knowledge_sends_limit_not_top_k() -> None:
    # #341: the handler reads `limit`; the previous `top_k` key was silently
    # dropped and the result count fell back to the server default.
    client, seen = _capture()
    client.search_knowledge("acme fraud", purpose="analytics", top_k=25)
    args = seen["arguments"]
    assert args["limit"] == 25
    assert "top_k" not in args


def test_recall_sends_adr145_retrieval_quality_params() -> None:
    # #2674: the low-level MCP `recall` wrapper must forward the five
    # ADR-145 retrieval-quality operators the server's `recall` handler reads
    # (crates/relata-cli/src/serve/mcp.rs, min_confidence/recency_half_life_secs/
    # budget_tokens/stability_days/cancel_threshold), not just top_k.
    client, seen = _capture()
    client.recall(
        "hello",
        purpose="x",
        min_confidence=0.5,
        recency_half_life_secs=1800.0,
        budget_tokens=256,
        stability_days=14.0,
        cancel_threshold=0.99,
    )
    args = seen["arguments"]
    assert args["min_confidence"] == 0.5
    assert args["recency_half_life_secs"] == 1800.0
    assert args["budget_tokens"] == 256
    assert args["stability_days"] == 14.0
    assert args["cancel_threshold"] == 0.99


def test_recall_omits_adr145_params_when_unset() -> None:
    client, seen = _capture()
    client.recall("hello", purpose="x")
    args = seen["arguments"]
    for key in (
        "min_confidence",
        "recency_half_life_secs",
        "budget_tokens",
        "stability_days",
        "cancel_threshold",
    ):
        assert key not in args


def test_suggest_extensions_sends_no_args() -> None:
    # #341: the handler takes no arguments (it ignores any) and returns pack
    # suggestions — the previous `prefix` arg was misleading and dropped.
    client, seen = _capture()
    client.suggest_extensions()
    # call_tool omits the `arguments` key entirely for an empty map, so no
    # `prefix` is ever sent (the handler takes no args).
    assert "arguments" not in seen
    assert seen.get("name") == "suggest_extensions"


# ---------------------------------------------------------------------------
# #2322 — 42 (+4) previously Rust-only MCP tool wrappers, ported to Python.
#
# Each test asserts the wire tool `name` and the `arguments` key set/values
# sent match the server's real registration (source of truth: the
# `input_schema` blocks in `crates/relata-cli/src/serve.rs` and
# `crates/relata-cli/src/serve/mcp/procedure.rs`), not merely that the
# method exists.
# ---------------------------------------------------------------------------


def test_remember_batch_sends_items() -> None:
    client, seen = _capture()
    client.remember_batch([{"content": "a"}, {"content": "b"}], purpose="analytics")
    assert seen["name"] == "remember_batch"
    args = seen["arguments"]
    assert args["items"] == [{"content": "a"}, {"content": "b"}]
    assert args["purpose"] == "analytics"


def test_recognize_sends_id() -> None:
    client, seen = _capture()
    client.recognize("mem-1")
    assert seen["name"] == "recognize"
    assert seen["arguments"]["id"] == "mem-1"


def test_episodes_in_sends_session_id_and_limit() -> None:
    client, seen = _capture()
    client.episodes_in("sess-1", limit=5)
    assert seen["name"] == "episodes_in"
    args = seen["arguments"]
    assert args["session_id"] == "sess-1"
    assert args["limit"] == 5


def test_justify_sends_id() -> None:
    client, seen = _capture()
    client.justify("mem-1")
    assert seen["name"] == "justify"
    assert seen["arguments"]["id"] == "mem-1"


def test_consolidate_sends_id_and_content() -> None:
    client, seen = _capture()
    client.consolidate("mem-1", "updated belief", confidence=0.5)
    assert seen["name"] == "consolidate"
    args = seen["arguments"]
    assert args["id"] == "mem-1"
    assert args["content"] == "updated belief"
    assert args["confidence"] == 0.5


def test_forget_sends_id_and_retain_days() -> None:
    client, seen = _capture()
    client.forget("mem-1", retain_days=30)
    assert seen["name"] == "forget"
    args = seen["arguments"]
    assert args["id"] == "mem-1"
    assert args["retain_days"] == 30


def test_remember_procedure_sends_schema_fields() -> None:
    client, seen = _capture()
    client.remember_procedure("agent-1", "onboarding", "do the thing")
    assert seen["name"] == "remember_procedure"
    args = seen["arguments"]
    assert args["agent_id"] == "agent-1"
    assert args["name"] == "onboarding"
    assert args["instruction_text"] == "do the thing"


def test_recall_procedure_sends_agent_id() -> None:
    client, seen = _capture()
    client.recall_procedure("agent-1", name="onboarding", all_versions=True)
    assert seen["name"] == "recall_procedure"
    args = seen["arguments"]
    assert args["agent_id"] == "agent-1"
    assert args["name"] == "onboarding"
    assert args["all_versions"] is True


def test_associate_sends_from_id_to_id_relation() -> None:
    client, seen = _capture()
    client.associate("mem-1", "mem-2", relation="contradicts")
    assert seen["name"] == "associate"
    args = seen["arguments"]
    assert args["from_id"] == "mem-1"
    assert args["to_id"] == "mem-2"
    assert args["relation"] == "contradicts"


def test_resolve_sends_id() -> None:
    client, seen = _capture()
    client.resolve("mem-1")
    assert seen["name"] == "resolve"
    assert seen["arguments"]["id"] == "mem-1"


def test_summarise_sends_ids() -> None:
    client, seen = _capture()
    client.summarise(ids=["mem-1", "mem-2"], scope="topic:fraud")
    assert seen["name"] == "summarise"
    args = seen["arguments"]
    assert args["ids"] == ["mem-1", "mem-2"]
    assert args["scope"] == "topic:fraud"
