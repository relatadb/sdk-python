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


def test_nl_query_sends_query() -> None:
    client, seen = _capture()
    client.nl_query("find all persons in Dublin", interpret=True)
    assert seen["name"] == "nl_query"
    args = seen["arguments"]
    assert args["query"] == "find all persons in Dublin"
    assert args["interpret"] is True


def test_nl_query_sends_max_sub_questions_when_provided() -> None:
    client, seen = _capture()
    client.nl_query("find alice, and who alice is linked to", max_sub_questions=3)
    assert seen["name"] == "nl_query"
    args = seen["arguments"]
    assert args["max_sub_questions"] == 3


def test_nl_query_omits_max_sub_questions_by_default() -> None:
    client, seen = _capture()
    client.nl_query("find all persons in Dublin")
    args = seen["arguments"]
    assert "max_sub_questions" not in args


def test_erase_subject_sends_subject_and_reason() -> None:
    client, seen = _capture()
    client.erase_subject("alice@example.com", reason="user-request")
    assert seen["name"] == "erase_subject"
    args = seen["arguments"]
    assert args["subject"] == "alice@example.com"
    assert args["reason"] == "user-request"


def test_ingest_media_sends_object_type() -> None:
    client, seen = _capture()
    client.ingest_media("MediaContent", modality="image", bytes_b64="Zm9v")
    assert seen["name"] == "ingest_media"
    args = seen["arguments"]
    assert args["object_type"] == "MediaContent"
    assert args["bytes_b64"] == "Zm9v"


def test_similar_multimodal_sends_entity_type_and_id() -> None:
    client, seen = _capture()
    client.similar_multimodal("Document", "doc-1", top_k=5)
    assert seen["name"] == "similar_multimodal"
    args = seen["arguments"]
    assert args["entity_type"] == "Document"
    assert args["id"] == "doc-1"
    assert "entity_id" not in args


def test_hybrid_search_sends_entity_type_and_query() -> None:
    client, seen = _capture()
    client.hybrid_search("Document", "fraud ring")
    assert seen["name"] == "hybrid_search"
    args = seen["arguments"]
    assert args["entity_type"] == "Document"
    assert args["query"] == "fraud ring"


def test_paths_between_sends_from_and_to() -> None:
    client, seen = _capture()
    client.paths_between("a", "b", max_hops=6)
    assert seen["name"] == "paths_between"
    args = seen["arguments"]
    assert args["from"] == "a"
    assert args["to"] == "b"
    assert args["max_hops"] == 6


def test_list_link_types_sends_no_args() -> None:
    client, seen = _capture()
    client.list_link_types()
    assert seen.get("name") == "list_link_types"
    assert "arguments" not in seen


def test_server_health_sends_no_args() -> None:
    client, seen = _capture()
    client.server_health()
    assert seen.get("name") == "server_health"


def test_job_status_sends_no_args() -> None:
    client, seen = _capture()
    client.job_status()
    assert seen.get("name") == "job_status"


def test_metrics_sends_no_args() -> None:
    client, seen = _capture()
    client.metrics()
    assert seen.get("name") == "metrics"


def test_list_rules_sends_no_args() -> None:
    client, seen = _capture()
    client.list_rules()
    assert seen.get("name") == "list_rules"


def test_create_rule_sends_name_and_condition_sql() -> None:
    client, seen = _capture()
    client.create_rule("high-value-txn", "amount > 10000", severity="high")
    assert seen["name"] == "create_rule"
    args = seen["arguments"]
    assert args["name"] == "high-value-txn"
    assert args["condition_sql"] == "amount > 10000"
    assert args["severity"] == "high"
    assert args["purpose"] == "security"


def test_import_sigma_sends_yaml_key() -> None:
    client, seen = _capture()
    client.import_sigma("title: test\n")
    assert seen["name"] == "import_sigma"
    args = seen["arguments"]
    assert args["yaml"] == "title: test\n"
    assert "sigma_yaml" not in args


def test_list_jobs_sends_no_args() -> None:
    client, seen = _capture()
    client.list_jobs()
    assert seen.get("name") == "list_jobs"


def test_schedule_job_sends_name() -> None:
    client, seen = _capture()
    client.schedule_job("transaction_ring")
    assert seen["name"] == "schedule_job"
    assert seen["arguments"]["name"] == "transaction_ring"


def test_list_workflows_sends_no_args() -> None:
    client, seen = _capture()
    client.list_workflows()
    assert seen.get("name") == "list_workflows"


def test_run_workflow_sends_name() -> None:
    client, seen = _capture()
    client.run_workflow("aml-review")
    assert seen["name"] == "run_workflow"
    assert seen["arguments"]["name"] == "aml-review"


def test_workflow_status_sends_run_id() -> None:
    client, seen = _capture()
    client.workflow_status("run-1")
    assert seen["name"] == "workflow_status"
    assert seen["arguments"]["run_id"] == "run-1"


def test_trace_crypto_sends_address() -> None:
    client, seen = _capture()
    client.trace_crypto("bc1qxyz", max_hops=3)
    assert seen["name"] == "trace_crypto"
    args = seen["arguments"]
    assert args["address"] == "bc1qxyz"
    assert args["max_hops"] == 3
    assert args["purpose"] == "analytics"


def test_beneficial_ownership_sends_party() -> None:
    client, seen = _capture()
    client.beneficial_ownership("Acme Holdings")
    assert seen["name"] == "beneficial_ownership"
    assert seen["arguments"]["party"] == "Acme Holdings"


def test_reconstruct_wire_sends_account() -> None:
    client, seen = _capture()
    client.reconstruct_wire("IE12BOFI90000112345678")
    assert seen["name"] == "reconstruct_wire"
    assert seen["arguments"]["account"] == "IE12BOFI90000112345678"


def test_trace_hawala_sends_seed() -> None:
    client, seen = _capture()
    client.trace_hawala("actor-1")
    assert seen["name"] == "trace_hawala"
    assert seen["arguments"]["seed"] == "actor-1"


def test_geofence_sends_lat_lon() -> None:
    client, seen = _capture()
    client.geofence(53.35, -6.26, radius_m=500)
    assert seen["name"] == "geofence"
    args = seen["arguments"]
    assert args["lat"] == 53.35
    assert args["lon"] == -6.26
    assert args["radius_m"] == 500


def test_resolve_entity_identity_sends_identity() -> None:
    client, seen = _capture()
    client.resolve_entity_identity("alice@example.com")
    assert seen["name"] == "resolve_entity_identity"
    assert seen["arguments"]["identity"] == "alice@example.com"


def test_detect_communities_sends_entity_type() -> None:
    client, seen = _capture()
    client.detect_communities("Person", algo="leiden")
    assert seen["name"] == "detect_communities"
    args = seen["arguments"]
    assert args["entity_type"] == "Person"
    assert args["algo"] == "leiden"


def test_rank_key_nodes_sends_entity_type_and_metric() -> None:
    client, seen = _capture()
    client.rank_key_nodes("Person", metric="betweenness")
    assert seen["name"] == "rank_key_nodes"
    args = seen["arguments"]
    assert args["entity_type"] == "Person"
    assert args["metric"] == "betweenness"


def test_hub_authority_sends_entity_type() -> None:
    client, seen = _capture()
    client.hub_authority("Person")
    assert seen["name"] == "hub_authority"
    assert seen["arguments"]["entity_type"] == "Person"


def test_predict_links_sends_entity_type() -> None:
    client, seen = _capture()
    client.predict_links("Person", from_id="p1", to_id="p2", method="jaccard")
    assert seen["name"] == "predict_links"
    args = seen["arguments"]
    assert args["entity_type"] == "Person"
    assert args["from_id"] == "p1"
    assert args["to_id"] == "p2"
    assert args["method"] == "jaccard"


def test_find_scc_sends_entity_type() -> None:
    client, seen = _capture()
    client.find_scc("Person")
    assert seen["name"] == "find_scc"
    assert seen["arguments"]["entity_type"] == "Person"


def test_screen_sanctions_sends_name() -> None:
    client, seen = _capture()
    client.screen_sanctions("John Doe", threshold=0.9)
    assert seen["name"] == "screen_sanctions"
    args = seen["arguments"]
    assert args["name"] == "John Doe"
    assert args["threshold"] == 0.9
    assert args["purpose"] == "compliance_review"


def test_aggregate_stats_sends_entity_type() -> None:
    client, seen = _capture()
    client.aggregate_stats("Transaction", agg="SUM", column="amount")
    assert seen["name"] == "aggregate_stats"
    args = seen["arguments"]
    assert args["entity_type"] == "Transaction"
    assert args["agg"] == "SUM"
    assert args["column"] == "amount"


def test_investigate_entity_sends_entity_type_and_id() -> None:
    client, seen = _capture()
    client.investigate_entity("Person", "p1")
    assert seen["name"] == "investigate_entity"
    args = seen["arguments"]
    assert args["entity_type"] == "Person"
    assert args["id"] == "p1"


def test_find_threats_sends_entity_type() -> None:
    client, seen = _capture()
    client.find_threats("Person")
    assert seen["name"] == "find_threats"
    assert seen["arguments"]["entity_type"] == "Person"


def test_search_video_frames_sends_query_id() -> None:
    client, seen = _capture()
    client.search_video_frames("frame-1", top_k=10)
    assert seen["name"] == "search_video_frames"
    args = seen["arguments"]
    assert args["query_id"] == "frame-1"
    assert args["top_k"] == 10


def test_face_match_sends_probe_id() -> None:
    client, seen = _capture()
    client.face_match("probe-1", threshold=0.9)
    assert seen["name"] == "face_match"
    args = seen["arguments"]
    assert args["probe_id"] == "probe-1"
    assert args["threshold"] == 0.9
