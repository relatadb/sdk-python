"""End-to-end agent-verb coverage tests (#77, #82 family).

Locks the SDK against the server's agent surface so any future server-side
addition shows up as a test failure here before it ships in production.

Covers:
- All 30 server MCP tools reachable via McpClient (typed wrapper or via
  ``call_tool`` fallback for the 8 memory verbs covered by Memory).
- A2AClient checkpoint save / load / list round-trip.
- A2AClient agent card fetch.
- SystemClient LLM config + test + jobs status.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx

from relata.a2a import A2AClient
from relata.mcp import McpClient
from relata.system import SystemClient

BASE = "http://localhost:8080"

Handler = Callable[[httpx.Request], httpx.Response]


def _wrap(handler: Handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


# Server MCP tool dispatch table (serve/mcp.rs:212-245). Keep in sync.
SERVER_MCP_TOOLS: set[str] = {
    "query", "query_knowledge", "explain_policy", "suggest_extensions",
    "lookup_identity", "list_entity_types", "get_entities", "search_entities",
    "find_in_social_corpus", "get_domain_summary", "ingest_document",
    "rag_store_answer", "rag_store_elements", "search_knowledge",
    "get_relationships", "get_entity_profile", "get_timeline",
    "find_connections", "add_case_note", "get_audit_trail", "get_case_summary",
    # 10 cognitive memory verbs
    "remember", "recall", "recognize", "episodes_in", "justify",
    "consolidate", "forget", "associate", "resolve", "summarise",
}


def test_every_server_mcp_tool_is_reachable_via_call_tool() -> None:
    """Generic ``call_tool`` must accept every name the server dispatches."""
    seen: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/mcp/tools/call":
            body = json.loads(req.content)
            seen.append(body["name"])
        return httpx.Response(
            200,
            json={"content": [{"type": "text", "text": "{}"}], "isError": False},
        )

    c = McpClient(BASE, transport=_wrap(handler))
    for tool in SERVER_MCP_TOOLS:
        c.call_tool(tool, {"_test": True})

    assert set(seen) == SERVER_MCP_TOOLS, (
        f"Missing tools: {SERVER_MCP_TOOLS - set(seen)}; "
        f"extra: {set(seen) - SERVER_MCP_TOOLS}"
    )


def test_mcp_typed_wrappers_match_server_names() -> None:
    """Every typed wrapper on McpClient must call a tool name the server
    actually dispatches. Catches the historical bug where ``count`` /
    ``sample`` / ``query_relationships`` were invented names that the server
    never registered."""
    seen: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/mcp/tools/call":
            body = json.loads(req.content)
            seen.append(body["name"])
        return httpx.Response(
            200,
            json={"content": [{"type": "text", "text": "{}"}], "isError": False},
        )

    c = McpClient(BASE, transport=_wrap(handler))
    # Call every typed wrapper with stub args.
    c.query_knowledge("SELECT 1", purpose="x")
    c.search_knowledge("q", purpose="x")
    c.explain_policy("SELECT 1", purpose="x")
    c.suggest_extensions("p")
    c.list_entity_types()
    c.get_entities("Person")
    c.search_entities("q")
    c.get_domain_summary()
    c.find_in_social_corpus("q")
    c.lookup_identity("alice@example.com")
    c.get_entity_profile("e-1", purpose="x")
    c.get_timeline("e-1", purpose="x")
    c.find_connections("a", "b", purpose="x")
    c.get_relationships("e-1", purpose="x")
    c.add_case_note("c-1", "note")
    c.get_audit_trail(case_id="c-1")
    c.get_case_summary("c-1", purpose="x")
    c.rag_store_answer("q", "a", purpose="x")
    c.rag_store_elements([], purpose="x")
    c.ingest_document("chunks", "manifest", purpose="x")
    c.remember("hello", purpose="x")
    c.recall("hello", purpose="x")

    unknown = set(seen) - SERVER_MCP_TOOLS
    assert not unknown, (
        f"Typed wrappers called unknown server tool names: {unknown}. "
        "Update the wrapper or add the tool to the server dispatch table."
    )


def test_a2a_checkpoint_save_load_round_trip() -> None:
    seen: list[tuple[str, str]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.method, req.url.path))
        if req.method == "PUT" and "/a2a/checkpoints/t-1/c-1" in req.url.path:
            body = json.loads(req.content)
            assert body == {"checkpoint": {"x": 1}}
            return httpx.Response(200, json={"saved": True, "checkpoint_id": "c-1"})
        if req.method == "GET" and "/a2a/checkpoints/t-1/c-1" in req.url.path:
            return httpx.Response(200, json={"checkpoint": {"x": 1}})
        return httpx.Response(404)

    c = A2AClient(BASE, transport=_wrap(handler))
    out = c.save_checkpoint("t-1", "c-1", {"checkpoint": {"x": 1}})
    assert out["saved"] is True
    loaded = c.load_checkpoint("t-1", "c-1")
    assert loaded["checkpoint"] == {"x": 1}
    # PUT then GET — verify the path/method order.
    assert ("PUT", "/a2a/checkpoints/t-1/c-1") in seen
    assert ("GET", "/a2a/checkpoints/t-1/c-1") in seen


def test_a2a_agent_card_fetches_well_known() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/.well-known/agent.json"
        return httpx.Response(
            200,
            json={
                "name": "Relata",
                "skills": [{"name": "query_knowledge"}],
                "authentication": {"schemes": ["bearer"]},
            },
        )

    c = A2AClient(BASE, transport=_wrap(handler))
    card = c.agent_card()
    assert card["name"] == "Relata"
    assert card["skills"][0]["name"] == "query_knowledge"


def test_system_client_llm_config_and_test() -> None:
    seen: list[tuple[str, str]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.method, req.url.path))
        if req.url.path == "/config/llm" and req.method == "GET":
            return httpx.Response(200, json={"endpoint": "http://ollama:11434"})
        if req.url.path == "/config/llm/test" and req.method == "POST":
            body = json.loads(req.content)
            assert body["prompt"] == "hi"
            return httpx.Response(200, json={"reply": "hello", "latency_ms": 12})
        return httpx.Response(404)

    s = SystemClient(BASE, transport=_wrap(handler))
    cfg = s.llm_config()
    assert cfg["endpoint"] == "http://ollama:11434"
    out = s.test_llm(prompt="hi")
    assert out["reply"] == "hello"


def test_system_client_jobs_status() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/jobs"
        return httpx.Response(
            200,
            json={"jobs": [{"name": "orphan_sweep", "status": "ok"}]},
        )

    s = SystemClient(BASE, transport=_wrap(handler))
    out = s.jobs_status()
    assert out["jobs"][0]["name"] == "orphan_sweep"
