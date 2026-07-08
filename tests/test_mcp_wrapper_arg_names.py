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

BASE = "http://localhost:8080"


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
