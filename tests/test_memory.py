"""Tests for the Mem0-style :class:`relata.Memory` client (#684).

Uses httpx's built-in ``MockTransport`` (no live server, no extra deps) to
verify request shapes and the MCP-envelope unwrapping.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx

from relata import Memory

BASE = "http://localhost:8080"

Handler = Callable[[httpx.Request], httpx.Response]


def _mcp(result: dict[str, Any]) -> dict[str, Any]:
    """Wrap a result the way the server's mcp_ok() does."""
    return {"content": [{"type": "text", "text": json.dumps(result)}], "isError": False}


def _memory(handler: Handler) -> Memory:
    return Memory(BASE, purpose="agent-notes", transport=httpx.MockTransport(handler))


def test_add_returns_id_and_sends_governed_payload() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        return httpx.Response(200, json=_mcp({"stored": "MemoryItem", "id": "mem-123"}))

    with _memory(handler) as m:
        mem_id = m.add("Alice prefers dark mode", confidence=0.8, memory_class="episodic")

    assert mem_id == "mem-123"
    sent = json.loads(seen["request"].content)
    assert sent["content"] == "Alice prefers dark mode"
    assert sent["purpose"] == "agent-notes"  # governance: purpose always sent
    assert sent["confidence"] == 0.8
    assert sent["memory_class"] == "episodic"
    assert str(seen["request"].url).endswith("/memory/remember")


def test_search_unwraps_rows() -> None:
    rows = [
        {
            "id": "a",
            "content": "dark mode",
            "confidence": 0.9,
            "score": 1.2,
            "memory_class": "semantic",
        },
        {
            "id": "b",
            "content": "light mode",
            "confidence": 0.5,
            "score": 0.3,
            "memory_class": "semantic",
        },
    ]
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json=_mcp({"rows": rows, "count": 2, "query": "ui"}))

    with _memory(handler) as m:
        hits = m.search("ui preferences", top_k=2)

    assert [h["id"] for h in hits] == ["a", "b"]
    assert "/memory/recall" in seen["url"]
    assert "q=ui+preferences" in seen["url"]
    assert "purpose=agent-notes" in seen["url"]
    assert "top_k=2" in seen["url"]


def test_search_empty_when_no_rows() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_mcp({"count": 0, "query": "x"}))

    with _memory(handler) as m:
        assert m.search("nothing") == []


def test_forget_is_governed_retract() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        return httpx.Response(
            200,
            json=_mcp({"memory_item_id": "mem-123", "policy": "default", "forget_at_ns": 42}),
        )

    with _memory(handler) as m:
        out = m.forget("mem-123")

    assert out["memory_item_id"] == "mem-123"
    assert out["forget_at_ns"] == 42
    assert seen["request"].method == "DELETE"
    url = str(seen["request"].url)
    assert "/memory/forget/mem-123" in url
    assert "purpose=agent-notes" in url


def test_get_returns_memory_or_none() -> None:
    def found(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=_mcp({"recognized": True, "memory": {"id": "m1", "content": "hi"}})
        )

    with _memory(found) as m:
        got = m.get("m1")
        assert got is not None and got["content"] == "hi"

    def missing(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_mcp({"recognized": False}))

    with _memory(missing) as m:
        assert m.get("nope") is None


def test_update_returns_new_id_and_sends_supersede() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        return httpx.Response(200, json=_mcp({"superseded": "old", "new_id": "new-7"}))

    with _memory(handler) as m:
        new_id = m.update("old", "revised content")

    assert new_id == "new-7"
    sent = json.loads(seen["request"].content)
    assert sent == {"id": "old", "content": "revised content", "purpose": "agent-notes"}
    assert str(seen["request"].url).endswith("/memory/consolidate")


def test_purpose_required() -> None:
    try:
        Memory(BASE, purpose="")
    except ValueError as exc:
        assert "purpose" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("empty purpose should raise (governance is on)")
