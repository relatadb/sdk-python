"""Tests for the async :class:`relata.AsyncMemory` client (#688).

Uses httpx's ``MockTransport`` (async). pytest-asyncio runs the coroutines
(``asyncio_mode = "auto"``).
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from relata import AsyncMemory

BASE = "http://localhost:9090"


def _mcp(result: dict[str, Any]) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(result)}], "isError": False}


async def test_async_add_search_forget_roundtrip() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        path = request.url.path
        if path == "/v1/memory/remember":
            return httpx.Response(200, json=_mcp({"id": "mem-9"}))
        if path == "/v1/memory/recall":
            return httpx.Response(200, json=_mcp({"rows": [{"id": "mem-9", "content": "x"}]}))
        return httpx.Response(200, json=_mcp({"memory_item_id": path.rsplit("/", 1)[-1]}))

    async with AsyncMemory(BASE, purpose="agent", transport=httpx.MockTransport(handler)) as m:
        assert await m.add("hello") == "mem-9"
        hits = await m.search("hello", top_k=3)
        assert hits[0]["id"] == "mem-9"
        out = await m.forget("mem-9")
        assert out["memory_item_id"] == "mem-9"

    paths = [r.url.path for r in seen]
    assert paths == ["/v1/memory/remember", "/v1/memory/recall", "/v1/memory/forget/mem-9"]
    assert json.loads(seen[1].content)["purpose"] == "agent"


async def test_async_search_sends_adr145_params_and_detailed_surfaces_metadata() -> None:
    """#2674: async mirror of the sync ADR-145 retrieval-quality coverage."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json=_mcp(
                {
                    "rows": [{"id": "mem-9", "content": "x"}],
                    "recall_cost_tokens": 17,
                    "cancelled": False,
                }
            ),
        )

    async with AsyncMemory(BASE, purpose="agent", transport=httpx.MockTransport(handler)) as m:
        hits = await m.search("hello", min_confidence=0.4, budget_tokens=100)
        assert hits[0]["id"] == "mem-9"
        detailed = await m.search_detailed("hello", stability_days=3.0, cancel_threshold=0.8)

    body = json.loads(seen[0].content)
    assert body["min_confidence"] == 0.4
    assert body["budget_tokens"] == 100
    assert detailed["recall_cost_tokens"] == 17
    assert detailed["cancelled"] is False


async def test_async_get_and_update() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.startswith("/v1/memory/recognize/"):
            return httpx.Response(200, json=_mcp({"memory": {"id": "m1", "content": "hi"}}))
        return httpx.Response(200, json=_mcp({"new_id": "new-2"}))

    async with AsyncMemory(BASE, purpose="agent", transport=httpx.MockTransport(handler)) as m:
        got = await m.get("m1")
        assert got is not None and got["content"] == "hi"
        assert await m.update("m1", "revised") == "new-2"


async def test_async_purpose_required() -> None:
    try:
        AsyncMemory(BASE, purpose="")
    except ValueError as exc:
        assert "purpose" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("empty purpose should raise")
