"""Tests for the async :class:`relata.AsyncMemory` client (#688).

Uses httpx's ``MockTransport`` (async). pytest-asyncio runs the coroutines
(``asyncio_mode = "auto"``).
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from relata import AsyncMemory

BASE = "http://localhost:8080"


def _mcp(result: dict[str, Any]) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(result)}], "isError": False}


async def test_async_add_search_forget_roundtrip() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        path = request.url.path
        if path == "/memory/remember":
            return httpx.Response(200, json=_mcp({"id": "mem-9"}))
        if path == "/memory/recall":
            return httpx.Response(200, json=_mcp({"rows": [{"id": "mem-9", "content": "x"}]}))
        return httpx.Response(200, json=_mcp({"memory_item_id": path.rsplit("/", 1)[-1]}))

    async with AsyncMemory(BASE, purpose="agent", transport=httpx.MockTransport(handler)) as m:
        assert await m.add("hello") == "mem-9"
        hits = await m.search("hello", top_k=3)
        assert hits[0]["id"] == "mem-9"
        out = await m.forget("mem-9")
        assert out["memory_item_id"] == "mem-9"

    paths = [r.url.path for r in seen]
    assert paths == ["/memory/remember", "/memory/recall", "/memory/forget/mem-9"]
    assert "purpose=agent" in str(seen[1].url)


async def test_async_get_and_update() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.startswith("/memory/recognize/"):
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
