"""Tests for the agent-framework memory adapters (#683).

The frameworks themselves are not imported (the adapters are framework-agnostic);
these tests drive each adapter through a mocked Relata `/memory/*` surface via
httpx.MockTransport and assert the store/recall/retract behaviour.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx

from relata import Memory
from relata_adapters import message_parts
from relata_adapters.autogen import RelataMemory as AutoGenMemory
from relata_adapters.crewai import RelataStorage
from relata_adapters.langchain import RelataMemory as LangChainMemory
from relata_adapters.llamaindex import RelataMemory as LlamaIndexMemory

BASE = "http://localhost:9090"
Handler = Callable[[httpx.Request], httpx.Response]


def _mcp(result: dict[str, Any]) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(result)}], "isError": False}


class Recorder:
    """A MockTransport handler that records calls and serves canned replies."""

    def __init__(self) -> None:
        self.calls: list[httpx.Request] = []
        self._next_id = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        path = request.url.path
        if path == "/memory/remember":
            self._next_id += 1
            return httpx.Response(200, json=_mcp({"id": f"mem-{self._next_id}"}))
        if path == "/memory/recall":
            return httpx.Response(
                200,
                json=_mcp(
                    {"rows": [{"id": "a", "content": "dark mode", "score": 0.9}], "count": 1}
                ),
            )
        if path.startswith("/memory/forget/"):
            return httpx.Response(200, json=_mcp({"memory_item_id": path.rsplit("/", 1)[-1]}))
        return httpx.Response(404, json={"error": "not found"})

    def paths(self) -> list[str]:
        return [c.url.path for c in self.calls]


def _memory(rec: Recorder) -> Memory:
    return Memory(BASE, purpose="agent", transport=httpx.MockTransport(rec))


def test_message_parts_duck_types() -> None:
    assert message_parts({"role": "user", "content": "hi"}) == ("user", "hi")
    assert message_parts({"type": "ai", "content": "yo"}) == ("ai", "yo")

    class Msg:
        role = "human"
        content = "hello"

    assert message_parts(Msg()) == ("human", "hello")
    assert message_parts("plain")[1] == "plain"


def test_langchain_save_and_load() -> None:
    rec = Recorder()
    mem = LangChainMemory(_memory(rec))
    assert mem.memory_variables == ["history"]

    mem.save_context({"input": "what theme?"}, {"output": "dark mode"})
    # two remembers: the human input + the ai output
    assert rec.paths().count("/memory/remember") == 2

    out = mem.load_memory_variables({"input": "theme preference"})
    assert "dark mode" in out["history"]
    assert "/memory/recall" in rec.paths()

    mem.clear()
    assert rec.paths().count("/memory/forget/mem-1") == 1


def test_llamaindex_put_get_reset() -> None:
    rec = Recorder()
    mem = LlamaIndexMemory(_memory(rec))
    mem.put({"role": "user", "content": "remember dark mode"})
    assert rec.paths().count("/memory/remember") == 1

    hits = mem.get("theme")
    assert hits and hits[0]["content"] == "dark mode"
    assert mem.get(None) == []  # semantic memory: no query -> nothing
    assert mem.get_all() == []  # documented: not supported without a query

    mem.reset()
    assert any(p.startswith("/memory/forget/") for p in rec.paths())


def test_crewai_save_search_threshold() -> None:
    rec = Recorder()
    store = RelataStorage(_memory(rec))
    store.save("Alice prefers dark mode", metadata={"user": "alice"})
    assert rec.paths().count("/memory/remember") == 1

    hits = store.search("preferences", limit=5)
    assert len(hits) == 1
    # score 0.9 in the canned row -> filtered out by a higher threshold
    assert store.search("preferences", score_threshold=0.95) == []


def test_autogen_async_add_query_clear() -> None:
    import asyncio

    rec = Recorder()
    mem = AutoGenMemory(_memory(rec))

    async def run() -> None:
        await mem.add({"role": "user", "content": "dark mode"})
        rows = await mem.query("theme")
        assert rows and rows[0]["content"] == "dark mode"
        await mem.clear()
        await mem.close()

    asyncio.run(run())
    assert rec.paths().count("/memory/remember") == 1
    assert "/memory/recall" in rec.paths()
    assert any(p.startswith("/memory/forget/") for p in rec.paths())
