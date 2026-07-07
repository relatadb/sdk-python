"""Tests for the v1.1 Memory verb completion (#77) and QueryBuilder cursor (#75).

Covers:
- ``Memory.associate`` / ``episodes`` / ``justify`` / ``resolve`` / ``summarise``
  (sync + async).
- ``QueryBuilder.limit(n, after=cursor)`` emits ``LIMIT n AFTER 'cursor'``.
- ``QueryBuilder.since(cursor)`` emits a ``system_from > cursor`` predicate.
- ``QueryBuilder.since`` composes with ``where`` and ``limit``.

Uses ``httpx.MockTransport`` — no live server required.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx

from relata import AsyncMemory, Memory
from relata.query import select

BASE = "http://localhost:8080"

Handler = Callable[[httpx.Request], httpx.Response]


def _mcp(result: dict[str, Any]) -> dict[str, Any]:
    """Wrap a result the way the server's mcp_ok() does."""
    return {"content": [{"type": "text", "text": json.dumps(result)}], "isError": False}


def _memory(handler: Handler) -> Memory:
    return Memory(BASE, purpose="agent-notes", transport=httpx.MockTransport(handler))


def _async_memory(handler: Handler) -> AsyncMemory:
    return AsyncMemory(BASE, purpose="agent-notes", transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------------------
# Memory — five missing cognitive verbs (sync)
# ---------------------------------------------------------------------------


def test_associate_round_trip() -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        assert req.url.path == "/memory/associate"
        body = json.loads(req.content)
        # Server schema uses from_id / to_id (not source_id/target_id).
        assert body["from_id"] == "mem-1"
        assert body["to_id"] == "mem-2"
        assert body["relation"] == "supports"
        return httpx.Response(200, json=_mcp({"association_id": "a-1"}))

    m = _memory(handler)
    out = m.associate("mem-1", "mem-2", "supports", confidence=0.9)
    assert out["association_id"] == "a-1"


def test_episodes_filters_by_session() -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(
            200,
            json=_mcp({"rows": [{"id": "ep-1"}, {"id": "ep-2"}]}),
        )

    m = _memory(handler)
    rows = m.episodes(session_id="sess-7", as_of="2026-01-01")
    assert [r["id"] for r in rows] == ["ep-1", "ep-2"]
    # The query string carries purpose + session_id + as_of
    qs = str(seen[0].url)
    assert "session_id=sess-7" in qs
    assert "as_of=2026-01-01" in qs


def test_justify_returns_assertion_chain() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/memory/justify/mem-9"
        return httpx.Response(
            200,
            json=_mcp({"chain": [{"assertion": "wasGeneratedBy", "actor": "tool-1"}]}),
        )

    m = _memory(handler)
    chain = m.justify("mem-9")
    assert chain["chain"][0]["actor"] == "tool-1"


def test_resolve_round_trip() -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        # Server route is GET /memory/resolve/:id (supersession-chain resolve).
        assert req.method == "GET"
        assert req.url.path == "/memory/resolve/mem-9"
        return httpx.Response(200, json=_mcp({"resolved_id": "mem-10"}))

    m = _memory(handler)
    out = m.resolve("mem-9")
    assert out["resolved_id"] == "mem-10"


def test_summarise_round_trip() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        # Server schema reads ids + contents (not source_ids/summary_content).
        assert body["ids"] == ["m1", "m2", "m3"]
        assert body["contents"] == ["Alice and Bob met in Nairobi."]
        return httpx.Response(200, json=_mcp({"summary_id": "sum-1"}))

    m = _memory(handler)
    out = m.summarise(["m1", "m2", "m3"], summary_content="Alice and Bob met in Nairobi.")
    assert out["summary_id"] == "sum-1"


# ---------------------------------------------------------------------------
# Memory — five missing cognitive verbs (async mirrors)
# ---------------------------------------------------------------------------


async def test_async_associate_round_trip() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_mcp({"association_id": "a-2"}))

    m = _async_memory(handler)
    out = await m.associate("m1", "m2", "contradicts")
    assert out["association_id"] == "a-2"


async def test_async_episodes_round_trip() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_mcp({"rows": [{"id": "ep-9"}]}))

    m = _async_memory(handler)
    rows = await m.episodes()
    assert rows == [{"id": "ep-9"}]


async def test_async_justify_resolve_summarise() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.startswith("/memory/justify/"):
            return httpx.Response(200, json=_mcp({"chain": []}))
        if req.url.path.startswith("/memory/resolve/"):
            return httpx.Response(200, json=_mcp({"resolved_id": "r1"}))
        if req.url.path == "/memory/summarise":
            return httpx.Response(200, json=_mcp({"summary_id": "s1"}))
        return httpx.Response(404)

    m = _async_memory(handler)
    assert await m.justify("x") == {"chain": []}
    assert (await m.resolve("x"))["resolved_id"] == "r1"
    assert (await m.summarise(["a", "b"]))["summary_id"] == "s1"


# ---------------------------------------------------------------------------
# QueryBuilder cursor support (#75)
# ---------------------------------------------------------------------------


def test_limit_after_emits_keyset_form() -> None:
    sql = select("Person").limit(50, after="1760000000000000000").sql()
    assert sql == (
        "SELECT * FROM Person LIMIT 50 AFTER '1760000000000000000'"
    )


def test_limit_without_after_emits_plain_limit() -> None:
    sql = select("Person").limit(10).sql()
    assert sql == "SELECT * FROM Person LIMIT 10"


def test_since_emits_system_from_predicate() -> None:
    sql = select("Person").since("1760000000000000000").limit(100).sql()
    assert "WHERE (system_from > 1760000000000000000)" in sql
    assert sql.endswith("LIMIT 100")


def test_since_composes_with_where() -> None:
    sql = (
        select("Person")
        .where("age > 18")
        .since("1760000000000000000")
        .limit(10)
        .sql()
    )
    # Both predicates appear in WHERE, joined by AND.
    assert "WHERE (age > 18) AND (system_from > 1760000000000000000)" in sql


def test_limit_rejects_nonpositive() -> None:
    import pytest

    with pytest.raises(ValueError):
        select("Person").limit(0)
    with pytest.raises(ValueError):
        select("Person").limit(-5)
