"""Tests for the #2757 SDK-sync deltas: async MCP wrapper parity, the six
async-method gaps, version-string alignment, and the `__all__` / re-export
completeness fixes.

Uses httpx.MockTransport (works for both sync and async clients — see the
existing pattern in test_async_memory.py). pytest-asyncio runs the coroutine
tests (`asyncio_mode = "auto"`, see pyproject.toml).
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from typing import Any

import httpx

import relata
from relata._http import AsyncHttpTransport, HttpTransport
from relata.mcp import AsyncMcpClient, McpClient

BASE = "http://localhost:9090"

Handler = Callable[[httpx.Request], httpx.Response]


def _wrap(handler: Handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def _mcp(result: dict[str, Any]) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(result)}], "isError": False}


# ---------------------------------------------------------------------------
# Delta #1 — version-string drift
# ---------------------------------------------------------------------------


def test_package_version_is_canonical() -> None:
    assert relata.__version__ == "2.0.0"


def test_user_agent_header_matches_canonical_version() -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json={"status": "ok"})

    t = HttpTransport(BASE, None, 30.0, transport=_wrap(handler))
    t.get("/status")
    assert seen[0].headers["user-agent"] == f"relata-sdk-python/{relata.__version__}"


# ---------------------------------------------------------------------------
# Delta #4/#5 — __all__ completeness + top-level re-exports
# ---------------------------------------------------------------------------


def test_all_contains_previously_omitted_exception_and_model_symbols() -> None:
    for name in (
        "ForbiddenError",
        "NotFoundError",
        "ConflictError",
        "ValidationError",
        "RateLimitedError",
        "SearchHit",
        "SearchResponse",
    ):
        assert name in relata.__all__, f"{name} missing from relata.__all__"
        assert hasattr(relata, name)


def test_backup_token_log_clients_reexported_from_top_level() -> None:
    for name in (
        "BackupClient",
        "AsyncBackupClient",
        "TokenClient",
        "AsyncTokenClient",
        "LogClient",
        "AsyncLogClient",
    ):
        assert name in relata.__all__, f"{name} missing from relata.__all__"
        assert hasattr(relata, name)


# ---------------------------------------------------------------------------
# Delta #2 — AsyncMcpClient mirrors every McpClient convenience wrapper
# ---------------------------------------------------------------------------


def test_async_mcp_client_mirrors_every_sync_wrapper() -> None:
    """Structural parity: every non-lifecycle sync method has an async,
    identically-named, coroutine-function mirror with the same required
    positional parameters."""
    skip = {"close", "__enter__", "__exit__", "__init__", "from_client"}
    sync_methods = {
        name: fn
        for name, fn in inspect.getmembers(McpClient, predicate=inspect.isfunction)
        if name not in skip
    }
    async_methods = {
        name: fn
        for name, fn in inspect.getmembers(AsyncMcpClient, predicate=inspect.iscoroutinefunction)
    }
    missing = sorted(set(sync_methods) - set(async_methods))
    assert not missing, f"AsyncMcpClient is missing async mirrors for: {missing}"
    for name, sync_fn in sync_methods.items():
        sync_params = list(inspect.signature(sync_fn).parameters)
        async_params = list(inspect.signature(async_methods[name]).parameters)
        assert sync_params == async_params, f"{name} signature drifted between sync/async"


async def test_async_mcp_query_knowledge_sends_same_payload_as_sync() -> None:
    seen: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        return httpx.Response(200, json=_mcp({"rows": [{"x": 1}]}))

    async with AsyncMcpClient(BASE, transport=_wrap(handler)) as c:
        out = await c.query_knowledge("SELECT 1", purpose="x")
    assert out["rows"] == [{"x": 1}]
    assert seen[0] == {"name": "query_knowledge", "arguments": {"sql": "SELECT 1", "purpose": "x"}}


async def test_async_mcp_recall_forwards_adr145_params() -> None:
    seen: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        return httpx.Response(200, json=_mcp({"rows": []}))

    async with AsyncMcpClient(BASE, transport=_wrap(handler)) as c:
        await c.recall("hello", purpose="x", min_confidence=0.5, budget_tokens=100)
    args = seen[0]["arguments"]
    assert args["min_confidence"] == 0.5
    assert args["budget_tokens"] == 100


async def test_async_mcp_face_match_sends_probe_id() -> None:
    seen: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        return httpx.Response(200, json=_mcp({"matches": []}))

    async with AsyncMcpClient(BASE, transport=_wrap(handler)) as c:
        await c.face_match("probe-1", threshold=0.9)
    assert seen[0]["name"] == "face_match"
    assert seen[0]["arguments"]["probe_id"] == "probe-1"


# ---------------------------------------------------------------------------
# Delta #3 — the six async-method gaps
# ---------------------------------------------------------------------------


async def test_async_ingest_iter_batches_sync_and_async_iterables() -> None:
    from relata.ingest import AsyncIngestClient

    seen: list[int] = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = req.content.decode().strip()
        seen.append(len(body.split("\n")))
        return httpx.Response(200, json={"accepted": len(body.split("\n"))})

    c = AsyncIngestClient(BASE, purpose="onboarding", transport=_wrap(handler))
    total = await c.ingest_iter("Person", [{"name": f"n{i}"} for i in range(5)], batch_size=2)
    assert total == 5
    assert seen == [2, 2, 1]

    async def _agen() -> Any:
        for i in range(3):
            yield {"name": f"a{i}"}

    seen.clear()
    total2 = await c.ingest_iter("Person", _agen(), batch_size=2)
    assert total2 == 3
    assert seen == [2, 1]


async def test_async_object_client_typed_upsert() -> None:
    from relata.objects import AsyncObjectClient

    seen: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        assert "object_type=Person" in str(req.url)
        rows = [json.loads(line) for line in req.content.decode().strip().split("\n")]
        seen.extend(rows)
        return httpx.Response(200, json={"accepted": 1, "write_seq": 1})

    c = AsyncObjectClient(BASE, purpose="analytics", transport=_wrap(handler))
    out = await c.typed_upsert({"_type": "Person", "id": "p-1", "name": "Alice"})
    assert out["accepted"] == 1
    assert seen[0] == {"id": "p-1", "name": "Alice"}


async def test_async_memory_batch_search_merges_results() -> None:
    from relata import AsyncMemory

    def handler(req: httpx.Request) -> httpx.Response:
        q = dict(req.url.params)["q"]
        return httpx.Response(200, json=_mcp({"rows": [{"id": q}]}))

    async with AsyncMemory(BASE, purpose="agent", transport=_wrap(handler)) as m:
        out = await m.batch_search(["a", "b"], top_k=3)
    assert out == [[{"id": "a"}], [{"id": "b"}]]


async def test_async_governance_list_retention_policies() -> None:
    from relata.governance import AsyncGovernanceClient

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/retention/policies"
        return httpx.Response(200, json={"policies": [{"object_type": "Person"}]})

    async with AsyncGovernanceClient(BASE, transport=_wrap(handler)) as g:
        policies = await g.list_retention_policies()
    assert policies == [{"object_type": "Person"}]


async def test_async_backup_compact() -> None:
    from relata.backup import AsyncBackupClient

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/admin/compact"
        assert req.method == "POST"
        return httpx.Response(200, json={"status": "started"})

    async with AsyncBackupClient(BASE, transport=_wrap(handler)) as b:
        out = await b.compact()
    assert out["status"] == "started"


async def test_async_streaming_query_arrow_raw_yields_chunks() -> None:
    from relata import RelataClient
    from relata.streaming import AsyncStreamingClient

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/query/arrow"
        return httpx.Response(200, content=b"arrow-bytes-chunk")

    client = RelataClient(BASE, purpose="analytics")
    client._RelataClient__async_transport = (  # type: ignore[attr-defined]
        AsyncHttpTransport(BASE, None, 30.0, transport=_wrap(handler), extra_headers=None)
    )
    stream = AsyncStreamingClient.from_client(client)
    chunks = [c async for c in await stream.query_arrow_raw("SELECT 1", purpose="analytics")]
    assert b"".join(chunks) == b"arrow-bytes-chunk"
