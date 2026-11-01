"""Wire tests for the server operations that previously had no SDK method (Refs #5960)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from relata import RelataClient
from relata._http import HttpTransport
from relata.governance import AsyncGovernanceClient, GovernanceClient
from relata.ingest import AsyncIngestClient, IngestClient
from relata.streaming import AsyncStreamingClient, StreamingClient
from relata.system import AsyncSystemClient, SystemClient

BASE = "http://localhost:9090"

Seen = list[httpx.Request]


def _json_transport(seen: Seen, body: dict[str, Any] | None = None) -> httpx.MockTransport:
    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json=body or {"ok": True})

    return httpx.MockTransport(handler)


def test_system_ops_hit_expected_routes() -> None:
    seen: Seen = []
    c = SystemClient(BASE, transport=_json_transport(seen))
    c.system_config()
    c.system_specs()
    c.mcp_sessions()
    c.mcp_sessions(session_id="s 1", limit=5)
    c.update_notification_rule("r/1", {"enabled": False})
    got = [(r.method, r.url.raw_path.decode()) for r in seen]
    assert got == [
        ("GET", "/v1/system/config"),
        ("GET", "/v1/system/specs"),
        ("GET", "/v1/mcp/sessions"),
        ("GET", "/v1/mcp/sessions?session_id=s+1&limit=5"),
        ("PATCH", "/v1/notification-rules/r%2F1"),
    ]
    assert json.loads(seen[4].content) == {"enabled": False}


@pytest.mark.asyncio
async def test_system_ops_async_mirror() -> None:
    seen: Seen = []

    async def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json={"ok": True})

    c = AsyncSystemClient(BASE, transport=httpx.MockTransport(handler))
    await c.system_config()
    await c.system_specs()
    await c.mcp_sessions(limit=2)
    await c.update_notification_rule("r1", {"enabled": True})
    assert [(r.method, r.url.raw_path.decode()) for r in seen] == [
        ("GET", "/v1/system/config"),
        ("GET", "/v1/system/specs"),
        ("GET", "/v1/mcp/sessions?limit=2"),
        ("PATCH", "/v1/notification-rules/r1"),
    ]


def test_governance_ops() -> None:
    seen: Seen = []
    c = GovernanceClient(BASE, transport=_json_transport(seen))
    c.get_rule("a/b")
    c.remove_rule_exception("a/b", "e 1")
    c.set_retention_policy("TraceSpan", 30)
    assert [(r.method, r.url.raw_path.decode()) for r in seen] == [
        ("GET", "/v1/rules/a%2Fb"),
        ("DELETE", "/v1/rules/a%2Fb/exceptions/e%201"),
        ("POST", "/v1/retention/policies"),
    ]
    assert json.loads(seen[2].content) == {
        "object_type": "TraceSpan",
        "max_age_days": 30,
        "action": "delete",
    }


@pytest.mark.asyncio
async def test_governance_ops_async() -> None:
    seen: Seen = []

    async def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json={})

    c = AsyncGovernanceClient(BASE, transport=httpx.MockTransport(handler))
    await c.get_rule("r1")
    await c.remove_rule_exception("r1", "e1")
    await c.set_retention_policy("T", 7, "tier:cold")
    assert [(r.method, r.url.path) for r in seen] == [
        ("GET", "/v1/rules/r1"),
        ("DELETE", "/v1/rules/r1/exceptions/e1"),
        ("POST", "/v1/retention/policies"),
    ]
    assert json.loads(seen[2].content)["action"] == "tier:cold"


def test_import_stix_sends_bundle_with_format_and_purpose() -> None:
    seen: Seen = []
    c = IngestClient(BASE, purpose="intel", transport=_json_transport(seen))
    bundle = {"type": "bundle", "id": "bundle--1", "objects": []}
    c.import_stix(bundle)
    req = seen[0]
    assert req.method == "POST"
    assert req.url.path == "/v1/ingest/import"
    assert dict(req.url.params) == {"format": "stix", "purpose": "intel"}
    assert json.loads(req.content) == bundle


@pytest.mark.asyncio
async def test_import_stix_async() -> None:
    seen: Seen = []

    async def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json={})

    c = AsyncIngestClient(BASE, transport=httpx.MockTransport(handler))
    await c.import_stix({"type": "bundle"}, purpose="p")
    assert dict(seen[0].url.params) == {"format": "stix", "purpose": "p"}


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> RelataClient:
    client = RelataClient(BASE, purpose="analytics")
    client._RelataClient__sync_transport = HttpTransport(  # type: ignore[attr-defined]
        BASE, None, 30.0, transport=httpx.MockTransport(handler), extra_headers=None,
    )
    return client


def test_detect_config_list_and_set() -> None:
    seen: Seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json={"configs": []})

    c = _client(handler)
    c.list_type_detect_configs()
    c.set_type_detect_config("Person", ["contact"])
    assert [(r.method, r.url.path) for r in seen] == [
        ("GET", "/v1/types/detect-config"),
        ("POST", "/v1/types/detect-config"),
    ]
    assert json.loads(seen[1].content) == {"type": "Person", "packs": ["contact"]}


def _sse(seen: Seen) -> Callable[[httpx.Request], httpx.Response]:
    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(
            200,
            content=b'event: change\ndata: {"id": 1}\n\n',
            headers={"content-type": "text/event-stream"},
        )

    return handler


def test_graph_changes_and_a2a_stream_sync() -> None:
    seen: Seen = []
    stream = StreamingClient.from_client(_client(_sse(seen)))
    events = list(stream.graph_changes(type="Person", reconnect=False))
    assert events == [{"id": 1, "event": "change"}]
    assert seen[0].url.path == "/v1/graph/changes"
    assert dict(seen[0].url.params) == {"type": "Person"}
    assert list(stream.a2a_task_stream("t/1")) == [{"id": 1, "event": "change"}]
    assert seen[1].url.raw_path.decode() == "/v1/a2a/tasks/t%2F1/stream"


def test_graph_changes_without_filters_has_no_query() -> None:
    seen: Seen = []
    stream = StreamingClient.from_client(_client(_sse(seen)))
    list(stream.graph_changes(reconnect=False))
    assert seen[0].url.raw_path.decode() == "/v1/graph/changes"


@pytest.mark.asyncio
async def test_graph_changes_and_a2a_stream_async() -> None:
    seen: Seen = []
    client = RelataClient(BASE, purpose="analytics")
    from relata._http import AsyncHttpTransport

    client._RelataClient__async_transport = AsyncHttpTransport(  # type: ignore[attr-defined]
        BASE, None, 30.0, transport=httpx.MockTransport(_sse(seen)), extra_headers=None,
    )
    stream = AsyncStreamingClient(client)
    out = [e async for e in await stream.graph_changes(tenant_id="acme")]
    assert out == [{"id": 1, "event": "change"}]
    assert dict(seen[0].url.params) == {"tenant_id": "acme"}
    out = [e async for e in await stream.a2a_task_stream("t1")]
    assert seen[1].url.path == "/v1/a2a/tasks/t1/stream"
    assert out == [{"id": 1, "event": "change"}]
