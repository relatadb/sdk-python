"""Tests for schema-BRANCH CRUD + typed edge definitions (#2497).

These are distinct from schema-ALTER (#2476, ``PATCH /types/:name/schema``) —
this is the schema-BRANCH surface (``POST/DELETE /schema/branches/:name``)
plus the typed-edge door (``GET/POST /types/edges``), mirrored from the Rust
flat client at ``crates/relata-sdk-rust/src/http.rs:2535-2570``.

Uses httpx.MockTransport — no live server required.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx

from relata import RelataClient
from relata._http import AsyncHttpTransport, HttpTransport

BASE = "http://localhost:9090"
Handler = Callable[[httpx.Request], httpx.Response]


def _mock(handler: Handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def _client(handler: Handler) -> RelataClient:
    client = RelataClient(BASE, bearer_token="tok", purpose="schema")
    mock = _mock(handler)
    extra = client._extra_headers  # type: ignore[attr-defined]
    client._RelataClient__sync_transport = HttpTransport(  # type: ignore[attr-defined]
        BASE, client._bearer_token, client._timeout, transport=mock, extra_headers=extra
    )
    client._RelataClient__async_transport = AsyncHttpTransport(  # type: ignore[attr-defined]
        BASE, client._bearer_token, client._timeout, transport=mock, extra_headers=extra
    )
    return client


# ── list_edge_types / register_edge_type ───────────────────────────────────


def test_list_edge_types_gets_types_edges() -> None:
    seen: dict[str, str] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["method"] = req.method
        seen["path"] = req.url.path
        return httpx.Response(
            200,
            json={"edges": [{"from_type": "Person", "to_type": "Org", "label": "DIRECTOR_OF"}]},
        )

    client = _client(handler)
    out = client.list_edge_types()
    assert seen["method"] == "GET"
    assert seen["path"] == "/types/edges"
    assert out["edges"][0]["label"] == "DIRECTOR_OF"


def test_register_edge_type_posts_correct_body() -> None:
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["method"] = req.method
        seen["path"] = req.url.path
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"created": True})

    client = _client(handler)
    out = client.register_edge_type("Person", "Org", "DIRECTOR_OF")
    assert seen["method"] == "POST"
    assert seen["path"] == "/types/edges"
    assert seen["body"] == {"from_type": "Person", "to_type": "Org", "label": "DIRECTOR_OF"}
    assert out["created"] is True


# ── create_schema_branch / delete_schema_branch ────────────────────────────


def test_create_schema_branch_posts_to_branch_door() -> None:
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["method"] = req.method
        seen["path"] = req.url.path
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"name": "exp-branch", "created": True})

    client = _client(handler)
    out = client.create_schema_branch("exp-branch", "main")
    assert seen["method"] == "POST"
    assert seen["path"] == "/schema/branches/exp-branch"
    assert seen["body"] == {"from": "main"}
    assert out["name"] == "exp-branch"
    assert out["created"] is True


def test_delete_schema_branch_deletes_branch_door() -> None:
    seen: dict[str, str] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["method"] = req.method
        seen["path"] = req.url.path
        return httpx.Response(200, json={"name": "exp-branch", "deleted": True})

    client = _client(handler)
    out = client.delete_schema_branch("exp-branch")
    assert seen["method"] == "DELETE"
    assert seen["path"] == "/schema/branches/exp-branch"
    assert out["deleted"] is True


# ── async variants ─────────────────────────────────────────────────────────


def test_async_schema_branch_crud_round_trip() -> None:
    import asyncio

    captured: list[tuple[str, str]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append((req.method, req.url.path))
        if req.method == "POST" and "/schema/branches/" in req.url.path:
            return httpx.Response(200, json={"name": "dev", "created": True})
        if req.method == "DELETE" and "/schema/branches/" in req.url.path:
            return httpx.Response(200, json={"name": "dev", "deleted": True})
        if req.url.path == "/types/edges":
            if req.method == "GET":
                return httpx.Response(200, json={"edges": []})
            return httpx.Response(200, json={"created": True})
        return httpx.Response(404)

    client = _client(handler)

    async def run() -> None:
        await client.acreate_schema_branch("dev", "main")
        await client.aregister_edge_type("Person", "Org", "KNOWS")
        await client.alist_edge_types()
        await client.adelete_schema_branch("dev")

    asyncio.run(run())
    assert ("POST", "/schema/branches/dev") in captured
    assert ("POST", "/types/edges") in captured
    assert ("GET", "/types/edges") in captured
    assert ("DELETE", "/schema/branches/dev") in captured
