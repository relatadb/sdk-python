"""Regression test for #4704: ``RelataClient.schema_alter``'s documented

short action verbs (``"add"`` | ``"drop"`` | ``"rename"`` | ``"retype"``)
must be translated to the server's long-form verbs (``add_column`` |
``remove_column`` | ``rename_column`` | ``change_type``,
``crates/relata-cli/src/serve/types_routes.rs::schema_alter_handler``) on
the wire, since the server only ever accepted the long form and every call
following the method's own docstring previously 400'd.

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


def test_schema_alter_translates_documented_short_verbs() -> None:
    seen: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        return httpx.Response(200, json={"schema_generation": 2})

    client = _client(handler)
    client.schema_alter("Company", "add", "lei", col_type="text")
    client.schema_alter("Company", "drop", "old_col")
    client.schema_alter("Company", "rename", "old_col", new_column="new_col")
    client.schema_alter("Company", "retype", "lei", col_type="text")

    assert seen[0]["action"] == "add_column"
    assert seen[1]["action"] == "remove_column"
    assert seen[2]["action"] == "rename_column"
    assert seen[3]["action"] == "change_type"


def test_schema_alter_passes_through_long_form_verbs_unchanged() -> None:
    seen: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        return httpx.Response(200, json={"schema_generation": 2})

    client = _client(handler)
    client.schema_alter("Company", "add_column", "lei", col_type="text")

    assert seen[0]["action"] == "add_column"


def test_schema_alter_hits_patch_types_schema() -> None:
    seen: dict[str, str] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["method"] = req.method
        seen["path"] = req.url.path
        return httpx.Response(200, json={"schema_generation": 3})

    client = _client(handler)
    client.schema_alter("Company", "add", "lei", col_type="text")

    assert seen["method"] == "PATCH"
    assert seen["path"] == "/types/Company/schema"
