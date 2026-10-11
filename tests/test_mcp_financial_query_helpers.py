"""Financial MCP helpers are SDK-side SQL composition over generic query."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from relata.mcp import AsyncMcpClient, McpClient


def _capture() -> tuple[McpClient, list[dict[str, Any]]]:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"content": [{"type": "text", "text": '{"rows":[]}'}], "isError": False},
        )

    return McpClient("http://localhost", transport=httpx.MockTransport(handler)), seen


def _assert_query(seen: list[dict[str, Any]], sql: str, purpose: str) -> None:
    call = seen[-1]
    assert call["name"] == "query"
    assert call["arguments"] == {"sql": sql, "purpose": purpose}


def test_financial_helpers_compose_legacy_sql_and_clamps() -> None:
    client, seen = _capture()
    client.trace_crypto("bc1", max_hops=30, min_amount=1.25, purpose="audit")
    _assert_query(
        seen, "PURPOSE 'audit' CRYPTO_TRACE('bc1', MAX_HOPS => 10, MIN_AMOUNT => 1.25)", "audit"
    )
    client.beneficial_ownership("Acme", max_depth=0)
    _assert_query(
        seen, "PURPOSE 'analytics' BENEFICIAL_OWNERSHIP_CHAIN('Acme', MAX_DEPTH => 1)", "analytics"
    )
    client.reconstruct_wire("acct-1")
    _assert_query(
        seen, "PURPOSE 'analytics' WIRE_RECONSTRUCTION('acct-1', TOLERANCE_PCT => 5)", "analytics"
    )
    client.trace_hawala("actor-1")
    _assert_query(seen, "PURPOSE 'analytics' HAWALA_TRACE('actor-1', MAX_HOPS => 5)", "analytics")
    client.screen_sanctions("Jane Doe", threshold=0.9)
    _assert_query(
        seen,
        "PURPOSE 'compliance_review' SANCTIONS_SCREEN('Jane Doe', THRESHOLD => 0.9)",
        "compliance_review",
    )
    client.screen_sanctions("Jane Doe")
    _assert_query(
        seen, "PURPOSE 'compliance_review' SANCTIONS_SCREEN('Jane Doe')", "compliance_review"
    )


def test_financial_helpers_reject_sql_injection_before_http() -> None:
    client, seen = _capture()
    with pytest.raises(ValueError):
        client.trace_crypto("x'); SELECT * FROM Person; --")
    with pytest.raises(ValueError):
        client.trace_crypto("address", purpose="audit' SELECT")
    assert seen == []


@pytest.mark.asyncio
async def test_async_financial_helpers_use_generic_query() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"content": [{"type": "text", "text": '{"rows":[]}'}], "isError": False},
        )

    async with AsyncMcpClient("http://localhost", transport=httpx.MockTransport(handler)) as client:
        await client.trace_hawala("actor-1", max_hops=30)
    _assert_query(seen, "PURPOSE 'analytics' HAWALA_TRACE('actor-1', MAX_HOPS => 10)", "analytics")
