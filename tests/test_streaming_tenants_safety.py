"""Tests for streaming (#75), tenant admin (#73), and SQL-injection stopgap (#76)."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from relata import RelataClient
from relata._http import HttpTransport
from relata.query import select
from relata.tenants import TenantAdminClient

BASE = "http://localhost:9090"

Handler = Callable[[httpx.Request], httpx.Response]


def _wrap(handler: Handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def _client(handler: Handler) -> RelataClient:
    client = RelataClient(BASE, purpose="analytics")
    client._RelataClient__sync_transport = HttpTransport(  # type: ignore[attr-defined]
        BASE, None, 30.0, transport=_wrap(handler), extra_headers=None,
    )
    return client


# ---------------------------------------------------------------------------
# #76 — SQL injection allowlist stopgap
# ---------------------------------------------------------------------------


def test_from_rejects_invalid_table_name() -> None:
    """``from_()`` validates the table identifier."""
    with pytest.raises(ValueError, match="Invalid table name"):
        select("*").from_("Person; DROP TABLE--").sql()


def test_from_rejects_table_with_spaces() -> None:
    """``from_()`` rejects table names with spaces (injection attempt)."""
    with pytest.raises(ValueError, match="Invalid table name"):
        select("*").from_("Person WHERE 1=1").sql()


def test_where_rejects_semicolon() -> None:
    with pytest.raises(ValueError, match="forbidden token"):
        select("Person").where("age > 18; DROP TABLE Person").sql()


def test_where_rejects_drop_keyword() -> None:
    with pytest.raises(ValueError, match="forbidden token"):
        select("Person").where("1=1; DROP TABLE Person").sql()


def test_where_rejects_comment_injection() -> None:
    with pytest.raises(ValueError, match="forbidden token"):
        select("Person").where("age > 18 -- comment").sql()


def test_where_accepts_normal_condition() -> None:
    sql = select("Person").where("age > 18 AND name LIKE 'A%'").sql()
    assert "WHERE (age > 18 AND name LIKE 'A%')" in sql


def test_from_accepts_dotted_identifier() -> None:
    """``namespace.column`` is a valid dotted identifier."""
    sql = select("ns.col").sql()
    assert "FROM ns.col" in sql


# ---------------------------------------------------------------------------
# #75 — Streaming (row + SSE)
# ---------------------------------------------------------------------------


def test_query_rows_streams_ndjson() -> None:
    """The row streamer consumes NDJSON line by line without buffering."""

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/v1/query/stream"
        assert req.headers["accept"] == "application/x-ndjson"
        body = (
            '{"_meta":{"query_id":"q1","started_at":1,"columns":["id"]}}\n'
            '{"id": 1}\n{"id": 2}\n{"id": 3}\n'
            '{"_meta":{"processing_time_ms":1,"row_count":3,"truncated":false}}\n'
        )
        return httpx.Response(
            200,
            content=body.encode(),
            headers={"content-type": "application/x-ndjson"},
        )

    from relata.streaming import StreamingClient

    client = _client(handler)
    stream = StreamingClient.from_client(client)
    rows = list(stream.query_rows("SELECT * FROM Person"))
    assert [r["id"] for r in rows] == [1, 2, 3]


def test_watch_sse_parses_events() -> None:
    """The watch SSE consumer parses event/data framing."""
    sse_body = (
        "event: RowsAppended\n"
        'data: {"rows": [{"id": 1}]}\n'
        "\n"
        "event: RowsAppended\n"
        'data: {"rows": [{"id": 2}]}\n'
        "\n"
    )

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/v1/watch/stream"
        return httpx.Response(
            200,
            content=sse_body.encode(),
            headers={"content-type": "text/event-stream"},
        )

    from relata.streaming import StreamingClient

    client = _client(handler)
    stream = StreamingClient.from_client(client)
    events = list(stream.watch("SELECT * FROM Person", purpose="analytics", reconnect=False))
    assert len(events) == 2
    assert events[0]["event"] == "RowsAppended"
    assert events[0]["rows"] == [{"id": 1}]


# ---------------------------------------------------------------------------
# Tenant self-introspection (admin CRUD is Enterprise-only)
# ---------------------------------------------------------------------------


def test_tenant_client_me_only() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v1/tenants/me":
            return httpx.Response(200, json={"tenant_id": "acme", "tier": "enterprise"})
        return httpx.Response(404)

    c = TenantAdminClient(BASE, transport=_wrap(handler))
    me = c.me()
    assert me["tenant_id"] == "acme"
    for removed in ("usage", "create", "delete", "set_quota", "suspend", "platform_usage"):
        assert not hasattr(c, removed)
