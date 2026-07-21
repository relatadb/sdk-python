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
        assert req.url.path == "/query"
        # Return 3 NDJSON rows.
        body = '{"id": 1}\n{"id": 2}\n{"id": 3}\n'
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
        assert req.url.path == "/watch/stream"
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
# #73 — Tenant admin CRUD
# ---------------------------------------------------------------------------


def test_tenant_admin_create_posts_to_platform() -> None:
    seen: list[dict[str, object]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append({"method": req.method, "path": req.url.path})
        if req.method == "POST" and req.url.path == "/platform/tenants":
            body = json.loads(req.content)
            assert body == {"tenant_id": "acme", "tier": "enterprise"}
            return httpx.Response(200, json={"tenant_id": "acme", "status": "active"})
        return httpx.Response(404)

    c = TenantAdminClient(BASE, transport=_wrap(handler))
    out = c.create("acme", tier="enterprise")
    assert out["tenant_id"] == "acme"


def test_tenant_admin_me_and_usage() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/tenants/me":
            return httpx.Response(200, json={"tenant_id": "acme", "tier": "enterprise"})
        if req.url.path == "/tenants/usage":
            return httpx.Response(200, json={"rows": 1000, "cost_used": 42})
        return httpx.Response(404)

    c = TenantAdminClient(BASE, transport=_wrap(handler))
    me = c.me()
    assert me["tenant_id"] == "acme"
    usage = c.usage()
    assert usage["rows"] == 1000


def test_tenant_admin_set_quota() -> None:
    seen: list[dict[str, object]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append({"method": req.method, "path": req.url.path})
        if req.method == "PUT" and req.url.path == "/tenants/acme/quota":
            body = json.loads(req.content)
            assert body == {"quota": 50000}
            return httpx.Response(200, json={"quota": 50000})
        return httpx.Response(404)

    c = TenantAdminClient(BASE, transport=_wrap(handler))
    out = c.set_quota("acme", 50000)
    assert out["quota"] == 50000


def test_tenant_admin_create_sharing() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/tenants/acme/sharing"
        body = json.loads(req.content)
        assert body == {
            "partner_tenant": "partner-org",
            "object_type": "Person",
            "action": "read",
        }
        return httpx.Response(200, json={"agreement_id": "sa-1"})

    c = TenantAdminClient(BASE, transport=_wrap(handler))
    out = c.create_sharing("acme", "partner-org", object_type="Person")
    assert out["agreement_id"] == "sa-1"


def test_tenant_admin_suspend_reactivate() -> None:
    seen: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.url.path)
        if "suspend" in req.url.path:
            return httpx.Response(200, json={"status": "suspended"})
        if "reactivate" in req.url.path:
            return httpx.Response(200, json={"status": "active"})
        return httpx.Response(404)

    c = TenantAdminClient(BASE, transport=_wrap(handler))
    assert c.suspend("acme")["status"] == "suspended"
    assert c.reactivate("acme")["status"] == "active"
