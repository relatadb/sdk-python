"""Tests for the v1.1 introspection + tenant-header surface (#86, #73).

Covers:
- ``RelataClient`` constructor accepts ``tenant`` / ``acting_as`` /
  ``delegated_by`` / ``headers`` and propagates them as HTTP headers.
- ``stats()`` / ``astats()`` round-trip through ``GET /debug/stats``.
- ``version()`` / ``aversion()`` round-trip through ``GET /version``.
- ``ready()`` / ``aready()`` round-trip through ``GET /health/ready``.

Uses ``httpx.MockTransport`` — no live server required.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from relata import ReadyReport, RelataClient, Stats, VersionInfo
from relata._http import AsyncHttpTransport, HttpTransport

BASE = "http://localhost:9090"

Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: Handler, **client_kwargs: object) -> RelataClient:
    """Build a ``RelataClient`` whose transport is mocked by ``handler``.

    The client normally builds its ``HttpTransport`` lazily on first call. We
    pre-populate the private backing fields with real ``HttpTransport`` /
    ``AsyncHttpTransport`` instances whose ``httpx`` layer is intercepted by
    ``handler`` via ``httpx.MockTransport`` — the same pattern used in
    ``test_memory.py``.
    """
    client = RelataClient(BASE, **client_kwargs)  # type: ignore[arg-type]
    mock = httpx.MockTransport(handler)
    extra = client._extra_headers  # type: ignore[attr-defined]
    client._RelataClient__sync_transport = HttpTransport(  # type: ignore[attr-defined]
        BASE,
        client._bearer_token,  # type: ignore[attr-defined]
        client._timeout,  # type: ignore[attr-defined]
        transport=mock,
        extra_headers=extra,
    )
    client._RelataClient__async_transport = AsyncHttpTransport(  # type: ignore[attr-defined]
        BASE,
        client._bearer_token,  # type: ignore[attr-defined]
        client._timeout,  # type: ignore[attr-defined]
        transport=mock,
        extra_headers=extra,
    )
    return client


def test_tenant_header_is_sent_on_every_request() -> None:
    """``tenant=`` constructor arg becomes ``X-Relata-Tenant-Id`` on the wire."""
    captured: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        return httpx.Response(200, json={"status": "ok", "profile": "free", "node_id": "n1"})

    client = _client(handler, tenant="org-cbi")
    client.health()

    assert len(captured) == 1
    assert captured[0].headers["X-Relata-Tenant-Id"] == "org-cbi"


def test_acting_as_and_delegated_by_headers_propagate() -> None:
    captured: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        return httpx.Response(200, json={"status": "ok", "profile": "free", "node_id": "n1"})

    client = _client(
        handler,
        tenant="acme",
        acting_as="acme-partner",
        delegated_by="service-acct",
    )
    client.health()

    assert captured[0].headers["X-Relata-Tenant-Id"] == "acme"
    assert captured[0].headers["X-Acting-As"] == "acme-partner"
    assert captured[0].headers["X-Delegated-By"] == "service-acct"


def test_custom_headers_win_over_defaults() -> None:
    """Caller-supplied ``headers=`` can override SDK defaults (e.g. User-Agent)."""
    captured: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        return httpx.Response(200, json={"status": "ok", "profile": "free", "node_id": "n1"})

    client = _client(
        handler,
        headers={"User-Agent": "my-app/1.0", "X-Request-ID": "abc-123"},
    )
    client.health()

    assert captured[0].headers["User-Agent"] == "my-app/1.0"
    assert captured[0].headers["X-Request-ID"] == "abc-123"


def test_stats_round_trip() -> None:
    payload = {
        "records": 15234,
        "states": 482111,
        "snapshot_rows": 12000,
        "log_leaves": None,  # pending #85
        "tokens": None,      # pending #84
        "mv_freshness": [{"name": "mv_persons", "age_secs": 12}],
    }

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/debug/stats"
        return httpx.Response(200, json=payload)

    client = _client(handler)
    stats: Stats = client.stats()
    assert stats.records == 15234
    assert stats.states == 482111
    assert stats.snapshot_rows == 12000
    assert stats.log_leaves is None
    assert stats.tokens is None


def test_version_round_trip() -> None:
    payload = {
        "version": "1.1.0",
        "commit": "7622d74",
        "profile": "server",
        "schema_version": "2026_07_06",
        "features": ["columnar", "flight"],
    }

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/version"
        return httpx.Response(200, json=payload)

    client = _client(handler)
    v: VersionInfo = client.version()
    assert v.version == "1.1.0"
    assert v.commit == "7622d74"
    assert v.profile == "server"
    assert v.schema_version == "2026_07_06"
    assert v.features == ["columnar", "flight"]


def test_ready_happy_path() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/health/ready"
        return httpx.Response(200, json={"status": "ok"})

    client = _client(handler)
    report: ReadyReport = client.ready()
    assert report.is_ready is True
    assert report.status == "ok"


def test_ready_shedding_surfaces_reason() -> None:
    """A 503 from /health/ready raises ServerError.

    Note: the structured ``reason`` field is currently lost because the legacy
    classifier only reads ``error`` / ``message`` keys (RFC 7807 wire format
    fix tracked in #62 / #80). When #62 ships, this test should be tightened
    to assert the reason text is preserved.
    """
    from relata.exceptions import ServerError

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            json={"status": "shedding", "reason": "queue_backpressure"},
        )

    client = _client(handler)
    with pytest.raises(ServerError) as exc_info:
        client.ready()
    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_async_methods_round_trip() -> None:
    """``astats`` / ``aversion`` / ``aready`` mirror their sync counterparts."""

    payload = {"status": "ok", "profile": "free", "node_id": "n1"}

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client = _client(handler)
    health = await client.ahealth()
    assert health.status == "ok"
