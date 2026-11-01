"""#5702 investigation client: wire shapes, quoting and error surfacing."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from relata.investigation import AsyncInvestigationClient, InvestigationClient

BASE = "http://localhost:9090"


def _handler(
    seen: list[tuple[str, str, str, Any]], status: int = 200, body: object = None
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(req: httpx.Request) -> httpx.Response:
        payload = json.loads(req.content) if req.content else None
        seen.append(
            (req.method, req.url.raw_path.decode().split("?")[0], req.url.query.decode(), payload)
        )
        return httpx.Response(status, json=body if body is not None else {"ok": True})

    return handler


def test_every_call_hits_its_documented_route_with_exact_bodies() -> None:
    seen: list[tuple[str, str, str, Any]] = []
    c = InvestigationClient(BASE, transport=httpx.MockTransport(_handler(seen)))
    c.alert_evidence("a/1")
    c.alert_evidence("a1", max_events=5, deadline_ms=900)
    c.alert_timeline("a1")
    c.alert_timeline("a1", deadline_ms=10)
    c.pivot("a1")
    c.pivot(
        "a1",
        field="user",
        value="alice",
        target_field="owner",
        target_types=["Session"],
        limit=3,
        deadline_ms=50,
    )
    c.create_saved_search("n", "SELECT 1", purpose="audit", description="d", params=["x"])
    c.list_saved_searches()
    c.list_saved_searches(q="fail", limit=7)
    c.get_saved_search("ss/1")
    c.update_saved_search("ss-1", "n", "SELECT 2", expected_version=3)
    c.delete_saved_search("ss-1")
    c.run_saved_search("ss-1")
    c.run_saved_search("ss-1", params=[1, None])
    assert seen == [
        ("GET", "/v1/alerts/a%2F1/evidence", "", None),
        ("GET", "/v1/alerts/a1/evidence", "max_events=5&deadline_ms=900", None),
        ("GET", "/v1/alerts/a1/timeline", "", None),
        ("GET", "/v1/alerts/a1/timeline", "deadline_ms=10", None),
        ("POST", "/v1/alerts/a1/pivot", "", {}),
        (
            "POST",
            "/v1/alerts/a1/pivot",
            "",
            {
                "field": "user",
                "value": "alice",
                "target_field": "owner",
                "target_types": ["Session"],
                "limit": 3,
                "deadline_ms": 50,
            },
        ),
        (
            "POST",
            "/v1/search/saved",
            "",
            {
                "name": "n",
                "sql": "SELECT 1",
                "purpose": "audit",
                "description": "d",
                "params": ["x"],
            },
        ),
        ("GET", "/v1/search/saved", "", None),
        ("GET", "/v1/search/saved", "q=fail&limit=7", None),
        ("GET", "/v1/search/saved/ss%2F1", "", None),
        (
            "PUT",
            "/v1/search/saved/ss-1",
            "",
            {"name": "n", "sql": "SELECT 2", "expected_version": 3},
        ),
        ("DELETE", "/v1/search/saved/ss-1", "", None),
        ("POST", "/v1/search/saved/ss-1/run", "", {}),
        ("POST", "/v1/search/saved/ss-1/run", "", {"params": [1, None]}),
    ]


def test_errors_surface_with_their_problem_code() -> None:
    body = {"code": "pivot-not-authorized", "status": 403, "title": "Pivot not authorized"}
    c = InvestigationClient(BASE, transport=httpx.MockTransport(_handler([], 403, body)))
    with pytest.raises(Exception) as err:  # noqa: PT011
        c.pivot("a1", value="bob")
    assert (
        "pivot-not-authorized" in str(err.value) or getattr(err.value, "status_code", 0) == 403
    )


@pytest.mark.asyncio
async def test_async_client_mirrors_the_sync_surface() -> None:
    seen: list[tuple[str, str, str, Any]] = []
    c = AsyncInvestigationClient(BASE, transport=httpx.MockTransport(_handler(seen)))
    await c.alert_evidence("a1")
    await c.alert_timeline("a1")
    await c.pivot("a1", limit=2)
    await c.create_saved_search("n", "SELECT 1")
    await c.list_saved_searches(q="x")
    await c.get_saved_search("ss-1")
    await c.update_saved_search("ss-1", "n", "SELECT 2")
    await c.delete_saved_search("ss-1")
    await c.run_saved_search("ss-1")
    await c.close()
    assert [(m, p) for m, p, _, _ in seen] == [
        ("GET", "/v1/alerts/a1/evidence"),
        ("GET", "/v1/alerts/a1/timeline"),
        ("POST", "/v1/alerts/a1/pivot"),
        ("POST", "/v1/search/saved"),
        ("GET", "/v1/search/saved"),
        ("GET", "/v1/search/saved/ss-1"),
        ("PUT", "/v1/search/saved/ss-1"),
        ("DELETE", "/v1/search/saved/ss-1"),
        ("POST", "/v1/search/saved/ss-1/run"),
    ]
