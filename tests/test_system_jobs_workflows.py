"""Tests for the jobs/workflows SDK surface (#723) — SystemClient methods."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx

BASE = "http://localhost:8080"

Handler = Callable[[httpx.Request], httpx.Response]


def _wrap(handler: Handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def test_job_status_hits_named_route() -> None:
    from relata.system import SystemClient

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/jobs/c2_beacon"
        assert req.method == "GET"
        return httpx.Response(200, json={"name": "c2_beacon", "status": "Running"})

    c = SystemClient(BASE, transport=_wrap(handler))
    got = c.job_status("c2_beacon")
    assert got["status"] == "Running"


def test_run_workflow_returns_run_id() -> None:
    from relata.system import SystemClient

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/workflows/enrich/run"
        assert req.method == "POST"
        return httpx.Response(200, json={"run_id": "enrich/abc-123", "status": "Running"})

    c = SystemClient(BASE, transport=_wrap(handler))
    got = c.run_workflow("enrich")
    assert got["run_id"] == "enrich/abc-123"


def test_workflow_run_surfaces_retry_fields() -> None:
    from relata.system import SystemClient

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/workflows/runs/enrich/abc-123"
        assert req.method == "GET"
        return httpx.Response(
            200,
            json={
                "run_id": "enrich/abc-123",
                "workflow_name": "enrich",
                "status": "Failed",
                "attempt_count": 3,
                "last_error": "embed timeout",
            },
        )

    c = SystemClient(BASE, transport=_wrap(handler))
    got = c.workflow_run("enrich/abc-123")
    # #711/#713 made these fields accurate — the SDK must surface them.
    assert got["attempt_count"] == 3
    assert got["last_error"] == "embed timeout"


def test_register_workflow_posts_steps() -> None:
    from relata.system import SystemClient

    steps = [{"name": "q1", "kind": "query", "sql": "SELECT 1"}]

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/workflows"
        assert req.method == "POST"
        body = json.loads(req.content)
        assert body["name"] == "wf"
        assert body["steps"] == steps
        return httpx.Response(200, json={"name": "wf"})

    c = SystemClient(BASE, transport=_wrap(handler))
    got = c.register_workflow("wf", steps)
    assert got["name"] == "wf"


def test_async_run_workflow_returns_run_id() -> None:
    import asyncio

    from relata.system import AsyncSystemClient

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/workflows/enrich/run"
        return httpx.Response(200, json={"run_id": "enrich/x", "status": "Running"})

    async def main() -> None:
        c = AsyncSystemClient(BASE, transport=_wrap(handler))
        got = await c.run_workflow("enrich")
        assert got["run_id"] == "enrich/x"

    asyncio.run(main())
