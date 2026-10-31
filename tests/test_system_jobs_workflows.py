"""Tests for the jobs SDK surface (#723) — SystemClient methods."""

from __future__ import annotations

from collections.abc import Callable

import httpx

BASE = "http://localhost:9090"

Handler = Callable[[httpx.Request], httpx.Response]


def _wrap(handler: Handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def test_job_status_hits_named_route() -> None:
    from relata.system import SystemClient

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/v1/jobs/c2_beacon"
        assert req.method == "GET"
        return httpx.Response(200, json={"name": "c2_beacon", "status": "Running"})

    c = SystemClient(BASE, transport=_wrap(handler))
    got = c.job_status("c2_beacon")
    assert got["status"] == "Running"
