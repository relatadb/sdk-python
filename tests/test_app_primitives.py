"""Wire-shape tests for the application primitives client (#5768, #5783)."""

from __future__ import annotations

import json

import httpx
import pytest

from relata.app_primitives import AppPrimitivesClient

BASE = "http://localhost:9090"


def _client(handler):  # type: ignore[no-untyped-def]
    return AppPrimitivesClient(BASE, transport=httpx.MockTransport(handler))


def test_sequence_next_sends_camel_case_body() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "POST" and req.url.path == "/v1/sequences/inv/next"
        assert json.loads(req.content) == {"count": 5, "idempotencyKey": "k"}
        return httpx.Response(200, json={"start": 1, "end": 5, "next": 6, "duplicate": False})

    assert _client(handler).sequence_next("inv", count=5, idempotency_key="k")["end"] == 5


def test_claim_returns_jobs_and_complete_sends_fence() -> None:
    seen: list[tuple[str, dict]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.url.path, json.loads(req.content)))
        if req.url.path.endswith("/claim"):
            return httpx.Response(200, json={"jobs": [{"jobId": "j1", "fencingToken": 7}]})
        return httpx.Response(200, json={"state": "done"})

    c = _client(handler)
    jobs = c.claim("q", "w1", lease_ms=1000, max=2)
    assert jobs[0]["fencingToken"] == 7
    c.complete("j1", 7, result={"ok": True})
    assert seen[0] == ("/v1/queues/q/claim", {"workerId": "w1", "leaseMs": 1000, "max": 2})
    assert seen[1] == ("/v1/jobs/j1/complete", {"fencingToken": 7, "result": {"ok": True}})


def test_stale_token_surfaces_as_an_error() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            409,
            json={"type": "x", "title": "stale", "status": 409, "code": "stale-fencing-token"},
            headers={"content-type": "application/problem+json"},
        )

    with pytest.raises(Exception) as err:
        _client(handler).complete("j1", 1)
    assert "409" in repr(err.value) or "stale" in str(err.value).lower()


def test_list_jobs_builds_query_string() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/v1/queues/q/jobs"
        assert dict(req.url.params) == {"state": "dead", "limit": "5"}
        return httpx.Response(200, json={"queue": "q", "counts": {}, "jobs": []})

    assert _client(handler).list_jobs("q", state="dead", limit=5)["jobs"] == []
