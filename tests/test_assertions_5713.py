"""#5713 temporal-assertion client: wire shapes, quoting and error surfacing."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from relata.assertions import AssertionsClient, AsyncAssertionsClient

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
    c = AssertionsClient(BASE, transport=httpx.MockTransport(_handler(seen)))
    claim = {"predicate": "city", "value": "Rome"}
    c.record(claim)
    c.list()
    c.list(subject_type="Person", subject_id="p 1", predicate="city", state="confirmed", limit=5)
    c.get("asr_1")
    c.confirm("asr_1", "verified")
    c.reject("asr/2", "stale")
    c.revoke("asr_1", "error")
    c.publish_policy("trust", {"precedence": ["state"]}, expected_version=2)
    c.get_policy("trust")
    c.get_policy("trust", version=3)
    c.resolve("Person", "p1", "city", 5)
    c.resolve("Person", "p1", "city", 5, recorded_as_of=9, policy="trust", policy_version=2)
    assert seen == [
        ("POST", "/v1/assertions", "", claim),
        ("GET", "/v1/assertions", "", None),
        (
            "GET",
            "/v1/assertions",
            "subject_type=Person&subject_id=p+1&predicate=city&state=confirmed&limit=5",
            None,
        ),
        ("GET", "/v1/assertions/asr_1", "", None),
        (
            "POST",
            "/v1/assertions/asr_1/decision",
            "",
            {"decision": "confirm", "reason": "verified"},
        ),
        ("POST", "/v1/assertions/asr%2F2/decision", "", {"decision": "reject", "reason": "stale"}),
        ("POST", "/v1/assertions/asr_1/revoke", "", {"reason": "error"}),
        (
            "PUT",
            "/v1/assertions/policies/trust",
            "",
            {"expected_version": 2, "definition": {"precedence": ["state"]}},
        ),
        ("GET", "/v1/assertions/policies/trust", "", None),
        ("GET", "/v1/assertions/policies/trust", "version=3", None),
        (
            "GET",
            "/v1/assertions/resolve",
            "subject_type=Person&subject_id=p1&predicate=city&valid_at=5",
            None,
        ),
        (
            "GET",
            "/v1/assertions/resolve",
            "subject_type=Person&subject_id=p1&predicate=city&valid_at=5&recorded_as_of=9&policy=trust&policy_version=2",
            None,
        ),
    ]


def test_errors_surface_with_their_problem_code() -> None:
    body = {
        "code": "assertion-retention-blocked",
        "status": 409,
        "title": "Assertion Retention Blocked",
    }
    c = AssertionsClient(BASE, transport=httpx.MockTransport(_handler([], 409, body)))
    with pytest.raises(Exception) as err:  # noqa: PT011
        c.revoke("asr_1", "x")
    assert (
        "assertion-retention-blocked" in str(err.value)
        or getattr(err.value, "status_code", 0) == 409
    )


@pytest.mark.asyncio
async def test_async_client_mirrors_the_sync_surface() -> None:
    seen: list[tuple[str, str, str, Any]] = []
    async with AsyncAssertionsClient(BASE, transport=httpx.MockTransport(_handler(seen))) as c:
        await c.record({"predicate": "city"})
        await c.confirm("asr_1", "ok")
        await c.resolve("Person", "p1", "city", 7, policy="trust")
    assert [(m, p) for m, p, _q, _b in seen] == [
        ("POST", "/v1/assertions"),
        ("POST", "/v1/assertions/asr_1/decision"),
        ("GET", "/v1/assertions/resolve"),
    ]
    assert seen[2][2] == "subject_type=Person&subject_id=p1&predicate=city&valid_at=7&policy=trust"
