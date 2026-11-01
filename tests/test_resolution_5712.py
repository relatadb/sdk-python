"""#5712 entity-resolution client: wire shapes, quoting and error surfacing."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from relata.resolution import AsyncResolutionClient, ResolutionClient

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
    c = ResolutionClient(BASE, transport=httpx.MockTransport(_handler(seen)))
    d = {"entity_type": "Person"}
    c.list_rulesets()
    c.get_ruleset("people")
    c.get_ruleset("people", version=2)
    c.publish_ruleset("people", d, expected_version=1)
    c.run("people")
    c.run("people", proposals=[{"rule": "m", "a": "x", "b": "y", "score": 0.9}])
    c.evaluate("people", [{"a": "x", "b": "y", "same": True}])
    c.list_candidates()
    c.list_candidates(ruleset="people", state="pending", limit=5)
    c.get_candidate("cand_1")
    c.approve("cand_1", "same mailbox")
    c.reject("cand_2", "different people")
    c.unmerge("cand_1", "wrong")
    c.get_entity("ent_1")
    c.lookup("Person", "r 1")
    assert seen == [
        ("GET", "/v1/resolution/rulesets", "", None),
        ("GET", "/v1/resolution/rulesets/people", "", None),
        ("GET", "/v1/resolution/rulesets/people", "version=2", None),
        ("PUT", "/v1/resolution/rulesets/people", "", {"expected_version": 1, "definition": d}),
        ("POST", "/v1/resolution/rulesets/people/run", "", {}),
        (
            "POST",
            "/v1/resolution/rulesets/people/run",
            "",
            {"proposals": [{"rule": "m", "a": "x", "b": "y", "score": 0.9}]},
        ),
        (
            "POST",
            "/v1/resolution/rulesets/people/evaluate",
            "",
            {"labeled": [{"a": "x", "b": "y", "same": True}]},
        ),
        ("GET", "/v1/resolution/candidates", "", None),
        ("GET", "/v1/resolution/candidates", "ruleset=people&state=pending&limit=5", None),
        ("GET", "/v1/resolution/candidates/cand_1", "", None),
        (
            "POST",
            "/v1/resolution/candidates/cand_1/decision",
            "",
            {"decision": "approve", "reason": "same mailbox"},
        ),
        (
            "POST",
            "/v1/resolution/candidates/cand_2/decision",
            "",
            {"decision": "reject", "reason": "different people"},
        ),
        (
            "POST",
            "/v1/resolution/candidates/cand_1/decision",
            "",
            {"decision": "unmerge", "reason": "wrong"},
        ),
        ("GET", "/v1/resolution/entities/ent_1", "", None),
        ("GET", "/v1/resolution/lookup", "type=Person&id=r+1", None),
    ]


def test_names_are_path_quoted_and_a_conflict_raises() -> None:
    seen: list[tuple[str, str, str, Any]] = []
    c = ResolutionClient(BASE, transport=httpx.MockTransport(_handler(seen)))
    c.get_ruleset("a/b c")
    assert seen[0][1] == "/v1/resolution/rulesets/a%2Fb%20c"
    body = {"code": "resolution-version-conflict", "status": 412, "title": "x", "detail": "d"}
    failing = ResolutionClient(BASE, transport=httpx.MockTransport(_handler([], 412, body)))
    with pytest.raises(Exception) as err:
        failing.publish_ruleset("people", {}, expected_version=0)
    assert "412" in str(err.value) or "resolution-version-conflict" in str(err.value)


@pytest.mark.asyncio
async def test_async_client_uses_the_same_routes() -> None:
    seen: list[tuple[str, str, str, Any]] = []
    async with AsyncResolutionClient(BASE, transport=httpx.MockTransport(_handler(seen))) as c:
        await c.approve("cand_1", "ok")
        await c.lookup("Person", "r1")
    assert seen == [
        (
            "POST",
            "/v1/resolution/candidates/cand_1/decision",
            "",
            {"decision": "approve", "reason": "ok"},
        ),
        ("GET", "/v1/resolution/lookup", "type=Person&id=r1", None),
    ]
