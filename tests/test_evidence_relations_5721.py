"""#5721 evidence-relation client: wire shapes and quoting."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from relata.assertions import AssertionsClient, AsyncAssertionsClient

BASE = "http://localhost:9090"


def _handler(
    seen: list[tuple[str, str, str, Any]],
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(req: httpx.Request) -> httpx.Response:
        payload = json.loads(req.content) if req.content else None
        seen.append(
            (req.method, req.url.raw_path.decode().split("?")[0], req.url.query.decode(), payload)
        )
        return httpx.Response(200, json={"ok": True})

    return handler


def test_relation_calls_hit_their_documented_routes() -> None:
    seen: list[tuple[str, str, str, Any]] = []
    c = AssertionsClient(BASE, transport=httpx.MockTransport(_handler(seen)))
    c.relate("asr_a", "supports", "asr_b", "same passage", locator={"pointer": "page=3"})
    c.relate("asr_a", "contradicts", "asr_b", "dates differ")
    c.unrelate("rel/1", "misread")
    c.assess("asr_a", "disputed", "two witnesses disagree")
    c.evidence_graph("asr_a")
    c.evidence_graph("asr/a", depth=3, limit=10, policy="trust", policy_version=2)
    assert seen == [
        (
            "POST",
            "/v1/assertions/relations",
            "",
            {
                "from": "asr_a",
                "to": "asr_b",
                "kind": "supports",
                "rationale": "same passage",
                "locator": {"pointer": "page=3"},
            },
        ),
        (
            "POST",
            "/v1/assertions/relations",
            "",
            {"from": "asr_a", "to": "asr_b", "kind": "contradicts", "rationale": "dates differ"},
        ),
        ("POST", "/v1/assertions/relations/rel%2F1/revoke", "", {"reason": "misread"}),
        (
            "POST",
            "/v1/assertions/asr_a/assessment",
            "",
            {"judgment": "disputed", "reason": "two witnesses disagree"},
        ),
        ("GET", "/v1/assertions/asr_a/evidence-graph", "", None),
        (
            "GET",
            "/v1/assertions/asr%2Fa/evidence-graph",
            "depth=3&limit=10&policy=trust&policy_version=2",
            None,
        ),
    ]


@pytest.mark.asyncio
async def test_async_relation_calls_mirror_the_sync_surface() -> None:
    seen: list[tuple[str, str, str, Any]] = []
    async with AsyncAssertionsClient(BASE, transport=httpx.MockTransport(_handler(seen))) as c:
        await c.relate("a", "derived_from", "b", "copied")
        await c.unrelate("rel_1", "x")
        await c.assess("a", "inconclusive", "r")
        await c.evidence_graph("a", depth=1)
    assert [(m, p, q) for m, p, q, _b in seen] == [
        ("POST", "/v1/assertions/relations", ""),
        ("POST", "/v1/assertions/relations/rel_1/revoke", ""),
        ("POST", "/v1/assertions/a/assessment", ""),
        ("GET", "/v1/assertions/a/evidence-graph", "depth=1"),
    ]


def test_similarity_relation_and_threshold_wire_shapes() -> None:
    seen: list[tuple[str, str, str, Any]] = []
    c = AssertionsClient(BASE, transport=httpx.MockTransport(_handler(seen)))
    c.relate("a", "similar_to", "b", "same photo", similarity={"score": 0.93, "method": "phash-v1"})
    c.relate("a", "distinct_from", "c", "different frame")
    c.evidence_graph("a", min_similarity=0.5)
    assert seen[0][3]["similarity"] == {"score": 0.93, "method": "phash-v1"}
    assert "similarity" not in seen[1][3]
    assert seen[2][2] == "min_similarity=0.5"


def test_confidence_and_uncertainty_reach_the_wire_only_when_given() -> None:
    seen: list[tuple[str, str, str, Any]] = []
    c = AssertionsClient(BASE, transport=httpx.MockTransport(_handler(seen)))
    c.relate("a", "supports", "b", "same figure", confidence=0.4, uncertainty="one witness")
    c.assess("a", "inconclusive", "needs a second source", confidence=0.6)
    c.assess("a", "inconclusive", "plain")
    assert seen[0][3] == {
        "from": "a",
        "to": "b",
        "kind": "supports",
        "rationale": "same figure",
        "confidence": 0.4,
        "uncertainty": "one witness",
    }
    assert seen[1][3] == {
        "judgment": "inconclusive",
        "reason": "needs a second source",
        "confidence": 0.6,
    }
    assert seen[2][3] == {"judgment": "inconclusive", "reason": "plain"}
