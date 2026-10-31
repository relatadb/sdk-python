"""#5678 extraction-suggestion client: wire shapes, quoting and error surfacing."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from relata.extraction import AsyncExtractionClient, ExtractionClient

BASE = "http://localhost:9090"


def _handler(
    seen: list, status: int = 200, body: object = None
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(req: httpx.Request) -> httpx.Response:
        payload = json.loads(req.content) if req.content else None
        seen.append(
            (req.method, req.url.raw_path.decode().split("?")[0], req.url.query.decode(), payload)
        )
        return httpx.Response(status, json=body if body is not None else {"ok": True})

    return handler


def test_every_call_hits_its_documented_route_with_exact_bodies() -> None:
    seen: list = []
    c = ExtractionClient(BASE, transport=httpx.MockTransport(_handler(seen)))
    s = {"kind": "entity", "confidence": 0.9}
    fix = {"type": "Person", "properties": {"name": "Ada"}}
    c.submit([s])
    c.list()
    c.list(
        state="pending",
        kind="entity",
        definition="kb",
        low_confidence=True,
        stale=False,
        min_confidence=0.25,
        limit=5,
    )
    c.get("sug/1")
    c.accept("sug_1", "ok")
    c.accept("sug_1", accept_low_confidence=True)
    c.reject("sug_1", "wrong")
    c.correct("sug_1", fix, "typo")
    c.reopen("sug_1")
    c.apply("kb")
    c.apply("kb", suggestion_ids=["sug_1"])
    c.entities_exist([{"type": "Person", "match": {"email": "a@b.c"}}])
    c.entities_exist([], definition="kb")
    assert seen == [
        ("POST", "/ontology/suggestions", "", {"suggestions": [s]}),
        ("GET", "/ontology/suggestions", "", None),
        (
            "GET",
            "/ontology/suggestions",
            "state=pending&kind=entity&definition=kb&low_confidence=true&stale=false&min_confidence=0.25&limit=5",
            None,
        ),
        ("GET", "/ontology/suggestions/sug%2F1", "", None),
        (
            "POST",
            "/ontology/suggestions/sug_1/decision",
            "",
            {"decision": "accept", "reason": "ok"},
        ),
        (
            "POST",
            "/ontology/suggestions/sug_1/decision",
            "",
            {"decision": "accept", "reason": "", "accept_low_confidence": True},
        ),
        (
            "POST",
            "/ontology/suggestions/sug_1/decision",
            "",
            {"decision": "reject", "reason": "wrong"},
        ),
        (
            "POST",
            "/ontology/suggestions/sug_1/decision",
            "",
            {"decision": "correct", "reason": "typo", "correction": fix},
        ),
        ("POST", "/ontology/suggestions/sug_1/decision", "", {"decision": "reopen", "reason": ""}),
        ("POST", "/ontology/suggestions/apply", "", {"definition": "kb"}),
        (
            "POST",
            "/ontology/suggestions/apply",
            "",
            {"definition": "kb", "suggestion_ids": ["sug_1"]},
        ),
        (
            "POST",
            "/entities/exists",
            "",
            {"candidates": [{"type": "Person", "match": {"email": "a@b.c"}}]},
        ),
        ("POST", "/entities/exists", "", {"candidates": [], "definition": "kb"}),
    ]


def test_errors_surface_with_their_problem_code() -> None:
    body = {
        "code": "suggestion-low-confidence",
        "status": 409,
        "title": "Suggestion Low Confidence",
    }
    c = ExtractionClient(BASE, transport=httpx.MockTransport(_handler([], 409, body)))
    with pytest.raises(Exception) as err:  # noqa: PT011
        c.accept("sug_1")
    assert (
        "suggestion-low-confidence" in str(err.value) or getattr(err.value, "status_code", 0) == 409
    )


@pytest.mark.asyncio
async def test_async_client_mirrors_the_sync_surface() -> None:
    seen: list = []
    c = AsyncExtractionClient(BASE, transport=httpx.MockTransport(_handler(seen)))
    await c.submit([{"kind": "entity"}])
    await c.list(state="pending")
    await c.get("sug_1")
    await c.correct("sug_1", {"type": "Person"}, "typo")
    await c.apply("kb")
    await c.entities_exist([], definition="kb")
    await c.close()
    assert [(m, p) for m, p, _, _ in seen] == [
        ("POST", "/ontology/suggestions"),
        ("GET", "/ontology/suggestions"),
        ("GET", "/ontology/suggestions/sug_1"),
        ("POST", "/ontology/suggestions/sug_1/decision"),
        ("POST", "/ontology/suggestions/apply"),
        ("POST", "/entities/exists"),
    ]
