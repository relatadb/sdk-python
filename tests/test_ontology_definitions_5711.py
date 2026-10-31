"""#5711 ontology definitions client: wire shapes, quoting and error surfacing."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from relata.ontology_definitions import OntologyDefinitionsClient

BASE = "http://localhost:9090"


def _client(
    seen: list[tuple[str, str, str, Any]], status: int = 200, body: object = None
) -> OntologyDefinitionsClient:
    def handler(req: httpx.Request) -> httpx.Response:
        payload = json.loads(req.content) if req.content else None
        seen.append(
            (req.method, req.url.raw_path.decode().split("?")[0], req.url.query.decode(), payload)
        )
        return httpx.Response(status, json=body if body is not None else {"ok": True})

    return OntologyDefinitionsClient(BASE, transport=httpx.MockTransport(handler))


def test_every_call_hits_its_documented_route_with_exact_bodies() -> None:
    seen: list[tuple[str, str, str, Any]] = []
    c = _client(seen)
    d = {"objects": [{"name": "Person"}]}
    c.list()
    c.create("crm", d)
    c.get("crm")
    c.save_draft("crm", d)
    c.validate("crm")
    c.validate("crm", d)
    c.compare("crm", from_version=1, to="draft")
    c.compare("crm")
    c.publish("crm", expected_version=1, accept_migration=True, from_version=1)
    c.publish("crm", expected_version=0)
    c.version("crm", 2)
    c.status("crm")
    c.migration("crm", 2)
    c.run_migration("crm", 2, max_rows=10, retry_failed=True)
    c.run_migration("crm", 2)
    assert seen == [
        ("GET", "/v1/ontology/definitions", "", None),
        ("POST", "/v1/ontology/definitions", "", {"name": "crm", "definition": d}),
        ("GET", "/v1/ontology/definitions/crm", "", None),
        ("PUT", "/v1/ontology/definitions/crm/draft", "", {"definition": d}),
        ("POST", "/v1/ontology/definitions/crm/validate", "", {}),
        ("POST", "/v1/ontology/definitions/crm/validate", "", {"definition": d}),
        ("GET", "/v1/ontology/definitions/crm/compare", "from=1&to=draft", None),
        ("GET", "/v1/ontology/definitions/crm/compare", "", None),
        (
            "POST",
            "/v1/ontology/definitions/crm/publish",
            "",
            {"expected_version": 1, "accept_migration": True, "from_version": 1},
        ),
        ("POST", "/v1/ontology/definitions/crm/publish", "", {"expected_version": 0}),
        ("GET", "/v1/ontology/definitions/crm/versions/2", "", None),
        ("GET", "/v1/ontology/definitions/crm/status", "", None),
        ("GET", "/v1/ontology/definitions/crm/migrations/2", "", None),
        (
            "POST",
            "/v1/ontology/definitions/crm/migrations/2/run",
            "",
            {"max_rows": 10, "retry_failed": True},
        ),
        ("POST", "/v1/ontology/definitions/crm/migrations/2/run", "", {}),
    ]


def test_names_are_path_quoted() -> None:
    seen: list[tuple[str, str, str, Any]] = []
    _client(seen).get("a/b c")
    assert seen[0][1] == "/v1/ontology/definitions/a%2Fb%20c"


def test_a_precondition_failure_is_raised_not_returned() -> None:
    seen: list[tuple[str, str, str, Any]] = []
    c = _client(
        seen,
        412,
        {
            "code": "ontology-version-conflict",
            "status": 412,
            "title": "x",
            "detail": "d",
            "current_version": 3,
        },
    )
    with pytest.raises(Exception) as info:
        c.publish("crm", expected_version=1)
    assert "412" in str(info.value) or "ontology-version-conflict" in str(info.value)
