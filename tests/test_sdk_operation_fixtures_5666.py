"""#5666: cross-SDK shared operation fixtures for objects, search, memory and assertions (Python leg).

``sdks/shared/sdk-operation-fixtures.json`` pins, per case, the exact HTTP request an SDK call must
put on the wire (method, path, query, headers, body) and the normalized result its wire response
must decode to. TypeScript and Go run the same file; the Rust test
``contract_matrix::sdk_operation_fixtures_match_live_server`` keeps the wire bodies and requests
faithful to the real server.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl

import httpx
import pytest

from relata import RelataClient
from relata.exceptions import RelataError
from relata._http import HttpTransport
from relata.assertions import AssertionsClient
from relata.memory import Memory
from relata.objects import ObjectClient

FIXTURES = json.loads(
    (Path(__file__).resolve().parents[2] / "shared" / "sdk-operation-fixtures.json").read_text()
)
BASE = "http://relata.invalid"
SDK = "python"
CASES = [c for c in FIXTURES["cases"] if SDK in c.get("sdks", FIXTURES["sdks"])]


def _search_view(resp: Any) -> dict[str, Any]:
    return {
        "hit_ids": [h.id for h in resp.hits],
        "hit_fields": [h.fields for h in resp.hits],
        "hit_scores": [h.score for h in resp.hits],
        "hit_highlight_spans": [h.highlight_spans for h in resp.hits],
        "total": resp.total,
        "estimated_total_hits": resp.estimated_total_hits,
    }


def _dispatch(op: str, a: dict[str, Any], handler: Any) -> Any:
    transport = httpx.MockTransport(handler)
    fam = op.split(".")[0]
    if fam == "objects":
        o = ObjectClient(BASE, bearer_token="t", purpose="analytics", transport=transport)
        if op == "objects.put":
            return o.upsert(
                a["object_type"], a["object_id"], a["fields"],
                expected_version=a.get("expected_version"), if_absent=a.get("if_absent", False),
            )
        if op == "objects.upsert_ingest":
            return o.upsert(a["object_type"], a["object_id"], a["fields"])
        if op == "objects.get":
            return o.get_versioned(a["object_type"], a["object_id"])
        if op == "objects.delete":
            return o.delete(a["object_type"], a["object_id"], expected_version=a.get("expected_version"))
    if fam == "search":
        client = RelataClient(BASE, purpose="analytics")
        client._RelataClient__sync_transport = HttpTransport(  # type: ignore[attr-defined]
            BASE, client._bearer_token, client._timeout, transport=transport,
            extra_headers=client._extra_headers,
        )
        try:
            return _search_view(client.search(
                a["query"], a["type"], limit=a.get("limit"), highlight=a.get("highlight", False),
            ))
        finally:
            client.close()
    if fam == "memory":
        m = Memory(BASE, purpose="analytics", bearer_token="t", transport=transport)
        if op == "memory.add":
            return m.add(
                a["content"], confidence=a.get("confidence", 1.0),
                memory_class=a.get("memory_class", "semantic"),
            )
        if op == "memory.search":
            return m.search(a["query"], top_k=a.get("top_k", 5))
        if op == "memory.get":
            return m.get(a["memory_id"])
    if fam == "assertions":
        s = AssertionsClient(BASE, bearer_token="t", transport=transport)
        if op == "assertions.record":
            return s.record(a["assertion"])
        if op == "assertions.get":
            return s.get(a["id"])
        if op == "assertions.list":
            return s.list(
                subject_type=a.get("subject_type"), subject_id=a.get("subject_id"),
                predicate=a.get("predicate"), state=a.get("state"), limit=a.get("limit"),
            )
        if op == "assertions.confirm":
            return s.confirm(a["id"], a["reason"])
        if op == "assertions.revoke":
            return s.revoke(a["id"], a["reason"])
    raise AssertionError(f"fixture op {op} has no Python dispatch")


def _request_for(case: dict[str, Any]) -> dict[str, Any]:
    return {**case["request"], **case.get("request_override", {}).get(SDK, {})}


def test_fixture_file_is_the_documented_contract() -> None:
    assert FIXTURES["contract"] == "relata.sdk-operation-fixtures/v1"
    names = [c["name"] for c in FIXTURES["cases"]]
    assert len(names) == len(set(names))
    assert {c["family"] for c in FIXTURES["cases"]} == {"objects", "search", "memory", "assertions"}


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_request_and_result_match_the_shared_expectation(case: dict[str, Any]) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(case["wire"]["status"], json=case["wire"]["body"])

    exp = case["expect"]
    exp = {**exp, **case.get("expect_override", {}).get(SDK, {})}
    got: Any = None
    if "error" in exp:
        with pytest.raises(RelataError) as caught:
            _dispatch(case["call"]["op"], case["call"]["args"], handler)
        err = caught.value
        assert err.status_code == exp["error"]["status"]
        assert err.code == exp["error"]["code"]
        if "current_version" in exp["error"]:
            assert err.current_version == exp["error"]["current_version"]
    else:
        got = _dispatch(case["call"]["op"], case["call"]["args"], handler)

    assert len(seen) == 1
    req, want = seen[0], _request_for(case)
    assert req.method == want["method"]
    assert req.url.path == want["path"]
    assert dict(parse_qsl(req.url.query.decode())) == want.get("query", {})
    for name, value in want.get("headers", {}).items():
        assert req.headers.get(name) == value, name
    if "ndjson" in want:
        lines = req.content.decode().strip("\n").split("\n")
        assert [json.loads(line) for line in lines] == want["ndjson"]
    elif "body" in want:
        assert json.loads(req.content) == want["body"]
    else:
        assert req.content == b""

    if "error" not in exp:
        assert got == exp["result"]
