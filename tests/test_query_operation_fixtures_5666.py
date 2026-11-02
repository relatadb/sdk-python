"""#5666: cross-SDK shared query-operation fixtures (Python leg).

``sdks/shared/query-operation-fixtures.json`` pins, per case, the exact ``POST /v1/query`` request
an SDK call must produce and the normalized result its wire response must decode to. TypeScript
and Go run the same file; the Rust test ``contract_matrix::query_operation_fixtures_match_live_server``
keeps the wire bodies faithful to the real server.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from relata import RelataClient
from relata._http import HttpTransport

FIXTURES = json.loads(
    (Path(__file__).resolve().parents[2] / "shared" / "query-operation-fixtures.json").read_text()
)
BASE = "http://relata.invalid"


def test_fixture_file_is_the_documented_contract() -> None:
    assert FIXTURES["contract"] == "relata.query-operation-fixtures/v1"
    names = [c["name"] for c in FIXTURES["cases"]]
    assert len(names) == len(set(names)) and len(names) >= 4


@pytest.mark.parametrize("case", FIXTURES["cases"], ids=lambda c: c["name"])
def test_query_request_and_result_match_the_shared_expectation(case: dict[str, Any]) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(case["wire"]["status"], json=case["wire"]["body"])

    client = RelataClient(BASE, purpose="analytics")
    client._RelataClient__sync_transport = HttpTransport(  # type: ignore[attr-defined]
        BASE, client._bearer_token, client._timeout, transport=httpx.MockTransport(handler),
        extra_headers=client._extra_headers,
    )
    call = case["call"]
    got = client.query(
        call["sql"],
        purpose=call.get("purpose"),
        query_vector=call.get("query_vector"),
        query_vector_model=call.get("query_vector_model"),
    )
    client.close()

    assert len(seen) == 1
    req = seen[0]
    assert req.method == case["request"]["method"]
    assert req.url.path == case["request"]["path"]
    assert json.loads(req.content) == case["request"]["body"]

    exp = case["expect"]
    assert got.rows == exp["rows"]
    assert got.columns == exp["columns"]
    assert got.row_count == exp["row_count"]
    assert got.query_id == exp["query_id"]
    assert got.has_next_cursor_metadata is exp["next_cursor_present"]
    assert got.next_cursor == exp["next_cursor"]
    assert got.truncated is exp["truncated"]
