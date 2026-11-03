"""#5733: shared workflow-runner step fixtures (Python leg).

``sdks/shared/workflow-runner-fixtures.json`` is replayed by the TypeScript and Go SDKs and
checked against the Rust ``relata-workflow-runner`` request shape. One engine response must
normalize to the same error class, status, retry delay and request count in every consumer.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from relata import RelataClient
from relata._http import HttpTransport
from relata.exceptions import (
    ConnectionError as RelataConnectionError,
)
from relata.exceptions import (
    ForbiddenError,
    NotFoundError,
    RateLimitedError,
    ServerError,
)
from relata.mcp import McpClient

FIXTURES = json.loads(
    (Path(__file__).resolve().parents[2] / "shared" / "workflow-runner-fixtures.json").read_text()
)
BASE = "http://relata.invalid"
KINDS: dict[str, type[Exception]] = {
    "not_found": NotFoundError,
    "rate_limited": RateLimitedError,
    "server_error": ServerError,
    "forbidden": ForbiddenError,
    "connection": RelataConnectionError,
}


def _handler(case: dict[str, Any], seen: list[httpx.Request]):
    wire = case["wire"]

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if wire.get("reset"):
            raise httpx.RemoteProtocolError("server disconnected without a response")
        return httpx.Response(wire["status"], json=wire["body"], headers=wire.get("headers", {}))

    return handle


def _run_step(case: dict[str, Any], seen: list[httpx.Request]) -> Any:
    transport = httpx.MockTransport(_handler(case, seen))
    if case["step"] == "job":
        job = FIXTURES["requests"]["job"]
        client = McpClient(BASE, bearer_token="t", transport=transport)
        return client.call_tool(job["body"]["name"], job["body"]["arguments"])
    query = FIXTURES["requests"]["query"]
    relata = RelataClient(BASE, purpose=query["purpose"])
    relata._RelataClient__sync_transport = HttpTransport(  # type: ignore[attr-defined]
        BASE, "t", relata._timeout, transport=transport, extra_headers=relata._extra_headers,
    )
    return relata.query(query["sql"], purpose=query["purpose"])


def test_fixture_file_is_the_documented_contract() -> None:
    assert FIXTURES["contract"] == "relata.workflow-runner-fixtures/v1"
    names = [c["name"] for c in FIXTURES["cases"]]
    assert len(names) == len(set(names)) and len(names) >= 8
    assert {c["step"] for c in FIXTURES["cases"]} == {"job", "query"}
    assert FIXTURES["requests"]["job"]["body"]["name"] == "schedule_job"


@pytest.mark.parametrize("case", FIXTURES["cases"], ids=lambda c: c["name"])
def test_engine_response_normalizes_to_the_shared_outcome(case: dict[str, Any]) -> None:
    expect = case["expect"]
    seen: list[httpx.Request] = []
    if expect["outcome"] == "ok":
        _run_step(case, seen)
    else:
        with pytest.raises(KINDS[expect["error"]]) as caught:
            _run_step(case, seen)
        error = caught.value
        assert getattr(error, "status_code", None) == expect["status"]
        got_delay = getattr(error, "retry_after", None)
        want = expect["retry_after_seconds"]
        assert (got_delay is None) if want is None else got_delay == want
    assert len(seen) == expect["requests"], "a runner step is never replayed by the transport"
    if case["step"] == "job":
        job = FIXTURES["requests"]["job"]
        assert seen[0].method == job["method"] and seen[0].url.path == job["path"]
        assert json.loads(seen[0].content) == job["body"]
