"""#5666: cross-SDK shared RFC 9457 fixtures (Python leg).

``sdks/shared/problem-details-fixtures.json`` holds wire-exact error responses captured from a
live ``relata serve`` (the Rust test
``http_e2e::contract_matrix::problem_fixtures_match_live_server`` replays every fixture's
``live`` request against the real binary and fails on drift). The
TypeScript and Go SDKs run the same file, so one wire response must map to the same error kind,
``code``, ``type`` URL, ``retryable`` flag, request id and retry delay in all three.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from relata import exceptions as ex
from relata._http import HttpTransport

FIXTURES = json.loads(
    (Path(__file__).resolve().parent / "fixtures" / "problem-details-fixtures.json").read_text()
)

# fixture kind -> exact exception class for this SDK. 400 non-purpose and the unmapped statuses
# (405/413/415 ...) are the base class here.
KIND_TO_CLASS = {
    "purpose": ex.PurposeError,
    "auth": ex.AuthError,
    "forbidden": ex.ForbiddenError,
    "not_found": ex.NotFoundError,
    "conflict": ex.ConflictError,
    "precondition_failed": ex.PreconditionFailedError,
    "validation": ex.ValidationError,
    "rate_limited": ex.RateLimitedError,
    "server": ex.ServerError,
    "bad_request": ex.RelataError,
    "other": ex.RelataError,
}


def _raise_for(case: dict[str, Any]) -> ex.RelataError:
    wire = case["wire"]
    body = wire["body"]
    content = body if isinstance(body, str) else json.dumps(body)
    transport = HttpTransport(
        "http://relata.invalid",
        None,
        1,
        max_retries=0,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(wire["status"], headers=wire["headers"], content=content)
        ),
    )
    with pytest.raises(ex.RelataError) as caught:
        transport.get("/probe")
    return caught.value


def test_fixture_file_is_the_documented_contract() -> None:
    assert FIXTURES["contract"] == "relata.problem-details-fixtures/v1"
    assert len(FIXTURES["cases"]) >= 10
    names = [c["name"] for c in FIXTURES["cases"]]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("case", FIXTURES["cases"], ids=lambda c: c["name"])
def test_wire_response_maps_to_the_shared_expectation(case: dict[str, Any]) -> None:
    exp = case["expect"]
    err = _raise_for(case)
    want = KIND_TO_CLASS[exp["kind"]]
    assert type(err) is want, f"{case['name']}: {type(err).__name__} != {want.__name__}"
    assert err.status_code == exp["status"]
    assert err.code == exp["code"]
    assert err.type_url == exp["type"]
    assert err.retryable is exp["retryable"]
    assert err.request_id == exp["request_id"]
    if exp["retry_after_seconds"] is None:
        assert err.retry_after is None
    else:
        assert err.retry_after == pytest.approx(exp["retry_after_seconds"])
    for needle in exp.get("message_contains", []):
        assert needle in str(err), f"{case['name']}: {needle!r} missing from {str(err)!r}"
