"""#5666: cross-SDK shared queue-acknowledgement and task-status fixtures (Python leg).

``sdks/shared/task-receipt-fixtures.json`` holds the wire bodies of ``POST /v1/ingest/bulk``
(queue acknowledgement) and ``GET /v1/ingest/tasks/{id}`` (task status). The Rust test
``http_e2e::contract_matrix::task_receipt_fixtures_match_live_server`` replays the ``live``
cases against a real server. TypeScript and Go run the same file, so one wire body decodes to the
same typed fields, preserved extensions and un-coerced status in all three SDKs.
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
    (Path(__file__).resolve().parent / "fixtures" / "task-receipt-fixtures.json").read_text()
)
BASE = "http://relata.invalid"


def _client(case: dict[str, Any]) -> RelataClient:
    wire = case["wire"]
    transport = httpx.MockTransport(lambda request: httpx.Response(wire["status"], json=wire["body"]))
    client = RelataClient(BASE, purpose="analytics")
    client._RelataClient__sync_transport = HttpTransport(  # type: ignore[attr-defined]
        BASE, client._bearer_token, client._timeout, transport=transport,
        extra_headers=client._extra_headers,
    )
    return client


def test_fixture_file_is_the_documented_contract() -> None:
    assert FIXTURES["contract"] == "relata.task-receipt-fixtures/v1"
    names = [c["name"] for c in FIXTURES["cases"]]
    assert len(names) == len(set(names)) and len(names) >= 8
    assert {c["kind"] for c in FIXTURES["cases"]} == {"queue_ack", "task_status"}


@pytest.mark.parametrize("case", FIXTURES["cases"], ids=lambda c: c["name"])
def test_wire_body_decodes_to_the_shared_expectation(case: dict[str, Any]) -> None:
    exp = case["expect"]
    client = _client(case)
    if case["kind"] == "queue_ack":
        got = client.ingest_document([{"_type": "T", "_pk": "1"}])
        assert got.task_id == exp["task_id"]
        assert got.rows_queued == exp["rows_queued"]
        assert got.rows_rejected == exp["rows_rejected"]
        assert got.errors == exp["errors"]
        assert got.queue_depth == exp["queue_depth"]
    else:
        got = client.ingest_document_status(exp["task_id"])
        assert got.task_id == exp["task_id"]
        # The status string is surfaced verbatim: an unknown value is never coerced.
        assert got.status == exp["status"]
        assert got.rows_total == exp["rows_total"]
        assert got.rows_written == exp["rows_written"]
        assert got.rows_dropped == exp["rows_dropped"]
        assert got.drop_reasons == exp["drop_reasons"]
        assert got.outcome_uncertain == exp["outcome_uncertain"]
        assert got.code == exp["code"]
        assert got.derived_error == exp["derived_error"]
        assert got.retryable == exp["retryable"]
        token = exp["write_token"]
        assert (got.write_token if token is None else dict(got.write_token)) == token
    assert sorted((got.model_extra or {}).keys()) == exp["extension_keys"]
    client.close()
