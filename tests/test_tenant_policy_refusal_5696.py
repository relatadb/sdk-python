"""#5696: the SDK surfaces an ACL denial and a tenant refusal, never an empty success.

The server answers a caller without read on a type, and a caller naming a tenant this
single-tenant node does not serve, with a 403 problem. The wire bodies come from the shared
fixtures ``forbidden_403_acl_denied`` and ``forbidden_403_org_not_entitled``, which the Rust
test ``http_e2e::contract_matrix::problem_fixtures_match_live_server`` replays against the
live binary. Each high-level call on the probed surfaces (query, alerts, audit log, lookups,
quarantine) must raise ``ForbiddenError`` with the server's ``code``; returning an empty
list or page would hide the refusal from the application.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from relata import RelataClient
from relata import exceptions as ex
from relata._http import HttpTransport
from relata.audit import AuditClient
from relata.governance import GovernanceClient
from relata.identity import IdentityClient
from relata.ingest_control import IngestControlClient

BASE = "http://relata.invalid"
CASES = {
    c["name"]: c
    for c in json.loads(
        (Path(__file__).resolve().parent / "fixtures" / "problem-details-fixtures.json").read_text()
    )["cases"]
}


def _answer(case_name: str, seen: list[httpx.Request]) -> httpx.MockTransport:
    wire = CASES[case_name]["wire"]

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            wire["status"], headers=wire["headers"], content=json.dumps(wire["body"])
        )

    return httpx.MockTransport(handler)


def _calls(transport: httpx.MockTransport, tenant: str | None) -> dict[str, Callable[[], Any]]:
    headers = {"X-Relata-Tenant-Id": tenant} if tenant else None
    client = RelataClient(BASE, bearer_token="t", purpose="investigation", tenant=tenant)
    client._RelataClient__sync_transport = HttpTransport(  # type: ignore[attr-defined]
        BASE, "t", 5.0, max_retries=0, transport=transport, extra_headers=headers
    )
    kw: dict[str, Any] = {"bearer_token": "t", "transport": transport, "extra_headers": headers}
    gov = GovernanceClient(BASE, purpose="investigation", tenant=tenant, **kw)
    audit = AuditClient(BASE, **kw)
    identity = IdentityClient(BASE, **kw)
    ingest = IngestControlClient(BASE, **kw)
    return {
        "query": lambda: client.query("SELECT * FROM ProbeEvt"),
        "list_alerts": gov.list_alerts,
        "audit_log": audit.log,
        "audit_count": audit.count,
        "list_lookups": identity.list_lookups,
        "invoke_lookup": lambda: identity.invoke_lookup("hosts", "10.0.0.1"),
        "list_quarantine": ingest.list_quarantine,
    }


@pytest.mark.parametrize(
    ("case_name", "tenant"),
    [("forbidden_403_acl_denied", None), ("forbidden_403_org_not_entitled", "tenant-b")],
)
def test_refusal_raises_forbidden_with_the_server_code(case_name: str, tenant: str | None) -> None:
    want = CASES[case_name]["expect"]
    seen: list[httpx.Request] = []
    for name, call in _calls(_answer(case_name, seen), tenant).items():
        with pytest.raises(ex.ForbiddenError) as caught:
            result = call()
            pytest.fail(f"{name} returned {result!r} instead of raising")
        assert caught.value.status_code == 403, name
        assert caught.value.code == want["code"], name
        assert caught.value.retryable is False, name
    # The tenant the application named is what the server refused: it was sent on every call.
    assert seen, "no request reached the transport"
    for request in seen:
        assert request.headers.get("X-Relata-Tenant-Id") == tenant, request.url
