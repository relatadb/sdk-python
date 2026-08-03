"""URL path-segment / query-param injection regression tests (#3212).

The audit found ~40 helpers concatenating caller-supplied ids into request
URLs verbatim, letting ``../``, ``?``, ``#``, ``&`` rewrite the path or
smuggle query params (e.g. ``tenant.delete("../x")``,
``export_data("Person&purpose=admin")``).

The fix centralizes ``relata._http.path_segment`` (``quote(s, safe="")``) and
routes every ``f"/x/{id}"`` site through it; query params go through
``urlencode``. Each test feeds the traversal/smuggling payloads into a
representative helper and asserts the outgoing URL is the percent-encoded
form — the raw payload must never reach the wire.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from relata import RelataClient
from relata._http import AsyncHttpTransport, HttpTransport, path_segment

BASE = "http://localhost:9090"

Handler = Callable[[httpx.Request], httpx.Response]

# Path-traversal / query-smuggling payloads.
PAYLOADS = ["../x", "..%2Fx", "?admin=1", "#frag", "a&purpose=admin", "a/b", "a b"]


class _URLRecorder:
    def __init__(self) -> None:
        self.urls: list[str] = []
        self.raw_paths: list[str] = []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.urls.append(str(req.url))
        self.raw_paths.append(req.url.raw_path.decode("ascii"))
        return httpx.Response(200, json={"ok": True})


def _transport(rec: _URLRecorder) -> httpx.MockTransport:
    return httpx.MockTransport(rec)


def _assert_wire_is_encoded(raw_path: str) -> None:
    """The wire path must carry no raw query/fragment/param delimiters — a
    payload that smuggled ``?``, ``#`` or ``&`` would appear here verbatim.
    (``/`` traversal is pinned by the exact-equality assertions, since
    ``path_segment`` encodes ``/`` as ``%2F``.)"""
    assert "?" not in raw_path
    assert "#" not in raw_path
    assert "&" not in raw_path


# ---------------------------------------------------------------------------
# path_segment helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "encoded"),
    [
        ("../x", "..%2Fx"),
        ("?admin=1", "%3Fadmin%3D1"),
        ("#frag", "%23frag"),
        ("a&b", "a%26b"),
        ("a/b", "a%2Fb"),
        ("a b", "a%20b"),
        ("plain-id_1", "plain-id_1"),
    ],
)
def test_path_segment_percent_encodes(raw: str, encoded: str) -> None:
    assert path_segment(raw) == encoded
    # The encoded form can never contain a path/query/fragment delimiter.
    assert not any(c in path_segment(raw) for c in "/?#&")


# ---------------------------------------------------------------------------
# Representative helpers — tenant delete, governance disable, a2a get, export
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("payload", PAYLOADS)
def test_tenant_delete_percent_encodes(payload: str) -> None:
    from relata.tenants import TenantAdminClient

    rec = _URLRecorder()
    c = TenantAdminClient(BASE, transport=_transport(rec))
    c.delete(payload)
    assert rec.urls[0] == f"{BASE}/platform/tenants/{path_segment(payload)}"
    _assert_wire_is_encoded(rec.raw_paths[0])


@pytest.mark.parametrize("payload", PAYLOADS)
def test_governance_disable_rule_percent_encodes(payload: str) -> None:
    from relata.governance import GovernanceClient

    rec = _URLRecorder()
    c = GovernanceClient(BASE, transport=_transport(rec))
    c.disable_rule(payload)
    assert rec.urls[0] == f"{BASE}/rules/{path_segment(payload)}"
    _assert_wire_is_encoded(rec.raw_paths[0])


@pytest.mark.parametrize("payload", PAYLOADS)
def test_a2a_get_task_percent_encodes(payload: str) -> None:
    from relata.a2a import A2AClient

    rec = _URLRecorder()
    c = A2AClient(BASE, transport=_transport(rec))
    c.get_task(payload)
    assert rec.urls[0] == f"{BASE}/a2a/tasks/{path_segment(payload)}"
    _assert_wire_is_encoded(rec.raw_paths[0])


@pytest.mark.parametrize("payload", PAYLOADS)
def test_export_data_routes_type_through_query_params(payload: str) -> None:
    from urllib.parse import parse_qs, urlsplit

    rec = _URLRecorder()
    client = RelataClient(BASE, purpose="analytics")
    client._RelataClient__sync_transport = HttpTransport(  # type: ignore[attr-defined]
        BASE, None, 30.0, transport=_transport(rec), extra_headers=None
    )
    client.export_data(payload)
    parts = urlsplit(rec.urls[0])
    assert parts.path == "/export"
    # The payload round-trips through urlencode as ONE `type` value — it can
    # never inject a sibling param (e.g. `&purpose=admin`) or rewrite the path.
    params = parse_qs(parts.query)
    assert params["type"] == [payload]
    assert set(params) == {"type", "format", "purpose"}
    assert params["purpose"] == ["export"]


@pytest.mark.parametrize("payload", PAYLOADS)
def test_session_commit_percent_encodes(payload: str) -> None:
    rec = _URLRecorder()
    client = RelataClient(BASE, purpose="analytics")
    client._RelataClient__sync_transport = HttpTransport(  # type: ignore[attr-defined]
        BASE, None, 30.0, transport=_transport(rec), extra_headers=None
    )
    client.session_commit(payload)
    assert rec.urls[0] == f"{BASE}/session/{path_segment(payload)}/commit"
    _assert_wire_is_encoded(rec.raw_paths[0])


@pytest.mark.parametrize("payload", PAYLOADS)
def test_backup_restore_status_percent_encodes(payload: str) -> None:
    from relata.backup import BackupClient

    rec = _URLRecorder()
    c = BackupClient(BASE, transport=_transport(rec))
    c.restore_status(payload)
    assert rec.urls[0] == f"{BASE}/admin/restore/{path_segment(payload)}"
    _assert_wire_is_encoded(rec.raw_paths[0])


@pytest.mark.parametrize("payload", PAYLOADS)
def test_token_check_percent_encodes(payload: str) -> None:
    from relata.tokens import TokenClient

    rec = _URLRecorder()
    c = TokenClient(BASE, transport=_transport(rec))
    c.check(payload)
    assert rec.urls[0] == f"{BASE}/tokens/{path_segment(payload)}"
    _assert_wire_is_encoded(rec.raw_paths[0])


@pytest.mark.parametrize("payload", PAYLOADS)
def test_memory_forget_percent_encodes(payload: str) -> None:
    from relata.memory import Memory

    rec = _URLRecorder()
    mem = Memory(BASE, purpose="agent-notes", transport=_transport(rec))
    mem.forget(payload)
    # safe="" encoding — the embedded '/' payload can never span segments.
    assert f"/memory/forget/{path_segment(payload)}?" in rec.urls[0]
    _assert_wire_is_encoded(rec.raw_paths[0].split("?")[0])


# ---------------------------------------------------------------------------
# Async mirrors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("payload", PAYLOADS)
async def test_async_tenant_delete_percent_encodes(payload: str) -> None:
    from relata.tenants import AsyncTenantAdminClient

    rec = _URLRecorder()
    c = AsyncTenantAdminClient(BASE, transport=_transport(rec))
    await c.delete(payload)
    assert rec.urls[0] == f"{BASE}/platform/tenants/{path_segment(payload)}"
    _assert_wire_is_encoded(rec.raw_paths[0])


@pytest.mark.parametrize("payload", PAYLOADS)
async def test_async_a2a_load_checkpoint_percent_encodes(payload: str) -> None:
    from relata.a2a import AsyncA2AClient

    rec = _URLRecorder()
    c = AsyncA2AClient(BASE, transport=_transport(rec))
    await c.load_checkpoint(payload, payload)
    seg = path_segment(payload)
    assert rec.urls[0] == f"{BASE}/a2a/checkpoints/{seg}/{seg}"
    _assert_wire_is_encoded(rec.raw_paths[0])


def _async_client(rec: _URLRecorder) -> RelataClient:
    client = RelataClient(BASE, purpose="analytics")
    client._RelataClient__async_transport = AsyncHttpTransport(  # type: ignore[attr-defined]
        BASE, None, 30.0, transport=_transport(rec), extra_headers=None
    )
    return client


@pytest.mark.parametrize("payload", PAYLOADS)
async def test_async_schema_branch_delete_percent_encodes(payload: str) -> None:
    rec = _URLRecorder()
    client = _async_client(rec)
    await client.adelete_schema_branch(payload)
    assert rec.urls[0] == f"{BASE}/schema/branches/{path_segment(payload)}"
    _assert_wire_is_encoded(rec.raw_paths[0])
