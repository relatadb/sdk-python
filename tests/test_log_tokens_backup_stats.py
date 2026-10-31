"""Tests for the tenant self-service token client, backup (#88),
and the partner-exact stats verification (#86)."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx

BASE = "http://localhost:9090"

Handler = Callable[[httpx.Request], httpx.Response]


def _wrap(handler: Handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


# ---------------------------------------------------------------------------
# Tenant self-service tokens (/v1/auth/tokens/self)
# ---------------------------------------------------------------------------


def test_token_list_self() -> None:
    from relata.tokens import TokenClient

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "GET"
        assert req.url.path == "/v1/auth/tokens/self"
        return httpx.Response(200, json={"tokens": [{"id": "tok-1"}]})

    c = TokenClient(BASE, transport=_wrap(handler))
    assert c.list_self()["tokens"][0]["id"] == "tok-1"


def test_token_self_audit() -> None:
    from relata.tokens import TokenClient

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "GET"
        assert req.url.path == "/v1/auth/tokens/self/audit"
        return httpx.Response(200, json={"events": []})

    c = TokenClient(BASE, transport=_wrap(handler))
    assert c.self_audit() == {"events": []}


def test_token_rotate_self() -> None:
    from relata.tokens import TokenClient

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "POST"
        assert req.url.path == "/v1/auth/tokens/self/rotate"
        return httpx.Response(200, json={"secret": "new-secret"})

    c = TokenClient(BASE, transport=_wrap(handler))
    assert c.rotate_self()["secret"] == "new-secret"


def test_token_revoke_self() -> None:
    from relata.tokens import TokenClient

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "DELETE"
        assert req.url.path == "/v1/auth/tokens/self/tok-1"
        return httpx.Response(200, json={"revoked": True})

    c = TokenClient(BASE, transport=_wrap(handler))
    assert c.revoke_self("tok-1") == {"revoked": True}


async def test_async_token_self_service_mirrors_sync() -> None:
    from relata.tokens import AsyncTokenClient

    seen: list[tuple[str, str]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.method, req.url.path))
        return httpx.Response(200, json={})

    c = AsyncTokenClient(BASE, transport=httpx.MockTransport(handler))
    await c.list_self()
    await c.self_audit()
    await c.rotate_self()
    await c.revoke_self("tok-1")
    assert seen == [
        ("GET", "/v1/auth/tokens/self"),
        ("GET", "/v1/auth/tokens/self/audit"),
        ("POST", "/v1/auth/tokens/self/rotate"),
        ("DELETE", "/v1/auth/tokens/self/tok-1"),
    ]


# ---------------------------------------------------------------------------
# #88 — Backup / restore
# ---------------------------------------------------------------------------


def test_backup_create_returns_ref() -> None:
    from relata.backup import BackupClient, BackupRef

    def handler(req: httpx.Request) -> httpx.Response:  # noqa: E501
        assert req.url.path == "/admin/backup"
        body = json.loads(req.content)
        assert body == {"kind": "full"}
        return httpx.Response(
            201,
            json={
                "snapshot_id": "snap-1",
                "kind": "full",
                "written_at_ns": 1760000000,
                "size_bytes": 1048576,
                "sha256": "abc123",
                "row_count": 50000,
                "path": "/data/backups/snap-1.json",
            },
        )

    c = BackupClient(BASE, transport=_wrap(handler))
    ref = c.create()
    assert isinstance(ref, BackupRef)
    assert ref.snapshot_id == "snap-1"
    assert ref.size_bytes == 1048576
    assert ref.row_count == 50000


def test_backup_list() -> None:
    from relata.backup import BackupClient

    def handler(req: httpx.Request) -> httpx.Response:  # noqa: E501
        return httpx.Response(
            200,
            json={
                "backups": [
                    {"snapshot_id": "snap-1", "kind": "full", "size_bytes": 100},
                    {"snapshot_id": "snap-2", "kind": "incremental", "size_bytes": 50},
                ],
                "count": 2,
            },
        )

    c = BackupClient(BASE, transport=_wrap(handler))
    backups = c.list()
    assert len(backups) == 2
    assert backups[0].snapshot_id == "snap-1"
    assert backups[1].kind == "incremental"


def test_backup_restore_returns_id() -> None:
    from relata.backup import BackupClient

    def handler(req: httpx.Request) -> httpx.Response:  # noqa: E501
        assert req.url.path == "/admin/restore"
        body = json.loads(req.content)
        assert body["snapshot_id"] == "snap-1"
        assert body.get("point_in_time_ns") == 1760000500
        return httpx.Response(202, json={"restore_id": "restore-1", "status": "running"})

    c = BackupClient(BASE, transport=_wrap(handler))
    restore_id = c.restore("snap-1", point_in_time_ns=1760000500)
    assert restore_id == "restore-1"


def test_backup_restore_status_polls() -> None:
    from relata.backup import BackupClient, RestoreStatus

    def handler(req: httpx.Request) -> httpx.Response:  # noqa: E501
        return httpx.Response(
            200,
            json={
                "restore_id": "restore-1",
                "snapshot_id": "snap-1",
                "status": "completed",
                "rto_actual_secs": 12.5,
                "error": None,
            },
        )

    c = BackupClient(BASE, transport=_wrap(handler))
    status = c.restore_status("restore-1")
    assert isinstance(status, RestoreStatus)
    assert status.is_completed is True
    assert status.rto_actual_secs == 12.5
    assert status.error is None


def test_backup_incremental_create() -> None:
    from relata.backup import BackupClient

    seen: list[dict[str, object]] = []

    def handler(req: httpx.Request) -> httpx.Response:  # noqa: E501
        body = json.loads(req.content)
        seen.append(body)
        return httpx.Response(
            201,
            json={
                "snapshot_id": "snap-incr",
                "kind": "incremental",
                "size_bytes": 512,
                "sha256": "def",
                "row_count": 100,
                "written_at_ns": 1760000100,
                "path": "/d/incr.json",
            },
        )

    c = BackupClient(BASE, transport=_wrap(handler))
    ref = c.create(kind="incremental")
    assert ref.kind == "incremental"
    assert seen[0] == {"kind": "incremental"}


# ---------------------------------------------------------------------------
# #86 — Stats now populate log_leaves + tokens (server primitives landed)
# ---------------------------------------------------------------------------


def test_stats_populates_log_leaves_and_tokens() -> None:
    """Now that #83 + #84 server primitives shipped, the /v1/system/stats
    endpoint should emit non-None log_leaves and tokens counts."""
    from relata import RelataClient
    from relata._http import HttpTransport

    def handler(req: httpx.Request) -> httpx.Response:  # noqa: E501
        return httpx.Response(
            200,
            json={
                "records": 100,
                "states": 50000,
                "snapshot_rows": 5000,
                "log_leaves": 999,
                "tokens": 42,
            },
        )

    client = RelataClient(BASE, purpose="analytics")
    client._RelataClient__sync_transport = HttpTransport(  # type: ignore[attr-defined]
        BASE, None, 5.0, transport=_wrap(handler),
    )
    stats = client.stats()
    assert stats.log_leaves == 999
    assert stats.tokens == 42
    assert stats.records == 100
    assert stats.states == 50000
