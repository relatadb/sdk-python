"""Tests for the final 4 SDK modules: log (#83), tokens (#84), backup (#88),
and the partner-exact stats verification (#86)."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable

import httpx

BASE = "http://localhost:9090"

Handler = Callable[[httpx.Request], httpx.Response]


def _wrap(handler: Handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


# ---------------------------------------------------------------------------
# #83 — Ordered integrity log
# ---------------------------------------------------------------------------


def test_log_append_returns_index() -> None:
    from relata.log import LogClient

    def handler(req: httpx.Request) -> httpx.Response:  # noqa: E501
        assert req.url.path == "/log/leaf"
        assert req.method == "POST"
        body = json.loads(req.content)
        decoded = base64.b64decode(body["data_b64"])
        assert decoded == b"hello-log"
        return httpx.Response(201, json={"index": 42, "commit_ns": 1760000000})

    c = LogClient(BASE, transport=_wrap(handler))
    idx, commit_ns = c.append(b"hello-log")
    assert idx == 42
    assert commit_ns == 1760000000


def test_log_head() -> None:
    from relata.log import LogClient

    def handler(req: httpx.Request) -> httpx.Response:  # noqa: E501
        assert req.url.path == "/log/head"
        return httpx.Response(200, json={"head": 999})

    c = LogClient(BASE, transport=_wrap(handler))
    assert c.head() == 999


def test_log_load_leaves_decodes_base64() -> None:
    from relata.log import LogClient

    def handler(req: httpx.Request) -> httpx.Response:  # noqa: E501
        assert "since=5" in str(req.url)
        assert "limit=10" in str(req.url)
        return httpx.Response(
            200,
            json={
                "leaves": [
                    {"index": 5, "data_b64": base64.b64encode(b"aaa").decode(), "commit_ns": 1},
                    {"index": 6, "data_b64": base64.b64encode(b"bbb").decode(), "commit_ns": 2},
                ],
                "count": 2,
            },
        )

    c = LogClient(BASE, transport=_wrap(handler))
    leaves = c.load_leaves(since=5, limit=10)
    assert len(leaves) == 2
    assert leaves[0].index == 5
    assert leaves[0].data == b"aaa"
    assert leaves[1].data == b"bbb"


# ---------------------------------------------------------------------------
# #84 — Dedup / uniqueness tokens
# ---------------------------------------------------------------------------


def test_token_remember_new_returns_true() -> None:
    from relata.tokens import TokenClient

    def handler(req: httpx.Request) -> httpx.Response:  # noqa: E501
        assert req.url.path == "/tokens"
        body = json.loads(req.content)
        assert body == {"id": "replay-defence-1"}
        return httpx.Response(201, json={"newly_inserted": True, "token_id": "tok-1"})

    c = TokenClient(BASE, transport=_wrap(handler))
    assert c.remember("replay-defence-1") is True


def test_token_remember_duplicate_returns_false() -> None:
    from relata.tokens import TokenClient

    def handler(req: httpx.Request) -> httpx.Response:  # noqa: E501
        return httpx.Response(200, json={"newly_inserted": False, "token_id": "tok-1"})

    c = TokenClient(BASE, transport=_wrap(handler))
    assert c.remember("replay-defence-1") is False


def test_token_check_present() -> None:
    from relata.tokens import TokenClient

    def handler(req: httpx.Request) -> httpx.Response:  # noqa: E501
        assert req.url.path == "/tokens/tok-1"
        return httpx.Response(
            200,
            json={"present": True, "inserted_at_ns": 1760000000, "ttl_secs": 86400},
        )

    c = TokenClient(BASE, transport=_wrap(handler))
    result = c.check("tok-1")
    assert result["present"] is True
    assert result["inserted_at_ns"] == 1760000000


def test_token_check_absent() -> None:
    from relata.tokens import TokenClient

    def handler(req: httpx.Request) -> httpx.Response:  # noqa: E501
        return httpx.Response(200, json={"present": False})

    c = TokenClient(BASE, transport=_wrap(handler))
    result = c.check("missing-tok")
    assert result["present"] is False


def test_token_revoke() -> None:
    from relata.tokens import TokenClient

    def handler(req: httpx.Request) -> httpx.Response:  # noqa: E501
        assert req.method == "DELETE"
        assert req.url.path == "/tokens/tok-1"
        return httpx.Response(200, json={"revoked": True})

    c = TokenClient(BASE, transport=_wrap(handler))
    assert c.revoke("tok-1") is True


def test_token_stats() -> None:
    from relata.tokens import TokenClient, TokenStats

    def handler(req: httpx.Request) -> httpx.Response:  # noqa: E501
        assert req.url.path == "/tokens/stats"
        return httpx.Response(
            200,
            json={"count": 42, "oldest_ns": 1759000000, "eviction_policy": "min_age_secs=86400"},
        )

    c = TokenClient(BASE, transport=_wrap(handler))
    stats = c.stats()
    assert isinstance(stats, TokenStats)
    assert stats.count == 42
    assert stats.oldest_ns == 1759000000
    assert "86400" in stats.eviction_policy


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
            json={"snapshot_id": "snap-incr", "kind": "incremental", "size_bytes": 512,
                  "sha256": "def", "row_count": 100, "written_at_ns": 1760000100, "path": "/d/incr.json"},
        )

    c = BackupClient(BASE, transport=_wrap(handler))
    ref = c.create(kind="incremental")
    assert ref.kind == "incremental"
    assert seen[0] == {"kind": "incremental"}


# ---------------------------------------------------------------------------
# #86 — Stats now populate log_leaves + tokens (server primitives landed)
# ---------------------------------------------------------------------------


def test_stats_populates_log_leaves_and_tokens() -> None:
    """Now that #83 + #84 server primitives shipped, the /debug/stats
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
