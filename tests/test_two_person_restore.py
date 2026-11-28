"""Two-person restore: the refusal carries a confirmation id the SDK exposes,
and restore() can send it back (with force) to finish the operation."""

from __future__ import annotations

import json

import httpx
import pytest

BASE = "http://localhost:9090"

TWO_PERSON_BODY = {
    "type": "about:blank",
    "title": "Forbidden",
    "status": 403,
    "code": "TWO_PERSON_REQUIRED",
    "detail": "this operation is destructive and requires a second, distinct admin credential",
    "confirmation_id": "2P-01a21e15-8481-7f63-8906-a40bbeba95b6",
    "expires_at_ns": 1795801956385709000,
}


def test_two_person_refusal_exposes_confirmation_id() -> None:
    from relata.backup import BackupClient
    from relata.exceptions import ForbiddenError

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403, json=TWO_PERSON_BODY, headers={"content-type": "application/problem+json"}
        )

    c = BackupClient(BASE, transport=httpx.MockTransport(handler))
    with pytest.raises(ForbiddenError) as exc:
        c.restore("snap-1")
    assert exc.value.code == "TWO_PERSON_REQUIRED"
    assert exc.value.confirmation_id == TWO_PERSON_BODY["confirmation_id"]
    assert exc.value.confirmation_expires_at_ns == TWO_PERSON_BODY["expires_at_ns"]


def test_restore_sends_confirmation_id_and_force() -> None:
    from relata.backup import BackupClient

    seen: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(json.loads(req.content))
        return httpx.Response(202, json={"restore_id": "restore-2", "status": "running"})

    c = BackupClient(BASE, transport=httpx.MockTransport(handler))
    rid = c.restore("snap-1", force=True, confirmation_id="2P-abc")
    assert rid == "restore-2"
    assert seen == {"snapshot_id": "snap-1", "force": True, "confirmation_id": "2P-abc"}


def test_restore_omits_unset_two_person_fields() -> None:
    from relata.backup import BackupClient

    seen: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(json.loads(req.content))
        return httpx.Response(202, json={"restore_id": "restore-3"})

    BackupClient(BASE, transport=httpx.MockTransport(handler)).restore("snap-1")
    assert seen == {"snapshot_id": "snap-1"}


def test_async_restore_sends_confirmation_id() -> None:
    import asyncio

    from relata.backup import AsyncBackupClient

    seen: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(json.loads(req.content))
        return httpx.Response(202, json={"restore_id": "restore-4"})

    c = AsyncBackupClient(BASE, transport=httpx.MockTransport(handler))
    assert asyncio.run(c.restore("snap-1", confirmation_id="2P-xyz")) == "restore-4"
    assert seen == {"snapshot_id": "snap-1", "confirmation_id": "2P-xyz"}
