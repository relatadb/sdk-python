"""#5766/#5767: conditional object writes and atomic commit bundles."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from relata.exceptions import PreconditionFailedError
from relata.objects import AsyncObjectClient, ObjectClient

BASE = "http://localhost:9090"
Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: Handler) -> ObjectClient:
    return ObjectClient(BASE, purpose="audit", transport=httpx.MockTransport(handler))


def test_upsert_if_absent_puts_with_if_none_match() -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(201, json={"type": "Doc", "id": "a", "version": 7})

    out = _client(handler).upsert("Doc", "a", {"n": "x"}, if_absent=True)
    assert out["version"] == 7
    (req,) = seen
    assert req.method == "PUT"
    assert req.url.path == "/v1/objects/Doc/a"
    assert req.url.params["purpose"] == "audit"
    assert req.headers["if-none-match"] == "*"
    assert "if-match" not in req.headers
    assert json.loads(req.content) == {"n": "x"}


def test_upsert_expected_version_sends_if_match_and_raises_on_412() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.headers["if-match"] == '"41"'
        return httpx.Response(
            412,
            headers={"ETag": '"42"'},
            json={
                "type": "https://relatadb.dev/errors/precondition-failed",
                "title": "Precondition Failed",
                "status": 412,
                "detail": "'Doc/a' is at version 42, expected 41",
                "code": "precondition-failed",
                "current_version": 42,
                "failed_operation": 0,
            },
        )

    with pytest.raises(PreconditionFailedError) as info:
        _client(handler).upsert("Doc", "a", {"n": "x"}, expected_version=41)
    assert info.value.status_code == 412
    assert info.value.current_version == 42
    assert info.value.failed_operation == 0
    assert info.value.code == "precondition-failed"


def test_conditional_flags_are_mutually_exclusive() -> None:
    c = _client(lambda req: httpx.Response(500))
    with pytest.raises(ValueError):
        c.upsert("Doc", "a", {}, expected_version=1, if_absent=True)


def test_unconditional_upsert_still_uses_ingest() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "POST" and req.url.path == "/v1/ingest"
        return httpx.Response(200, json={"rows_queued": 1})

    assert _client(handler).upsert("Doc", "a", {"n": "x"})["rows_queued"] == 1


def test_delete_with_expected_version_is_a_conditional_delete() -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json={"deleted": True, "version": 9, "commitId": "c"})

    _client(handler).delete("Doc", "a", expected_version=8)
    (req,) = seen
    assert req.method == "DELETE" and req.url.path == "/v1/objects/Doc/a"
    assert req.headers["if-match"] == '"8"'


def test_get_versioned_reads_the_object_route_and_maps_404() -> None:
    def ok(req: httpx.Request) -> httpx.Response:
        assert req.method == "GET" and req.url.path == "/v1/objects/Doc/a"
        return httpx.Response(200, json={"id": "a", "_version": 5})

    assert _client(ok).get_versioned("Doc", "a") == {"id": "a", "_version": 5}
    assert _client(lambda req: httpx.Response(404, json={"error": "row not found"})).get_versioned(
        "Doc", "gone"
    ) is None


def test_commit_posts_the_bundle_and_translates_expected_version() -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(
            200,
            json={"commitId": "c1", "operations": [{"op": "update", "type": "Doc", "id": "a", "version": 3}], "replayed": False},
        )

    out = _client(handler).commit(
        [{"op": "update", "type": "Doc", "id": "a", "expected_version": 2, "data": {"n": "y"}}],
        idempotency_key="k1",
    )
    assert out["commitId"] == "c1"
    (req,) = seen
    assert req.method == "POST" and req.url.path == "/v1/commit"
    body = json.loads(req.content)
    assert body["idempotencyKey"] == "k1"
    assert body["purpose"] == "audit"
    assert body["operations"] == [
        {"op": "update", "type": "Doc", "id": "a", "expectedVersion": 2, "data": {"n": "y"}}
    ]


def test_commit_failure_reports_the_failing_operation() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            412,
            json={"status": 412, "detail": "missing", "code": "precondition-failed", "current_version": None, "failed_operation": 2},
        )

    with pytest.raises(PreconditionFailedError) as info:
        _client(handler).commit([{"op": "delete", "type": "Doc", "id": "z"}])
    assert info.value.failed_operation == 2
    assert info.value.current_version is None


@pytest.mark.asyncio
async def test_async_conditional_put_and_commit() -> None:
    seen: list[httpx.Request] = []

    async def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        if req.url.path == "/v1/commit":
            return httpx.Response(200, json={"commitId": "c", "operations": [], "replayed": True})
        return httpx.Response(200, json={"version": 2})

    c = AsyncObjectClient(BASE, purpose="audit", transport=httpx.MockTransport(handler))
    out = await c.upsert("Doc", "a", {"n": 1}, expected_version=1)
    assert out["version"] == 2
    assert seen[0].method == "PUT" and seen[0].headers["if-match"] == '"1"'
    done = await c.commit([{"op": "create", "type": "Doc", "id": "b", "data": {}}], idempotency_key="k")
    assert done["replayed"] is True
    await c.delete("Doc", "a", expected_version=2)
    assert seen[-1].method == "DELETE" and seen[-1].headers["if-match"] == '"2"'
