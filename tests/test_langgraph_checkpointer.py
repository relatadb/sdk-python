"""Tests for the LangGraph RelataCheckpointer (#1064).

Drives the checkpointer through a mocked Relata `/a2a/checkpoints/*` surface via
httpx.MockTransport and asserts put/get/list hit the right endpoints and
round-trip graph state. No live server needed.
"""

from __future__ import annotations

import json

import httpx

from relata_langgraph import RelataCheckpointer

BASE = "http://localhost:9090"


class FakeCheckpointStore:
    """A MockTransport handler emulating the server's checkpoint routes."""

    def __init__(self) -> None:
        # (thread_id, checkpoint_id) -> state
        self.store: dict[tuple[str, str], dict] = {}
        self.calls: list[tuple[str, str]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append((request.method, request.url.path))
        parts = request.url.path.strip("/").split("/")
        # /a2a/checkpoints/<thread>[/<id>]
        thread = parts[2]
        if request.method == "PUT":
            cid = parts[3]
            self.store[(thread, cid)] = json.loads(request.content)
            return httpx.Response(200, json={"saved": True, "checkpoint_id": cid})
        if request.method == "GET" and len(parts) == 4:
            cid = parts[3]
            state = self.store.get((thread, cid))
            if state is None:
                return httpx.Response(404, json={"error": "no such checkpoint"})
            return httpx.Response(200, json={"state": state})
        if request.method == "GET":  # list
            ids = [cid for (t, cid) in self.store if t == thread]
            return httpx.Response(200, json={"checkpoints": ids})
        return httpx.Response(405, json={"error": "method not allowed"})


def _checkpointer() -> tuple[RelataCheckpointer, FakeCheckpointStore]:
    fake = FakeCheckpointStore()
    cp = RelataCheckpointer(BASE, "tok", transport=httpx.MockTransport(fake))
    return cp, fake


def test_put_get_round_trip() -> None:
    cp, fake = _checkpointer()
    cp.put("t1", 0, {"input": "hello"})
    cp.put("t1", 1, {"input": "hello", "out": "world"})
    assert ("PUT", "/a2a/checkpoints/t1/0") in fake.calls
    assert cp.get("t1", 0) == {"input": "hello"}
    assert cp.get("t1", 1) == {"input": "hello", "out": "world"}


def test_get_missing_returns_none() -> None:
    cp, _ = _checkpointer()
    assert cp.get("t1", 99) is None


def test_list_orders_by_step_ascending() -> None:
    cp, _ = _checkpointer()
    # Insert out of order; list must return step-ascending.
    cp.put("t2", 2, {"s": 2})
    cp.put("t2", 0, {"s": 0})
    cp.put("t2", 1, {"s": 1})
    listed = cp.list("t2")
    assert [r["step"] for r in listed] == [0, 1, 2]
    assert listed[0]["state"] == {"s": 0}


def test_auth_header_sent() -> None:
    cp, fake = _checkpointer()

    def check(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("Authorization") == "Bearer tok"
        return fake(request)

    cp._client = httpx.Client(
        base_url=BASE,
        headers={"Authorization": "Bearer tok"},
        transport=httpx.MockTransport(check),
    )
    cp.put("t3", 0, {"x": 1})


def test_delete_is_noop() -> None:
    cp, fake = _checkpointer()
    cp.put("t4", 0, {"x": 1})
    cp.delete("t4", 0)  # must not raise; no server call
    assert cp.get("t4", 0) == {"x": 1}
