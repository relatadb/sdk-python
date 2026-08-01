"""Tests for the LangGraph RelataCheckpointer / AsyncRelataCheckpointer (#2229).

Drives both savers through a mocked Relata `/a2a/checkpoints/*` surface via
httpx.MockTransport — no live server needed. Covers:

- Direct `BaseCheckpointSaver` protocol conformance (put/get_tuple/list/put_writes).
- Real `langgraph.graph.StateGraph.compile(checkpointer=...)` round-trips —
  this is the acceptance-criteria test: a compiled graph invoked, then a
  brand-new checkpointer + brand-new compiled app (simulating a process
  restart) recovering the same state via `get_state()`/`aget_state()`.
- `delete_thread`/`adelete_thread` now raise `NotImplementedError` instead of
  silently no-op'ing (#340).

Requires the optional `langgraph` extra (`pip install relata-sdk[langgraph]`);
skipped automatically if it isn't installed.
"""

from __future__ import annotations

import json

import httpx
import pytest

langgraph = pytest.importorskip("langgraph")
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata  # noqa: E402
from langgraph.graph import END, StateGraph  # noqa: E402

from relata_langgraph import AsyncRelataCheckpointer, RelataCheckpointer  # noqa: E402

BASE = "http://localhost:9090"


class FakeCheckpointStore:
    """A MockTransport handler emulating the server's `/a2a/checkpoints/*` routes."""

    def __init__(self) -> None:
        # (thread_id, checkpoint_id_segment) -> body
        self.store: dict[tuple[str, str], dict] = {}
        self.calls: list[tuple[str, str]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append((request.method, request.url.path))
        parts = request.url.path.strip("/").split("/")
        # /a2a/checkpoints/<thread>[/<segment>]
        thread = parts[2]
        if request.method == "PUT":
            segment = parts[3]
            self.store[(thread, segment)] = json.loads(request.content)
            return httpx.Response(200, json={"saved": True, "checkpoint_id": segment})
        if request.method == "GET" and len(parts) == 4:
            segment = parts[3]
            state = self.store.get((thread, segment))
            if state is None:
                return httpx.Response(404, json={"error": "no such checkpoint"})
            return httpx.Response(200, json={"state": state})
        if request.method == "GET":  # list
            ids = [segment for (t, segment) in self.store if t == thread]
            return httpx.Response(200, json={"checkpoints": ids})
        return httpx.Response(405, json={"error": "method not allowed"})


def _sync_checkpointer() -> tuple[RelataCheckpointer, FakeCheckpointStore]:
    fake = FakeCheckpointStore()
    cp = RelataCheckpointer(BASE, "tok", transport=httpx.MockTransport(fake))
    return cp, fake


def _make_checkpoint(checkpoint_id: str) -> Checkpoint:
    return {
        "v": 2,
        "id": checkpoint_id,
        "ts": "2026-08-01T00:00:00+00:00",
        "channel_values": {"messages": ["hello"]},
        "channel_versions": {"messages": "1"},
        "versions_seen": {},
        "updated_channels": None,
    }


# -- isinstance / class-shape ------------------------------------------------


def test_is_a_base_checkpoint_saver() -> None:
    from langgraph.checkpoint.base import BaseCheckpointSaver

    cp, _ = _sync_checkpointer()
    assert isinstance(cp, BaseCheckpointSaver)
    async_cp = AsyncRelataCheckpointer(BASE, "tok")
    assert isinstance(async_cp, BaseCheckpointSaver)


# -- direct protocol conformance (sync) --------------------------------------


def test_put_and_get_tuple_round_trip() -> None:
    cp, fake = _sync_checkpointer()
    config = {"configurable": {"thread_id": "t1"}}
    checkpoint = _make_checkpoint("1f18dad7-92d5-61ca-9b74-c9a91c94b006")
    metadata: CheckpointMetadata = {"source": "loop", "step": 0}

    saved_config = cp.put(config, checkpoint, metadata, {})
    assert saved_config["configurable"]["checkpoint_id"] == checkpoint["id"]
    assert ("PUT", f"/a2a/checkpoints/t1/:{checkpoint['id']}") in fake.calls

    tup = cp.get_tuple(saved_config)
    assert tup is not None
    assert tup.checkpoint["id"] == checkpoint["id"]
    assert tup.checkpoint["channel_values"] == {"messages": ["hello"]}
    assert tup.metadata["step"] == 0


def test_get_tuple_missing_returns_none() -> None:
    cp, _ = _sync_checkpointer()
    config = {"configurable": {"thread_id": "t1", "checkpoint_id": "does-not-exist"}}
    assert cp.get_tuple(config) is None


def test_get_tuple_without_checkpoint_id_returns_latest() -> None:
    cp, _ = _sync_checkpointer()
    thread = {"configurable": {"thread_id": "t2"}}
    for cid in [
        "1f18dad7-92d5-61ca-0000-000000000001",
        "1f18dad7-92d5-61ca-0000-000000000002",
        "1f18dad7-92d5-61ca-0000-000000000003",
    ]:
        cp.put(thread, _make_checkpoint(cid), {"source": "loop", "step": 0}, {})
    latest = cp.get_tuple(thread)
    assert latest is not None
    assert latest.checkpoint["id"] == "1f18dad7-92d5-61ca-0000-000000000003"


def test_list_orders_newest_first_and_respects_limit() -> None:
    cp, _ = _sync_checkpointer()
    thread = {"configurable": {"thread_id": "t3"}}
    ids = [
        "1f18dad7-92d5-61ca-0000-000000000001",
        "1f18dad7-92d5-61ca-0000-000000000002",
        "1f18dad7-92d5-61ca-0000-000000000003",
    ]
    for cid in ids:
        cp.put(thread, _make_checkpoint(cid), {"source": "loop", "step": 0}, {})
    listed = list(cp.list(thread))
    assert [t.checkpoint["id"] for t in listed] == list(reversed(ids))

    limited = list(cp.list(thread, limit=2))
    assert len(limited) == 2
    assert limited[0].checkpoint["id"] == ids[-1]


def test_list_without_thread_id_raises() -> None:
    cp, _ = _sync_checkpointer()
    with pytest.raises(ValueError, match="thread_id"):
        list(cp.list(None))


def test_put_writes_attaches_before_and_after_put() -> None:
    cp, _ = _sync_checkpointer()
    checkpoint_id = "1f18dad7-92d5-61ca-0000-0000000000aa"
    config = {
        "configurable": {"thread_id": "t4", "checkpoint_ns": "", "checkpoint_id": checkpoint_id}
    }

    # LangGraph's Pregel loop may call put_writes() for a checkpoint id before
    # put() has been called for it (writes for the *next* tick, attached
    # pre-emptively) — must not raise.
    cp.put_writes(config, [("messages", "partial")], task_id="task-1")

    cp.put(config, _make_checkpoint(checkpoint_id), {"source": "loop", "step": 1}, {})
    cp.put_writes(config, [("other", "value")], task_id="task-2")

    tup = cp.get_tuple(config)
    assert tup is not None
    channels = {w[1] for w in tup.pending_writes}
    assert channels == {"messages", "other"}


def test_delete_thread_raises_not_implemented() -> None:
    cp, _ = _sync_checkpointer()
    with pytest.raises(NotImplementedError, match="does not support thread deletion"):
        cp.delete_thread("t1")


async def test_async_delete_thread_raises_not_implemented() -> None:
    cp = AsyncRelataCheckpointer(BASE, "tok", transport=httpx.MockTransport(FakeCheckpointStore()))
    with pytest.raises(NotImplementedError, match="does not support thread deletion"):
        await cp.adelete_thread("t1")
    with pytest.raises(NotImplementedError, match="does not support thread deletion"):
        cp.delete_thread("t1")


def test_async_saver_sync_methods_unsupported() -> None:
    cp = AsyncRelataCheckpointer(BASE, "tok")
    with pytest.raises(NotImplementedError, match="async-only"):
        cp.get_tuple({"configurable": {"thread_id": "t1"}})
    with pytest.raises(NotImplementedError, match="async-only"):
        cp.put({"configurable": {"thread_id": "t1"}}, _make_checkpoint("x"), {}, {})


def test_auth_header_sent() -> None:
    fake = FakeCheckpointStore()

    def check(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("Authorization") == "Bearer tok"
        return fake(request)

    cp = RelataCheckpointer(BASE, "tok", transport=httpx.MockTransport(check))
    config = {"configurable": {"thread_id": "t5"}}
    cp.put(config, _make_checkpoint("1f18dad7-92d5-61ca-0000-0000000000bb"), {}, {})


# -- real StateGraph.compile(checkpointer=...) conformance -------------------


def _build_graph() -> StateGraph:
    def node_a(state: dict) -> dict:
        return {"count": state["count"] + 1, "log": [*state["log"], "a"]}

    def node_b(state: dict) -> dict:
        return {"count": state["count"] + 1, "log": [*state["log"], "b"]}

    graph = StateGraph(dict)
    graph.add_node("a", node_a)
    graph.add_node("b", node_b)
    graph.set_entry_point("a")
    graph.add_edge("a", "b")
    graph.add_edge("b", END)
    return graph


def test_compiled_graph_invoke_and_restart_recovery() -> None:
    """Acceptance test: `compile(checkpointer=RelataCheckpointer(...))` works,
    and a brand-new checkpointer + brand-new compiled app (simulating a
    process restart) recovers the state via `get_state()`."""
    fake = FakeCheckpointStore()
    transport = httpx.MockTransport(fake)
    config = {"configurable": {"thread_id": "session-42"}}

    app = _build_graph().compile(checkpointer=RelataCheckpointer(BASE, "tok", transport=transport))
    result = app.invoke({"count": 0, "log": []}, config)
    assert result == {"count": 2, "log": ["a", "b"]}

    # "Restart": fresh checkpointer instance, fresh compiled app, same store.
    restarted_app = _build_graph().compile(
        checkpointer=RelataCheckpointer(BASE, "tok", transport=transport)
    )
    recovered = restarted_app.get_state(config)
    assert recovered.values == {"count": 2, "log": ["a", "b"]}
    assert len(list(restarted_app.get_state_history(config))) >= 1


async def test_async_compiled_graph_ainvoke_and_restart_recovery() -> None:
    fake = FakeCheckpointStore()
    transport = httpx.MockTransport(fake)
    config = {"configurable": {"thread_id": "session-async-1"}}

    async with AsyncRelataCheckpointer(BASE, "tok", transport=transport) as checkpointer:
        app = _build_graph().compile(checkpointer=checkpointer)
        result = await app.ainvoke({"count": 0, "log": []}, config)
        assert result == {"count": 2, "log": ["a", "b"]}

    async with AsyncRelataCheckpointer(BASE, "tok", transport=transport) as checkpointer:
        restarted_app = _build_graph().compile(checkpointer=checkpointer)
        recovered = await restarted_app.aget_state(config)
        assert recovered.values == {"count": 2, "log": ["a", "b"]}
