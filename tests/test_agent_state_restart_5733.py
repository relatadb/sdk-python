"""SDK-level restart and uncertain-effect behavior for retained agent state (#5733).

The A2A task and checkpoint routes keep bounded, process-local state on the
server (docs/src/guides/workflow-runner.md). These tests pin how the SDK
behaves around that boundary so application code can rely on it:

* a task submitted over a connection that resets mid-request is an *uncertain*
  effect: the SDK raises, does not replay the POST, and a follow-up status read
  after a server restart reports not-found rather than inventing a result;
* a LangGraph checkpoint saved before a server restart is gone afterwards:
  the saver reports "no checkpoint" instead of raising or returning stale data,
  so the application's own durable checkpointer remains the recovery path.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from relata.a2a import A2AClient
from relata.exceptions import ConnectionError as RelataConnectionError
from relata.exceptions import NotFoundError

BASE = "http://localhost:9090"


class _AgentServer:
    """A2A routes with an explicit 'restart' that drops all process-local state."""

    def __init__(self) -> None:
        self.tasks: dict[str, dict[str, Any]] = {}
        self.checkpoints: dict[tuple[str, str], dict[str, Any]] = {}
        self.submit_attempts = 0
        self.drop_next_submit_response = False

    def restart(self) -> None:
        self.tasks.clear()
        self.checkpoints.clear()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        parts = request.url.path.strip("/").split("/")  # v1 a2a tasks|checkpoints ...
        if parts[2] == "tasks":
            if request.method == "POST":
                self.submit_attempts += 1
                task_id = f"task-{self.submit_attempts}"
                # The memory operation runs synchronously inside the request.
                self.tasks[task_id] = {"id": task_id, "status": "completed", "result": {"ok": True}}
                if self.drop_next_submit_response:
                    self.drop_next_submit_response = False
                    raise httpx.RemoteProtocolError("server disconnected without a response")
                return httpx.Response(200, json=self.tasks[task_id])
            task = self.tasks.get(parts[3])
            if task is None:
                return httpx.Response(404, json={"error": "task not found"})
            return httpx.Response(200, json=task)
        thread = parts[3]
        if request.method == "PUT":
            self.checkpoints[(thread, parts[4])] = json.loads(request.content)
            return httpx.Response(200, json={"saved": True})
        if len(parts) == 5:
            state = self.checkpoints.get((thread, parts[4]))
            if state is None:
                return httpx.Response(404, json={"error": "no such checkpoint"})
            return httpx.Response(200, json={"state": state})
        ids = [seg for (t, seg) in self.checkpoints if t == thread]
        return httpx.Response(200, json={"checkpoints": ids})


def test_a2a_dropped_submit_is_uncertain_and_never_replayed_then_status_is_gone_after_restart() -> None:
    server = _AgentServer()
    server.drop_next_submit_response = True
    client = A2AClient(BASE, transport=httpx.MockTransport(server))

    with pytest.raises(RelataConnectionError) as exc_info:
        client.submit_task({"name": "remember", "input": {"text": "x"}})
    # The effect ran on the server but the caller never saw the receipt; the SDK
    # must not resubmit on its own.
    assert server.submit_attempts == 1
    assert "may have been applied" in str(exc_info.value)

    # Reconciliation path while the server is up: the recorded task is visible.
    assert client.get_task("task-1")["status"] == "completed"

    # After a restart the bounded in-process history is gone: not-found, never a
    # fabricated terminal state.
    server.restart()
    with pytest.raises(NotFoundError):
        client.get_task("task-1")


def test_langgraph_checkpoint_is_absent_after_server_restart() -> None:
    pytest.importorskip("langgraph")
    from langgraph.checkpoint.base import Checkpoint

    from relata_langgraph import RelataCheckpointer

    server = _AgentServer()
    saver = RelataCheckpointer(BASE, "tok", transport=httpx.MockTransport(server))
    config: Any = {"configurable": {"thread_id": "thread-1"}}
    checkpoint: Checkpoint = {
        "v": 2,
        "id": "1f18dad7-92d5-61ca-9b74-c9a91c94b006",
        "ts": "2026-08-01T00:00:00+00:00",
        "channel_values": {"messages": ["hello"]},
        "channel_versions": {"messages": "1"},
        "versions_seen": {},
        "updated_channels": None,
    }
    saved = saver.put(config, checkpoint, {"source": "loop", "step": 0}, {})
    assert saver.get_tuple(saved) is not None  # present while the server lives

    server.restart()

    assert saver.get_tuple(saved) is None
    assert saver.get_tuple(config) is None
    assert list(saver.list(config)) == []
