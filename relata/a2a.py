"""A2A client — Agent-to-Agent task surface (#77 Phase 3).

Wraps ``/a2a/tasks`` (create), ``/a2a/tasks/:id`` (status),
``/a2a/tasks/:id/stream`` (SSE tail — pairs with the streaming ticket), and
``/a2a/checkpoints/:thread_id`` (LangGraph checkpointer backing store).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from relata._http import AsyncHttpTransport, HttpTransport

if TYPE_CHECKING:
    import httpx

    from relata.client import RelataClient


class A2AClient:
    """Synchronous A2A client — submit / poll / list tasks; manage checkpoints."""

    def __init__(
        self,
        base_url: str,
        *,
        bearer_token: str | None = None,
        timeout: float = 30.0,
        extra_headers: dict[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._t = HttpTransport(
            base_url, bearer_token, timeout, transport=transport, extra_headers=extra_headers
        )

    @classmethod
    def from_client(cls, client: RelataClient) -> A2AClient:
        return cls(
            client._base_url,
            bearer_token=client._bearer_token,
            timeout=client._timeout,
            extra_headers=client._extra_headers,
        )

    def submit_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Submit a long-running A2A task. The ``payload`` matches the A2A
        protocol — typically ``{"name": "...", "input": {...}, "skill": ...}``.
        Returns the task record with its ``id`` and initial ``status``."""
        return self._t.post("/a2a/tasks", payload)

    def get_task(self, task_id: str) -> dict[str, Any]:
        """Poll a task's status. Returns ``{"id", "status", "result", "error", ...}``."""
        return self._t.get(f"/a2a/tasks/{task_id}")

    def list_checkpoints(self, thread_id: str) -> list[dict[str, Any]]:
        """List the LangGraph checkpoints for a thread (used by
        :class:`relata_langgraph.RelataCheckpointer`)."""
        data = self._t.get(f"/a2a/checkpoints/{thread_id}")
        checkpoints = data.get("checkpoints") if isinstance(data, dict) else data
        return checkpoints if isinstance(checkpoints, list) else []

    def load_checkpoint(self, thread_id: str, checkpoint_id: str) -> dict[str, Any]:
        """Load a specific checkpoint by id (``GET /a2a/checkpoints/:t/:c``)."""
        return self._t.get(f"/a2a/checkpoints/{thread_id}/{checkpoint_id}")

    def save_checkpoint(
        self,
        thread_id: str,
        checkpoint_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Save (PUT) a checkpoint. ``payload`` matches the LangGraph
        checkpointer shape (``{"checkpoint": ..., "metadata": ...}``).
        Returns the server's save receipt."""
        return self._t.put(
            f"/a2a/checkpoints/{thread_id}/{checkpoint_id}",
            payload,
        )

    def agent_card(self) -> dict[str, Any]:
        """Fetch the A2A agent card (``GET /.well-known/agent.json``).

        The card advertises the server's A2A capabilities, skills, and
        authentication modes to discovery clients."""
        return self._t.get("/.well-known/agent.json")

    def close(self) -> None:
        self._t.close()

    def __enter__(self) -> A2AClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class AsyncA2AClient:
    """Asynchronous A2A client — see :class:`A2AClient`."""

    def __init__(
        self,
        base_url: str,
        *,
        bearer_token: str | None = None,
        timeout: float = 30.0,
        extra_headers: dict[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._t = AsyncHttpTransport(
            base_url, bearer_token, timeout, transport=transport, extra_headers=extra_headers
        )

    @classmethod
    def from_client(cls, client: RelataClient) -> AsyncA2AClient:
        return cls(
            client._base_url,
            bearer_token=client._bearer_token,
            timeout=client._timeout,
            extra_headers=client._extra_headers,
        )

    async def submit_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._t.post("/a2a/tasks", payload)

    async def get_task(self, task_id: str) -> dict[str, Any]:
        return await self._t.get(f"/a2a/tasks/{task_id}")

    async def list_checkpoints(self, thread_id: str) -> list[dict[str, Any]]:
        data = await self._t.get(f"/a2a/checkpoints/{thread_id}")
        checkpoints = data.get("checkpoints") if isinstance(data, dict) else data
        return checkpoints if isinstance(checkpoints, list) else []

    async def load_checkpoint(self, thread_id: str, checkpoint_id: str) -> dict[str, Any]:
        return await self._t.get(f"/a2a/checkpoints/{thread_id}/{checkpoint_id}")

    async def save_checkpoint(
        self,
        thread_id: str,
        checkpoint_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return await self._t.put(
            f"/a2a/checkpoints/{thread_id}/{checkpoint_id}",
            payload,
        )

    async def agent_card(self) -> dict[str, Any]:
        return await self._t.get("/.well-known/agent.json")

    async def close(self) -> None:
        await self._t.aclose()

    async def __aenter__(self) -> AsyncA2AClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()
