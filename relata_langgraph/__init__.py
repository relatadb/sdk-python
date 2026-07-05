"""RelataCheckpointer for LangGraph — persists graph state in RelataDB.

Usage::

    from relata_langgraph import RelataCheckpointer

    checkpointer = RelataCheckpointer(
        endpoint="http://localhost:8080",
        token="my-bearer-token",
    )

    # Pass to your LangGraph compiled graph:
    # graph = workflow.compile(checkpointer=checkpointer)

See docs/src/guides/langgraph-checkpointer.md for full documentation.
"""

from __future__ import annotations

import httpx


class RelataCheckpointer:
    """LangGraph checkpointer backend backed by RelataDB.

    Persists LangGraph checkpoints via the governed A2A checkpoint door:
    ``PUT/GET /a2a/checkpoints/{thread_id}/{step}`` and
    ``GET /a2a/checkpoints/{thread_id}``. The server keeps an exact, lossless
    read-back store and mirrors each checkpoint as a governed MemoryItem
    (audit + provenance). Retained checkpoints FIFO-evict server-side.

    Requirements:
        - ``httpx`` (already a package dependency)
        - ``RELATA_ENDPOINT`` or explicit ``endpoint`` argument
        - ``RELATA_BEARER_TOKEN`` or explicit ``token`` argument

    Example (Python)::

        import os
        from relata_langgraph import RelataCheckpointer

        checkpointer = RelataCheckpointer(
            endpoint=os.environ["RELATA_ENDPOINT"],
            token=os.environ["RELATA_BEARER_TOKEN"],
        )

    Example (with LangGraph)::

        from langgraph.graph import StateGraph
        from relata_langgraph import RelataCheckpointer

        checkpointer = RelataCheckpointer(endpoint="...", token="...")
        workflow = StateGraph(...)
        # ... add nodes and edges ...
        app = workflow.compile(checkpointer=checkpointer)

        # Invoke with thread persistence:
        result = app.invoke({"input": "hello"}, config={"configurable": {"thread_id": "42"}})
    """

    def __init__(
        self,
        endpoint: str,
        token: str,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Create a checkpointer connected to a RelataDB instance.

        Args:
            endpoint: Base URL of the Relata HTTP endpoint
                      (e.g. ``"http://localhost:8080"``).
            token: Bearer token for authentication. Set to an empty string
                   in unauthenticated dev mode.
            transport: Optional httpx transport (injected in tests).
            timeout: Per-request timeout in seconds.
        """
        self.endpoint = endpoint.rstrip("/")
        self.token = token
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.Client(
            base_url=self.endpoint,
            headers=headers,
            transport=transport,
            timeout=timeout,
        )

    def _memory_key(self, thread_id: str, step: int) -> str:
        """Canonical memory key for a (thread_id, step) pair."""
        return f"langgraph:checkpoint:{thread_id}:{step}"

    def put(self, thread_id: str, step: int, state: dict) -> None:
        """Persist checkpoint state to RelataDB.

        ``PUT /a2a/checkpoints/{thread_id}/{step}`` with the graph state as the
        JSON body. Overwrites any existing checkpoint for the same
        (thread_id, step) pair (the server keeps an exact, lossless read-back
        store plus a governed MemoryItem mirror for audit/provenance).

        Args:
            thread_id: LangGraph thread identifier.
            step: Step index within the thread.
            state: Serialisable graph state dict.
        """
        resp = self._client.put(f"/a2a/checkpoints/{thread_id}/{step}", json=state)
        resp.raise_for_status()

    def get(self, thread_id: str, step: int) -> dict | None:
        """Retrieve checkpoint state from RelataDB.

        ``GET /a2a/checkpoints/{thread_id}/{step}``. Returns ``None`` (on 404)
        when no checkpoint exists for the given (thread_id, step).

        Args:
            thread_id: LangGraph thread identifier.
            step: Step index within the thread.

        Returns:
            The previously persisted state dict, or ``None``.
        """
        resp = self._client.get(f"/a2a/checkpoints/{thread_id}/{step}")
        if resp.status_code == httpx.codes.NOT_FOUND:
            return None
        resp.raise_for_status()
        return resp.json().get("state")

    def list(self, thread_id: str) -> list[dict]:
        """List all checkpoints for a thread, ordered by step ascending.

        ``GET /a2a/checkpoints/{thread_id}`` returns the retained checkpoint
        ids; each is fetched and returned as ``{"step": int, "state": dict}``.
        Non-integer ids (written by other clients) are skipped.

        Args:
            thread_id: LangGraph thread identifier.

        Returns:
            List of ``{"step": int, "state": dict}`` dicts, step-ascending.
        """
        resp = self._client.get(f"/a2a/checkpoints/{thread_id}")
        resp.raise_for_status()
        ids = resp.json().get("checkpoints", [])
        out: list[dict] = []
        for cid in ids:
            try:
                step = int(cid)
            except (ValueError, TypeError):
                continue
            state = self.get(thread_id, step)
            if state is not None:
                out.append({"step": step, "state": state})
        out.sort(key=lambda r: r["step"])
        return out

    def delete(self, thread_id: str, step: int | None = None) -> None:
        """Delete checkpoints for a thread.

        No-op: the server exposes no checkpoint-delete route — retained
        checkpoints are FIFO-evicted server-side, so explicit deletion is
        unnecessary for the checkpointer contract. Kept for interface
        compatibility with LangGraph's cleanup calls.

        Args:
            thread_id: LangGraph thread identifier.
            step: Step index to delete, or ``None`` to delete all steps.
        """
        # ponytail: no server delete-checkpoint endpoint; checkpoints FIFO-evict.

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> RelataCheckpointer:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
