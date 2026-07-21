"""Pydantic-AI adapter (#89).

Wraps :class:`relata.memory.Memory` as a Pydantic-AI ``MemoryBackend`` so a
Pydantic-AI agent can use Relata as its governed memory store.

Usage::

    from relata import RelataClient
    from relata_adapters.pydantic_ai import RelataMemoryBackend

    client = RelataClient("http://localhost:9090", purpose="agent")
    backend = RelataMemoryBackend(client)
    # Pass to a Pydantic-AI Agent(memory=backend).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from relata.client import RelataClient


class RelataMemoryBackend:
    """Pydantic-AI memory backend backed by Relata's governed Memory surface.

    Implements the duck-typed protocol that Pydantic-AI's ``Agent`` expects:
    ``add(message: str)`` and ``search(query: str, limit: int) -> list[dict]``.
    Does NOT import pydantic-ai at module load.

    Args:
        client: A :class:`~relata.client.RelataClient` with auth + purpose.
        session_id: Optional session id for multi-session agents.

    Example::

        from relata import RelataClient
        from relata_adapters.pydantic_ai import RelataMemoryBackend

        client = RelataClient(
            "http://localhost:9090", purpose="agent", bearer_token="..."
        )
        backend = RelataMemoryBackend(client, session_id="session-42")
        backend.add("User prefers dark mode")
        hits = backend.search("ui preferences", limit=5)
    """

    def __init__(
        self,
        client: RelataClient,
        *,
        session_id: str = "",
        transport: Any = None,  # noqa: ANN401
    ) -> None:
        from relata.memory import Memory

        self._memory = Memory(
            client._base_url,  # noqa: SLF001
            purpose=client._default_purpose or "agent",  # noqa: SLF001
            bearer_token=client._bearer_token,  # noqa: SLF001
            session_id=session_id,
            extra_headers=client._extra_headers,  # noqa: SLF001
            transport=transport,
        )
        self._client = client

    def add(self, message: str, **kwargs: Any) -> str:  # noqa: ANN401
        """Store a memory. Returns the memory id."""
        confidence = kwargs.get("confidence", 1.0)
        memory_class = kwargs.get("memory_class", "semantic")
        return self._memory.add(
            message, confidence=confidence, memory_class=memory_class
        )

    def search(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        """Search memories by hybrid recall (BM25 + vector)."""
        return self._memory.search(query, top_k=limit)

    def get(self, memory_id: str) -> dict[str, Any] | None:
        """Fetch a single memory by id."""
        return self._memory.get(memory_id)

    def forget(self, memory_id: str) -> dict[str, Any]:
        """Governed forget (retention-policy retract, not hard delete)."""
        return self._memory.forget(memory_id)

    def close(self) -> None:
        self._memory.close()
