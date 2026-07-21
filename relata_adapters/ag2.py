"""AG2 (AutoGen v0.4+) adapter (#89).

Wraps :class:`relata.memory.Memory` as an AG2 ``Memory``-protocol object so
an AG2 ``ConversableAgent`` can use Relata as its governed memory.

AG2's memory protocol expects ``add``, ``get``, ``search`` (similar to the
legacy AutoGen v0.2 adapter in ``relata_adapters.autogen``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from relata.client import RelataClient


class RelataAG2Memory:
    """AG2 (AutoGen v0.4+) memory adapter.

    Implements the duck-typed ``MemoryProtocol``:
    ``add(content, metadata)`` / ``search(query, n)`` / ``clear()``.

    Args:
        client: A :class:`~relata.client.RelataClient` with auth + purpose.
        session_id: Optional session id for multi-session agents.

    Example::

        from relata import RelataClient
        from relata_adapters.ag2 import RelataAG2Memory

        client = RelataClient(
            "http://localhost:9090", purpose="agent"
        )
        memory = RelataAG2Memory(client, session_id="case-42")
        memory.add("Alice called Bob from +254...", metadata={"type": "call"})
        hits = memory.search("Alice phone", n=5)
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

    def add(self, content: str, *, metadata: dict[str, Any] | None = None) -> str:
        """Store a memory. ``metadata`` is merged into the memory's fields."""
        confidence = (metadata or {}).get("confidence", 1.0)
        memory_class = (metadata or {}).get("memory_class", "semantic")
        return self._memory.add(
            content, confidence=confidence, memory_class=memory_class
        )

    def search(self, query: str, *, n: int = 5) -> list[dict[str, Any]]:
        """Search memories. AG2 uses ``n`` for the result count."""
        return self._memory.search(query, top_k=n)

    def get(self, memory_id: str) -> dict[str, Any] | None:
        return self._memory.get(memory_id)

    def clear(self) -> None:
        """AG2's ``clear()`` is a no-op here — governed memory uses retention
        policies, not hard deletes. To forget a specific memory, use
        :meth:`forget`."""
        pass

    def forget(self, memory_id: str) -> dict[str, Any]:
        return self._memory.forget(memory_id)

    def close(self) -> None:
        self._memory.close()
