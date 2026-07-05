"""LlamaIndex memory adapter (#683) — governed semantic memory backed by Relata.

Implements the LlamaIndex ``BaseMemory`` interface shape (``put`` / ``get`` /
``get_all`` / ``set`` / ``reset``) without importing LlamaIndex, so it loads with
or without the framework. ``get`` performs a relevance recall against the
incoming input; governance (purpose + ACL) is on.
"""

from __future__ import annotations

from typing import Any

from relata import Memory
from relata_adapters._base import RelataMemoryBackend, message_parts


class RelataMemory:
    """LlamaIndex ``BaseMemory``-shaped semantic memory over Relata."""

    def __init__(
        self,
        memory: Memory | None = None,
        *,
        base_url: str | None = None,
        purpose: str | None = None,
        session_id: str = "",
        top_k: int = 5,
    ) -> None:
        self._backend = RelataMemoryBackend(
            memory, base_url=base_url, purpose=purpose, session_id=session_id
        )
        self.top_k = top_k

    def put(self, message: object) -> None:
        """Store a chat message (object with ``.role``/``.content``, or a dict)."""
        role, content = message_parts(message)
        self._backend.record(content, role=role)

    def set(self, messages: list[Any]) -> None:
        """Store a list of messages."""
        for message in messages:
            self.put(message)

    def get(self, input: str | None = None, **_kwargs: object) -> list[dict[str, Any]]:  # noqa: A002
        """Recall memories relevant to ``input`` (the rows from recall)."""
        if not input:
            return []
        return self._backend.recall(input, top_k=self.top_k)

    def get_all(self) -> list[dict[str, Any]]:
        """Not supported for semantic memory — recall is relevance-ranked.

        Use :meth:`get` with a query. Returns an empty list rather than a
        misleading partial scan.
        """
        return []

    def reset(self) -> None:
        """Retract every memory this instance stored (governed)."""
        self._backend.wipe()
