"""AutoGen memory adapter (#683) — governed semantic memory backed by Relata.

Implements the AutoGen ``Memory`` protocol shape (async ``add`` / ``query`` /
``clear`` / ``close``) without importing AutoGen, so it loads with or without the
framework. The sync Relata client is run off the event loop via
``asyncio.to_thread``. Governance (purpose + ACL) is on.
"""

from __future__ import annotations

import asyncio
from typing import Any

from relata import Memory
from relata_adapters._base import RelataMemoryBackend, message_parts


class RelataMemory:
    """AutoGen ``Memory``-protocol-shaped semantic memory over Relata."""

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

    async def add(self, content: object) -> None:
        """Store a ``MemoryContent``-like object, a dict, or a plain string."""
        role, text = message_parts(content)
        await asyncio.to_thread(self._backend.record, text, role=role)

    async def query(self, query: str, **_kwargs: object) -> list[dict[str, Any]]:
        """Recall memories relevant to ``query`` (the rows from recall)."""
        return await asyncio.to_thread(self._backend.recall, query, top_k=self.top_k)

    async def clear(self) -> None:
        """Retract every memory this instance stored (governed)."""
        await asyncio.to_thread(self._backend.wipe)

    async def close(self) -> None:
        """Close the underlying HTTP connection."""
        await asyncio.to_thread(self._backend.close)
