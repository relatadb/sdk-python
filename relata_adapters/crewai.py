"""CrewAI storage adapter (#683) — governed semantic memory backed by Relata.

Implements the CrewAI ``Storage`` interface shape (``save`` / ``search`` /
``reset``) without importing CrewAI, so it loads with or without the framework.
Use it as the storage backend for a CrewAI memory:

    from relata_adapters.crewai import RelataStorage

    storage = RelataStorage(base_url="http://localhost:9090", purpose="crew")

Governance (purpose + ACL) is on.
"""

from __future__ import annotations

from typing import Any

from relata import Memory
from relata_adapters._base import RelataMemoryBackend


class RelataStorage:
    """CrewAI ``Storage``-shaped semantic memory over Relata."""

    def __init__(
        self,
        memory: Memory | None = None,
        *,
        base_url: str | None = None,
        purpose: str | None = None,
        session_id: str = "",
    ) -> None:
        self._backend = RelataMemoryBackend(
            memory, base_url=base_url, purpose=purpose, session_id=session_id
        )

    def save(self, value: object, metadata: dict[str, Any] | None = None) -> None:
        """Store ``value`` as a memory (``metadata`` is accepted for interface parity)."""
        _ = metadata
        self._backend.record(str(value))

    def search(
        self,
        query: str,
        limit: int = 3,
        score_threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Recall up to ``limit`` memories for ``query`` above ``score_threshold``."""
        rows = self._backend.recall(query, top_k=limit)
        if score_threshold > 0.0:
            rows = [r for r in rows if float(r.get("score", 0.0)) >= score_threshold]
        return rows

    def reset(self) -> None:
        """Retract every memory this instance stored (governed)."""
        self._backend.wipe()
