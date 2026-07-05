"""Shared, framework-agnostic backend for the agent-framework memory adapters (#683).

Each adapter (LangChain / LlamaIndex / CrewAI / AutoGen) maps its framework's
memory interface onto these primitives, which in turn call Relata's governed
``/memory/*`` verbs via :class:`relata.Memory`. The adapters never import their
framework, so they install and import cleanly with or without it (mirroring the
existing ``relata_langgraph`` adapter).

The mapping is the semantic-memory model — ``record`` (store) + ``recall``
(relevance search) — which is the Mem0 paradigm every one of these frameworks'
memory/storage abstractions supports.
"""

from __future__ import annotations

from typing import Any

from relata import Memory


def message_parts(message: object) -> tuple[str | None, str]:
    """Duck-type a framework message into ``(role, content)``.

    Works for dicts (``{"role"/"type", "content"}``) and objects exposing
    ``.role``/``.type`` + ``.content`` (LangChain, LlamaIndex, AutoGen messages)
    without importing any framework. Falls back to ``str(message)``.
    """
    if isinstance(message, dict):
        role = message.get("role") or message.get("type")
        return role, str(message.get("content", ""))
    role = getattr(message, "role", None) or getattr(message, "type", None)
    content = getattr(message, "content", None)
    return role, str(content if content is not None else message)


class RelataMemoryBackend:
    """Framework-agnostic store/recall backend over :class:`relata.Memory`.

    Either pass an existing :class:`relata.Memory`, or ``base_url`` + ``purpose``
    to construct one. ``record`` remembers the ids it stores so ``wipe`` can
    issue a governed retract for exactly those memories.
    """

    def __init__(
        self,
        memory: Memory | None = None,
        *,
        base_url: str | None = None,
        purpose: str | None = None,
        session_id: str = "",
    ) -> None:
        if memory is None:
            if not base_url or not purpose:
                raise ValueError("pass either memory= or both base_url= and purpose=")
            memory = Memory(base_url, purpose=purpose, session_id=session_id)
        self._memory = memory
        self._ids: list[str] = []

    def record(self, text: str, *, role: str | None = None) -> str:
        """Store ``text`` (optionally tagged with a ``role``) and return its id."""
        content = f"{role}: {text}" if role else text
        mem_id = self._memory.add(content)
        if mem_id:
            self._ids.append(mem_id)
        return mem_id

    def recall(self, query: str, *, top_k: int = 5) -> list[dict[str, Any]]:
        """Return memories relevant to ``query`` (ranked rows from recall)."""
        return self._memory.search(query, top_k=top_k)

    def wipe(self) -> int:
        """Retract every memory this backend stored; return the count.

        Best-effort governed retract (not a hard delete) of exactly the ids this
        instance recorded — reliable for a session's lifetime without a
        bulk-delete endpoint.
        """
        count = 0
        for mem_id in self._ids:
            self._memory.forget(mem_id)
            count += 1
        self._ids.clear()
        return count

    def close(self) -> None:
        """Close the underlying HTTP connection."""
        self._memory.close()
