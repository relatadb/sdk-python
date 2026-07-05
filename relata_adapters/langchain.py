"""LangChain memory adapter (#683) — governed semantic memory backed by Relata.

Implements the LangChain ``BaseMemory`` interface shape without importing
LangChain, so it loads with or without the framework installed. Plug it into a
chain as the ``memory=`` argument:

    from relata_adapters.langchain import RelataMemory

    mem = RelataMemory(base_url="http://localhost:8080", purpose="agent")
    # chain = ConversationChain(llm=..., memory=mem)

``save_context`` stores the turn; ``load_memory_variables`` recalls the most
relevant prior memories for the incoming input. Governance (purpose + ACL) is on.
"""

from __future__ import annotations

from typing import Any

from relata import Memory
from relata_adapters._base import RelataMemoryBackend


class RelataMemory:
    """LangChain ``BaseMemory``-shaped semantic memory over Relata."""

    def __init__(
        self,
        memory: Memory | None = None,
        *,
        base_url: str | None = None,
        purpose: str | None = None,
        session_id: str = "",
        memory_key: str = "history",
        top_k: int = 5,
    ) -> None:
        self._backend = RelataMemoryBackend(
            memory, base_url=base_url, purpose=purpose, session_id=session_id
        )
        self.memory_key = memory_key
        self.top_k = top_k

    @property
    def memory_variables(self) -> list[str]:
        """The keys this memory injects into the chain context."""
        return [self.memory_key]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        """Recall memories relevant to ``inputs`` and return them under ``memory_key``."""
        query = " ".join(str(v) for v in inputs.values()) if inputs else ""
        rows = self._backend.recall(query, top_k=self.top_k) if query else []
        history = "\n".join(str(r.get("content", "")) for r in rows)
        return {self.memory_key: history}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        """Store the human input and the AI output of a turn as memories."""
        for value in inputs.values():
            self._backend.record(str(value), role="human")
        for value in outputs.values():
            self._backend.record(str(value), role="ai")

    def clear(self) -> None:
        """Retract every memory this instance stored (governed)."""
        self._backend.wipe()
