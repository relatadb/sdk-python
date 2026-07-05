"""Agent-framework memory adapters for Relata (#683, epic #669).

Governed semantic-memory backends for the major agent frameworks, each mapping
the framework's memory/storage interface onto Relata's `/memory/*` verbs via
:class:`relata.Memory`. None of the adapters import their framework, so they
install and load with or without it (mirroring `relata_langgraph`).

- ``relata_adapters.langchain.RelataMemory``   — LangChain ``BaseMemory``
- ``relata_adapters.llamaindex.RelataMemory``  — LlamaIndex ``BaseMemory``
- ``relata_adapters.crewai.RelataStorage``     — CrewAI ``Storage``
- ``relata_adapters.autogen.RelataMemory``     — AutoGen ``Memory`` (async)

LangGraph ships separately as ``relata_langgraph.RelataCheckpointer``.
"""

from __future__ import annotations

from relata_adapters._base import RelataMemoryBackend, message_parts

__all__ = ["RelataMemoryBackend", "message_parts"]
