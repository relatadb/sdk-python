"""Adapter registry (#89) — auto-detect installed frameworks and return the
right adapter class.

Usage::

    from relata import RelataClient
    from relata_adapters.registry import get_memory_adapter

    client = RelataClient("http://localhost:8080", purpose="agent")
    AdapterClass = get_memory_adapter()  # auto-detects langchain/crewai/etc.
    if AdapterClass:
        adapter = AdapterClass(client)
    else:
        print("No supported agent framework installed")
"""

from __future__ import annotations

import importlib.util
from typing import Any


def _is_installed(package_name: str) -> bool:
    """Return ``True`` if ``package_name`` is importable."""
    return importlib.util.find_spec(package_name) is not None


def get_memory_adapter() -> type[Any] | None:
    """Return the first available memory-adapter class for a detected framework.

    Detection order matches framework popularity (LangChain first). Returns
    ``None`` if no supported framework is installed.
    """
    if _is_installed("langchain"):
        from relata_adapters.langchain import RelataMemory

        return RelataMemory
    if _is_installed("pydantic_ai"):
        from relata_adapters.pydantic_ai import RelataMemoryBackend

        return RelataMemoryBackend
    if _is_installed("ag2"):
        from relata_adapters.ag2 import RelataAG2Memory

        return RelataAG2Memory
    if _is_installed("autogen"):
        from relata_adapters.autogen import RelataMemory

        return RelataMemory  # type: ignore[no-any-return]
    if _is_installed("crewai"):
        from relata_adapters.crewai import RelataStorage

        return RelataStorage
    if _is_installed("smolagents"):
        from relata_adapters.smolagents import RelataTool

        return RelataTool
    if _is_installed("llama_index"):
        from relata_adapters.llamaindex import RelataMemory

        return RelataMemory
    return None


def list_available_adapters() -> dict[str, bool]:
    """Return a map of framework → installed for every supported framework."""
    frameworks = {
        "langchain": "langchain",
        "llama_index": "llama_index",
        "crewai": "crewai",
        "autogen": "autogen",
        "ag2": "ag2",
        "pydantic_ai": "pydantic_ai",
        "smolagents": "smolagents",
    }
    return {name: _is_installed(pkg) for name, pkg in frameworks.items()}
