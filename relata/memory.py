"""
Mem0-style high-level memory client (#684).

A thin convenience layer over Relata's governed ``/memory/*`` verbs (ADR-144)
so the common agent-memory case is a one-liner instead of hand-built requests::

    from relata import Memory

    with Memory("http://localhost:8080", purpose="agent-notes") as m:
        mem_id = m.add("Alice prefers dark mode")
        hits = m.search("ui preferences", top_k=5)
        m.forget(mem_id)

Governance stays on: a non-empty ``purpose`` is required (the server records it
for audit and enforces ACL). ``forget`` is a governed, retention-policy retract,
not a hard delete.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlencode

from relata._http import AsyncHttpTransport, HttpTransport

if TYPE_CHECKING:
    import httpx


def _unwrap(resp: dict[str, Any]) -> dict[str, Any]:
    """Unwrap the MCP envelope the memory verbs return.

    The server replies ``{"content": [{"type": "text", "text": "<json>"}],
    "isError": false}`` where the actual result is JSON encoded inside
    ``content[0].text``. Returns the decoded inner object, or the response
    unchanged when it is already flat.
    """
    content = resp.get("content")
    if isinstance(content, list) and content:
        text = content[0].get("text") if isinstance(content[0], dict) else None
        if isinstance(text, str):
            try:
                inner = json.loads(text)
            except json.JSONDecodeError:
                return {"text": text}
            return inner if isinstance(inner, dict) else {"value": inner}
    return resp


class Memory:
    """High-level memory client — ``add`` / ``search`` / ``forget``.

    Args:
        base_url: Relata server URL, e.g. ``"http://localhost:8080"``.
        purpose: Declared purpose token, recorded for audit and enforced by
            ACL. Required and non-empty.
        bearer_token: Optional bearer token (required when the server sets
            ``RELATA_BEARER_TOKEN``).
        session_id: Default session id applied to ``add``/``search`` when not
            overridden per call.
        timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        *,
        purpose: str,
        bearer_token: str | None = None,
        session_id: str = "",
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        if not purpose:
            raise ValueError("purpose is required and must be non-empty (governance is on)")
        self._t = HttpTransport(
            base_url, bearer_token, timeout,
            transport=transport, extra_headers=extra_headers,
        )
        self._purpose = purpose
        self._session_id = session_id

    def add(
        self,
        content: str,
        *,
        confidence: float = 1.0,
        memory_class: str = "semantic",
        session_id: str | None = None,
    ) -> str:
        """Store a memory and return its id.

        ``memory_class`` is one of ``episodic`` | ``semantic`` | ``procedural``
        (defaults to ``semantic``); ``confidence`` is clamped to ``[0, 1]``.
        """
        payload = {
            "content": content,
            "purpose": self._purpose,
            "confidence": confidence,
            "memory_class": memory_class,
            "session_id": session_id if session_id is not None else self._session_id,
        }
        result = _unwrap(self._t.post("/memory/remember", payload))
        return str(result.get("id", ""))

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        session_id: str | None = None,
        as_of: str | None = None,
    ) -> list[dict[str, Any]]:
        """Recall memories ranked by confidence × recency × relevance.

        Returns the ranked rows (each with ``id``, ``content``, ``confidence``,
        ``score``, ``memory_class``).
        """
        params: dict[str, str] = {
            "q": query,
            "purpose": self._purpose,
            "top_k": str(top_k),
        }
        sid = session_id if session_id is not None else self._session_id
        if sid:
            params["session_id"] = sid
        if as_of:
            params["as_of"] = as_of
        result = _unwrap(self._t.get("/memory/recall?" + urlencode(params)))
        rows = result.get("rows")
        return rows if isinstance(rows, list) else []

    def get(self, memory_id: str) -> dict[str, Any] | None:
        """Fetch a single memory by id, or ``None`` if not found."""
        path = f"/memory/recognize/{quote(memory_id)}?purpose={quote(self._purpose)}"
        memory = _unwrap(self._t.get(path)).get("memory")
        return memory if isinstance(memory, dict) else None

    def update(self, memory_id: str, content: str) -> str:
        """Supersede ``memory_id`` with new ``content`` (governed); return the new id."""
        payload = {"id": memory_id, "content": content, "purpose": self._purpose}
        return str(_unwrap(self._t.post("/memory/consolidate", payload)).get("new_id", ""))

    def forget(self, memory_id: str) -> dict[str, Any]:
        """Schedule a governed retention-policy retract for ``memory_id``.

        Not a hard delete — returns the policy decision (``memory_item_id``,
        ``policy``, ``forget_at_ns``).
        """
        path = f"/memory/forget/{quote(memory_id)}?purpose={quote(self._purpose)}"
        return _unwrap(self._t.delete(path))

    # ------------------------------------------------------------------
    # v1.1 — five missing cognitive verbs (#77)
    # ------------------------------------------------------------------

    def associate(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        *,
        confidence: float = 1.0,
    ) -> dict[str, Any]:
        """Link two memories with a typed relation.

        Wraps ``POST /memory/associate``. Returns the association record
        (``source_id``, ``target_id``, ``relation``, ``confidence``).
        """
        payload = {
            "source_id": source_id,
            "target_id": target_id,
            "relation": relation,
            "confidence": confidence,
            "purpose": self._purpose,
        }
        return _unwrap(self._t.post("/memory/associate", payload))

    def episodes(
        self,
        *,
        session_id: str | None = None,
        as_of: str | None = None,
    ) -> list[dict[str, Any]]:
        """List episodes with full ``Episode`` detail.

        Wraps ``GET /memory/episodes``. Optional ``session_id`` filters to one
        session; ``as_of`` makes the lookup bi-temporal.
        """
        params: dict[str, str] = {"purpose": self._purpose}
        sid = session_id if session_id is not None else self._session_id
        if sid:
            params["session_id"] = sid
        if as_of:
            params["as_of"] = as_of
        result = _unwrap(self._t.get("/memory/episodes?" + urlencode(params)))
        rows = result.get("rows") if isinstance(result, dict) else result
        return rows if isinstance(rows, list) else []

    def justify(self, memory_id: str) -> dict[str, Any]:
        """Return the PROV-O assertion chain for ``memory_id``.

        Wraps ``GET /memory/justify/<id>``. The chain explains how the memory
        was derived — source session, tool call, prior belief, etc.
        """
        path = f"/memory/justify/{quote(memory_id)}?purpose={quote(self._purpose)}"
        return _unwrap(self._t.get(path))

    def resolve(self, memory_id: str, *, policy: str = "latest_wins") -> dict[str, Any]:
        """Resolve a contradiction between ``memory_id`` and a newer belief.

        Wraps ``POST /memory/resolve/<id>``. The ``policy`` selects the
        resolver (``latest_wins`` / ``highest_confidence`` / ``manual``).
        Returns the resolution record.
        """
        payload = {"policy": policy, "purpose": self._purpose}
        path = f"/memory/resolve/{quote(memory_id)}"
        return _unwrap(self._t.post(path, payload))

    def summarise(
        self,
        source_ids: list[str],
        *,
        summary_content: str | None = None,
    ) -> dict[str, Any]:
        """Produce a summary belief from a set of source memories.

        Wraps ``POST /memory/summarise``. ``source_ids`` is the list of memory
        ids to summarise; ``summary_content`` lets the caller supply the
        summary text (the server otherwise derives one). Returns the new
        summary memory.
        """
        payload: dict[str, Any] = {
            "source_ids": source_ids,
            "purpose": self._purpose,
        }
        if summary_content is not None:
            payload["summary_content"] = summary_content
        return _unwrap(self._t.post("/memory/summarise", payload))

    def close(self) -> None:
        """Close the underlying HTTP connection."""
        self._t.close()

    def __enter__(self) -> Memory:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class AsyncMemory:
    """Async counterpart of :class:`Memory` (#688).

    Same surface and semantics — `await m.add(...)` / `await m.search(...)` /
    `await m.forget(...)` — over the governed `/memory/*` verbs, for async apps
    and the AutoGen adapter. Use as an async context manager.
    """

    def __init__(
        self,
        base_url: str,
        *,
        purpose: str,
        bearer_token: str | None = None,
        session_id: str = "",
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        if not purpose:
            raise ValueError("purpose is required and must be non-empty (governance is on)")
        self._t = AsyncHttpTransport(
            base_url, bearer_token, timeout,
            transport=transport, extra_headers=extra_headers,
        )
        self._purpose = purpose
        self._session_id = session_id

    async def add(
        self,
        content: str,
        *,
        confidence: float = 1.0,
        memory_class: str = "semantic",
        session_id: str | None = None,
    ) -> str:
        """Store a memory and return its id (see :meth:`Memory.add`)."""
        payload = {
            "content": content,
            "purpose": self._purpose,
            "confidence": confidence,
            "memory_class": memory_class,
            "session_id": session_id if session_id is not None else self._session_id,
        }
        result = _unwrap(await self._t.post("/memory/remember", payload))
        return str(result.get("id", ""))

    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        session_id: str | None = None,
        as_of: str | None = None,
    ) -> list[dict[str, Any]]:
        """Recall memories ranked by confidence × recency × relevance."""
        params: dict[str, str] = {"q": query, "purpose": self._purpose, "top_k": str(top_k)}
        sid = session_id if session_id is not None else self._session_id
        if sid:
            params["session_id"] = sid
        if as_of:
            params["as_of"] = as_of
        result = _unwrap(await self._t.get("/memory/recall?" + urlencode(params)))
        rows = result.get("rows")
        return rows if isinstance(rows, list) else []

    async def get(self, memory_id: str) -> dict[str, Any] | None:
        """Fetch a single memory by id, or ``None`` if not found."""
        path = f"/memory/recognize/{quote(memory_id)}?purpose={quote(self._purpose)}"
        memory = _unwrap(await self._t.get(path)).get("memory")
        return memory if isinstance(memory, dict) else None

    async def update(self, memory_id: str, content: str) -> str:
        """Supersede ``memory_id`` with new ``content`` (governed); return the new id."""
        payload = {"id": memory_id, "content": content, "purpose": self._purpose}
        return str(_unwrap(await self._t.post("/memory/consolidate", payload)).get("new_id", ""))

    async def forget(self, memory_id: str) -> dict[str, Any]:
        """Schedule a governed retention-policy retract for ``memory_id``."""
        path = f"/memory/forget/{quote(memory_id)}?purpose={quote(self._purpose)}"
        return _unwrap(await self._t.delete(path))

    # ------------------------------------------------------------------
    # v1.1 — five missing cognitive verbs (async mirrors, #77)
    # ------------------------------------------------------------------

    async def associate(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        *,
        confidence: float = 1.0,
    ) -> dict[str, Any]:
        """Link two memories with a typed relation (async)."""
        payload = {
            "source_id": source_id,
            "target_id": target_id,
            "relation": relation,
            "confidence": confidence,
            "purpose": self._purpose,
        }
        return _unwrap(await self._t.post("/memory/associate", payload))

    async def episodes(
        self,
        *,
        session_id: str | None = None,
        as_of: str | None = None,
    ) -> list[dict[str, Any]]:
        """List episodes with full ``Episode`` detail (async)."""
        params: dict[str, str] = {"purpose": self._purpose}
        sid = session_id if session_id is not None else self._session_id
        if sid:
            params["session_id"] = sid
        if as_of:
            params["as_of"] = as_of
        result = _unwrap(await self._t.get("/memory/episodes?" + urlencode(params)))
        rows = result.get("rows") if isinstance(result, dict) else result
        return rows if isinstance(rows, list) else []

    async def justify(self, memory_id: str) -> dict[str, Any]:
        """Return the PROV-O assertion chain for ``memory_id`` (async)."""
        path = f"/memory/justify/{quote(memory_id)}?purpose={quote(self._purpose)}"
        return _unwrap(await self._t.get(path))

    async def resolve(self, memory_id: str, *, policy: str = "latest_wins") -> dict[str, Any]:
        """Resolve a contradiction between ``memory_id`` and a newer belief (async)."""
        payload = {"policy": policy, "purpose": self._purpose}
        path = f"/memory/resolve/{quote(memory_id)}"
        return _unwrap(await self._t.post(path, payload))

    async def summarise(
        self,
        source_ids: list[str],
        *,
        summary_content: str | None = None,
    ) -> dict[str, Any]:
        """Produce a summary belief from a set of source memories (async)."""
        payload: dict[str, Any] = {
            "source_ids": source_ids,
            "purpose": self._purpose,
        }
        if summary_content is not None:
            payload["summary_content"] = summary_content
        return _unwrap(await self._t.post("/memory/summarise", payload))

    async def close(self) -> None:
        """Close the underlying HTTP connection."""
        await self._t.aclose()

    async def __aenter__(self) -> AsyncMemory:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()
