"""
Mem0-style high-level memory client (#684).

A thin convenience layer over Relata's governed ``/memory/*`` verbs (ADR-144)
so the common agent-memory case is a one-liner instead of hand-built requests::

    from relata import Memory

    with Memory("http://localhost:9090", purpose="agent-notes") as m:
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


def _recall_params(
    query: str,
    purpose: str,
    *,
    top_k: int,
    session_id: str | None,
    default_session_id: str,
    as_of: str | None,
    min_confidence: float | None,
    recency_half_life_secs: float | None,
    budget_tokens: int | None,
    stability_days: float | None,
    cancel_threshold: float | None,
) -> dict[str, str]:
    """Build the ``GET /memory/recall`` query params.

    Includes the five ADR-145 retrieval-quality operators — ``min_confidence``
    (CONFIDENCE), ``recency_half_life_secs`` (RECENCY), ``budget_tokens``
    (BUDGET), ``stability_days`` (FORGETTING_CURVE), ``cancel_threshold``
    (CANCEL_WHEN) — each omitted when ``None`` so the server applies its own
    default. Shared by :class:`Memory` and :class:`AsyncMemory`.
    """
    params: dict[str, str] = {"q": query, "purpose": purpose, "top_k": str(top_k)}
    sid = session_id if session_id is not None else default_session_id
    if sid:
        params["session_id"] = sid
    if as_of:
        params["as_of"] = as_of
    if min_confidence is not None:
        params["min_confidence"] = str(min_confidence)
    if recency_half_life_secs is not None:
        params["recency_half_life_secs"] = str(recency_half_life_secs)
    if budget_tokens is not None:
        params["budget_tokens"] = str(budget_tokens)
    if stability_days is not None:
        params["stability_days"] = str(stability_days)
    if cancel_threshold is not None:
        params["cancel_threshold"] = str(cancel_threshold)
    return params


def _batch_items(
    items: list[str | dict[str, Any]],
    default_session: str,
) -> list[dict[str, Any]]:
    """Normalise ``add_batch`` inputs into ``remember_batch`` item dicts.

    Each input is either a plain ``str`` (content) or a dict with ``content``
    and optional ``confidence`` / ``memory_class`` / ``session_id`` /
    ``purpose``. A missing ``session_id`` inherits ``default_session``.
    """
    out: list[dict[str, Any]] = []
    for it in items:
        if isinstance(it, str):
            out.append({"content": it, "session_id": default_session})
        else:
            d = dict(it)
            d.setdefault("session_id", default_session)
            out.append(d)
    return out


def _batch_ids(result: dict[str, Any]) -> list[str]:
    """Pull the per-item ids out of a ``remember_batch`` response.

    Positions that failed validation carry no ``id`` and yield ``""`` so the
    returned list stays index-aligned with the submitted items.
    """
    rows = result.get("results")
    if not isinstance(rows, list):
        return []
    return [str(r.get("id", "")) if isinstance(r, dict) else "" for r in rows]


class Memory:
    """High-level memory client — ``add`` / ``search`` / ``forget``.

    Args:
        base_url: Relata server URL, e.g. ``"http://localhost:9090"``.
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

    def add_batch(
        self,
        items: list[str | dict[str, Any]],
        *,
        session_id: str | None = None,
    ) -> list[str]:
        """Store many memories in one request; return their ids in order.

        Each item is either a ``str`` (content) or a dict with ``content`` and
        optional ``confidence`` / ``memory_class`` / ``session_id`` /
        ``purpose``. This is the high-throughput write path: the server
        amortises the full-text index flush across the whole batch instead of
        once per record. Items that fail validation yield ``""`` at their
        position; valid items still commit.
        """
        default_session = session_id if session_id is not None else self._session_id
        payload = {
            "purpose": self._purpose,
            "items": _batch_items(items, default_session),
        }
        return _batch_ids(_unwrap(self._t.post("/memory/remember/batch", payload)))

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        session_id: str | None = None,
        as_of: str | None = None,
        min_confidence: float | None = None,
        recency_half_life_secs: float | None = None,
        budget_tokens: int | None = None,
        stability_days: float | None = None,
        cancel_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Recall memories ranked by confidence × recency × relevance.

        Returns the ranked rows (each with ``id``, ``content``, ``confidence``,
        ``score``, ``memory_class``).

        The five keyword-only knobs are the ADR-145 retrieval-quality
        operators (REST/MCP params on ``recall``): ``min_confidence``
        (CONFIDENCE), ``recency_half_life_secs`` (RECENCY), ``budget_tokens``
        (BUDGET), ``stability_days`` (FORGETTING_CURVE), and
        ``cancel_threshold`` (CANCEL_WHEN). Each defaults to ``None``, which
        omits it from the request so the server applies its own default. Use
        :meth:`search_detailed` instead of ``search`` to also observe the
        read-only ``recall_cost_tokens``/``cancelled`` response fields these
        knobs produce.
        """
        result = self._recall(
            query,
            top_k=top_k,
            session_id=session_id,
            as_of=as_of,
            min_confidence=min_confidence,
            recency_half_life_secs=recency_half_life_secs,
            budget_tokens=budget_tokens,
            stability_days=stability_days,
            cancel_threshold=cancel_threshold,
        )
        rows = result.get("rows")
        return rows if isinstance(rows, list) else []

    def search_detailed(
        self,
        query: str,
        *,
        top_k: int = 5,
        session_id: str | None = None,
        as_of: str | None = None,
        min_confidence: float | None = None,
        recency_half_life_secs: float | None = None,
        budget_tokens: int | None = None,
        stability_days: float | None = None,
        cancel_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Like :meth:`search` but returns the full recall envelope.

        Includes ``rows`` plus the read-only ``recall_cost_tokens`` (BUDGET's
        running token total) and ``cancelled`` (CANCEL_WHEN short-circuit
        flag) fields, so callers can observe the effect of the ADR-145 knobs,
        not just set them.
        """
        return self._recall(
            query,
            top_k=top_k,
            session_id=session_id,
            as_of=as_of,
            min_confidence=min_confidence,
            recency_half_life_secs=recency_half_life_secs,
            budget_tokens=budget_tokens,
            stability_days=stability_days,
            cancel_threshold=cancel_threshold,
        )

    def _recall(
        self,
        query: str,
        *,
        top_k: int,
        session_id: str | None,
        as_of: str | None,
        min_confidence: float | None,
        recency_half_life_secs: float | None,
        budget_tokens: int | None,
        stability_days: float | None,
        cancel_threshold: float | None,
    ) -> dict[str, Any]:
        params = _recall_params(
            query,
            self._purpose,
            top_k=top_k,
            session_id=session_id,
            default_session_id=self._session_id,
            as_of=as_of,
            min_confidence=min_confidence,
            recency_half_life_secs=recency_half_life_secs,
            budget_tokens=budget_tokens,
            stability_days=stability_days,
            cancel_threshold=cancel_threshold,
        )
        return _unwrap(self._t.get("/memory/recall?" + urlencode(params)))

    def batch_search(self, queries: list[str], *, top_k: int = 5) -> list[list[dict[str, Any]]]:
        """Run multiple recall queries and return merged results (#967 Tier 4c)."""
        return [self.search(q, top_k=top_k) for q in queries]

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

        Wraps ``POST /memory/associate``. The server reads ``from_id`` /
        ``to_id`` / ``relation`` (``confidence`` is not part of the server
        schema and is ignored). Returns the association record.
        """
        payload = {
            "from_id": source_id,
            "to_id": target_id,
            "relation": relation,
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

    def resolve(self, memory_id: str) -> dict[str, Any]:
        """Resolve a memory's supersession chain to its canonical head.

        Wraps ``GET /memory/resolve/<id>``. Follows the supersession chain from
        ``memory_id`` to the current canonical belief and returns it.
        """
        path = f"/memory/resolve/{quote(memory_id)}?purpose={quote(self._purpose)}"
        return _unwrap(self._t.get(path))

    def summarise(
        self,
        source_ids: list[str],
        *,
        summary_content: str | None = None,
    ) -> dict[str, Any]:
        """Produce a summary belief from a set of source memories.

        Wraps ``POST /memory/summarise``. The server reads ``ids`` (memory
        UUIDs to fetch and summarise) and optional ``contents`` (explicit
        source text). Returns the new summary memory.
        """
        payload: dict[str, Any] = {
            "ids": source_ids,
            "purpose": self._purpose,
        }
        if summary_content is not None:
            payload["contents"] = [summary_content]
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

    async def add_batch(
        self,
        items: list[str | dict[str, Any]],
        *,
        session_id: str | None = None,
    ) -> list[str]:
        """Store many memories in one request; return their ids (see
        :meth:`Memory.add_batch`)."""
        default_session = session_id if session_id is not None else self._session_id
        payload = {
            "purpose": self._purpose,
            "items": _batch_items(items, default_session),
        }
        return _batch_ids(_unwrap(await self._t.post("/memory/remember/batch", payload)))

    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        session_id: str | None = None,
        as_of: str | None = None,
        min_confidence: float | None = None,
        recency_half_life_secs: float | None = None,
        budget_tokens: int | None = None,
        stability_days: float | None = None,
        cancel_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Recall memories ranked by confidence × recency × relevance.

        See :meth:`Memory.search` for the five ADR-145 retrieval-quality
        keyword knobs. Use :meth:`search_detailed` to also observe
        ``recall_cost_tokens``/``cancelled``.
        """
        result = await self._recall(
            query,
            top_k=top_k,
            session_id=session_id,
            as_of=as_of,
            min_confidence=min_confidence,
            recency_half_life_secs=recency_half_life_secs,
            budget_tokens=budget_tokens,
            stability_days=stability_days,
            cancel_threshold=cancel_threshold,
        )
        rows = result.get("rows")
        return rows if isinstance(rows, list) else []

    async def search_detailed(
        self,
        query: str,
        *,
        top_k: int = 5,
        session_id: str | None = None,
        as_of: str | None = None,
        min_confidence: float | None = None,
        recency_half_life_secs: float | None = None,
        budget_tokens: int | None = None,
        stability_days: float | None = None,
        cancel_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Like :meth:`search` but returns the full recall envelope (see
        :meth:`Memory.search_detailed`)."""
        return await self._recall(
            query,
            top_k=top_k,
            session_id=session_id,
            as_of=as_of,
            min_confidence=min_confidence,
            recency_half_life_secs=recency_half_life_secs,
            budget_tokens=budget_tokens,
            stability_days=stability_days,
            cancel_threshold=cancel_threshold,
        )

    async def _recall(
        self,
        query: str,
        *,
        top_k: int,
        session_id: str | None,
        as_of: str | None,
        min_confidence: float | None,
        recency_half_life_secs: float | None,
        budget_tokens: int | None,
        stability_days: float | None,
        cancel_threshold: float | None,
    ) -> dict[str, Any]:
        params = _recall_params(
            query,
            self._purpose,
            top_k=top_k,
            session_id=session_id,
            default_session_id=self._session_id,
            as_of=as_of,
            min_confidence=min_confidence,
            recency_half_life_secs=recency_half_life_secs,
            budget_tokens=budget_tokens,
            stability_days=stability_days,
            cancel_threshold=cancel_threshold,
        )
        return _unwrap(await self._t.get("/memory/recall?" + urlencode(params)))

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
            "from_id": source_id,
            "to_id": target_id,
            "relation": relation,
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

    async def resolve(self, memory_id: str) -> dict[str, Any]:
        """Resolve a memory's supersession chain to its canonical head (async)."""
        path = f"/memory/resolve/{quote(memory_id)}?purpose={quote(self._purpose)}"
        return _unwrap(await self._t.get(path))

    async def summarise(
        self,
        source_ids: list[str],
        *,
        summary_content: str | None = None,
    ) -> dict[str, Any]:
        """Produce a summary belief from a set of source memories (async)."""
        payload: dict[str, Any] = {
            "ids": source_ids,
            "purpose": self._purpose,
        }
        if summary_content is not None:
            payload["contents"] = [summary_content]
        return _unwrap(await self._t.post("/memory/summarise", payload))

    async def close(self) -> None:
        """Close the underlying HTTP connection."""
        await self._t.aclose()

    async def __aenter__(self) -> AsyncMemory:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()
