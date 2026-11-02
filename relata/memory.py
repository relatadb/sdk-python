"""
Mem0-style high-level memory client (#684).

A thin convenience layer over Relata's governed ``/v1/memory/*`` verbs (ADR-144)
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
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

from relata._http import AsyncHttpTransport, HttpTransport, path_segment

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


def _associate_batch_body(items: list[dict[str, Any]], purpose: str) -> dict[str, Any]:
    """Body for ``POST /v1/memory/associate/batch``: each item needs ``from_id`` and
    ``to_id``; ``relation`` / ``confidence`` / ``purpose`` are optional and the
    client's purpose fills any item that omits it."""
    out: list[dict[str, Any]] = []
    for item in items:
        if not item.get("from_id") or not item.get("to_id"):
            raise ValueError("each association needs non-empty 'from_id' and 'to_id'")
        out.append({"purpose": purpose, **item})
    return {"items": out}


def _recall_body(
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
) -> dict[str, Any]:
    """Build the ``POST /v1/memory/recall`` JSON body.

    Includes the five ADR-145 retrieval-quality operators — ``min_confidence``
    (CONFIDENCE), ``recency_half_life_secs`` (RECENCY), ``budget_tokens``
    (BUDGET), ``stability_days`` (FORGETTING_CURVE), ``cancel_threshold``
    (CANCEL_WHEN) — each omitted when ``None`` so the server applies its own
    default. Shared by :class:`Memory` and :class:`AsyncMemory`.
    """
    body: dict[str, Any] = {"q": query, "purpose": purpose, "top_k": top_k}
    sid = session_id if session_id is not None else default_session_id
    if sid:
        body["session_id"] = sid
    if as_of:
        body["as_of"] = as_of
    if min_confidence is not None:
        body["min_confidence"] = min_confidence
    if recency_half_life_secs is not None:
        body["recency_half_life_secs"] = recency_half_life_secs
    if budget_tokens is not None:
        body["budget_tokens"] = budget_tokens
    if stability_days is not None:
        body["stability_days"] = stability_days
    if cancel_threshold is not None:
        body["cancel_threshold"] = cancel_threshold
    return body


def _recall_stream_path(body: dict[str, Any]) -> str:
    """``GET /v1/memory/recall/stream?...`` path for a recall body (query-string form)."""
    return "/v1/memory/recall/stream?" + urlencode({k: str(v) for k, v in body.items()})


def _procedure_params(
    agent_id: str,
    purpose: str,
    name: str | None,
    all_versions: bool | None,
    limit: int | None,
) -> dict[str, str]:
    """Query params for ``GET /v1/memory/recall-procedure``."""
    params: dict[str, str] = {"agent_id": agent_id, "purpose": purpose}
    if name:
        params["name"] = name
    if all_versions is not None:
        params["all_versions"] = "true" if all_versions else "false"
    if limit is not None:
        params["limit"] = str(limit)
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


def _episode_rows(result: Any) -> list[dict[str, Any]]:
    """Episodes from ``GET /v1/memory/episodes``.

    The server answers ``{"session_id", "episodes": [...], "count"}``; an older
    shape used ``rows``. Reading only ``rows`` made a live call return ``[]``.
    """
    if isinstance(result, dict):
        rows = result.get("episodes")
        if not isinstance(rows, list):
            rows = result.get("rows")
    else:
        rows = result
    return rows if isinstance(rows, list) else []


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
        result = _unwrap(self._t.post("/v1/memory/remember", payload))
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
        return _batch_ids(_unwrap(self._t.post("/v1/memory/remember/batch", payload)))

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
        body = _recall_body(
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
        return _unwrap(self._t.post("/v1/memory/recall", body))

    def batch_search(self, queries: list[str], *, top_k: int = 5) -> list[list[dict[str, Any]]]:
        """Run multiple recall queries and return merged results (#967 Tier 4c)."""
        return [self.search(q, top_k=top_k) for q in queries]

    def get(self, memory_id: str) -> dict[str, Any] | None:
        """Fetch a single memory by id, or ``None`` if not found."""
        path = (
            f"/v1/memory/recognize/{path_segment(memory_id)}?purpose={path_segment(self._purpose)}"
        )
        memory = _unwrap(self._t.get(path)).get("memory")
        return memory if isinstance(memory, dict) else None

    def update(self, memory_id: str, content: str) -> str:
        """Supersede ``memory_id`` with new ``content`` (governed); return the new id."""
        payload = {"id": memory_id, "content": content, "purpose": self._purpose}
        return str(_unwrap(self._t.post("/v1/memory/consolidate", payload)).get("new_id", ""))

    def forget(self, memory_id: str) -> dict[str, Any]:
        """Schedule a governed retention-policy retract for ``memory_id``.

        Not a hard delete — returns the policy decision (``memory_item_id``,
        ``policy``, ``forget_at_ns``).
        """
        path = f"/v1/memory/forget/{path_segment(memory_id)}?purpose={path_segment(self._purpose)}"
        return _unwrap(self._t.delete(path))

    def review(
        self,
        memory_id: str,
        decision: str,
        *,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Approve or reject a pending memory (``decision`` is ``"approve"`` or ``"reject"``).

        Wraps ``POST /v1/memory/review/{id}``. Memory stored with an ``external:``
        origin is ``pending`` and hidden from recall until reviewed. The row is
        superseded by a reviewed copy (history stays readable with ``as_of``).
        Returns ``id``, ``review_status``, ``previous_review_status`` and
        ``changed`` (``False`` when the memory already had that status).
        """
        payload: dict[str, Any] = {"decision": decision, "purpose": self._purpose}
        if reason:
            payload["reason"] = reason
        return _unwrap(self._t.post(f"/v1/memory/review/{path_segment(memory_id)}", payload))

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

        Wraps ``POST /v1/memory/associate``. The server reads ``from_id`` /
        ``to_id`` / ``relation`` (``confidence`` is not part of the server
        schema and is ignored). Returns the association record.
        """
        payload = {
            "from_id": source_id,
            "to_id": target_id,
            "relation": relation,
            "purpose": self._purpose,
        }
        return _unwrap(self._t.post("/v1/memory/associate", payload))

    def associate_batch(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        """Link many memory pairs in one request.

        Wraps ``POST /v1/memory/associate/batch``. Each item is
        ``{"from_id", "to_id", "relation"?, "confidence"?, "purpose"?}``. Returns
        ``{"stored": N, "submitted": N, "results": [...]}``; ``stored`` below
        ``submitted`` means some items were rejected (see ``results``).
        """
        return dict(
            self._t.post("/v1/memory/associate/batch", _associate_batch_body(items, self._purpose))
        )

    def episodes(
        self,
        *,
        session_id: str | None = None,
        as_of: str | None = None,
    ) -> list[dict[str, Any]]:
        """List episodes with full ``Episode`` detail.

        Wraps ``GET /v1/memory/episodes``. Optional ``session_id`` filters to one
        session; ``as_of`` makes the lookup bi-temporal.
        """
        params: dict[str, str] = {"purpose": self._purpose}
        sid = session_id if session_id is not None else self._session_id
        if sid:
            params["session_id"] = sid
        if as_of:
            params["as_of"] = as_of
        result = _unwrap(self._t.get("/v1/memory/episodes?" + urlencode(params)))
        return _episode_rows(result)

    def justify(self, memory_id: str) -> dict[str, Any]:
        """Return the PROV-O assertion chain for ``memory_id``.

        Wraps ``GET /v1/memory/justify/<id>``. The chain explains how the memory
        was derived — source session, tool call, prior belief, etc.
        """
        path = f"/v1/memory/justify/{path_segment(memory_id)}?purpose={path_segment(self._purpose)}"
        return _unwrap(self._t.get(path))

    def resolve(self, memory_id: str) -> dict[str, Any]:
        """Resolve a memory's supersession chain to its canonical head.

        Wraps ``GET /v1/memory/resolve/<id>``. Follows the supersession chain from
        ``memory_id`` to the current canonical belief and returns it.
        """
        path = f"/v1/memory/resolve/{path_segment(memory_id)}?purpose={path_segment(self._purpose)}"
        return _unwrap(self._t.get(path))

    def summarise(
        self,
        source_ids: list[str],
        *,
        summary_content: str | None = None,
    ) -> dict[str, Any]:
        """Produce a summary belief from a set of source memories.

        Wraps ``POST /v1/memory/summarise``. The server reads ``ids`` (memory
        UUIDs to fetch and summarise) and optional ``contents`` (explicit
        source text). Returns the new summary memory.
        """
        payload: dict[str, Any] = {
            "ids": source_ids,
            "purpose": self._purpose,
        }
        if summary_content is not None:
            payload["contents"] = [summary_content]
        return _unwrap(self._t.post("/v1/memory/summarise", payload))

    def recall_stream(
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
    ) -> Iterator[dict[str, Any]]:
        """Stream a recall as server-sent events (``GET /v1/memory/recall/stream``).

        Same retrieval and ranking as :meth:`search`; each yielded dict is one
        parsed SSE ``data:`` frame (one memory per frame). An error is framed as
        a ``{"type": "error", ...}`` event rather than raised mid-stream.
        """
        from relata.streaming import _classify_capped_error

        body = _recall_body(
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
        with self._t._client.stream("GET", _recall_stream_path(body)) as resp:  # noqa: SLF001
            if not resp.is_success and resp.headers.get("content-type", "").split(";")[0] != (
                "text/event-stream"
            ):
                _classify_capped_error(self._t, resp)
            for line in resp.iter_lines():
                if line.startswith("data:"):
                    raw = line[5:].strip()
                    try:
                        yield json.loads(raw)
                    except json.JSONDecodeError:
                        yield {"raw": raw}

    def remember_procedure(
        self,
        agent_id: str,
        name: str,
        instruction_text: str,
    ) -> dict[str, Any]:
        """Store a versioned agent procedure (``POST /v1/memory/remember-procedure``)."""
        payload = {
            "agent_id": agent_id,
            "name": name,
            "instruction_text": instruction_text,
            "purpose": self._purpose,
        }
        return _unwrap(self._t.post("/v1/memory/remember-procedure", payload))

    def recall_procedure(
        self,
        agent_id: str,
        *,
        name: str | None = None,
        all_versions: bool | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Read back stored procedures for an agent (``GET /v1/memory/recall-procedure``).

        Only the latest version per name is returned unless ``all_versions``.
        """
        params = _procedure_params(agent_id, self._purpose, name, all_versions, limit)
        return _unwrap(self._t.get("/v1/memory/recall-procedure?" + urlencode(params)))

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
    `await m.forget(...)` — over the governed `/v1/memory/*` verbs, for async apps
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
        result = _unwrap(await self._t.post("/v1/memory/remember", payload))
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
        return _batch_ids(_unwrap(await self._t.post("/v1/memory/remember/batch", payload)))

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

    async def batch_search(
        self, queries: list[str], *, top_k: int = 5
    ) -> list[list[dict[str, Any]]]:
        """Async equivalent of :meth:`Memory.batch_search` — runs multiple
        recall queries and returns the merged results (#967 Tier 4c)."""
        return [await self.search(q, top_k=top_k) for q in queries]

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
        body = _recall_body(
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
        return _unwrap(await self._t.post("/v1/memory/recall", body))

    async def get(self, memory_id: str) -> dict[str, Any] | None:
        """Fetch a single memory by id, or ``None`` if not found."""
        path = (
            f"/v1/memory/recognize/{path_segment(memory_id)}?purpose={path_segment(self._purpose)}"
        )
        memory = _unwrap(await self._t.get(path)).get("memory")
        return memory if isinstance(memory, dict) else None

    async def update(self, memory_id: str, content: str) -> str:
        """Supersede ``memory_id`` with new ``content`` (governed); return the new id."""
        payload = {"id": memory_id, "content": content, "purpose": self._purpose}
        return str(_unwrap(await self._t.post("/v1/memory/consolidate", payload)).get("new_id", ""))

    async def forget(self, memory_id: str) -> dict[str, Any]:
        """Schedule a governed retention-policy retract for ``memory_id``."""
        path = f"/v1/memory/forget/{path_segment(memory_id)}?purpose={path_segment(self._purpose)}"
        return _unwrap(await self._t.delete(path))

    async def review(
        self,
        memory_id: str,
        decision: str,
        *,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Approve or reject a pending memory (async).

        Wraps ``POST /v1/memory/review/{id}``. Memory stored with an ``external:``
        origin is ``pending`` and hidden from recall until reviewed. The row is
        superseded by a reviewed copy (history stays readable with ``as_of``).
        Returns ``id``, ``review_status``, ``previous_review_status`` and
        ``changed`` (``False`` when the memory already had that status).
        """
        payload: dict[str, Any] = {"decision": decision, "purpose": self._purpose}
        if reason:
            payload["reason"] = reason
        return _unwrap(await self._t.post(f"/v1/memory/review/{path_segment(memory_id)}", payload))

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
        return _unwrap(await self._t.post("/v1/memory/associate", payload))

    async def associate_batch(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        """Link many memory pairs in one request (async) -- see :meth:`Memory.associate_batch`."""
        return dict(
            await self._t.post(
                "/v1/memory/associate/batch", _associate_batch_body(items, self._purpose)
            )
        )

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
        result = _unwrap(await self._t.get("/v1/memory/episodes?" + urlencode(params)))
        return _episode_rows(result)

    async def justify(self, memory_id: str) -> dict[str, Any]:
        """Return the PROV-O assertion chain for ``memory_id`` (async)."""
        path = f"/v1/memory/justify/{path_segment(memory_id)}?purpose={path_segment(self._purpose)}"
        return _unwrap(await self._t.get(path))

    async def resolve(self, memory_id: str) -> dict[str, Any]:
        """Resolve a memory's supersession chain to its canonical head (async)."""
        path = f"/v1/memory/resolve/{path_segment(memory_id)}?purpose={path_segment(self._purpose)}"
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
        return _unwrap(await self._t.post("/v1/memory/summarise", payload))

    async def recall_stream(
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
    ) -> AsyncIterator[dict[str, Any]]:
        """Async variant of :meth:`Memory.recall_stream` (use with ``async for``)."""
        from relata.streaming import _aclassify_capped_error

        body = _recall_body(
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
        async with self._t._client.stream("GET", _recall_stream_path(body)) as resp:  # noqa: SLF001
            if not resp.is_success and resp.headers.get("content-type", "").split(";")[0] != (
                "text/event-stream"
            ):
                await _aclassify_capped_error(self._t, resp)
            async for line in resp.aiter_lines():
                if line.startswith("data:"):
                    raw = line[5:].strip()
                    try:
                        yield json.loads(raw)
                    except json.JSONDecodeError:
                        yield {"raw": raw}

    async def remember_procedure(
        self,
        agent_id: str,
        name: str,
        instruction_text: str,
    ) -> dict[str, Any]:
        """Async variant of :meth:`Memory.remember_procedure`."""
        payload = {
            "agent_id": agent_id,
            "name": name,
            "instruction_text": instruction_text,
            "purpose": self._purpose,
        }
        return _unwrap(await self._t.post("/v1/memory/remember-procedure", payload))

    async def recall_procedure(
        self,
        agent_id: str,
        *,
        name: str | None = None,
        all_versions: bool | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Async variant of :meth:`Memory.recall_procedure`."""
        params = _procedure_params(agent_id, self._purpose, name, all_versions, limit)
        return _unwrap(await self._t.get("/v1/memory/recall-procedure?" + urlencode(params)))

    async def close(self) -> None:
        """Close the underlying HTTP connection."""
        await self._t.aclose()

    async def __aenter__(self) -> AsyncMemory:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()
