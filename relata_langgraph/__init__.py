"""RelataCheckpointer — a real LangGraph `BaseCheckpointSaver` backed by RelataDB.

`RelataCheckpointer` (sync) and `AsyncRelataCheckpointer` (async) both subclass
``langgraph.checkpoint.base.BaseCheckpointSaver`` and implement its full
required surface (``get_tuple``/``list``/``put``/``put_writes`` + async
mirrors), so either can be passed directly to ``workflow.compile(checkpointer=...)``
(#2229, fixing #340 — the previous bespoke ``put``/``get``/``list``/``delete``
client was not usable there). Checkpoints persist via the governed A2A
checkpoint door (``PUT/GET /a2a/checkpoints/{thread_id}/{checkpoint_id}``,
``GET /a2a/checkpoints/{thread_id}``) — the server keeps an exact, lossless
read-back store and mirrors each write as a governed ``MemoryItem`` (audit +
provenance). Retained checkpoints FIFO-evict server-side.

Unlike the framework-agnostic adapters in ``relata_adapters`` (which
deliberately duck-type their target framework's interface so the package
loads with or without it installed), this module imports ``langgraph`` at
class-definition time. That is required, not incidental: LangGraph's own
Pregel runtime gates checkpoint behavior behind
``isinstance(checkpointer, BaseCheckpointSaver)`` in several places, so a
duck-typed (non-subclassing) checkpointer is silently treated as absent.
``langgraph`` is therefore an optional *extra*, not a hard dependency of the
``relata`` package: install it with ``pip install relata-sdk[langgraph]``.

Requirements:
    - ``httpx`` (already a package dependency)
    - ``langgraph`` (optional extra: ``pip install relata-sdk[langgraph]``)
    - ``RELATA_ENDPOINT`` or explicit ``endpoint`` argument
    - ``RELATA_BEARER_TOKEN`` or explicit ``token`` argument

Verified against ``langgraph`` 1.2.x / ``langgraph-checkpoint`` 4.1.x.

Example (Python, sync graph)::

    import os
    from langgraph.graph import StateGraph, END
    from relata_langgraph import RelataCheckpointer

    checkpointer = RelataCheckpointer(
        endpoint=os.environ["RELATA_ENDPOINT"],
        token=os.environ.get("RELATA_BEARER_TOKEN", ""),
    )

    workflow = StateGraph(dict)
    workflow.add_node("echo", lambda state: {"output": f"echo: {state['input']}"})
    workflow.set_entry_point("echo")
    workflow.add_edge("echo", END)

    app = workflow.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "session-42"}}
    app.invoke({"input": "hello"}, config)

Example (Python, async graph)::

    from relata_langgraph import AsyncRelataCheckpointer

    async with AsyncRelataCheckpointer(endpoint=..., token=...) as checkpointer:
        app = workflow.compile(checkpointer=checkpointer)
        await app.ainvoke({"input": "hello"}, config)

See docs/src/guides/langgraph-checkpointer.md for full documentation, storage
format, and the current known limitations (namespaced/cross-thread listing,
thread deletion).
"""

from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any
from urllib.parse import quote, unquote

import httpx

try:
    from langgraph.checkpoint.base import (
        WRITES_IDX_MAP,
        BaseCheckpointSaver,
        ChannelVersions,
        Checkpoint,
        CheckpointMetadata,
        CheckpointTuple,
        get_checkpoint_id,
        get_checkpoint_metadata,
    )
    from langgraph.checkpoint.serde.base import SerializerProtocol
except ImportError as exc:  # pragma: no cover - exercised only without langgraph installed
    raise ImportError(
        "relata_langgraph requires the optional `langgraph` dependency. Install it "
        "with `pip install relata-sdk[langgraph]` (or `pip install langgraph` directly)."
    ) from exc

from langchain_core.runnables import RunnableConfig

__all__ = ["AsyncRelataCheckpointer", "RelataCheckpointer"]

_NS_SEP = ":"
_NOT_FOUND = httpx.codes.NOT_FOUND

_DELETE_THREAD_UNSUPPORTED = (
    "RelataCheckpointer does not support thread deletion: the Relata "
    "/a2a/checkpoints door has no delete route (checkpoints FIFO-evict "
    "server-side — see docs/src/guides/langgraph-checkpointer.md). For "
    "explicit governed erasure of the audit-trail MemoryItem mirror, call "
    "`DELETE /memory/forget/:id` directly (relata.A2AClient) against the "
    "underlying memory item id."
)


def _encode_typed(typed: tuple[str, bytes]) -> dict[str, str]:
    """Wrap a `serde.dumps_typed()` `(type, bytes)` pair for JSON transport."""
    kind, raw = typed
    return {"type": kind, "data": base64.b64encode(raw).decode("ascii")}


def _decode_typed(blob: dict[str, Any]) -> tuple[str, bytes]:
    """Inverse of `_encode_typed`, ready for `serde.loads_typed()`."""
    return (blob["type"], base64.b64decode(blob["data"]))


def _thread_segment(thread_id: str) -> str:
    return quote(thread_id, safe="")


def _checkpoint_segment(checkpoint_ns: str, checkpoint_id: str) -> str:
    """Encode `(checkpoint_ns, checkpoint_id)` into the single path segment the
    `/a2a/checkpoints/{thread_id}/{checkpoint_id}` route expects.

    `checkpoint_ns` is percent-quoted (it may be empty, or a subgraph
    namespace containing arbitrary characters); `checkpoint_id` is a LangGraph
    `uuid6` string (hex + hyphens only) and passes through verbatim.
    """
    return f"{quote(checkpoint_ns, safe='')}{_NS_SEP}{checkpoint_id}"


def _split_checkpoint_segment(segment: str) -> tuple[str, str]:
    ns_part, _, id_part = segment.partition(_NS_SEP)
    return unquote(ns_part), id_part


_WRITES_SUFFIX = "__writes"


def _writes_segment(checkpoint_ns: str, checkpoint_id: str) -> str:
    """Pending task writes for a checkpoint are stored as their own record,
    independent of (and possibly created before) the checkpoint record
    itself: LangGraph's Pregel loop may call `put_writes` for a checkpoint id
    that hasn't been `put()` yet (writes for the *next* tick are attached
    pre-emptively). `uuid6` checkpoint ids never contain `_`, so this suffix
    can't collide with a real checkpoint id.
    """
    return _checkpoint_segment(checkpoint_ns, f"{checkpoint_id}{_WRITES_SUFFIX}")


class _RelataCodec:
    """Pure (de)serialization shared by the sync and async savers — no I/O."""

    def __init__(self, serde: SerializerProtocol) -> None:
        self._serde = serde

    def encode_put_body(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> dict[str, Any]:
        return {
            "checkpoint": _encode_typed(self._serde.dumps_typed(checkpoint)),
            "metadata": _encode_typed(
                self._serde.dumps_typed(get_checkpoint_metadata(config, metadata))
            ),
            "parent_checkpoint_id": config["configurable"].get("checkpoint_id"),
        }

    def decode_tuple(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        body: dict[str, Any],
        writes_body: dict[str, Any] | None,
    ) -> CheckpointTuple:
        checkpoint: Checkpoint = self._serde.loads_typed(_decode_typed(body["checkpoint"]))
        metadata: CheckpointMetadata = self._serde.loads_typed(_decode_typed(body["metadata"]))
        pending_writes = [
            (
                write["task_id"],
                write["channel"],
                self._serde.loads_typed(_decode_typed(write["value"])),
            )
            for write in (writes_body or {}).get("writes", [])
        ]
        parent_id = body.get("parent_checkpoint_id")
        parent_config: RunnableConfig | None = (
            {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_id,
                }
            }
            if parent_id
            else None
        )
        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    def apply_writes(
        self,
        writes_body: dict[str, Any] | None,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str,
    ) -> dict[str, Any]:
        """Merge `writes` into `writes_body` (a `{"writes": [...]}` blob, or
        `None`/`{}` if none exists yet — writes may arrive before the
        checkpoint they'll attach to is `put()`, so this blob is independent
        of the checkpoint blob and always safe to create on first write)."""
        body: dict[str, Any] = writes_body if writes_body is not None else {}
        existing: list[dict[str, Any]] = body.setdefault("writes", [])
        existing_keys = {(w["task_id"], w["idx"]) for w in existing}
        for idx, (channel, value) in enumerate(writes):
            key_idx = WRITES_IDX_MAP.get(channel, idx)
            if key_idx >= 0 and (task_id, key_idx) in existing_keys:
                # Matches BaseCheckpointSaver's own dedup rule (see InMemorySaver.put_writes):
                # non-special writes only ever get their first attempt recorded.
                continue
            existing.append(
                {
                    "task_id": task_id,
                    "idx": key_idx,
                    "channel": channel,
                    "task_path": task_path,
                    "value": _encode_typed(self._serde.dumps_typed(value)),
                }
            )
        return body


def _matches_filter(metadata: CheckpointMetadata, filter_: dict[str, Any] | None) -> bool:
    if not filter_:
        return True
    return all(value == metadata.get(key) for key, value in filter_.items())


def _require_thread_id(config: RunnableConfig | None) -> tuple[str, str]:
    if config is None or not config.get("configurable", {}).get("thread_id"):
        raise ValueError(
            "RelataCheckpointer.list()/alist() require a thread_id in config; the "
            "Relata /a2a/checkpoints door has no cross-thread listing route."
        )
    configurable = config["configurable"]
    return configurable["thread_id"], configurable.get("checkpoint_ns", "")


class RelataCheckpointer(BaseCheckpointSaver[int]):
    """Sync LangGraph `BaseCheckpointSaver` backed by RelataDB.

    The `MemorySaver`/`InMemorySaver`-equivalent of this integration: works
    with `.invoke()` directly, and also implements the async surface (`aget_tuple`
    / `alist` / `aput` / `aput_writes`) by offloading the blocking HTTP calls to
    a thread via `asyncio.to_thread`, so it also works with `.ainvoke()`. For a
    natively async client (no thread-pool hop), use `AsyncRelataCheckpointer`.
    """

    def __init__(
        self,
        endpoint: str,
        token: str | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 30.0,
        serde: SerializerProtocol | None = None,
    ) -> None:
        """Create a checkpointer connected to a RelataDB instance.

        Args:
            endpoint: Base URL of the Relata HTTP endpoint
                      (e.g. ``"http://localhost:9090"``).
            token: Bearer token for authentication. Omit (or pass an empty
                   string) in unauthenticated dev mode.
            transport: Optional httpx transport (injected in tests).
            timeout: Per-request timeout in seconds.
            serde: Optional LangGraph serializer override; defaults to
                   `BaseCheckpointSaver`'s `JsonPlusSerializer`.
        """
        super().__init__(serde=serde)
        self.endpoint = endpoint.rstrip("/")
        self.token = token or ""
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        self._client = httpx.Client(
            base_url=self.endpoint,
            headers=headers,
            transport=transport,
            timeout=timeout,
        )
        self._codec = _RelataCodec(self.serde)

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> RelataCheckpointer:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- internal HTTP helpers ------------------------------------------------

    def _put(self, path: str, body: dict[str, Any]) -> None:
        resp = self._client.put(path, json=body)
        resp.raise_for_status()

    def _get(self, path: str) -> dict[str, Any] | None:
        resp = self._client.get(path)
        if resp.status_code == _NOT_FOUND:
            return None
        resp.raise_for_status()
        state: dict[str, Any] = resp.json()["state"]
        return state

    def _list_ids(self, thread_id: str, checkpoint_ns: str) -> list[str]:
        resp = self._client.get(f"/a2a/checkpoints/{_thread_segment(thread_id)}")
        resp.raise_for_status()
        out: list[str] = []
        for seg in resp.json().get("checkpoints", []):
            ns, checkpoint_id = _split_checkpoint_segment(seg)
            if ns == checkpoint_ns and not checkpoint_id.endswith(_WRITES_SUFFIX):
                out.append(checkpoint_id)
        return out

    def _fetch_tuple(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> CheckpointTuple | None:
        body = self._get(
            f"/a2a/checkpoints/{_thread_segment(thread_id)}/"
            f"{_checkpoint_segment(checkpoint_ns, checkpoint_id)}"
        )
        if body is None:
            return None
        writes_body = self._get(
            f"/a2a/checkpoints/{_thread_segment(thread_id)}/"
            f"{_writes_segment(checkpoint_ns, checkpoint_id)}"
        )
        return self._codec.decode_tuple(thread_id, checkpoint_ns, checkpoint_id, body, writes_body)

    # -- BaseCheckpointSaver (sync) --------------------------------------------

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Fetch a checkpoint tuple, or the latest one for the thread if no
        `checkpoint_id` is given in `config`."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)
        if checkpoint_id is None:
            ids = self._list_ids(thread_id, checkpoint_ns)
            if not ids:
                return None
            checkpoint_id = max(ids)  # uuid6 ids sort lexicographically by time
        return self._fetch_tuple(thread_id, checkpoint_ns, checkpoint_id)

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints for `config`'s thread, newest first.

        `config` (with a `thread_id`) is required — the Relata checkpoint
        store has no cross-thread listing route, unlike `InMemorySaver`.
        """
        thread_id, checkpoint_ns = _require_thread_id(config)
        before_id = get_checkpoint_id(before) if before else None
        yielded = 0
        for checkpoint_id in sorted(self._list_ids(thread_id, checkpoint_ns), reverse=True):
            if before_id is not None and checkpoint_id >= before_id:
                continue
            if limit is not None and yielded >= limit:
                break
            tup = self._fetch_tuple(thread_id, checkpoint_ns, checkpoint_id)
            if tup is None or not _matches_filter(tup.metadata, filter):
                continue
            yielded += 1
            yield tup

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Persist a checkpoint. `PUT /a2a/checkpoints/{thread_id}/{checkpoint_id}`
        stores the checkpoint + metadata verbatim; overwrites any existing
        checkpoint for the same id (re-saves don't accumulate duplicates)."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        body = self._codec.encode_put_body(config, checkpoint, metadata)
        self._put(
            f"/a2a/checkpoints/{_thread_segment(thread_id)}/"
            f"{_checkpoint_segment(checkpoint_ns, checkpoint_id)}",
            body,
        )
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Attach intermediate task writes to a checkpoint.

        Stored as an independent record (`_writes_segment`) from the
        checkpoint itself: LangGraph may call this for a checkpoint id that
        hasn't been `put()` yet (writes for the *next* tick, attached
        pre-emptively), so this must not depend on the checkpoint existing.
        Read-modify-write over that record (no server-side CAS — acceptable
        for the same reason `InMemorySaver` needs no locking: concurrent
        `put_writes` calls to the *same* checkpoint id from different tasks
        could still race, a known limitation of a REST-backed adapter with no
        compare-and-swap primitive server-side).
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]
        path = (
            f"/a2a/checkpoints/{_thread_segment(thread_id)}/"
            f"{_writes_segment(checkpoint_ns, checkpoint_id)}"
        )
        writes_body = self._codec.apply_writes(self._get(path), writes, task_id, task_path)
        self._put(path, writes_body)

    def delete_thread(self, thread_id: str) -> None:
        """Not supported — see `AsyncRelataCheckpointer.adelete_thread`."""
        raise NotImplementedError(_DELETE_THREAD_UNSUPPORTED)

    # -- BaseCheckpointSaver (async — offloaded to a thread) -------------------

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        items = await asyncio.to_thread(
            lambda: list(self.list(config, filter=filter, before=before, limit=limit))
        )
        for item in items:
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        await asyncio.to_thread(self.put_writes, config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        """Not supported — see class docstring / `_DELETE_THREAD_UNSUPPORTED`."""
        raise NotImplementedError(_DELETE_THREAD_UNSUPPORTED)


class AsyncRelataCheckpointer(BaseCheckpointSaver[int]):
    """Native-async LangGraph `BaseCheckpointSaver` backed by RelataDB.

    Uses `httpx.AsyncClient` directly (no thread-pool hop), for graphs driven
    exclusively via `.ainvoke()`/`.astream()`. Sync methods are intentionally
    left unimplemented — mirroring the sync/async split most REST- and
    DB-backed LangGraph savers ship (e.g. `AsyncSqliteSaver`/`AsyncPostgresSaver`
    in the `langgraph-checkpoint-*` packages); use `RelataCheckpointer` for
    sync graphs.
    """

    def __init__(
        self,
        endpoint: str,
        token: str | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 30.0,
        serde: SerializerProtocol | None = None,
    ) -> None:
        """See `RelataCheckpointer.__init__` — identical arguments, async transport."""
        super().__init__(serde=serde)
        self.endpoint = endpoint.rstrip("/")
        self.token = token or ""
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        self._client = httpx.AsyncClient(
            base_url=self.endpoint,
            headers=headers,
            transport=transport,
            timeout=timeout,
        )
        self._codec = _RelataCodec(self.serde)

    async def aclose(self) -> None:
        """Close the underlying async HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> AsyncRelataCheckpointer:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    # -- internal HTTP helpers ------------------------------------------------

    async def _put(self, path: str, body: dict[str, Any]) -> None:
        resp = await self._client.put(path, json=body)
        resp.raise_for_status()

    async def _get(self, path: str) -> dict[str, Any] | None:
        resp = await self._client.get(path)
        if resp.status_code == _NOT_FOUND:
            return None
        resp.raise_for_status()
        state: dict[str, Any] = resp.json()["state"]
        return state

    async def _list_ids(self, thread_id: str, checkpoint_ns: str) -> list[str]:
        resp = await self._client.get(f"/a2a/checkpoints/{_thread_segment(thread_id)}")
        resp.raise_for_status()
        out: list[str] = []
        for seg in resp.json().get("checkpoints", []):
            ns, checkpoint_id = _split_checkpoint_segment(seg)
            if ns == checkpoint_ns and not checkpoint_id.endswith(_WRITES_SUFFIX):
                out.append(checkpoint_id)
        return out

    async def _fetch_tuple(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> CheckpointTuple | None:
        body = await self._get(
            f"/a2a/checkpoints/{_thread_segment(thread_id)}/"
            f"{_checkpoint_segment(checkpoint_ns, checkpoint_id)}"
        )
        if body is None:
            return None
        writes_body = await self._get(
            f"/a2a/checkpoints/{_thread_segment(thread_id)}/"
            f"{_writes_segment(checkpoint_ns, checkpoint_id)}"
        )
        return self._codec.decode_tuple(thread_id, checkpoint_ns, checkpoint_id, body, writes_body)

    # -- BaseCheckpointSaver (async) --------------------------------------------

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)
        if checkpoint_id is None:
            ids = await self._list_ids(thread_id, checkpoint_ns)
            if not ids:
                return None
            checkpoint_id = max(ids)
        return await self._fetch_tuple(thread_id, checkpoint_ns, checkpoint_id)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        thread_id, checkpoint_ns = _require_thread_id(config)
        before_id = get_checkpoint_id(before) if before else None
        ids = await self._list_ids(thread_id, checkpoint_ns)
        yielded = 0
        for checkpoint_id in sorted(ids, reverse=True):
            if before_id is not None and checkpoint_id >= before_id:
                continue
            if limit is not None and yielded >= limit:
                break
            tup = await self._fetch_tuple(thread_id, checkpoint_ns, checkpoint_id)
            if tup is None or not _matches_filter(tup.metadata, filter):
                continue
            yielded += 1
            yield tup

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        body = self._codec.encode_put_body(config, checkpoint, metadata)
        await self._put(
            f"/a2a/checkpoints/{_thread_segment(thread_id)}/"
            f"{_checkpoint_segment(checkpoint_ns, checkpoint_id)}",
            body,
        )
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]
        path = (
            f"/a2a/checkpoints/{_thread_segment(thread_id)}/"
            f"{_writes_segment(checkpoint_ns, checkpoint_id)}"
        )
        writes_body = self._codec.apply_writes(await self._get(path), writes, task_id, task_path)
        await self._put(path, writes_body)

    async def adelete_thread(self, thread_id: str) -> None:
        """Not supported — see class docstring / `_DELETE_THREAD_UNSUPPORTED`."""
        raise NotImplementedError(_DELETE_THREAD_UNSUPPORTED)

    # -- BaseCheckpointSaver (sync) — intentionally unsupported ----------------

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Not supported on the async saver — use `RelataCheckpointer` for sync
        graphs, or call `aget_tuple` from an event loop."""
        raise NotImplementedError(
            "AsyncRelataCheckpointer is async-only; use RelataCheckpointer for "
            "sync graphs, or the async methods (aget_tuple/alist/aput/aput_writes)."
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """Not supported on the async saver — see `get_tuple`."""
        raise NotImplementedError(
            "AsyncRelataCheckpointer is async-only; use RelataCheckpointer for "
            "sync graphs, or the async methods (aget_tuple/alist/aput/aput_writes)."
        )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Not supported on the async saver — see `get_tuple`."""
        raise NotImplementedError(
            "AsyncRelataCheckpointer is async-only; use RelataCheckpointer for "
            "sync graphs, or the async methods (aget_tuple/alist/aput/aput_writes)."
        )

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Not supported on the async saver — see `get_tuple`."""
        raise NotImplementedError(
            "AsyncRelataCheckpointer is async-only; use RelataCheckpointer for "
            "sync graphs, or the async methods (aget_tuple/alist/aput/aput_writes)."
        )

    def delete_thread(self, thread_id: str) -> None:
        """Not supported — see class docstring / `_DELETE_THREAD_UNSUPPORTED`."""
        raise NotImplementedError(_DELETE_THREAD_UNSUPPORTED)
