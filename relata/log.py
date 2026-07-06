"""Ordered integrity log SDK (#83 — server routes now shipped).

Wraps ``POST /log/leaf``, ``GET /log/leaves``, ``GET /log/head``.

Server contract:
- ``POST /log/leaf`` body: ``{"data_b64": "<base64>"}`` → ``201 {"index": u64, "commit_ns": i64}``
- ``GET /log/leaves?since=<u64>&limit=<n>`` → ``200 {"leaves": [...], "count": n}``
- ``GET /log/head`` → ``200 {"head": u64}``
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

from relata._http import AsyncHttpTransport, HttpTransport

if TYPE_CHECKING:
    import httpx

    from relata.client import RelataClient


class LogLeaf:
    """A single log leaf entry."""

    __slots__ = ("index", "data", "commit_ns")

    def __init__(self, index: int, data: bytes, commit_ns: int) -> None:
        self.index = index
        self.data = data
        self.commit_ns = commit_ns

    def __repr__(self) -> str:
        return f"LogLeaf(index={self.index}, commit_ns={self.commit_ns}, data_len={len(self.data)})"


def _decode_leaf(raw: dict[str, Any]) -> LogLeaf:
    return LogLeaf(
        index=int(raw["index"]),
        data=base64.b64decode(raw.get("data_b64", "")),
        commit_ns=int(raw.get("commit_ns", 0)),
    )


class LogClient:
    """Synchronous ordered integrity-log client."""

    def __init__(
        self,
        base_url: str,
        *,
        bearer_token: str | None = None,
        timeout: float = 30.0,
        extra_headers: dict[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._t = HttpTransport(
            base_url, bearer_token, timeout, transport=transport, extra_headers=extra_headers
        )

    @classmethod
    def from_client(cls, client: RelataClient) -> LogClient:
        return cls(
            client._base_url,  # noqa: SLF001
            bearer_token=client._bearer_token,  # noqa: SLF001
            timeout=client._timeout,  # noqa: SLF001
            extra_headers=client._extra_headers,  # noqa: SLF001
        )

    def append(self, data: bytes) -> tuple[int, int]:
        """Append ``data`` as a new log leaf. Returns ``(index, commit_ns)``.

        The index is strictly monotonic and gap-free under the single-writer
        model. The server fsyncs the WAL before returning the index.
        """
        payload = {"data_b64": base64.b64encode(data).decode("ascii")}
        resp = self._t.post("/log/leaf", payload)
        return int(resp["index"]), int(resp.get("commit_ns", 0))

    def head(self) -> int:
        """Return the current ``write_seq`` (the highest committed index)."""
        resp = self._t.get("/log/head")
        return int(resp["head"])

    def load_leaves(
        self,
        *,
        since: int = 0,
        limit: int = 1000,
    ) -> list[LogLeaf]:
        """Load leaves in append order. ``since`` is an exclusive lower bound —
        leaves with ``index > since`` are returned. ``limit`` caps the response."""
        params = urlencode({"since": str(since), "limit": str(limit)})
        resp = self._t.get(f"/log/leaves?{params}")
        raw_leaves = resp.get("leaves", []) if isinstance(resp, dict) else []
        return [_decode_leaf(r) for r in raw_leaves]

    def close(self) -> None:
        self._t.close()

    def __enter__(self) -> LogClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class AsyncLogClient:
    """Asynchronous ordered integrity-log client — see :class:`LogClient`."""

    def __init__(
        self,
        base_url: str,
        *,
        bearer_token: str | None = None,
        timeout: float = 30.0,
        extra_headers: dict[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._t = AsyncHttpTransport(
            base_url, bearer_token, timeout, transport=transport, extra_headers=extra_headers
        )

    @classmethod
    def from_client(cls, client: RelataClient) -> AsyncLogClient:
        return cls(
            client._base_url,  # noqa: SLF001
            bearer_token=client._bearer_token,  # noqa: SLF001
            timeout=client._timeout,  # noqa: SLF001
            extra_headers=client._extra_headers,  # noqa: SLF001
        )

    async def append(self, data: bytes) -> tuple[int, int]:
        payload = {"data_b64": base64.b64encode(data).decode("ascii")}
        resp = await self._t.post("/log/leaf", payload)
        return int(resp["index"]), int(resp.get("commit_ns", 0))

    async def head(self) -> int:
        resp = await self._t.get("/log/head")
        return int(resp["head"])

    async def load_leaves(
        self,
        *,
        since: int = 0,
        limit: int = 1000,
    ) -> list[LogLeaf]:
        params = urlencode({"since": str(since), "limit": str(limit)})
        resp = await self._t.get(f"/log/leaves?{params}")
        raw_leaves = resp.get("leaves", []) if isinstance(resp, dict) else []
        return [_decode_leaf(r) for r in raw_leaves]

    async def close(self) -> None:
        await self._t.aclose()

    async def __aenter__(self) -> AsyncLogClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()
