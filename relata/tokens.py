"""Dedup / uniqueness token SDK (#84 — server routes now shipped).

Wraps ``POST /tokens``, ``GET /tokens/:id``, ``DELETE /tokens/:id``,
``GET /tokens/stats``.

Server contract:
Wraps ``POST /tokens``, ``GET /tokens/:id``, ``DELETE``
- ``GET /tokens/:id`` → ``200 {"present": bool, "inserted_at_ns"?: i64, "ttl_secs"?: u64|null}``
- ``DELETE /tokens/:id`` → ``200 {"revoked": bool}`` (admin only)
- ``GET /tokens/stats`` → ``200 {"count": usize, "oldest_ns"?: i64|null, "eviction_policy": str}``
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from relata._http import AsyncHttpTransport, HttpTransport

if TYPE_CHECKING:
    import httpx

    from relata.client import RelataClient


class TokenStats:
    """Dedup token set statistics."""

    __slots__ = ("count", "oldest_ns", "eviction_policy")

    def __init__(self, count: int, oldest_ns: int | None, eviction_policy: str) -> None:
        self.count = count
        self.oldest_ns = oldest_ns
        self.eviction_policy = eviction_policy

    def __repr__(self) -> str:
        return (
            f"TokenStats(count={self.count}, "
            f"oldest_ns={self.oldest_ns}, "
            f"eviction_policy={self.eviction_policy!r})"
        )


class TokenClient:
    """Synchronous dedup / uniqueness token client.

    Provides atomic test-and-set (``remember``), presence check (``check``),
    explicit revoke (``revoke`` — admin only), and stats.
    """

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
    def from_client(cls, client: RelataClient) -> TokenClient:
        return cls(
            client._base_url,  # noqa: SLF001
            bearer_token=client._bearer_token,  # noqa: SLF001
            timeout=client._timeout,  # noqa: SLF001
            extra_headers=client._extra_headers,  # noqa: SLF001
        )

    def remember(self, token_id: str, *, ttl_secs: int | None = None) -> bool:
        """Atomic test-and-set. Returns ``True`` if newly inserted, ``False``
        if already present. This is the replay-defence primitive — security
        relevant; a backend that buffers the write asynchronously or loses
        the token on restart fails the §7 contract."""
        payload: dict[str, Any] = {"id": token_id}
        if ttl_secs is not None:
            payload["ttl_secs"] = ttl_secs
        resp = self._t.post("/tokens", payload)
        return bool(resp.get("newly_inserted", False))

    def check(self, token_id: str) -> dict[str, Any]:
        """Non-mutating presence check. Returns ``{"present": bool,
        "inserted_at_ns"?: int, "ttl_secs"?: int|None}``."""
        return self._t.get(f"/tokens/{token_id}")

    def revoke(self, token_id: str) -> bool:
        """Explicit revoke (admin only). Returns ``True`` if a token was
        actually removed."""
        resp = self._t.delete(f"/tokens/{token_id}")
        return bool(resp.get("revoked", False))

    def stats(self) -> TokenStats:
        """Return dedup-set statistics."""
        resp = self._t.get("/tokens/stats")
        return TokenStats(
            count=int(resp.get("count", 0)),
            oldest_ns=int(resp["oldest_ns"]) if resp.get("oldest_ns") is not None else None,
            eviction_policy=str(resp.get("eviction_policy", "")),
        )

    def close(self) -> None:
        self._t.close()

    def __enter__(self) -> TokenClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class AsyncTokenClient:
    """Asynchronous dedup / uniqueness token client — see :class:`TokenClient`."""

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
    def from_client(cls, client: RelataClient) -> AsyncTokenClient:
        return cls(
            client._base_url,  # noqa: SLF001
            bearer_token=client._bearer_token,  # noqa: SLF001
            timeout=client._timeout,  # noqa: SLF001
            extra_headers=client._extra_headers,  # noqa: SLF001
        )

    async def remember(self, token_id: str, *, ttl_secs: int | None = None) -> bool:
        payload: dict[str, Any] = {"id": token_id}
        if ttl_secs is not None:
            payload["ttl_secs"] = ttl_secs
        resp = await self._t.post("/tokens", payload)
        return bool(resp.get("newly_inserted", False))

    async def check(self, token_id: str) -> dict[str, Any]:
        return await self._t.get(f"/tokens/{token_id}")

    async def revoke(self, token_id: str) -> bool:
        resp = await self._t.delete(f"/tokens/{token_id}")
        return bool(resp.get("revoked", False))

    async def stats(self) -> TokenStats:
        resp = await self._t.get("/tokens/stats")
        return TokenStats(
            count=int(resp.get("count", 0)),
            oldest_ns=int(resp["oldest_ns"]) if resp.get("oldest_ns") is not None else None,
            eviction_policy=str(resp.get("eviction_policy", "")),
        )

    async def close(self) -> None:
        await self._t.aclose()

    async def __aenter__(self) -> AsyncTokenClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()
