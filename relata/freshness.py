"""Durable-write and cross-index freshness contract (``relata.freshness/v1``, #5706).

A write returns an ``operation`` block carrying an opaque ``token``. Pass it
to :meth:`FreshnessClient.barrier` (or send it as ``X-Relata-Min-Version``,
see :func:`freshness_headers`) to wait, boundedly, until the write is visible
in the lexical, secondary, vector, graph and extraction indexes a read
depends on. A read is never served as complete while an index it needs is
behind: it fails with HTTP 503 ``freshness-not-ready`` carrying exact
per-index progress, or, in ``partial`` mode, returns with
``freshness.status == "incomplete"``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from relata._http import AsyncHttpTransport, HttpTransport

if TYPE_CHECKING:
    import httpx

    from relata.client import RelataClient

INDEXES = ("lexical", "secondary", "vector", "graph", "extraction")


def freshness_headers(
    token: str,
    *,
    indexes: list[str] | None = None,
    wait_ms: int | None = None,
    require_durable: bool = False,
    partial: bool = False,
) -> dict[str, str]:
    """Headers that gate ``/query``, ``/search``, ``/graphql`` or MCP reads on ``token``."""
    headers = {"X-Relata-Min-Version": token}
    if indexes is not None:
        headers["X-Relata-Require-Index"] = ",".join(indexes) or "none"
    if wait_ms is not None:
        headers["X-Relata-Wait-Ms"] = str(int(wait_ms))
    if require_durable:
        headers["X-Relata-Require-Durable"] = "true"
    if partial:
        headers["X-Relata-Freshness-Mode"] = "partial"
    return headers


def _barrier_payload(
    token: str, indexes: list[str] | None, wait_ms: int | None, require_durable: bool
) -> dict[str, Any]:
    payload: dict[str, Any] = {"token": token}
    if indexes is not None:
        payload["indexes"] = list(indexes)
    if wait_ms is not None:
        payload["wait_ms"] = int(wait_ms)
    if require_durable:
        payload["require_durable"] = True
    return payload


class FreshnessClient:
    """Synchronous freshness client."""

    def __init__(
        self,
        base_url: str,
        *,
        bearer_token: str | None = None,
        timeout: float = 60.0,
        extra_headers: dict[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._t = HttpTransport(
            base_url, bearer_token, timeout, transport=transport, extra_headers=extra_headers
        )

    @classmethod
    def from_client(cls, client: RelataClient) -> FreshnessClient:
        return cls(
            client._base_url,  # noqa: SLF001
            bearer_token=client._bearer_token,  # noqa: SLF001
            timeout=client._timeout,  # noqa: SLF001
            extra_headers=client._extra_headers,  # noqa: SLF001
        )

    def status(self) -> dict[str, Any]:
        """``GET /freshness/status``: per-index state plus a token for the latest write."""
        return dict(self._t.get("/freshness/status"))

    def barrier(
        self,
        token: str,
        *,
        indexes: list[str] | None = None,
        wait_ms: int | None = None,
        require_durable: bool = False,
    ) -> dict[str, Any]:
        """``POST /freshness/barrier``: wait until ``token`` is visible in ``indexes``.

        Returns the freshness report on success. A write that is not yet visible
        within ``wait_ms`` (max 30000) raises with the server's ``freshness-not-ready``
        problem, whose body names the lagging index and its pending count.
        """
        return dict(
            self._t.post("/freshness/barrier", _barrier_payload(token, indexes, wait_ms, require_durable))
        )

    def close(self) -> None:
        self._t.close()

    def __enter__(self) -> FreshnessClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class AsyncFreshnessClient:
    """Asynchronous freshness client -- see :class:`FreshnessClient`."""

    def __init__(
        self,
        base_url: str,
        *,
        bearer_token: str | None = None,
        timeout: float = 60.0,
        extra_headers: dict[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._t = AsyncHttpTransport(
            base_url, bearer_token, timeout, transport=transport, extra_headers=extra_headers
        )

    @classmethod
    def from_client(cls, client: RelataClient) -> AsyncFreshnessClient:
        return cls(
            client._base_url,  # noqa: SLF001
            bearer_token=client._bearer_token,  # noqa: SLF001
            timeout=client._timeout,  # noqa: SLF001
            extra_headers=client._extra_headers,  # noqa: SLF001
        )

    async def status(self) -> dict[str, Any]:
        return dict(await self._t.get("/freshness/status"))

    async def barrier(
        self,
        token: str,
        *,
        indexes: list[str] | None = None,
        wait_ms: int | None = None,
        require_durable: bool = False,
    ) -> dict[str, Any]:
        return dict(
            await self._t.post(
                "/freshness/barrier", _barrier_payload(token, indexes, wait_ms, require_durable)
            )
        )

    async def close(self) -> None:
        await self._t.aclose()

    async def __aenter__(self) -> AsyncFreshnessClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()
