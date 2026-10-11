"""Client for the optional RIFN feed broker HTTP service.

The broker has its own address and lifecycle; this client deliberately does
not derive a URL or credentials from :class:`relata.client.RelataClient`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from relata._http import AsyncHttpTransport, HttpTransport

if TYPE_CHECKING:
    import httpx


class FeedBrokerClient:
    """Synchronous client for a separately deployed feed broker."""

    def __init__(
        self,
        broker_url: str,
        *,
        bearer_token: str | None = None,
        timeout: float = 30.0,
        extra_headers: dict[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._t = HttpTransport(
            broker_url, bearer_token, timeout, transport=transport, extra_headers=extra_headers
        )

    def health(self) -> dict[str, Any]:
        """Return broker health from ``GET /feed/health``."""
        return self._t.get("/feed/health")

    def channels(self) -> list[dict[str, Any]]:
        """List public broker channels from ``GET /feed/channels``."""
        return cast(list[dict[str, Any]], self._t.get("/feed/channels"))

    def publish(self, publisher_did: str, entry: dict[str, Any]) -> dict[str, Any]:
        """Publish a complete signed FeedEntry.

        The broker requires the caller's publisher DID in both the entry and
        ``X-Publisher-Did``. The SDK does not construct or sign the entry.
        """
        if not publisher_did.strip():
            raise ValueError("publisher_did is required")
        return self._t.post(
            "/feed/publish", entry, extra_headers={"X-Publisher-Did": publisher_did}
        )

    def close(self) -> None:
        self._t.close()

    def __enter__(self) -> FeedBrokerClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class AsyncFeedBrokerClient:
    """Asynchronous client for a separately deployed feed broker."""

    def __init__(
        self,
        broker_url: str,
        *,
        bearer_token: str | None = None,
        timeout: float = 30.0,
        extra_headers: dict[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._t = AsyncHttpTransport(
            broker_url, bearer_token, timeout, transport=transport, extra_headers=extra_headers
        )

    async def health(self) -> dict[str, Any]:
        return await self._t.get("/feed/health")

    async def channels(self) -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]], await self._t.get("/feed/channels"))

    async def publish(self, publisher_did: str, entry: dict[str, Any]) -> dict[str, Any]:
        if not publisher_did.strip():
            raise ValueError("publisher_did is required")
        return await self._t.post(
            "/feed/publish", entry, extra_headers={"X-Publisher-Did": publisher_did}
        )

    async def close(self) -> None:
        await self._t.aclose()

    async def __aenter__(self) -> AsyncFeedBrokerClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()
