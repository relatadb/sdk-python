"""Tenant self-service bearer-token SDK.

Wraps the caller's own auth-token routes:

- ``GET /v1/auth/tokens/self`` -- list the caller's own tenant-scoped tokens
- ``GET /v1/auth/tokens/self/audit`` -- audit trail of the caller's token lifecycle
- ``POST /v1/auth/tokens/self/rotate`` -- rotate the bearer token in use
- ``DELETE /v1/auth/tokens/self/{id}`` -- revoke one of the caller's own tokens

Minting and cluster-wide token administration live on the loopback admin
listener (``/admin/tokens``) and are intentionally not part of this client.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from relata._http import AsyncHttpTransport, HttpTransport, path_segment

if TYPE_CHECKING:
    import httpx

    from relata.client import RelataClient


class TokenClient:
    """Synchronous tenant self-service token client."""

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

    def list_self(self) -> dict[str, Any]:
        """List the caller's own tenant-scoped bearer tokens (``GET /v1/auth/tokens/self``)."""
        return self._t.get("/v1/auth/tokens/self")

    def self_audit(self) -> dict[str, Any]:
        """Audit trail of the caller's own token lifecycle events
        (``GET /v1/auth/tokens/self/audit``)."""
        return self._t.get("/v1/auth/tokens/self/audit")

    def rotate_self(self) -> dict[str, Any]:
        """Rotate the bearer token in use (``POST /v1/auth/tokens/self/rotate``).

        The response carries the fresh secret; the old token is invalidated
        immediately. Only per-tenant registry tokens may self-rotate.
        """
        return self._t.post("/v1/auth/tokens/self/rotate", {})

    def revoke_self(self, token_id: str) -> dict[str, Any]:
        """Revoke one of the caller's own tokens (``DELETE /v1/auth/tokens/self/{id}``)."""
        return self._t.delete(f"/v1/auth/tokens/self/{path_segment(token_id)}")

    def close(self) -> None:
        self._t.close()

    def __enter__(self) -> TokenClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class AsyncTokenClient:
    """Asynchronous tenant self-service token client -- see :class:`TokenClient`."""

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

    async def list_self(self) -> dict[str, Any]:
        """Async variant of :meth:`TokenClient.list_self`."""
        return await self._t.get("/v1/auth/tokens/self")

    async def self_audit(self) -> dict[str, Any]:
        """Async variant of :meth:`TokenClient.self_audit`."""
        return await self._t.get("/v1/auth/tokens/self/audit")

    async def rotate_self(self) -> dict[str, Any]:
        """Async variant of :meth:`TokenClient.rotate_self`."""
        return await self._t.post("/v1/auth/tokens/self/rotate", {})

    async def revoke_self(self, token_id: str) -> dict[str, Any]:
        """Async variant of :meth:`TokenClient.revoke_self`."""
        return await self._t.delete(f"/v1/auth/tokens/self/{path_segment(token_id)}")

    async def close(self) -> None:
        await self._t.aclose()

    async def __aenter__(self) -> AsyncTokenClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()
