"""Tenant self-introspection SDK.

Wraps ``GET /v1/tenants/me`` so a caller can read its own tenant record.
Tenant provisioning, quotas, sharing and platform administration are
Enterprise features and are not part of the public SDK.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from relata._http import AsyncHttpTransport, HttpTransport

if TYPE_CHECKING:
    import httpx

    from relata.client import RelataClient


class TenantAdminClient:
    """Synchronous tenant client -- ``me`` only."""

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
            base_url,
            bearer_token,
            timeout,
            transport=transport,
            extra_headers=extra_headers,
        )

    @classmethod
    def from_client(cls, client: RelataClient) -> TenantAdminClient:
        return cls(
            client._base_url,  # noqa: SLF001
            bearer_token=client._bearer_token,  # noqa: SLF001
            timeout=client._timeout,  # noqa: SLF001
            extra_headers=client._extra_headers,  # noqa: SLF001
        )

    def me(self) -> dict[str, Any]:
        """Return the caller's own tenant record (``GET /v1/tenants/me``)."""
        return self._t.get("/v1/tenants/me")

    def close(self) -> None:
        self._t.close()

    def __enter__(self) -> TenantAdminClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class AsyncTenantAdminClient:
    """Asynchronous tenant client -- see :class:`TenantAdminClient`."""

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
            base_url,
            bearer_token,
            timeout,
            transport=transport,
            extra_headers=extra_headers,
        )

    @classmethod
    def from_client(cls, client: RelataClient) -> AsyncTenantAdminClient:
        return cls(
            client._base_url,  # noqa: SLF001
            bearer_token=client._bearer_token,  # noqa: SLF001
            timeout=client._timeout,  # noqa: SLF001
            extra_headers=client._extra_headers,  # noqa: SLF001
        )

    async def me(self) -> dict[str, Any]:
        """Async variant of :meth:`TenantAdminClient.me`."""
        return await self._t.get("/v1/tenants/me")

    async def close(self) -> None:
        await self._t.aclose()

    async def __aenter__(self) -> AsyncTenantAdminClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()
