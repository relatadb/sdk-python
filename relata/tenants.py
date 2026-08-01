"""Tenant admin SDK (#73 — completes the tenant surface).

Wraps ``/tenants/*`` and ``/platform/tenants/*`` so a Python operator can
provision, suspend, reactivate, quota-manage, and configure sharing
agreements for tenants from code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from relata._http import AsyncHttpTransport, HttpTransport

if TYPE_CHECKING:
    import httpx

    from relata.client import RelataClient


class TenantAdminClient:
    """Synchronous tenant-admin client — CRUD + quota + sharing."""

    def __init__(
        self,
        base_url: str,
        *,
        bearer_token: str | None = None,
        timeout: float = 30.0,
        extra_headers: dict[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
        admin_base_url: str | None = None,
    ) -> None:
        # #2321 (ADR-0261): this client's routes are split across two planes
        # -- /tenants/* is an ordinary data-plane route, but /platform/* is
        # mounted only on the loopbound admin control-plane listener
        # (RELATA_ADMIN_BIND, default 127.0.0.1:9091) on a hardened
        # server/cluster deployment. Pass admin_base_url to route the
        # /platform/* methods there; /tenants/* methods are unaffected.
        # Unset (the default) preserves prior behaviour: every request goes
        # to base_url.
        self._t = HttpTransport(
            base_url,
            bearer_token,
            timeout,
            transport=transport,
            extra_headers=extra_headers,
            admin_base_url=admin_base_url,
        )

    @classmethod
    def from_client(cls, client: RelataClient) -> TenantAdminClient:
        return cls(
            client._base_url,  # noqa: SLF001
            bearer_token=client._bearer_token,  # noqa: SLF001
            timeout=client._timeout,  # noqa: SLF001
            extra_headers=client._extra_headers,  # noqa: SLF001
            admin_base_url=client._admin_base_url,  # noqa: SLF001
        )

    # ------------------------------------------------------------------
    # Tenant introspection
    # ------------------------------------------------------------------

    def me(self) -> dict[str, Any]:
        """Return the caller's own tenant record (``GET /tenants/me``)."""
        return self._t.get("/tenants/me")

    def usage(self) -> dict[str, Any]:
        """Return the caller's tenant usage (``GET /tenants/usage``)."""
        return self._t.get("/tenants/usage")

    # ------------------------------------------------------------------
    # Tenant CRUD (platform admin)
    # ------------------------------------------------------------------

    def create(
        self, tenant_id: str, *, tier: str = "standard", **fields: Any  # noqa: ANN401
    ) -> dict[str, Any]:
        """Provision a new tenant. Wraps ``POST /platform/tenants``."""
        payload: dict[str, Any] = {"tenant_id": tenant_id, "tier": tier, **fields}
        return self._t.post("/platform/tenants", payload)

    def get(self, tenant_id: str) -> dict[str, Any]:
        """Get a tenant's details. Wraps ``GET /platform/tenants/:id``."""
        return self._t.get(f"/platform/tenants/{tenant_id}")

    def delete(self, tenant_id: str) -> dict[str, Any]:
        """Delete a tenant. Wraps ``DELETE /platform/tenants/:id``."""
        return self._t.delete(f"/platform/tenants/{tenant_id}")

    def set_tier(self, tenant_id: str, tier: str) -> dict[str, Any]:
        """Change a tenant's tier. Wraps ``PATCH /platform/tenants/:id/tier``."""
        return self._t.patch(f"/platform/tenants/{tenant_id}/tier", {"tier": tier})

    def suspend(self, tenant_id: str) -> dict[str, Any]:
        """Suspend a tenant. Wraps ``POST /platform/tenants/:id/suspend``."""
        return self._t.post(f"/platform/tenants/{tenant_id}/suspend", {})

    def reactivate(self, tenant_id: str) -> dict[str, Any]:
        """Reactivate a suspended tenant."""
        return self._t.post(f"/platform/tenants/{tenant_id}/reactivate", {})

    # ------------------------------------------------------------------
    # Per-tenant quota + usage
    # ------------------------------------------------------------------

    def set_quota(self, tenant_id: str, quota: int) -> dict[str, Any]:
        """Set the cost-unit quota for a tenant. Wraps ``PUT /tenants/:id/quota``."""
        return self._t.put(f"/tenants/{tenant_id}/quota", {"quota": quota})

    def tenant_usage(self, tenant_id: str) -> dict[str, Any]:
        """Get a specific tenant's usage."""
        return self._t.get(f"/tenants/{tenant_id}/usage")

    # ------------------------------------------------------------------
    # Sharing agreements (ADR-132)
    # ------------------------------------------------------------------

    def list_sharing(self, tenant_id: str) -> list[dict[str, Any]]:
        """List the sharing agreements for a tenant."""
        data = self._t.get(f"/tenants/{tenant_id}/sharing")
        agreements = data.get("agreements") if isinstance(data, dict) else data
        return agreements if isinstance(agreements, list) else []

    def create_sharing(
        self,
        tenant_id: str,
        partner_tenant: str,
        *,
        object_type: str,
        action: str = "read",
        expires_ns: int | None = None,
    ) -> dict[str, Any]:
        """Register a SharingAgreement so ``tenant_id`` can read ``partner_tenant``'s
        rows of ``object_type`` (ADR-132 cross-org reads)."""
        payload: dict[str, Any] = {
            "partner_tenant": partner_tenant,
            "object_type": object_type,
            "action": action,
        }
        if expires_ns is not None:
            payload["expires_ns"] = expires_ns
        return self._t.post(f"/tenants/{tenant_id}/sharing", payload)

    # ------------------------------------------------------------------
    # Platform-wide
    # ------------------------------------------------------------------

    def platform_usage(self) -> dict[str, Any]:
        """Return platform-wide usage (all tenants). Wraps ``GET /platform/usage``."""
        return self._t.get("/platform/usage")

    def platform_license(self) -> dict[str, Any]:
        """Return the platform license info (tier, max_tenants, expiry)."""
        return self._t.get("/platform/license")

    def close(self) -> None:
        self._t.close()

    def __enter__(self) -> TenantAdminClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class AsyncTenantAdminClient:
    """Asynchronous tenant-admin client — see :class:`TenantAdminClient`."""

    def __init__(
        self,
        base_url: str,
        *,
        bearer_token: str | None = None,
        timeout: float = 30.0,
        extra_headers: dict[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        admin_base_url: str | None = None,
    ) -> None:
        # See TenantAdminClient.__init__ (#2321, ADR-0261).
        self._t = AsyncHttpTransport(
            base_url,
            bearer_token,
            timeout,
            transport=transport,
            extra_headers=extra_headers,
            admin_base_url=admin_base_url,
        )

    @classmethod
    def from_client(cls, client: RelataClient) -> AsyncTenantAdminClient:
        return cls(
            client._base_url,  # noqa: SLF001
            bearer_token=client._bearer_token,  # noqa: SLF001
            timeout=client._timeout,  # noqa: SLF001
            extra_headers=client._extra_headers,  # noqa: SLF001
            admin_base_url=client._admin_base_url,  # noqa: SLF001
        )

    async def me(self) -> dict[str, Any]:
        return await self._t.get("/tenants/me")

    async def usage(self) -> dict[str, Any]:
        return await self._t.get("/tenants/usage")

    async def create(
        self, tenant_id: str, *, tier: str = "standard", **fields: Any  # noqa: ANN401
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"tenant_id": tenant_id, "tier": tier, **fields}
        return await self._t.post("/platform/tenants", payload)

    async def get(self, tenant_id: str) -> dict[str, Any]:
        return await self._t.get(f"/platform/tenants/{tenant_id}")

    async def delete(self, tenant_id: str) -> dict[str, Any]:
        return await self._t.delete(f"/platform/tenants/{tenant_id}")

    async def set_tier(self, tenant_id: str, tier: str) -> dict[str, Any]:
        return await self._t.patch(f"/platform/tenants/{tenant_id}/tier", {"tier": tier})

    async def suspend(self, tenant_id: str) -> dict[str, Any]:
        return await self._t.post(f"/platform/tenants/{tenant_id}/suspend", {})

    async def reactivate(self, tenant_id: str) -> dict[str, Any]:
        return await self._t.post(f"/platform/tenants/{tenant_id}/reactivate", {})

    async def set_quota(self, tenant_id: str, quota: int) -> dict[str, Any]:
        return await self._t.put(f"/tenants/{tenant_id}/quota", {"quota": quota})

    async def tenant_usage(self, tenant_id: str) -> dict[str, Any]:
        return await self._t.get(f"/tenants/{tenant_id}/usage")

    async def list_sharing(self, tenant_id: str) -> list[dict[str, Any]]:
        data = await self._t.get(f"/tenants/{tenant_id}/sharing")
        agreements = data.get("agreements") if isinstance(data, dict) else data
        return agreements if isinstance(agreements, list) else []

    async def create_sharing(
        self,
        tenant_id: str,
        partner_tenant: str,
        *,
        object_type: str,
        action: str = "read",
        expires_ns: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "partner_tenant": partner_tenant,
            "object_type": object_type,
            "action": action,
        }
        if expires_ns is not None:
            payload["expires_ns"] = expires_ns
        return await self._t.post(f"/tenants/{tenant_id}/sharing", payload)

    async def platform_usage(self) -> dict[str, Any]:
        return await self._t.get("/platform/usage")

    async def platform_license(self) -> dict[str, Any]:
        return await self._t.get("/platform/license")

    async def close(self) -> None:
        await self._t.aclose()

    async def __aenter__(self) -> AsyncTenantAdminClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()
