"""Capture tenant-create wire payloads, including the asynchronous client (#5652)."""
import json

import httpx
import pytest

from relata.tenants import AsyncTenantAdminClient, TenantAdminClient


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("options", [{}, {"tier": "enterprise", "name": "Acme"}])
@pytest.mark.asyncio
async def test_tenant_create_wire_contract(asynchronous: bool, options: dict) -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.method == "POST"
        assert str(request.url) == "http://admin/platform/tenants"
        assert json.loads(request.content) == {"id": "acme", "tier": "standard", **options}
        return httpx.Response(201, json={"id": "acme"})

    args = {"admin_base_url": "http://admin", "transport": httpx.MockTransport(handler)}
    if asynchronous:
        async with AsyncTenantAdminClient("http://data", **args) as client:
            assert await client.create("acme", **options) == {"id": "acme"}
    else:
        with TenantAdminClient("http://data", **args) as client:
            assert client.create("acme", **options) == {"id": "acme"}
    assert len(seen) == 1
