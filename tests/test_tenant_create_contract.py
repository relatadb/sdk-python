"""Tenant client wire contract: only ``GET /v1/tenants/me`` remains public."""
import httpx
import pytest

from relata.tenants import AsyncTenantAdminClient, TenantAdminClient


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_tenant_me_wire_contract(asynchronous: bool) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.method == "GET"
        assert str(request.url) == "http://data/v1/tenants/me"
        return httpx.Response(200, json={"id": "acme"})

    transport = httpx.MockTransport(handler)
    if asynchronous:
        async with AsyncTenantAdminClient("http://data", transport=transport) as async_client:
            assert await async_client.me() == {"id": "acme"}
    else:
        with TenantAdminClient("http://data", transport=transport) as sync_client:
            assert sync_client.me() == {"id": "acme"}
    assert len(seen) == 1
