"""Retired financial MCP conveniences stay absent from both clients."""

import pytest

from relata.mcp import AsyncMcpClient, McpClient

RETIRED = ("trace_crypto", "beneficial_ownership", "reconstruct_wire", "trace_hawala")

def test_sync_financial_mcp_aliases_are_absent() -> None:
    client = McpClient("http://localhost")
    assert all(not hasattr(client, name) for name in RETIRED)

@pytest.mark.asyncio
async def test_async_financial_mcp_aliases_are_absent() -> None:
    async with AsyncMcpClient("http://localhost") as client:
        assert all(not hasattr(client, name) for name in RETIRED)
