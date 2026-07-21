"""HuggingFace smolagents adapter (#89).

Wraps :class:`relata.mcp.McpClient` as a smolagents ``Tool`` so a
``HfAgent`` can invoke any of the 30+ server MCP tools (query_knowledge,
lookup_identity, search_entities, etc.) by name.

Usage::

    from relata import RelataClient
    from relata_adapters.smolagents import RelataTool

    client = RelataClient("http://localhost:9090", purpose="analytics")
    tool = RelataTool(client)
    # Pass to an HfAgent's toolbox.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from relata.client import RelataClient


class RelataTool:
    """smolagents tool wrapper around :class:`~relata.mcp.McpClient`.

    This class is duck-typed to match the smolagents ``Tool`` protocol
    (``name``, ``description``, ``inputs``, ``__call__``). It does NOT import
    smolagents at module load — the caller passes it to the agent's toolbox,
    and the agent calls it as a plain callable.

    Args:
        client: A :class:`~relata.client.RelataClient` with auth + tenant
            pre-configured.
        tool_name: The MCP tool name to expose (default: ``"query_knowledge"``).
        description: Optional human-readable description for the agent.

    Example::

        from relata import RelataClient
        from relata_adapters.smolagents import RelataTool

        client = RelataClient("http://localhost:9090", purpose="analytics")
        query_tool = RelataTool(client, tool_name="query_knowledge")
        result = query_tool(sql="SELECT * FROM Person LIMIT 5")
    """

    def __init__(
        self,
        client: RelataClient,
        *,
        tool_name: str = "query_knowledge",
        description: str | None = None,
        transport: Any = None,  # noqa: ANN401
    ) -> None:
        from relata.mcp import McpClient

        if transport is not None:
            self._mcp = McpClient(
                client._base_url,  # noqa: SLF001
                bearer_token=client._bearer_token,  # noqa: SLF001
                extra_headers=client._extra_headers,  # noqa: SLF001
                transport=transport,
            )
        else:
            self._mcp = McpClient.from_client(client)
        self.name = tool_name
        self.description = description or f"Relata MCP tool: {tool_name}"
        self.inputs: dict[str, dict[str, str]] = {
            "kwargs": {"type": "object", "description": "Tool arguments as JSON"},
        }
        self.output_type = "string"
        # Keep a reference to the client for cleanup.
        self._client = client

    def __call__(self, **kwargs: Any) -> str:  # noqa: ANN401
        """Invoke the MCP tool with ``kwargs`` as the arguments map.

        Returns the JSON-stringified result so the agent can parse it.
        """
        import json

        result = self._mcp.call_tool(self.name, kwargs)
        return json.dumps(result)

    def close(self) -> None:
        self._mcp.close()
