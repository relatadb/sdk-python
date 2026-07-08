"""MCP client — Model Context Protocol tool surface (#77 Phase 2).

Wraps ``/mcp/initialize``, ``/mcp/tools``, ``/mcp/tools/call``. The server
responds with the MCP envelope ``{"content": [{"type":"text","text":"..."}],
"isError": false}``; the :func:`unwrap_mcp` helper extracts the inner JSON.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from relata._http import AsyncHttpTransport, HttpTransport

if TYPE_CHECKING:
    import httpx

    from relata.client import RelataClient


def unwrap_mcp(resp: dict[str, Any]) -> dict[str, Any]:
    """Unwrap the MCP envelope. Extracted from ``relata.memory._unwrap`` so
    every MCP caller shares one implementation."""
    content = resp.get("content")
    if isinstance(content, list) and content:
        text = content[0].get("text") if isinstance(content[0], dict) else None
        if isinstance(text, str):
            try:
                inner = json.loads(text)
            except json.JSONDecodeError:
                return {"text": text}
            return inner if isinstance(inner, dict) else {"value": inner}
    return resp


class McpClient:
    """Synchronous MCP client — initialize / list_tools / call_tool."""

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
    def from_client(cls, client: RelataClient) -> McpClient:
        return cls(
            client._base_url,
            bearer_token=client._bearer_token,
            timeout=client._timeout,
            extra_headers=client._extra_headers,
        )

    def initialize(
        self,
        *,
        client_id: str = "relata-python-sdk",
        version: str = "1.0",
    ) -> dict[str, Any]:
        """Send the MCP initialize handshake."""
        return unwrap_mcp(
            self._t.post("/mcp/initialize", {"client_id": client_id, "version": version})
        )

    def list_tools(self) -> list[dict[str, Any]]:
        """List every MCP tool the server exposes (30+)."""
        data = unwrap_mcp(self._t.get("/mcp/tools"))
        tools = data.get("tools") if isinstance(data, dict) else data
        return tools if isinstance(tools, list) else []

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Invoke an MCP tool by name with a typed arguments map."""
        payload: dict[str, Any] = {"name": name}
        if arguments:
            payload["arguments"] = arguments
        return unwrap_mcp(self._t.post("/mcp/tools/call", payload))

    # ------------------------------------------------------------------
    # Convenience wrappers for the most-used MCP tools.
    # Server dispatch table: crates/relata-cli/src/serve/mcp.rs:212-245.
    # ------------------------------------------------------------------

    # --- Knowledge / query ---

    def query_knowledge(self, sql: str, *, purpose: str) -> dict[str, Any]:
        """``query`` / ``query_knowledge`` — governed SQL query."""
        return self.call_tool("query_knowledge", {"sql": sql, "purpose": purpose})

    def search_knowledge(
        self,
        query: str,
        *,
        purpose: str,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """``search_knowledge`` — hybrid BM25 + vector search."""
        return self.call_tool(
            "search_knowledge",
            {"query": query, "purpose": purpose, "top_k": top_k},
        )

    def explain_policy(self, sql: str, *, purpose: str) -> dict[str, Any]:
        """``explain_policy`` — show the ACL / org-isolation policy that would
        apply to ``sql`` without executing it."""
        return self.call_tool("explain_policy", {"sql": sql, "purpose": purpose})

    def suggest_extensions(self, prefix: str) -> dict[str, Any]:
        """``suggest_extensions`` — type/canonical-kind autocomplete."""
        return self.call_tool("suggest_extensions", {"prefix": prefix})

    # --- Entity / type discovery ---

    def list_entity_types(self) -> dict[str, Any]:
        """``list_entity_types`` — every registered ontology type."""
        return self.call_tool("list_entity_types", {})

    def get_entities(
        self,
        object_type: str,
        *,
        filters: dict[str, str] | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """``get_entities`` — paginated entity list.

        #253: the registered schema declares ``entity_type`` (not ``object_type``)
        and ``filters`` as a key-value object (not a ``filter`` string); sending
        the old names 400'd before dispatch.
        """
        args: dict[str, Any] = {"entity_type": object_type, "limit": limit}
        if filters:
            args["filters"] = filters
        return self.call_tool("get_entities", args)

    def search_entities(
        self, query: str, *, entity_types: list[str] | None = None
    ) -> dict[str, Any]:
        """``search_entities`` — free-text entity search.

        #253: the schema declares ``entity_types`` (an array), not ``object_type``.
        """
        args: dict[str, Any] = {"query": query}
        if entity_types:
            args["entity_types"] = entity_types
        return self.call_tool("search_entities", args)

    def get_domain_summary(self, domain: str) -> dict[str, Any]:
        """``get_domain_summary`` — counts + freshness for a domain.

        #253: the schema requires ``domain`` (enum: financial, telco, cyber,
        humint, narcotics, fara, maritime, border, sanctions, all), not
        ``object_type``.
        """
        return self.call_tool("get_domain_summary", {"domain": domain})

    def find_in_social_corpus(
        self,
        object_type: str,
        *,
        text_query: str | None = None,
        user: str | None = None,
        top_k: int = 20,
    ) -> dict[str, Any]:
        """``find_in_social_corpus`` — search the ingested social-media corpus.

        #253: the schema requires ``object_type`` (the post type) and takes
        ``text_query`` / ``user`` / ``top_k`` — not the old ``query`` / ``corpus``.
        """
        args: dict[str, Any] = {"object_type": object_type, "top_k": top_k}
        if text_query:
            args["text_query"] = text_query
        if user:
            args["user"] = user
        return self.call_tool("find_in_social_corpus", args)

    # --- Identity ---

    def lookup_identity(self, value: str, *, purpose: str = "analytics") -> dict[str, Any]:
        """``lookup_identity`` — universal identity lookup.

        #253: the schema declares the raw identifier under ``raw``, not ``value``.
        """
        return self.call_tool("lookup_identity", {"raw": value, "purpose": purpose})

    # --- Case / investigation ---

    def get_entity_profile(self, entity_id: str, *, purpose: str) -> dict[str, Any]:
        """``get_entity_profile`` — rich per-entity dossier.

        #253: the schema declares the entity under ``name``, not ``entity_id``.
        """
        return self.call_tool("get_entity_profile", {"name": entity_id, "purpose": purpose})

    def get_timeline(
        self,
        entity_id: str,
        *,
        purpose: str,
        since_ns: int | None = None,
        until_ns: int | None = None,
    ) -> dict[str, Any]:
        """``get_timeline`` — chronological event list for an entity.

        #253: the schema declares the entity under ``entity``, not ``entity_id``.
        """
        args: dict[str, Any] = {"entity": entity_id, "purpose": purpose}
        if since_ns is not None:
            args["since_ns"] = since_ns
        if until_ns is not None:
            args["until_ns"] = until_ns
        return self.call_tool("get_timeline", args)

    def find_connections(
        self,
        entity: str,
        *,
        purpose: str,
        limit: int = 50,
    ) -> dict[str, Any]:
        """``find_connections`` — surface entities connected to a target via
        relationships or shared attributes.

        Args:
            entity: The target entity ID to find connections for.
            purpose: Declared purpose token.
            limit: Max results (default 50, max 200).
        """
        return self.call_tool(
            "find_connections",
            {"entity": entity, "limit": limit, "purpose": purpose},
        )

    def get_relationships(
        self,
        entity_id: str,
        *,
        purpose: str,
        depth: int = 1,
    ) -> dict[str, Any]:
        """``get_relationships`` — direct neighbours of ``entity_id``."""
        return self.call_tool(
            "get_relationships",
            {"entity_id": entity_id, "depth": depth, "purpose": purpose},
        )

    def add_case_note(
        self,
        case_id: str,
        note: str,
        *,
        author: str | None = None,
    ) -> dict[str, Any]:
        """``add_case_note`` — append an investigative note to a case."""
        args: dict[str, Any] = {"case_id": case_id, "note": note}
        if author:
            args["author"] = author
        return self.call_tool("add_case_note", args)

    def get_audit_trail(
        self,
        *,
        case_id: str | None = None,
        entity_id: str | None = None,
    ) -> dict[str, Any]:
        """``get_audit_trail`` — provenance chain for a case or entity."""
        args: dict[str, Any] = {}
        if case_id:
            args["case_id"] = case_id
        if entity_id:
            args["entity_id"] = entity_id
        return self.call_tool("get_audit_trail", args)

    def get_case_summary(self, case_id: str, *, purpose: str) -> dict[str, Any]:
        """``get_case_summary`` — LLM-generated narrative summary of a case."""
        return self.call_tool("get_case_summary", {"case_id": case_id, "purpose": purpose})

    # --- RAG / ingest ---

    def rag_store_answer(
        self,
        question: str,
        answer: str,
        *,
        source_ids: list[str] | None = None,
        purpose: str,
    ) -> dict[str, Any]:
        """``rag_store_answer`` — persist a Q&A pair for downstream RAG."""
        args: dict[str, Any] = {"question": question, "answer": answer, "purpose": purpose}
        if source_ids:
            args["source_ids"] = source_ids
        return self.call_tool("rag_store_answer", args)

    def rag_store_elements(
        self,
        elements: list[dict[str, Any]],
        *,
        purpose: str,
    ) -> dict[str, Any]:
        """``rag_store_elements`` — bulk persist structured RAG elements."""
        return self.call_tool("rag_store_elements", {"elements": elements, "purpose": purpose})

    def ingest_document(
        self,
        chunks_jsonl: str,
        manifest_json: str,
        *,
        purpose: str,
    ) -> dict[str, Any]:
        """``ingest_document`` — datagrep-envelope document ingest via MCP."""
        return self.call_tool(
            "ingest_document",
            {"chunks_jsonl": chunks_jsonl, "manifest_json": manifest_json, "purpose": purpose},
        )

    # --- Memory (the 10 cognitive verbs are reachable via MCP too) ---
    # The dedicated Memory client is the typed surface; these MCP wrappers
    # exist for parity when an agent drives everything through /mcp/tools/call.

    def remember(
        self,
        content: str,
        *,
        purpose: str,
        confidence: float = 1.0,
        memory_class: str = "semantic",
    ) -> dict[str, Any]:
        """``remember`` MCP tool — store a memory (same shape as ``Memory.add``)."""
        return self.call_tool(
            "remember",
            {
                "content": content,
                "purpose": purpose,
                "confidence": confidence,
                "memory_class": memory_class,
            },
        )

    def recall(
        self,
        query: str,
        *,
        purpose: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """``recall`` MCP tool."""
        return self.call_tool("recall", {"query": query, "purpose": purpose, "top_k": top_k})

    def close(self) -> None:
        self._t.close()

    def __enter__(self) -> McpClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class AsyncMcpClient:
    """Asynchronous MCP client — see :class:`McpClient`."""

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
    def from_client(cls, client: RelataClient) -> AsyncMcpClient:
        return cls(
            client._base_url,
            bearer_token=client._bearer_token,
            timeout=client._timeout,
            extra_headers=client._extra_headers,
        )

    async def initialize(
        self, *, client_id: str = "relata-python-sdk", version: str = "1.0"
    ) -> dict[str, Any]:
        return unwrap_mcp(
            await self._t.post("/mcp/initialize", {"client_id": client_id, "version": version})
        )

    async def list_tools(self) -> list[dict[str, Any]]:
        data = unwrap_mcp(await self._t.get("/mcp/tools"))
        tools = data.get("tools") if isinstance(data, dict) else data
        return tools if isinstance(tools, list) else []

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": name}
        if arguments:
            payload["arguments"] = arguments
        return unwrap_mcp(await self._t.post("/mcp/tools/call", payload))

    async def close(self) -> None:
        await self._t.aclose()

    async def __aenter__(self) -> AsyncMcpClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()
