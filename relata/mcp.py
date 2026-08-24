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
        # #341: the handler reads `limit` (mcp_tool_search_knowledge), so send
        # `limit` — the previous `top_k` key was silently dropped and the result
        # count fell back to the server default.
        return self.call_tool(
            "search_knowledge",
            {"query": query, "purpose": purpose, "limit": top_k},
        )

    def explain_policy(self, sql: str, *, purpose: str) -> dict[str, Any]:
        """``explain_policy`` — show the ACL / org-isolation policy that would
        apply to ``sql`` without executing it."""
        return self.call_tool("explain_policy", {"sql": sql, "purpose": purpose})

    def suggest_extensions(self) -> dict[str, Any]:
        """``suggest_extensions`` — list extension packs and their data
        availability. #341: takes no arguments (the handler ignores any) and
        returns pack suggestions, not type/kind autocomplete."""
        return self.call_tool("suggest_extensions", {})

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
        subject: str | None = None,
        *,
        purpose: str,
        predicate: str | None = None,
        object: str | None = None,  # noqa: A002 — matches the schema field name
        source: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """``get_relationships`` — governed (subject, predicate, object) triples.

        #253: the schema declares ``subject`` / ``predicate`` / ``object`` /
        ``source`` filters (not ``entity_id`` / ``depth`` — that neighbours-at-depth
        shape is ``find_connections``). Sending the old names 400'd before dispatch.
        """
        args: dict[str, Any] = {"purpose": purpose, "limit": limit}
        if subject:
            args["subject"] = subject
        if predicate:
            args["predicate"] = predicate
        if object:
            args["object"] = object
        if source:
            args["source"] = source
        return self.call_tool("get_relationships", args)

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
        purpose: str | None = None,
        principal_filter: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """``get_audit_trail`` — the governed audit trail.

        #253: the schema declares ``purpose`` / ``principal_filter`` / ``limit``
        (not ``case_id`` / ``entity_id`` — those are not audit-trail parameters).
        Sending the old names 400'd before dispatch.
        """
        args: dict[str, Any] = {"limit": limit}
        if purpose:
            args["purpose"] = purpose
        if principal_filter:
            args["principal_filter"] = principal_filter
        return self.call_tool("get_audit_trail", args)

    def get_case_summary(
        self,
        *,
        purpose: str,
        include_graph: bool | None = None,
        include_notes: bool | None = None,
        include_answers: bool | None = None,
    ) -> dict[str, Any]:
        """``get_case_summary`` — tenant-wide data inventory + knowledge-graph
        stats + analyst notes.

        #4658: NOT case-scoped — ``mcp_tool_get_case_summary``
        (``crates/relata-cli/src/serve/mcp.rs``) never reads a ``case_id``
        argument anywhere (confirmed by reading the full handler body), so
        the previously-required ``case_id`` parameter was dropped rather
        than kept as a misleading filter (same treatment as TS #4651).
        ``include_graph``/``include_notes``/``include_answers`` are real,
        server-read toggles (each defaults ``True`` server-side when
        omitted).
        """
        args: dict[str, Any] = {"purpose": purpose}
        if include_graph is not None:
            args["include_graph"] = include_graph
        if include_notes is not None:
            args["include_notes"] = include_notes
        if include_answers is not None:
            args["include_answers"] = include_answers
        return self.call_tool("get_case_summary", args)

    # --- RAG / ingest ---

    def rag_store_answer(
        self,
        question: str,
        answer: str,
        *,
        sources: list[dict[str, Any]] | None = None,
        purpose: str,
    ) -> dict[str, Any]:
        """``rag_store_answer`` — persist a Q&A pair for downstream RAG.

        #4659: the server's ``mcp_tool_rag_store_answer_with_gate``
        (``crates/relata-cli/src/serve/mcp/doc_writes.rs``) reads a
        ``sources`` array of objects (each with an ``id``/``url``,
        ``source``/``title``, and ``score``/``relevance`` field) — the
        previous ``source_ids`` key was never read anywhere in the handler,
        so citation data supplied that way was silently dropped.
        """
        args: dict[str, Any] = {"question": question, "answer": answer, "purpose": purpose}
        if sources:
            args["sources"] = sources
        return self.call_tool("rag_store_answer", args)

    def rag_store_elements(
        self,
        elements: list[dict[str, Any]],
        source_filename: str,
        *,
        purpose: str,
        source_sha256: str | None = None,
        label: str | None = None,
    ) -> dict[str, Any]:
        """``rag_store_elements`` — bulk persist structured RAG elements.

        #4660: the server's ``mcp_tool_rag_store_elements_with_gate``
        (``crates/relata-cli/src/serve/mcp/doc_writes.rs``) requires
        ``source_filename`` (no default) — every call 400ed without it.
        ``source_sha256``/``label`` mirror the handler's other optional
        fields.
        """
        args: dict[str, Any] = {
            "elements": elements,
            "source_filename": source_filename,
            "purpose": purpose,
        }
        if source_sha256 is not None:
            args["source_sha256"] = source_sha256
        if label is not None:
            args["label"] = label
        return self.call_tool("rag_store_elements", args)

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
        min_confidence: float | None = None,
        recency_half_life_secs: float | None = None,
        budget_tokens: int | None = None,
        stability_days: float | None = None,
        cancel_threshold: float | None = None,
    ) -> dict[str, Any]:
        """``recall`` MCP tool.

        The five keyword-only knobs are the ADR-145 retrieval-quality
        operators: ``min_confidence`` (CONFIDENCE), ``recency_half_life_secs``
        (RECENCY), ``budget_tokens`` (BUDGET), ``stability_days``
        (FORGETTING_CURVE), ``cancel_threshold`` (CANCEL_WHEN). Each defaults
        to ``None``, which omits it from the tool call so the server applies
        its own default. The raw envelope this returns carries the read-only
        ``recall_cost_tokens``/``cancelled`` fields once unwrapped (see
        :func:`unwrap_mcp`).
        """
        args: dict[str, Any] = {"query": query, "purpose": purpose, "top_k": top_k}
        if min_confidence is not None:
            args["min_confidence"] = min_confidence
        if recency_half_life_secs is not None:
            args["recency_half_life_secs"] = recency_half_life_secs
        if budget_tokens is not None:
            args["budget_tokens"] = budget_tokens
        if stability_days is not None:
            args["stability_days"] = stability_days
        if cancel_threshold is not None:
            args["cancel_threshold"] = cancel_threshold
        return self.call_tool("recall", args)

    def remember_batch(
        self, items: list[dict[str, Any]], *, purpose: str | None = None
    ) -> dict[str, Any]:
        """``remember_batch`` — bulk ``remember`` write (#2322)."""
        args: dict[str, Any] = {"items": items}
        if purpose:
            args["purpose"] = purpose
        return self.call_tool("remember_batch", args)

    def recognize(self, memory_id: str, *, purpose: str | None = None) -> dict[str, Any]:
        """``recognize`` — look up a stored memory item by id (#2322)."""
        args: dict[str, Any] = {"id": memory_id}
        if purpose:
            args["purpose"] = purpose
        return self.call_tool("recognize", args)

    def episodes_in(
        self, session_id: str, *, limit: int = 20, purpose: str | None = None
    ) -> dict[str, Any]:
        """``episodes_in`` — list Episodes within an AgentSession (#2322)."""
        args: dict[str, Any] = {"session_id": session_id, "limit": limit}
        if purpose:
            args["purpose"] = purpose
        return self.call_tool("episodes_in", args)

    def justify(self, memory_id: str, *, purpose: str | None = None) -> dict[str, Any]:
        """``justify`` — provenance/audit chain for a memory object (#2322)."""
        args: dict[str, Any] = {"id": memory_id}
        if purpose:
            args["purpose"] = purpose
        return self.call_tool("justify", args)

    def consolidate(
        self,
        memory_id: str,
        content: str,
        *,
        confidence: float = 1.0,
        purpose: str | None = None,
    ) -> dict[str, Any]:
        """``consolidate`` — supersede a memory item with an updated belief (#2322)."""
        args: dict[str, Any] = {"id": memory_id, "content": content, "confidence": confidence}
        if purpose:
            args["purpose"] = purpose
        return self.call_tool("consolidate", args)

    def forget(
        self, memory_id: str, *, retain_days: int = 0, purpose: str | None = None
    ) -> dict[str, Any]:
        """``forget`` — apply a retention policy to a memory item (#2322)."""
        args: dict[str, Any] = {"id": memory_id, "retain_days": retain_days}
        if purpose:
            args["purpose"] = purpose
        return self.call_tool("forget", args)

    def remember_procedure(
        self,
        agent_id: str,
        name: str,
        instruction_text: str,
        *,
        purpose: str | None = None,
    ) -> dict[str, Any]:
        """``remember_procedure`` — store a versioned agent procedure (T16/#2233, #2322)."""
        args: dict[str, Any] = {
            "agent_id": agent_id,
            "name": name,
            "instruction_text": instruction_text,
        }
        if purpose:
            args["purpose"] = purpose
        return self.call_tool("remember_procedure", args)

    def recall_procedure(
        self,
        agent_id: str,
        *,
        name: str | None = None,
        all_versions: bool = False,
        limit: int = 20,
        purpose: str | None = None,
    ) -> dict[str, Any]:
        """``recall_procedure`` — read back stored procedures for an agent (T16/#2233, #2322)."""
        args: dict[str, Any] = {"agent_id": agent_id, "all_versions": all_versions, "limit": limit}
        if name:
            args["name"] = name
        if purpose:
            args["purpose"] = purpose
        return self.call_tool("recall_procedure", args)

    def associate(
        self,
        from_id: str,
        to_id: str,
        *,
        relation: str = "related_to",
        purpose: str | None = None,
    ) -> dict[str, Any]:
        """``associate`` — link two memory items/entities with a typed relation (#2322)."""
        args: dict[str, Any] = {"from_id": from_id, "to_id": to_id, "relation": relation}
        if purpose:
            args["purpose"] = purpose
        return self.call_tool("associate", args)

    def resolve(self, memory_id: str, *, purpose: str | None = None) -> dict[str, Any]:
        """``resolve`` — follow a memory's supersession chain to its canonical head (#2322)."""
        args: dict[str, Any] = {"id": memory_id}
        if purpose:
            args["purpose"] = purpose
        return self.call_tool("resolve", args)

    def summarise(
        self,
        *,
        ids: list[str] | None = None,
        session_id: str | None = None,
        scope: str | None = None,
        contents: list[str] | None = None,
        purpose: str | None = None,
    ) -> dict[str, Any]:
        """``summarise`` — governed, provenance-stamped summary of a session/topic (#2322)."""
        args: dict[str, Any] = {}
        if ids:
            args["ids"] = ids
        if session_id:
            args["session_id"] = session_id
        if scope:
            args["scope"] = scope
        if contents:
            args["contents"] = contents
        if purpose:
            args["purpose"] = purpose
        return self.call_tool("summarise", args)

    def nl_query(
        self,
        query: str,
        *,
        purpose: str | None = None,
        interpret: bool = False,
        max_sub_questions: int | None = None,
    ) -> dict[str, Any]:
        """``nl_query`` — natural-language question translated to SQL and executed (#2322).

        A dialect router (#3267) classifies each question as SQL, Cypher, or a governed
        graph operator and prompts accordingly; the response carries a ``dialect`` field.
        Pass ``max_sub_questions > 1`` to decompose a multi-part question (e.g. "find X,
        and who X is linked to") into independently-routed sub-questions — the response
        then carries ``decomposed: true`` and a ``sub_results`` array instead of a single
        result. Capped at 5 server-side regardless of the value given.
        """
        args: dict[str, Any] = {"query": query, "interpret": interpret}
        if purpose:
            args["purpose"] = purpose
        if max_sub_questions is not None:
            args["max_sub_questions"] = max_sub_questions
        return self.call_tool("nl_query", args)

    def erase_subject(self, subject: str, *, reason: str = "gdpr-art17-request") -> dict[str, Any]:
        """``erase_subject`` — GDPR Art. 17 crypto-shred erasure with a signed receipt (#2322)."""
        return self.call_tool("erase_subject", {"subject": subject, "reason": reason})

    def ingest_media(
        self,
        object_type: str,
        *,
        modality: str = "image",
        codec: str | None = None,
        bytes_b64: str | None = None,
        text: str | None = None,
        tenant_id: str | None = None,
        partition_key: str | None = None,
    ) -> dict[str, Any]:
        """``ingest_media`` — ingest an image/audio/video (base64) or text payload (#2322)."""
        args: dict[str, Any] = {"object_type": object_type, "modality": modality}
        if codec:
            args["codec"] = codec
        if bytes_b64:
            args["bytes_b64"] = bytes_b64
        if text:
            args["text"] = text
        if tenant_id:
            args["tenant_id"] = tenant_id
        if partition_key:
            args["partition_key"] = partition_key
        return self.call_tool("ingest_media", args)

    def similar_multimodal(
        self,
        entity_type: str,
        entity_id: str,
        *,
        top_k: int = 10,
        modality: str = "text",
        purpose: str | None = None,
    ) -> dict[str, Any]:
        """``similar_multimodal`` — governed cross-modal similarity search (ADR-106, #2322)."""
        args: dict[str, Any] = {
            "entity_type": entity_type,
            "id": entity_id,
            "top_k": top_k,
            "modality": modality,
        }
        if purpose:
            args["purpose"] = purpose
        return self.call_tool("similar_multimodal", args)

    def hybrid_search(
        self,
        entity_type: str,
        query: str,
        *,
        top_k: int = 10,
        purpose: str | None = None,
    ) -> dict[str, Any]:
        """``hybrid_search`` — governed BM25 ⊕ vector retrieval fused via RRF (#2322)."""
        args: dict[str, Any] = {"entity_type": entity_type, "query": query, "top_k": top_k}
        if purpose:
            args["purpose"] = purpose
        return self.call_tool("hybrid_search", args)

    def paths_between(
        self,
        from_id: str,
        to_id: str,
        *,
        max_hops: int = 4,
        purpose: str | None = None,
    ) -> dict[str, Any]:
        """``paths_between`` — governed relationship/identity graph walk (#2322)."""
        args: dict[str, Any] = {"from": from_id, "to": to_id, "max_hops": max_hops}
        if purpose:
            args["purpose"] = purpose
        return self.call_tool("paths_between", args)

    def list_link_types(self) -> dict[str, Any]:
        """``list_link_types`` — every governed edge type defined in the ontology (#2322)."""
        return self.call_tool("list_link_types", {})

    def server_health(self) -> dict[str, Any]:
        """``server_health`` — readiness snapshot for an ops agent (#2322)."""
        return self.call_tool("server_health", {})

    def job_status(self) -> dict[str, Any]:
        """``job_status`` — list continuous detection jobs with live status (#2322)."""
        return self.call_tool("job_status", {})

    def metrics(self) -> dict[str, Any]:
        """``metrics`` — operational counters for a monitoring agent (#2322)."""
        return self.call_tool("metrics", {})

    def list_rules(self) -> dict[str, Any]:
        """``list_rules`` — list detection rules (ADR-162, #2322)."""
        return self.call_tool("list_rules", {})

    def create_rule(
        self,
        name: str,
        condition_sql: str,
        *,
        severity: str | None = None,
        description: str | None = None,
        purpose: str = "security",
    ) -> dict[str, Any]:
        """``create_rule`` — create a detection rule (ADR-162, #2322)."""
        args: dict[str, Any] = {
            "name": name,
            "condition_sql": condition_sql,
            "purpose": purpose,
        }
        if severity:
            args["severity"] = severity
        if description:
            args["description"] = description
        return self.call_tool("create_rule", args)

    def import_sigma(self, sigma_yaml: str, *, purpose: str = "security") -> dict[str, Any]:
        """``import_sigma`` — import a Sigma detection rule (YAML) (#2322)."""
        return self.call_tool("import_sigma", {"yaml": sigma_yaml, "purpose": purpose})

    def list_jobs(self) -> dict[str, Any]:
        """``list_jobs`` — list all registered detection jobs with live status (#2322)."""
        return self.call_tool("list_jobs", {})

    def schedule_job(self, name: str) -> dict[str, Any]:
        """``schedule_job`` — trigger an immediate run of a named detection job (#2322)."""
        return self.call_tool("schedule_job", {"name": name})

    def list_workflows(self) -> dict[str, Any]:
        """``list_workflows`` — list all registered workflow definitions (#2322)."""
        return self.call_tool("list_workflows", {})

    def run_workflow(self, name: str) -> dict[str, Any]:
        """``run_workflow`` — start a workflow execution by name (#2322)."""
        return self.call_tool("run_workflow", {"name": name})

    def workflow_status(self, run_id: str) -> dict[str, Any]:
        """``workflow_status`` — query step-level status of a workflow run (#2322)."""
        return self.call_tool("workflow_status", {"run_id": run_id})

    def trace_crypto(
        self,
        address: str,
        *,
        max_hops: int = 5,
        min_amount: float = 0,
        purpose: str = "analytics",
    ) -> dict[str, Any]:
        """``trace_crypto`` — follow a cryptocurrency address hop-by-hop (#2322)."""
        return self.call_tool(
            "trace_crypto",
            {
                "address": address,
                "max_hops": max_hops,
                "min_amount": min_amount,
                "purpose": purpose,
            },
        )

    def beneficial_ownership(
        self, party: str, *, max_depth: int = 6, purpose: str = "analytics"
    ) -> dict[str, Any]:
        """``beneficial_ownership`` — trace the beneficial ownership chain for a party (#2322)."""
        return self.call_tool(
            "beneficial_ownership",
            {"party": party, "max_depth": max_depth, "purpose": purpose},
        )

    def reconstruct_wire(
        self, account: str, *, tolerance_pct: float = 5.0, purpose: str = "analytics"
    ) -> dict[str, Any]:
        """``reconstruct_wire`` — reconstruct a wire-transfer chain for an account (#2322)."""
        return self.call_tool(
            "reconstruct_wire",
            {"account": account, "tolerance_pct": tolerance_pct, "purpose": purpose},
        )

    def trace_hawala(
        self, seed: str, *, max_hops: int = 5, purpose: str = "analytics"
    ) -> dict[str, Any]:
        """``trace_hawala`` — trace informal hawala value-transfer networks (#2322)."""
        return self.call_tool(
            "trace_hawala", {"seed": seed, "max_hops": max_hops, "purpose": purpose}
        )

    def geofence(
        self,
        lat: float,
        lon: float,
        *,
        radius_m: float = 1000,
        target_type: str = "MovementEvent",
        purpose: str = "analytics",
    ) -> dict[str, Any]:
        """``geofence`` — spatial fence query over a circular area (#2322)."""
        return self.call_tool(
            "geofence",
            {
                "lat": lat,
                "lon": lon,
                "radius_m": radius_m,
                "target_type": target_type,
                "purpose": purpose,
            },
        )

    def resolve_entity_identity(
        self, identity: str, *, purpose: str = "analytics"
    ) -> dict[str, Any]:
        """``resolve_entity_identity`` — resolve the canonical identity cluster (#2322)."""
        return self.call_tool(
            "resolve_entity_identity", {"identity": identity, "purpose": purpose}
        )

    def detect_communities(
        self, entity_type: str, *, algo: str = "louvain", purpose: str = "analytics"
    ) -> dict[str, Any]:
        """``detect_communities`` — Louvain/Leiden community detection (#2322)."""
        return self.call_tool(
            "detect_communities",
            {"entity_type": entity_type, "algo": algo, "purpose": purpose},
        )

    def rank_key_nodes(
        self,
        entity_type: str,
        *,
        metric: str = "pagerank",
        damping: float = 0.85,
        max_iter: int = 20,
        purpose: str = "analytics",
    ) -> dict[str, Any]:
        """``rank_key_nodes`` — rank entities by PageRank or centrality metric (#2322)."""
        return self.call_tool(
            "rank_key_nodes",
            {
                "entity_type": entity_type,
                "metric": metric,
                "damping": damping,
                "max_iter": max_iter,
                "purpose": purpose,
            },
        )

    def hub_authority(
        self, entity_type: str, *, max_iter: int = 20, purpose: str = "analytics"
    ) -> dict[str, Any]:
        """``hub_authority`` — HITS hub/authority scores for an entity type (#2322)."""
        return self.call_tool(
            "hub_authority",
            {"entity_type": entity_type, "max_iter": max_iter, "purpose": purpose},
        )

    def predict_links(
        self,
        entity_type: str,
        *,
        from_id: str | None = None,
        to_id: str | None = None,
        method: str = "common_neighbors",
        purpose: str = "analytics",
    ) -> dict[str, Any]:
        """``predict_links`` — score candidate edges between entities (#2322)."""
        args: dict[str, Any] = {"entity_type": entity_type, "method": method, "purpose": purpose}
        if from_id:
            args["from_id"] = from_id
        if to_id:
            args["to_id"] = to_id
        return self.call_tool("predict_links", args)

    def find_scc(self, entity_type: str, *, purpose: str = "analytics") -> dict[str, Any]:
        """``find_scc`` — find strongly connected components in an entity type's graph (#2322)."""
        return self.call_tool("find_scc", {"entity_type": entity_type, "purpose": purpose})

    def screen_sanctions(
        self,
        name: str,
        *,
        threshold: float | None = None,
        purpose: str = "compliance_review",
    ) -> dict[str, Any]:
        """``screen_sanctions`` — screen a name/entity against sanctions lists (#2322)."""
        args: dict[str, Any] = {"name": name, "purpose": purpose}
        if threshold is not None:
            args["threshold"] = threshold
        return self.call_tool("screen_sanctions", args)

    def aggregate_stats(
        self,
        entity_type: str,
        *,
        agg: str = "COUNT",
        column: str = "*",
        purpose: str = "analytics",
    ) -> dict[str, Any]:
        """``aggregate_stats`` — governed aggregate query over an entity type (#2322)."""
        return self.call_tool(
            "aggregate_stats",
            {"entity_type": entity_type, "agg": agg, "column": column, "purpose": purpose},
        )

    def investigate_entity(
        self, entity_type: str, entity_id: str, *, purpose: str = "security_incident"
    ) -> dict[str, Any]:
        """``investigate_entity`` — composite profile + timeline + connections + risk (#2322)."""
        return self.call_tool(
            "investigate_entity",
            {"entity_type": entity_type, "id": entity_id, "purpose": purpose},
        )

    def find_threats(
        self, entity_type: str, *, purpose: str = "security_incident"
    ) -> dict[str, Any]:
        """``find_threats`` — composite threat hunt over an entity type (#2322)."""
        return self.call_tool(
            "find_threats", {"entity_type": entity_type, "purpose": purpose}
        )

    def search_video_frames(
        self,
        query_id: str,
        *,
        media_type: str = "VideoFrame",
        top_k: int = 20,
        purpose: str = "security_incident",
    ) -> dict[str, Any]:
        """``search_video_frames`` — similarity search over video frames/segments (#2322)."""
        return self.call_tool(
            "search_video_frames",
            {
                "query_id": query_id,
                "media_type": media_type,
                "top_k": top_k,
                "purpose": purpose,
            },
        )

    def face_match(
        self,
        probe_id: str,
        *,
        threshold: float = 0.8,
        top_k: int = 10,
        purpose: str = "security_incident",
    ) -> dict[str, Any]:
        """``face_match`` — GATED: match a probe face against indexed media (#2322)."""
        return self.call_tool(
            "face_match",
            {"probe_id": probe_id, "threshold": threshold, "top_k": top_k, "purpose": purpose},
        )

    def close(self) -> None:
        self._t.close()

    def __enter__(self) -> McpClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class AsyncMcpClient:
    """Asynchronous MCP client — see :class:`McpClient`.

    Mirrors every sync convenience wrapper on :class:`McpClient` (#2757) in
    addition to the three core methods (``initialize``/``list_tools``/
    ``call_tool``) — the same method name, `async def`, same arguments.
    """

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

    # ------------------------------------------------------------------
    # Async mirrors of every McpClient convenience wrapper (#2757).
    # Same argument shapes / server tool-name mapping as the sync class.
    # ------------------------------------------------------------------

    # --- Knowledge / query ---

    async def query_knowledge(self, sql: str, *, purpose: str) -> dict[str, Any]:
        """``query`` / ``query_knowledge`` — governed SQL query."""
        return await self.call_tool("query_knowledge", {"sql": sql, "purpose": purpose})

    async def search_knowledge(
        self,
        query: str,
        *,
        purpose: str,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """``search_knowledge`` — hybrid BM25 + vector search."""
        return await self.call_tool(
            "search_knowledge",
            {"query": query, "purpose": purpose, "limit": top_k},
        )

    async def explain_policy(self, sql: str, *, purpose: str) -> dict[str, Any]:
        """``explain_policy`` — show the ACL / org-isolation policy that would
        apply to ``sql`` without executing it."""
        return await self.call_tool("explain_policy", {"sql": sql, "purpose": purpose})

    async def suggest_extensions(self) -> dict[str, Any]:
        """``suggest_extensions`` — list extension packs and their data
        availability."""
        return await self.call_tool("suggest_extensions", {})

    # --- Entity / type discovery ---

    async def list_entity_types(self) -> dict[str, Any]:
        """``list_entity_types`` — every registered ontology type."""
        return await self.call_tool("list_entity_types", {})

    async def get_entities(
        self,
        object_type: str,
        *,
        filters: dict[str, str] | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """``get_entities`` — paginated entity list."""
        args: dict[str, Any] = {"entity_type": object_type, "limit": limit}
        if filters:
            args["filters"] = filters
        return await self.call_tool("get_entities", args)

    async def search_entities(
        self, query: str, *, entity_types: list[str] | None = None
    ) -> dict[str, Any]:
        """``search_entities`` — free-text entity search."""
        args: dict[str, Any] = {"query": query}
        if entity_types:
            args["entity_types"] = entity_types
        return await self.call_tool("search_entities", args)

    async def get_domain_summary(self, domain: str) -> dict[str, Any]:
        """``get_domain_summary`` — counts + freshness for a domain."""
        return await self.call_tool("get_domain_summary", {"domain": domain})

    async def find_in_social_corpus(
        self,
        object_type: str,
        *,
        text_query: str | None = None,
        user: str | None = None,
        top_k: int = 20,
    ) -> dict[str, Any]:
        """``find_in_social_corpus`` — search the ingested social-media corpus."""
        args: dict[str, Any] = {"object_type": object_type, "top_k": top_k}
        if text_query:
            args["text_query"] = text_query
        if user:
            args["user"] = user
        return await self.call_tool("find_in_social_corpus", args)

    # --- Identity ---

    async def lookup_identity(self, value: str, *, purpose: str = "analytics") -> dict[str, Any]:
        """``lookup_identity`` — universal identity lookup."""
        return await self.call_tool("lookup_identity", {"raw": value, "purpose": purpose})

    # --- Case / investigation ---

    async def get_entity_profile(self, entity_id: str, *, purpose: str) -> dict[str, Any]:
        """``get_entity_profile`` — rich per-entity dossier."""
        return await self.call_tool("get_entity_profile", {"name": entity_id, "purpose": purpose})

    async def get_timeline(
        self,
        entity_id: str,
        *,
        purpose: str,
        since_ns: int | None = None,
        until_ns: int | None = None,
    ) -> dict[str, Any]:
        """``get_timeline`` — chronological event list for an entity."""
        args: dict[str, Any] = {"entity": entity_id, "purpose": purpose}
        if since_ns is not None:
            args["since_ns"] = since_ns
        if until_ns is not None:
            args["until_ns"] = until_ns
        return await self.call_tool("get_timeline", args)

    async def find_connections(
        self,
        entity: str,
        *,
        purpose: str,
        limit: int = 50,
    ) -> dict[str, Any]:
        """``find_connections`` — surface entities connected to a target via
        relationships or shared attributes."""
        return await self.call_tool(
            "find_connections",
            {"entity": entity, "limit": limit, "purpose": purpose},
        )

    async def get_relationships(
        self,
        subject: str | None = None,
        *,
        purpose: str,
        predicate: str | None = None,
        object: str | None = None,  # noqa: A002 — matches the schema field name
        source: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """``get_relationships`` — governed (subject, predicate, object) triples."""
        args: dict[str, Any] = {"purpose": purpose, "limit": limit}
        if subject:
            args["subject"] = subject
        if predicate:
            args["predicate"] = predicate
        if object:
            args["object"] = object
        if source:
            args["source"] = source
        return await self.call_tool("get_relationships", args)

    async def add_case_note(
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
        return await self.call_tool("add_case_note", args)

    async def get_audit_trail(
        self,
        *,
        purpose: str | None = None,
        principal_filter: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """``get_audit_trail`` — the governed audit trail."""
        args: dict[str, Any] = {"limit": limit}
        if purpose:
            args["purpose"] = purpose
        if principal_filter:
            args["principal_filter"] = principal_filter
        return await self.call_tool("get_audit_trail", args)

    async def get_case_summary(
        self,
        *,
        purpose: str,
        include_graph: bool | None = None,
        include_notes: bool | None = None,
        include_answers: bool | None = None,
    ) -> dict[str, Any]:
        """``get_case_summary`` — tenant-wide data inventory + knowledge-graph
        stats + analyst notes.

        #4658: NOT case-scoped — ``mcp_tool_get_case_summary``
        (``crates/relata-cli/src/serve/mcp.rs``) never reads a ``case_id``
        argument anywhere (confirmed by reading the full handler body), so
        the previously-required ``case_id`` parameter was dropped rather
        than kept as a misleading filter (same treatment as TS #4651).
        ``include_graph``/``include_notes``/``include_answers`` are real,
        server-read toggles (each defaults ``True`` server-side when
        omitted).
        """
        args: dict[str, Any] = {"purpose": purpose}
        if include_graph is not None:
            args["include_graph"] = include_graph
        if include_notes is not None:
            args["include_notes"] = include_notes
        if include_answers is not None:
            args["include_answers"] = include_answers
        return await self.call_tool("get_case_summary", args)

    # --- RAG / ingest ---

    async def rag_store_answer(
        self,
        question: str,
        answer: str,
        *,
        sources: list[dict[str, Any]] | None = None,
        purpose: str,
    ) -> dict[str, Any]:
        """``rag_store_answer`` — persist a Q&A pair for downstream RAG.

        #4659: the server's ``mcp_tool_rag_store_answer_with_gate``
        (``crates/relata-cli/src/serve/mcp/doc_writes.rs``) reads a
        ``sources`` array of objects (each with an ``id``/``url``,
        ``source``/``title``, and ``score``/``relevance`` field) — the
        previous ``source_ids`` key was never read anywhere in the handler,
        so citation data supplied that way was silently dropped.
        """
        args: dict[str, Any] = {"question": question, "answer": answer, "purpose": purpose}
        if sources:
            args["sources"] = sources
        return await self.call_tool("rag_store_answer", args)

    async def rag_store_elements(
        self,
        elements: list[dict[str, Any]],
        source_filename: str,
        *,
        purpose: str,
        source_sha256: str | None = None,
        label: str | None = None,
    ) -> dict[str, Any]:
        """``rag_store_elements`` — bulk persist structured RAG elements.

        #4660: the server's ``mcp_tool_rag_store_elements_with_gate``
        (``crates/relata-cli/src/serve/mcp/doc_writes.rs``) requires
        ``source_filename`` (no default) — every call 400ed without it.
        ``source_sha256``/``label`` mirror the handler's other optional
        fields.
        """
        args: dict[str, Any] = {
            "elements": elements,
            "source_filename": source_filename,
            "purpose": purpose,
        }
        if source_sha256 is not None:
            args["source_sha256"] = source_sha256
        if label is not None:
            args["label"] = label
        return await self.call_tool("rag_store_elements", args)

    async def ingest_document(
        self,
        chunks_jsonl: str,
        manifest_json: str,
        *,
        purpose: str,
    ) -> dict[str, Any]:
        """``ingest_document`` — datagrep-envelope document ingest via MCP."""
        return await self.call_tool(
            "ingest_document",
            {"chunks_jsonl": chunks_jsonl, "manifest_json": manifest_json, "purpose": purpose},
        )

    # --- Memory (the 10 cognitive verbs are reachable via MCP too) ---

    async def remember(
        self,
        content: str,
        *,
        purpose: str,
        confidence: float = 1.0,
        memory_class: str = "semantic",
    ) -> dict[str, Any]:
        """``remember`` MCP tool — store a memory (same shape as ``Memory.add``)."""
        return await self.call_tool(
            "remember",
            {
                "content": content,
                "purpose": purpose,
                "confidence": confidence,
                "memory_class": memory_class,
            },
        )

    async def recall(
        self,
        query: str,
        *,
        purpose: str,
        top_k: int = 5,
        min_confidence: float | None = None,
        recency_half_life_secs: float | None = None,
        budget_tokens: int | None = None,
        stability_days: float | None = None,
        cancel_threshold: float | None = None,
    ) -> dict[str, Any]:
        """``recall`` MCP tool. See :meth:`McpClient.recall` for the five
        ADR-145 retrieval-quality keyword knobs."""
        args: dict[str, Any] = {"query": query, "purpose": purpose, "top_k": top_k}
        if min_confidence is not None:
            args["min_confidence"] = min_confidence
        if recency_half_life_secs is not None:
            args["recency_half_life_secs"] = recency_half_life_secs
        if budget_tokens is not None:
            args["budget_tokens"] = budget_tokens
        if stability_days is not None:
            args["stability_days"] = stability_days
        if cancel_threshold is not None:
            args["cancel_threshold"] = cancel_threshold
        return await self.call_tool("recall", args)

    async def remember_batch(
        self, items: list[dict[str, Any]], *, purpose: str | None = None
    ) -> dict[str, Any]:
        """``remember_batch`` — bulk ``remember`` write (#2322)."""
        args: dict[str, Any] = {"items": items}
        if purpose:
            args["purpose"] = purpose
        return await self.call_tool("remember_batch", args)

    async def recognize(self, memory_id: str, *, purpose: str | None = None) -> dict[str, Any]:
        """``recognize`` — look up a stored memory item by id (#2322)."""
        args: dict[str, Any] = {"id": memory_id}
        if purpose:
            args["purpose"] = purpose
        return await self.call_tool("recognize", args)

    async def episodes_in(
        self, session_id: str, *, limit: int = 20, purpose: str | None = None
    ) -> dict[str, Any]:
        """``episodes_in`` — list Episodes within an AgentSession (#2322)."""
        args: dict[str, Any] = {"session_id": session_id, "limit": limit}
        if purpose:
            args["purpose"] = purpose
        return await self.call_tool("episodes_in", args)

    async def justify(self, memory_id: str, *, purpose: str | None = None) -> dict[str, Any]:
        """``justify`` — provenance/audit chain for a memory object (#2322)."""
        args: dict[str, Any] = {"id": memory_id}
        if purpose:
            args["purpose"] = purpose
        return await self.call_tool("justify", args)

    async def consolidate(
        self,
        memory_id: str,
        content: str,
        *,
        confidence: float = 1.0,
        purpose: str | None = None,
    ) -> dict[str, Any]:
        """``consolidate`` — supersede a memory item with an updated belief (#2322)."""
        args: dict[str, Any] = {"id": memory_id, "content": content, "confidence": confidence}
        if purpose:
            args["purpose"] = purpose
        return await self.call_tool("consolidate", args)

    async def forget(
        self, memory_id: str, *, retain_days: int = 0, purpose: str | None = None
    ) -> dict[str, Any]:
        """``forget`` — apply a retention policy to a memory item (#2322)."""
        args: dict[str, Any] = {"id": memory_id, "retain_days": retain_days}
        if purpose:
            args["purpose"] = purpose
        return await self.call_tool("forget", args)

    async def remember_procedure(
        self,
        agent_id: str,
        name: str,
        instruction_text: str,
        *,
        purpose: str | None = None,
    ) -> dict[str, Any]:
        """``remember_procedure`` — store a versioned agent procedure (T16/#2233, #2322)."""
        args: dict[str, Any] = {
            "agent_id": agent_id,
            "name": name,
            "instruction_text": instruction_text,
        }
        if purpose:
            args["purpose"] = purpose
        return await self.call_tool("remember_procedure", args)

    async def recall_procedure(
        self,
        agent_id: str,
        *,
        name: str | None = None,
        all_versions: bool = False,
        limit: int = 20,
        purpose: str | None = None,
    ) -> dict[str, Any]:
        """``recall_procedure`` — read back stored procedures for an agent (T16/#2233, #2322)."""
        args: dict[str, Any] = {"agent_id": agent_id, "all_versions": all_versions, "limit": limit}
        if name:
            args["name"] = name
        if purpose:
            args["purpose"] = purpose
        return await self.call_tool("recall_procedure", args)

    async def associate(
        self,
        from_id: str,
        to_id: str,
        *,
        relation: str = "related_to",
        purpose: str | None = None,
    ) -> dict[str, Any]:
        """``associate`` — link two memory items/entities with a typed relation (#2322)."""
        args: dict[str, Any] = {"from_id": from_id, "to_id": to_id, "relation": relation}
        if purpose:
            args["purpose"] = purpose
        return await self.call_tool("associate", args)

    async def resolve(self, memory_id: str, *, purpose: str | None = None) -> dict[str, Any]:
        """``resolve`` — follow a memory's supersession chain to its canonical head (#2322)."""
        args: dict[str, Any] = {"id": memory_id}
        if purpose:
            args["purpose"] = purpose
        return await self.call_tool("resolve", args)

    async def summarise(
        self,
        *,
        ids: list[str] | None = None,
        session_id: str | None = None,
        scope: str | None = None,
        contents: list[str] | None = None,
        purpose: str | None = None,
    ) -> dict[str, Any]:
        """``summarise`` — governed, provenance-stamped summary of a session/topic (#2322)."""
        args: dict[str, Any] = {}
        if ids:
            args["ids"] = ids
        if session_id:
            args["session_id"] = session_id
        if scope:
            args["scope"] = scope
        if contents:
            args["contents"] = contents
        if purpose:
            args["purpose"] = purpose
        return await self.call_tool("summarise", args)

    async def nl_query(
        self,
        query: str,
        *,
        purpose: str | None = None,
        interpret: bool = False,
        max_sub_questions: int | None = None,
    ) -> dict[str, Any]:
        """``nl_query`` — natural-language question translated to SQL and executed (#2322).

        A dialect router (#3267) classifies each question as SQL, Cypher, or a governed
        graph operator and prompts accordingly; the response carries a ``dialect`` field.
        Pass ``max_sub_questions > 1`` to decompose a multi-part question (e.g. "find X,
        and who X is linked to") into independently-routed sub-questions — the response
        then carries ``decomposed: true`` and a ``sub_results`` array instead of a single
        result. Capped at 5 server-side regardless of the value given.
        """
        args: dict[str, Any] = {"query": query, "interpret": interpret}
        if purpose:
            args["purpose"] = purpose
        if max_sub_questions is not None:
            args["max_sub_questions"] = max_sub_questions
        return await self.call_tool("nl_query", args)

    async def erase_subject(
        self, subject: str, *, reason: str = "gdpr-art17-request"
    ) -> dict[str, Any]:
        """``erase_subject`` — GDPR Art. 17 crypto-shred erasure with a signed receipt (#2322)."""
        return await self.call_tool("erase_subject", {"subject": subject, "reason": reason})

    async def ingest_media(
        self,
        object_type: str,
        *,
        modality: str = "image",
        codec: str | None = None,
        bytes_b64: str | None = None,
        text: str | None = None,
        tenant_id: str | None = None,
        partition_key: str | None = None,
    ) -> dict[str, Any]:
        """``ingest_media`` — ingest an image/audio/video (base64) or text payload (#2322)."""
        args: dict[str, Any] = {"object_type": object_type, "modality": modality}
        if codec:
            args["codec"] = codec
        if bytes_b64:
            args["bytes_b64"] = bytes_b64
        if text:
            args["text"] = text
        if tenant_id:
            args["tenant_id"] = tenant_id
        if partition_key:
            args["partition_key"] = partition_key
        return await self.call_tool("ingest_media", args)

    async def similar_multimodal(
        self,
        entity_type: str,
        entity_id: str,
        *,
        top_k: int = 10,
        modality: str = "text",
        purpose: str | None = None,
    ) -> dict[str, Any]:
        """``similar_multimodal`` — governed cross-modal similarity search (ADR-106, #2322)."""
        args: dict[str, Any] = {
            "entity_type": entity_type,
            "id": entity_id,
            "top_k": top_k,
            "modality": modality,
        }
        if purpose:
            args["purpose"] = purpose
        return await self.call_tool("similar_multimodal", args)

    async def hybrid_search(
        self,
        entity_type: str,
        query: str,
        *,
        top_k: int = 10,
        purpose: str | None = None,
    ) -> dict[str, Any]:
        """``hybrid_search`` — governed BM25 ⊕ vector retrieval fused via RRF (#2322)."""
        args: dict[str, Any] = {"entity_type": entity_type, "query": query, "top_k": top_k}
        if purpose:
            args["purpose"] = purpose
        return await self.call_tool("hybrid_search", args)

    async def paths_between(
        self,
        from_id: str,
        to_id: str,
        *,
        max_hops: int = 4,
        purpose: str | None = None,
    ) -> dict[str, Any]:
        """``paths_between`` — governed relationship/identity graph walk (#2322)."""
        args: dict[str, Any] = {"from": from_id, "to": to_id, "max_hops": max_hops}
        if purpose:
            args["purpose"] = purpose
        return await self.call_tool("paths_between", args)

    async def list_link_types(self) -> dict[str, Any]:
        """``list_link_types`` — every governed edge type defined in the ontology (#2322)."""
        return await self.call_tool("list_link_types", {})

    async def server_health(self) -> dict[str, Any]:
        """``server_health`` — readiness snapshot for an ops agent (#2322)."""
        return await self.call_tool("server_health", {})

    async def job_status(self) -> dict[str, Any]:
        """``job_status`` — list continuous detection jobs with live status (#2322)."""
        return await self.call_tool("job_status", {})

    async def metrics(self) -> dict[str, Any]:
        """``metrics`` — operational counters for a monitoring agent (#2322)."""
        return await self.call_tool("metrics", {})

    async def list_rules(self) -> dict[str, Any]:
        """``list_rules`` — list detection rules (ADR-162, #2322)."""
        return await self.call_tool("list_rules", {})

    async def create_rule(
        self,
        name: str,
        condition_sql: str,
        *,
        severity: str | None = None,
        description: str | None = None,
        purpose: str = "security",
    ) -> dict[str, Any]:
        """``create_rule`` — create a detection rule (ADR-162, #2322)."""
        args: dict[str, Any] = {
            "name": name,
            "condition_sql": condition_sql,
            "purpose": purpose,
        }
        if severity:
            args["severity"] = severity
        if description:
            args["description"] = description
        return await self.call_tool("create_rule", args)

    async def import_sigma(self, sigma_yaml: str, *, purpose: str = "security") -> dict[str, Any]:
        """``import_sigma`` — import a Sigma detection rule (YAML) (#2322)."""
        return await self.call_tool("import_sigma", {"yaml": sigma_yaml, "purpose": purpose})

    async def list_jobs(self) -> dict[str, Any]:
        """``list_jobs`` — list all registered detection jobs with live status (#2322)."""
        return await self.call_tool("list_jobs", {})

    async def schedule_job(self, name: str) -> dict[str, Any]:
        """``schedule_job`` — trigger an immediate run of a named detection job (#2322)."""
        return await self.call_tool("schedule_job", {"name": name})

    async def list_workflows(self) -> dict[str, Any]:
        """``list_workflows`` — list all registered workflow definitions (#2322)."""
        return await self.call_tool("list_workflows", {})

    async def run_workflow(self, name: str) -> dict[str, Any]:
        """``run_workflow`` — start a workflow execution by name (#2322)."""
        return await self.call_tool("run_workflow", {"name": name})

    async def workflow_status(self, run_id: str) -> dict[str, Any]:
        """``workflow_status`` — query step-level status of a workflow run (#2322)."""
        return await self.call_tool("workflow_status", {"run_id": run_id})

    async def trace_crypto(
        self,
        address: str,
        *,
        max_hops: int = 5,
        min_amount: float = 0,
        purpose: str = "analytics",
    ) -> dict[str, Any]:
        """``trace_crypto`` — follow a cryptocurrency address hop-by-hop (#2322)."""
        return await self.call_tool(
            "trace_crypto",
            {
                "address": address,
                "max_hops": max_hops,
                "min_amount": min_amount,
                "purpose": purpose,
            },
        )

    async def beneficial_ownership(
        self, party: str, *, max_depth: int = 6, purpose: str = "analytics"
    ) -> dict[str, Any]:
        """``beneficial_ownership`` — trace the beneficial ownership chain for a party (#2322)."""
        return await self.call_tool(
            "beneficial_ownership",
            {"party": party, "max_depth": max_depth, "purpose": purpose},
        )

    async def reconstruct_wire(
        self, account: str, *, tolerance_pct: float = 5.0, purpose: str = "analytics"
    ) -> dict[str, Any]:
        """``reconstruct_wire`` — reconstruct a wire-transfer chain for an account (#2322)."""
        return await self.call_tool(
            "reconstruct_wire",
            {"account": account, "tolerance_pct": tolerance_pct, "purpose": purpose},
        )

    async def trace_hawala(
        self, seed: str, *, max_hops: int = 5, purpose: str = "analytics"
    ) -> dict[str, Any]:
        """``trace_hawala`` — trace informal hawala value-transfer networks (#2322)."""
        return await self.call_tool(
            "trace_hawala", {"seed": seed, "max_hops": max_hops, "purpose": purpose}
        )

    async def geofence(
        self,
        lat: float,
        lon: float,
        *,
        radius_m: float = 1000,
        target_type: str = "MovementEvent",
        purpose: str = "analytics",
    ) -> dict[str, Any]:
        """``geofence`` — spatial fence query over a circular area (#2322)."""
        return await self.call_tool(
            "geofence",
            {
                "lat": lat,
                "lon": lon,
                "radius_m": radius_m,
                "target_type": target_type,
                "purpose": purpose,
            },
        )

    async def resolve_entity_identity(
        self, identity: str, *, purpose: str = "analytics"
    ) -> dict[str, Any]:
        """``resolve_entity_identity`` — resolve the canonical identity cluster (#2322)."""
        return await self.call_tool(
            "resolve_entity_identity", {"identity": identity, "purpose": purpose}
        )

    async def detect_communities(
        self, entity_type: str, *, algo: str = "louvain", purpose: str = "analytics"
    ) -> dict[str, Any]:
        """``detect_communities`` — Louvain/Leiden community detection (#2322)."""
        return await self.call_tool(
            "detect_communities",
            {"entity_type": entity_type, "algo": algo, "purpose": purpose},
        )

    async def rank_key_nodes(
        self,
        entity_type: str,
        *,
        metric: str = "pagerank",
        damping: float = 0.85,
        max_iter: int = 20,
        purpose: str = "analytics",
    ) -> dict[str, Any]:
        """``rank_key_nodes`` — rank entities by PageRank or centrality metric (#2322)."""
        return await self.call_tool(
            "rank_key_nodes",
            {
                "entity_type": entity_type,
                "metric": metric,
                "damping": damping,
                "max_iter": max_iter,
                "purpose": purpose,
            },
        )

    async def hub_authority(
        self, entity_type: str, *, max_iter: int = 20, purpose: str = "analytics"
    ) -> dict[str, Any]:
        """``hub_authority`` — HITS hub/authority scores for an entity type (#2322)."""
        return await self.call_tool(
            "hub_authority",
            {"entity_type": entity_type, "max_iter": max_iter, "purpose": purpose},
        )

    async def predict_links(
        self,
        entity_type: str,
        *,
        from_id: str | None = None,
        to_id: str | None = None,
        method: str = "common_neighbors",
        purpose: str = "analytics",
    ) -> dict[str, Any]:
        """``predict_links`` — score candidate edges between entities (#2322)."""
        args: dict[str, Any] = {"entity_type": entity_type, "method": method, "purpose": purpose}
        if from_id:
            args["from_id"] = from_id
        if to_id:
            args["to_id"] = to_id
        return await self.call_tool("predict_links", args)

    async def find_scc(self, entity_type: str, *, purpose: str = "analytics") -> dict[str, Any]:
        """``find_scc`` — find strongly connected components in an entity type's graph (#2322)."""
        return await self.call_tool("find_scc", {"entity_type": entity_type, "purpose": purpose})

    async def screen_sanctions(
        self,
        name: str,
        *,
        threshold: float | None = None,
        purpose: str = "compliance_review",
    ) -> dict[str, Any]:
        """``screen_sanctions`` — screen a name/entity against sanctions lists (#2322)."""
        args: dict[str, Any] = {"name": name, "purpose": purpose}
        if threshold is not None:
            args["threshold"] = threshold
        return await self.call_tool("screen_sanctions", args)

    async def aggregate_stats(
        self,
        entity_type: str,
        *,
        agg: str = "COUNT",
        column: str = "*",
        purpose: str = "analytics",
    ) -> dict[str, Any]:
        """``aggregate_stats`` — governed aggregate query over an entity type (#2322)."""
        return await self.call_tool(
            "aggregate_stats",
            {"entity_type": entity_type, "agg": agg, "column": column, "purpose": purpose},
        )

    async def investigate_entity(
        self, entity_type: str, entity_id: str, *, purpose: str = "security_incident"
    ) -> dict[str, Any]:
        """``investigate_entity`` — composite profile + timeline + connections + risk (#2322)."""
        return await self.call_tool(
            "investigate_entity",
            {"entity_type": entity_type, "id": entity_id, "purpose": purpose},
        )

    async def find_threats(
        self, entity_type: str, *, purpose: str = "security_incident"
    ) -> dict[str, Any]:
        """``find_threats`` — composite threat hunt over an entity type (#2322)."""
        return await self.call_tool(
            "find_threats", {"entity_type": entity_type, "purpose": purpose}
        )

    async def search_video_frames(
        self,
        query_id: str,
        *,
        media_type: str = "VideoFrame",
        top_k: int = 20,
        purpose: str = "security_incident",
    ) -> dict[str, Any]:
        """``search_video_frames`` — similarity search over video frames/segments (#2322)."""
        return await self.call_tool(
            "search_video_frames",
            {
                "query_id": query_id,
                "media_type": media_type,
                "top_k": top_k,
                "purpose": purpose,
            },
        )

    async def face_match(
        self,
        probe_id: str,
        *,
        threshold: float = 0.8,
        top_k: int = 10,
        purpose: str = "security_incident",
    ) -> dict[str, Any]:
        """``face_match`` — GATED: match a probe face against indexed media (#2322)."""
        return await self.call_tool(
            "face_match",
            {"probe_id": probe_id, "threshold": threshold, "top_k": top_k, "purpose": purpose},
        )

    async def close(self) -> None:
        await self._t.aclose()

    async def __aenter__(self) -> AsyncMcpClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()
