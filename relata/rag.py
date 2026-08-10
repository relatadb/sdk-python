"""Typed ``/rag/query`` client — RAG epic foundational SDK ticket (#4523).

Wraps ``POST /rag/query`` (#4514, the ADR-0299 substrate contract) with a
typed request builder and response model. Per ADR-0298, RelataDB has no
server-side agent loop — orchestration lives in the SDK — and ``/rag/query``
itself is deliberately stateless, idempotent, and LLM-free: this client is a
thin, honest wrapper around that contract, nothing more.

**Entity disambiguation (#4534).** A query naming an ambiguous entity (e.g.
"Tell me about Ahmad" when the corpus has several distinct Ahmads) may come
back with :attr:`~relata.models.RagQueryResponse.clarification` set instead
of (or alongside empty) ``hits`` — see :class:`~relata.models.Clarification`.
``/rag/query`` itself stays stateless: the caller resumes with an explicit
pick by re-issuing the call with a ``filters`` entry scoping to the chosen
``entity_id``. :func:`filters_for_entity_ids` builds that entry, and
:meth:`RagClient.resume_with_selection`/:meth:`AsyncRagClient.resume_with_selection`
do the full re-run in one call — mirroring dgrep-rag's ``dgrep_resolve``
resume shape (prior-clarification pick -> ``MetadataFilter[]`` -> full
re-run), proven UX rather than speculative design.

**Frozen contract — do not deviate without re-opening ADR-0299 (#4514):**

Request fields: ``query``, ``type``, ``top_k`` (default 8, matches
``RAG_RETRIEVE_DEFAULT_TOP_K``), ``rerank`` (default ``False``),
``search_mode`` (``lexical|dense|hybrid|structural``, default ``hybrid`` —
``structural`` added by #4542, see :mod:`relata.structural_navigation`),
``embedding_slot`` (``text|summary|keyword|question``, default ``text``),
``filters``, ``as_of``, ``purpose`` (required), ``expand_window`` (default
``False``), ``graph_hops`` (default ``0``).

Response: a list of hits, each carrying ``bm25_score``/``vector_score``
**separately** (never fused into one number client-side — see
:class:`~relata.models.RagHit`), plus citation-grade fields
(``chunk_id``/``report_id``/``text``/``section_path``/``page_start``/
``page_end``), chunk adjacency (``prev_chunk_id``/``next_chunk_id``), and
``entity_ids``.

This client is built and tested against the documented contract via a mocked
HTTP transport — #4514 need not have merged first, since the contract is
frozen and any change to it requires reopening ADR-0299 rather than silently
drifting the client.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from relata.client import RelataClient

from relata.exceptions import PurposeError
from relata.models import RagQueryResponse
from relata.search import _normalise_filters

#: ``search_mode`` values #4514's contract accepts (default ``"hybrid"``).
#: ``"structural"`` (#4542) is a distinct navigation path — see
#: ``crates/relata-cli/src/serve/query/rag_query.rs``'s module doc — that
#: retrieves via the persisted ``DocumentStructureNode`` table-of-contents
#: index instead of flat embedding/BM25 ranking; it is still one governed
#: ``/rag/query`` call, so it needs no new client method, only this extra
#: enum value.
_SEARCH_MODES = frozenset({"lexical", "dense", "hybrid", "structural"})

#: ``embedding_slot`` values #4514's contract accepts (default ``"text"``).
_EMBEDDING_SLOTS = frozenset({"text", "summary", "keyword", "question"})

SearchMode = Literal["lexical", "dense", "hybrid", "structural"]
EmbeddingSlot = Literal["text", "summary", "keyword", "question"]

#: Matches ``RAG_RETRIEVE_DEFAULT_TOP_K`` (crates/relata-query/src/parser.rs:2854).
DEFAULT_TOP_K = 8


def _validate_search_mode(search_mode: str) -> str:
    if search_mode not in _SEARCH_MODES:
        raise ValueError(
            f"Invalid search_mode: {search_mode!r}. Must be one of: "
            f"{', '.join(sorted(_SEARCH_MODES))}."
        )
    return search_mode


def _validate_embedding_slot(embedding_slot: str) -> str:
    if embedding_slot not in _EMBEDDING_SLOTS:
        raise ValueError(
            f"Invalid embedding_slot: {embedding_slot!r}. Must be one of: "
            f"{', '.join(sorted(_EMBEDDING_SLOTS))}."
        )
    return embedding_slot


def _build_rag_payload(
    *,
    query: str,
    type: str,  # noqa: A002 — matches the wire field name verbatim (#4514)
    top_k: int,
    rerank: bool,
    search_mode: str,
    embedding_slot: str,
    filters: list[dict[str, Any]] | dict[str, Any] | None,
    as_of: str | None,
    purpose: str,
    expand_window: bool,
    graph_hops: int,
) -> dict[str, object]:
    """Assemble the ``POST /rag/query`` JSON body per #4514's exact table.

    Every field is sent explicitly (no field is ever silently dropped) even
    when it matches the server default, so a caller inspecting the outgoing
    payload sees the full, frozen contract rather than a partial one.
    """
    if not query:
        raise ValueError("query is required")
    if not type:
        raise ValueError("type is required")
    if not purpose:
        raise PurposeError(
            "rag_query requires a purpose. Pass purpose= to the call or set a "
            "default on the RelataClient."
        )

    payload: dict[str, object] = {
        "query": query,
        "type": type,
        "top_k": int(top_k),
        "rerank": bool(rerank),
        "search_mode": _validate_search_mode(search_mode),
        "embedding_slot": _validate_embedding_slot(embedding_slot),
        "filters": _normalise_filters(filters) or [],
        "as_of": as_of,
        "purpose": purpose,
        "expand_window": bool(expand_window),
        "graph_hops": int(graph_hops),
    }
    return payload


#: The filter field a resumed query scopes to — #4534's design section: "the
#: same `MetadataFilter[]` field `#4514` already defines, no new field
#: needed."
CANONICAL_ENTITY_ID_FIELD = "canonical_entity_id"


def filters_for_entity_ids(
    entity_ids: list[str],
    *,
    field: str = CANONICAL_ENTITY_ID_FIELD,
) -> list[dict[str, Any]]:
    """Build the ``filters`` entry a resumed query carries after a
    :class:`~relata.models.Clarification` pick (#4534).

    Returns ``[{"field": field, "op": "in", "value": entity_ids}]`` — the
    standard filter-list shape every ``RagClient.query``/``search`` call
    already accepts, so this needs no new wire field. Raises
    :class:`ValueError` on an empty ``entity_ids`` list (a resume call with no
    selection is a caller bug, not a valid "match nothing" filter).
    """
    if not entity_ids:
        raise ValueError("entity_ids must not be empty — nothing was selected to resume with")
    return [{"field": field, "op": "in", "value": list(entity_ids)}]


class RagClient:
    """Synchronous typed ``/rag/query`` client.

    Construct with :meth:`from_client` so the request executes under the
    parent client's auth/tenant context.
    """

    def __init__(self, client: RelataClient) -> None:
        self._client = client

    @classmethod
    def from_client(cls, client: RelataClient) -> RagClient:
        return cls(client)

    def _purpose(self, purpose: str | None) -> str | None:
        return purpose or self._client._default_purpose  # noqa: SLF001

    def query(
        self,
        query: str,
        type: str,  # noqa: A002 — matches the wire field name verbatim (#4514)
        *,
        top_k: int = DEFAULT_TOP_K,
        rerank: bool = False,
        search_mode: SearchMode = "hybrid",
        embedding_slot: EmbeddingSlot = "text",
        filters: list[dict[str, Any]] | dict[str, Any] | None = None,
        as_of: str | None = None,
        purpose: str | None = None,
        expand_window: bool = False,
        graph_hops: int = 0,
    ) -> RagQueryResponse:
        """Run a governed retrieval via ``POST /rag/query`` (#4514).

        Args:
            query: Query text.
            type: Object type to search (e.g. ``"DocumentChunk"``).
            top_k: Max hits (server default 8, ``RAG_RETRIEVE_DEFAULT_TOP_K``).
            rerank: Dispatch to the sidecar cross-encoder.
            search_mode: ``"lexical" | "dense" | "hybrid"`` (default hybrid).
            embedding_slot: ``"text" | "summary" | "keyword" | "question"``.
            filters: ``[{field, op, value}]`` or equality dict — standard
                ``WHERE``-equivalent filter list.
            as_of: Bi-temporal point-in-time query (``None`` = latest).
            purpose: PURPOSE token — required (falls back to the client's
                default purpose when unset).
            expand_window: Sentence-window expansion via adjacency links.
            graph_hops: Graph-augmented expansion depth.

        Returns:
            :class:`~relata.models.RagQueryResponse` — ``hits`` carries
            ``bm25_score``/``vector_score`` separately per hit, never fused.
        """
        payload = _build_rag_payload(
            query=query,
            type=type,
            top_k=top_k,
            rerank=rerank,
            search_mode=search_mode,
            embedding_slot=embedding_slot,
            filters=filters,
            as_of=as_of,
            purpose=self._purpose(purpose),
            expand_window=expand_window,
            graph_hops=graph_hops,
        )
        data = self._client._sync.post("/rag/query", payload)  # noqa: SLF001
        return RagQueryResponse.model_validate(data)

    async def aquery(
        self,
        query: str,
        type: str,  # noqa: A002
        *,
        top_k: int = DEFAULT_TOP_K,
        rerank: bool = False,
        search_mode: SearchMode = "hybrid",
        embedding_slot: EmbeddingSlot = "text",
        filters: list[dict[str, Any]] | dict[str, Any] | None = None,
        as_of: str | None = None,
        purpose: str | None = None,
        expand_window: bool = False,
        graph_hops: int = 0,
    ) -> RagQueryResponse:
        """Async variant of :meth:`query`."""
        payload = _build_rag_payload(
            query=query,
            type=type,
            top_k=top_k,
            rerank=rerank,
            search_mode=search_mode,
            embedding_slot=embedding_slot,
            filters=filters,
            as_of=as_of,
            purpose=self._purpose(purpose),
            expand_window=expand_window,
            graph_hops=graph_hops,
        )
        data = await self._client._async.post("/rag/query", payload)  # noqa: SLF001
        return RagQueryResponse.model_validate(data)

    def resume_with_selection(
        self,
        query: str,
        type: str,  # noqa: A002
        entity_ids: list[str],
        *,
        top_k: int = DEFAULT_TOP_K,
        rerank: bool = False,
        search_mode: SearchMode = "hybrid",
        embedding_slot: EmbeddingSlot = "text",
        as_of: str | None = None,
        purpose: str | None = None,
        expand_window: bool = False,
        graph_hops: int = 0,
    ) -> RagQueryResponse:
        """Resume an ambiguous query after a human/agent picks candidate(s)
        from a prior :class:`~relata.models.Clarification` (#4534).

        Re-runs :meth:`query` with ``filters`` scoped to
        ``entity_ids`` (typically ``[clarification.candidates[i].entity_id]``
        for a single pick, or several for a "these are all the same"
        selection) via :func:`filters_for_entity_ids`. ``/rag/query`` stays
        stateless — this is a client-side convenience over the same request
        shape, not a new server contract.

        Args:
            query: Same query text as the original ambiguous call (or a
                refined follow-up).
            type: Same object type as the original call.
            entity_ids: One or more ``candidates[i].entity_id`` values picked
                from the prior response's ``clarification``. Must be
                non-empty.
            top_k, rerank, search_mode, embedding_slot, as_of, purpose,
                expand_window, graph_hops: Same as :meth:`query`.

        Returns:
            :class:`~relata.models.RagQueryResponse` scoped to exactly the
            selected entity/entities.
        """
        return self.query(
            query,
            type,
            top_k=top_k,
            rerank=rerank,
            search_mode=search_mode,
            embedding_slot=embedding_slot,
            filters=filters_for_entity_ids(entity_ids),
            as_of=as_of,
            purpose=purpose,
            expand_window=expand_window,
            graph_hops=graph_hops,
        )

    async def aresume_with_selection(
        self,
        query: str,
        type: str,  # noqa: A002
        entity_ids: list[str],
        *,
        top_k: int = DEFAULT_TOP_K,
        rerank: bool = False,
        search_mode: SearchMode = "hybrid",
        embedding_slot: EmbeddingSlot = "text",
        as_of: str | None = None,
        purpose: str | None = None,
        expand_window: bool = False,
        graph_hops: int = 0,
    ) -> RagQueryResponse:
        """Async variant of :meth:`resume_with_selection`."""
        return await self.aquery(
            query,
            type,
            top_k=top_k,
            rerank=rerank,
            search_mode=search_mode,
            embedding_slot=embedding_slot,
            filters=filters_for_entity_ids(entity_ids),
            as_of=as_of,
            purpose=purpose,
            expand_window=expand_window,
            graph_hops=graph_hops,
        )


class AsyncRagClient:
    """Asynchronous typed ``/rag/query`` client — see :class:`RagClient`."""

    def __init__(self, client: RelataClient) -> None:
        self._inner = RagClient(client)

    @classmethod
    def from_client(cls, client: RelataClient) -> AsyncRagClient:
        return cls(client)

    async def query(
        self,
        query: str,
        type: str,  # noqa: A002
        *,
        top_k: int = DEFAULT_TOP_K,
        rerank: bool = False,
        search_mode: SearchMode = "hybrid",
        embedding_slot: EmbeddingSlot = "text",
        filters: list[dict[str, Any]] | dict[str, Any] | None = None,
        as_of: str | None = None,
        purpose: str | None = None,
        expand_window: bool = False,
        graph_hops: int = 0,
    ) -> RagQueryResponse:
        return await self._inner.aquery(
            query,
            type,
            top_k=top_k,
            rerank=rerank,
            search_mode=search_mode,
            embedding_slot=embedding_slot,
            filters=filters,
            as_of=as_of,
            purpose=purpose,
            expand_window=expand_window,
            graph_hops=graph_hops,
        )

    async def resume_with_selection(
        self,
        query: str,
        type: str,  # noqa: A002
        entity_ids: list[str],
        *,
        top_k: int = DEFAULT_TOP_K,
        rerank: bool = False,
        search_mode: SearchMode = "hybrid",
        embedding_slot: EmbeddingSlot = "text",
        as_of: str | None = None,
        purpose: str | None = None,
        expand_window: bool = False,
        graph_hops: int = 0,
    ) -> RagQueryResponse:
        """Async variant of :meth:`RagClient.resume_with_selection` (#4534)."""
        return await self._inner.aresume_with_selection(
            query,
            type,
            entity_ids,
            top_k=top_k,
            rerank=rerank,
            search_mode=search_mode,
            embedding_slot=embedding_slot,
            as_of=as_of,
            purpose=purpose,
            expand_window=expand_window,
            graph_hops=graph_hops,
        )
