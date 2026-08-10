"""Query-shape dispatch + HyDE + decomposition — RAG epic SDK-side query
understanding (#4524).

Three cooperating techniques that run **before** the first ``/rag/query``
call, all confirmed live in the reference RAG toolkit studied for ADR-0299:

1. **Query-shape dispatch** (:func:`classify_query_shape`) — regex
   classification of the query text. Conjunction shapes (``"who ... and
   ..."``, ``"with X and Y"``) route toward ``expand_window=True`` so the
   answer's supporting evidence can span more of a single document.
   Enumeration shapes (``"which"``/``"list"``/``"how many"``) route toward a
   wider ``top_k`` so document-level aggregation over the hits has enough
   candidates to work with.
2. **HyDE** (:func:`expand_query_hyde`) — generate a hypothetical answer via
   one caller-supplied LLM call, then send that hypothetical answer as the
   ``/rag/query`` search text instead of the bare query — **except** when
   numeric intent is detected (:func:`is_numeric_intent`: revenue,
   percentage, count, dosage, ...), where HyDE is skipped entirely since a
   hallucinated hypothetical answer would invent a plausible-but-wrong
   number.
3. **Decomposition** (:func:`decompose_query`, :func:`rrf_merge`) — split a
   multi-part question into sub-queries, issue one ``/rag/query`` call per
   sub-query (in parallel), and Reciprocal-Rank-Fuse the results back
   together using an auto-scaling dampening constant,
   ``k = max(10, 60 / n)`` (:func:`rrf_k_for_fanout`) — **not** the fixed
   ``k=60`` RelataDB uses server-side to fuse BM25⊕vector *within* one
   ``/rag/query`` call (``crates/relata-query/src/hybrid.rs::RRF_K``). That
   is a different merge at a different layer; reusing its fixed constant
   here would under-weight top ranks as the fan-out grows.

:func:`smart_rag_query` / :func:`asmart_rag_query` are the one entry point
that composes all three ahead of calling
:class:`~relata.rag.RagClient`/:class:`~relata.rag.AsyncRagClient`.

Per ADR-0298, RelataDB has no server-side agent loop and orchestration lives
in the SDK — this module is deliberately Python-only (the canonical SDK for
this epic's orchestration layer, per #4523's ticket body); ``RagClient``
itself stays a thin pass-through on the TypeScript/Go/Rust surfaces with no
equivalent of this module.

This module never calls an LLM itself: :data:`HypothesisGenerator` is a
caller-supplied callable (RelataDB runs no LLM, ADR-013).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import re
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from relata.rag import AsyncRagClient, RagClient

from relata.models import RagHit, RagQueryResponse
from relata.rag import DEFAULT_TOP_K

# ---------------------------------------------------------------------------
# 1. Query-shape dispatch
# ---------------------------------------------------------------------------


class QueryShape(str, Enum):
    """Detected shape of a query text, per :func:`classify_query_shape`."""

    CONJUNCTION = "conjunction"
    ENUMERATION = "enumeration"
    SIMPLE = "simple"


#: Conjunction patterns: "who ... and ...", "with X and Y", "both ... and ...".
_CONJUNCTION_RE = re.compile(
    r"\bwho\b.*\band\b|\bwith\b\s+\S+\s+\band\b\s+\S+|\bboth\b.*\band\b",
    re.IGNORECASE,
)

#: Enumeration patterns: a query opening with "which"/"list"/"how many"/etc.
_ENUMERATION_RE = re.compile(
    r"^\s*(which|what\s+are\s+all|list|enumerate|name\s+all|how\s+many)\b",
    re.IGNORECASE,
)

#: Enumeration queries aggregate across more of the corpus than a single-fact
#: lookup, so they widen top_k past #4514's server default
#: (``RAG_RETRIEVE_DEFAULT_TOP_K`` = 8) to give document-level aggregation
#: enough candidates to work with.
ENUMERATION_TOP_K = 25


def classify_query_shape(query: str) -> QueryShape:
    """Classify ``query`` as :class:`QueryShape` CONJUNCTION, ENUMERATION, or
    SIMPLE via regex — no LLM call, cheap enough to run on every query."""
    if _CONJUNCTION_RE.search(query):
        return QueryShape.CONJUNCTION
    if _ENUMERATION_RE.search(query):
        return QueryShape.ENUMERATION
    return QueryShape.SIMPLE


def _apply_query_shape(shape: QueryShape, rag_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``rag_kwargs`` adjusted for the detected ``shape``.

    Conjunction queries route toward ``expand_window=True`` (only if the
    caller did not already set it explicitly). Enumeration queries widen
    ``top_k`` to at least :data:`ENUMERATION_TOP_K`, taking the max with
    whatever the caller already requested so an explicit larger value is
    never shrunk.
    """
    out = dict(rag_kwargs)
    if shape is QueryShape.CONJUNCTION:
        out.setdefault("expand_window", True)
    elif shape is QueryShape.ENUMERATION:
        out["top_k"] = max(out.get("top_k", DEFAULT_TOP_K), ENUMERATION_TOP_K)
    return out


# ---------------------------------------------------------------------------
# 2. HyDE, with the numeric-intent guard
# ---------------------------------------------------------------------------

#: Signature for a caller-supplied hypothetical-answer generator. RelataDB
#: runs no LLM (ADR-013) — this is always provided by the application, never
#: called by RelataDB itself.
HypothesisGenerator = Callable[[str], str]

#: Numeric-intent trigger words (verified list from the reference toolkit).
#: A query containing one of these usually wants an exact number; HyDE's
#: hallucinated hypothetical answer would invent a plausible-but-wrong one,
#: so HyDE is skipped entirely for these rather than risk it.
NUMERIC_INTENT_WORDS = frozenset(
    {
        "revenue",
        "percentage",
        "percent",
        "count",
        "how many",
        "total",
        "sum",
        "average",
        "mean",
        "median",
        "dosage",
        "dose",
        "amount",
        "quantity",
        "number of",
        "price",
        "cost",
        "rate",
        "ratio",
        "margin",
        "growth rate",
    }
)

_NUMERIC_INTENT_ALTERNATION = "|".join(
    re.escape(w) for w in sorted(NUMERIC_INTENT_WORDS, key=len, reverse=True)
)
_NUMERIC_INTENT_RE = re.compile(r"\b(" + _NUMERIC_INTENT_ALTERNATION + r")\b", re.IGNORECASE)


def is_numeric_intent(query: str) -> bool:
    """``True`` when ``query`` contains a :data:`NUMERIC_INTENT_WORDS` trigger
    word — the guard :func:`expand_query_hyde` uses to skip HyDE entirely."""
    return bool(_NUMERIC_INTENT_RE.search(query))


def expand_query_hyde(query: str, *, hypothesis_fn: HypothesisGenerator) -> str:
    """Return the HyDE-expanded search text for ``query``, or ``query``
    unchanged when :func:`is_numeric_intent` trips.

    ``hypothesis_fn`` is called with ``query`` and must return a hypothetical
    answer (plain text); that hypothetical answer becomes the ``query`` field
    sent to ``POST /rag/query`` (#4514), which embeds and searches it
    server-side — this function does not embed anything itself.
    """
    if is_numeric_intent(query):
        return query
    hypothetical = hypothesis_fn(query)
    return hypothetical or query


# ---------------------------------------------------------------------------
# 3. Decomposition + RRF merge with auto-scaling k
# ---------------------------------------------------------------------------

#: Splits a multi-part question into sub-queries on (a) a `?` immediately
#: followed by a new capitalised sentence, or (b) an "and" conjoining two
#: question clauses ("... and when ...", "... and how ...").
_DECOMPOSITION_SPLIT_RE = re.compile(
    r"\?\s+(?=[A-Z])|\s+and\s+(?=(?:what|when|where|who|which|how|why)\b)",
    re.IGNORECASE,
)


def decompose_query(query: str) -> list[str]:
    """Split ``query`` into independently-retrievable sub-queries.

    Returns ``[query]`` unchanged (a single-element list) when no split
    point is detected — callers should treat ``len(...) == 1`` as "no
    decomposition happened," not as an error.
    """
    parts = [p.strip() for p in _DECOMPOSITION_SPLIT_RE.split(query) if p.strip()]
    return parts if parts else [query]


def rrf_k_for_fanout(n_subqueries: int) -> float:
    """Auto-scaling RRF dampening constant for merging ``n_subqueries``
    parallel result lists: ``max(10, 60 / n)`` (verified in the reference
    toolkit). Deliberately *not* RelataDB's fixed internal ``RRF_K=60``
    (``crates/relata-query/src/hybrid.rs``), which fuses BM25⊕vector within
    one ``/rag/query`` call — a different merge at a different layer.
    """
    if n_subqueries <= 0:
        raise ValueError("n_subqueries must be positive")
    return max(10.0, 60.0 / n_subqueries)


def _rrf_scores(responses: list[RagQueryResponse], k: float) -> dict[str, float]:
    """Per-``chunk_id`` summed RRF contribution ``1 / (k + rank)`` across
    ``responses``. Split out from :func:`rrf_merge` so the exact arithmetic
    (and, in particular, which ``k`` it was actually called with) is directly
    assertable in tests without depending on final hit ordering."""
    scores: dict[str, float] = {}
    for response in responses:
        for rank, hit in enumerate(response.hits, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


def rrf_merge(responses: list[RagQueryResponse], *, k: float) -> RagQueryResponse:
    """Reciprocal-Rank-Fuse ``responses`` (one per sub-query) into a single
    :class:`~relata.models.RagQueryResponse`, deduped by ``chunk_id`` and
    re-ordered by the summed RRF contribution ``1 / (k + rank)`` (see
    :func:`_rrf_scores`).

    Each hit's own ``bm25_score``/``vector_score`` (#4514's per-channel
    contract) is passed through untouched — RRF only decides final
    ordering/dedup here, it never fuses or overwrites those fields.
    """
    scores = _rrf_scores(responses, k)
    first_seen: dict[str, RagHit] = {}
    for response in responses:
        for hit in response.hits:
            first_seen.setdefault(hit.chunk_id, hit)
    ordered_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    hits = [first_seen[cid] for cid in ordered_ids]
    return RagQueryResponse(hits=hits, total=len(hits))


# ---------------------------------------------------------------------------
# Entry point — composes all three ahead of the first /rag/query call
# ---------------------------------------------------------------------------


def smart_rag_query(
    rag_client: RagClient,
    query: str,
    type: str,  # noqa: A002 — matches RagClient.query's wire-facing param name
    *,
    purpose: str | None = None,
    hypothesis_fn: HypothesisGenerator | None = None,
    max_workers: int = 4,
    **rag_kwargs: Any,
) -> RagQueryResponse:
    """Run ``query`` through decomposition, per-sub-query shape dispatch, and
    optional HyDE, then issue the (possibly several, parallel) ``/rag/query``
    calls via ``rag_client`` and RRF-merge the results.

    Args:
        rag_client: A :class:`~relata.rag.RagClient` (sync).
        query: The raw user question, possibly multi-part.
        type: Object type to search (e.g. ``"DocumentChunk"``).
        purpose: PURPOSE token, passed through to every underlying call.
        hypothesis_fn: When given, each sub-query's search text is HyDE-
            expanded via :func:`expand_query_hyde` before the call (skipped
            per sub-query when :func:`is_numeric_intent` trips).
        max_workers: Thread-pool size for the parallel sub-query fan-out
            when decomposition produces more than one sub-query.
        **rag_kwargs: Forwarded to :meth:`RagClient.query` for every
            sub-query call (``top_k``, ``rerank``, ``search_mode``,
            ``embedding_slot``, ``filters``, ``as_of``, ``expand_window``,
            ``graph_hops``), after :func:`classify_query_shape`'s per-shape
            adjustments are applied on top.

    Returns:
        A single merged :class:`~relata.models.RagQueryResponse`. When
        decomposition produced only one sub-query, this is that call's
        response unchanged (no RRF merge is performed for n=1).
    """
    sub_queries = decompose_query(query)
    n = len(sub_queries)

    def _run(sub_query: str) -> RagQueryResponse:
        shape = classify_query_shape(sub_query)
        call_kwargs = _apply_query_shape(shape, rag_kwargs)
        search_text = sub_query
        if hypothesis_fn is not None:
            search_text = expand_query_hyde(sub_query, hypothesis_fn=hypothesis_fn)
        return rag_client.query(search_text, type, purpose=purpose, **call_kwargs)

    if n == 1:
        return _run(sub_queries[0])

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, n)) as pool:
        responses = list(pool.map(_run, sub_queries))
    return rrf_merge(responses, k=rrf_k_for_fanout(n))


async def asmart_rag_query(
    rag_client: AsyncRagClient,
    query: str,
    type: str,  # noqa: A002
    *,
    purpose: str | None = None,
    hypothesis_fn: HypothesisGenerator | None = None,
    **rag_kwargs: Any,
) -> RagQueryResponse:
    """Async variant of :func:`smart_rag_query` — issues the sub-query fan-out
    concurrently via ``asyncio.gather`` instead of a thread pool."""
    sub_queries = decompose_query(query)
    n = len(sub_queries)

    async def _run(sub_query: str) -> RagQueryResponse:
        shape = classify_query_shape(sub_query)
        call_kwargs = _apply_query_shape(shape, rag_kwargs)
        search_text = sub_query
        if hypothesis_fn is not None:
            search_text = expand_query_hyde(sub_query, hypothesis_fn=hypothesis_fn)
        return await rag_client.query(search_text, type, purpose=purpose, **call_kwargs)

    if n == 1:
        return await _run(sub_queries[0])

    responses = list(await asyncio.gather(*(_run(sq) for sq in sub_queries)))
    return rrf_merge(responses, k=rrf_k_for_fanout(n))
