"""MMR (Maximal Marginal Relevance) diversity selection — RAG epic (#4526),
grouped with :mod:`relata.rag_loop`'s sub-agent fan-out because MMR is
typically applied to the fan-out's merged candidate set
(:attr:`~relata.rag_loop.FanoutResult.merged_response`), though it works
over any list of :class:`~relata.models.RagHit` (a single ``/rag/query``
call's hits, a decomposition merge, etc.).

Greedy selection maximizing::

    lambda_mult * relevance - (1 - lambda_mult) * max_similarity_to_selected

over the candidate set, verified live in the reference retrieval toolkit
studied for ADR-0299. Both terms are **pure functions of what
``POST /rag/query`` already returned** — RelataDB's wire contract carries
per-channel scores (``bm25_score``/``vector_score``, #4514) and the non-LLM
``relevance_confidence`` signal (#4520), never raw embedding vectors, so
:func:`mmr_select` needs no embedding call, no LLM call, and no network I/O:
it is deterministic given its inputs, matching :mod:`relata.rag_loop`'s
zero-LLM-calls-in-the-merge-path property.

- **Relevance** (:func:`default_relevance`) defaults to a hit's
  ``relevance_confidence`` when the server provides it, falling back to the
  plain average of ``bm25_score``/``vector_score`` for a server that
  predates #4520.
- **Similarity** (:func:`default_text_similarity`) defaults to Jaccard
  token-set overlap over each hit's ``text`` — the one per-hit field that
  can stand in for "how similar are these two chunks" without a vector.
  Callers with a real embedding function available (e.g. the same one
  passed to :func:`relata.rag_loop.heuristic_gate`) can pass a smarter
  ``similarity_fn`` instead; :func:`mmr_select` takes both as plain
  callables so neither default is load-bearing.

**λ is domain-configurable, not a single global constant** (#4526's AC #3):
:data:`MMR_LAMBDA_BY_PURPOSE` keys the verified reference-toolkit defaults
by RelataDB's existing ``purpose`` governance dimension (already threaded
through every ``/rag/query`` call) via :func:`mmr_lambda_for_purpose`.
"""

from __future__ import annotations

import re
from typing import Callable, Sequence

from relata.models import RagHit

# ---------------------------------------------------------------------------
# Domain-configurable lambda (verified defaults from the reference toolkit)
# ---------------------------------------------------------------------------

#: Purpose-keyed MMR lambda defaults (case-insensitive lookup via
#: :func:`mmr_lambda_for_purpose`). Higher lambda weights relevance over
#: diversity — legal/clinical/medical domains want the most-relevant chunks
#: even if similar; code retrieval wants the most diverse set even at some
#: relevance cost.
MMR_LAMBDA_BY_PURPOSE: dict[str, float] = {
    "legal": 0.7,
    "clinical": 0.7,
    "medical": 0.7,
    "finance": 0.65,
    "research": 0.6,
    "general": 0.5,
    "code": 0.4,
}

#: Fallback lambda for a ``purpose`` not present in
#: :data:`MMR_LAMBDA_BY_PURPOSE` (including ``None``) — the "general" value,
#: an even relevance/diversity split.
DEFAULT_MMR_LAMBDA = MMR_LAMBDA_BY_PURPOSE["general"]


def mmr_lambda_for_purpose(purpose: str | None) -> float:
    """Return the verified MMR lambda for ``purpose`` (case-insensitive),
    or :data:`DEFAULT_MMR_LAMBDA` when ``purpose`` is ``None`` or not one of
    :data:`MMR_LAMBDA_BY_PURPOSE`'s keys."""
    if purpose is None:
        return DEFAULT_MMR_LAMBDA
    return MMR_LAMBDA_BY_PURPOSE.get(purpose.lower(), DEFAULT_MMR_LAMBDA)


# ---------------------------------------------------------------------------
# Pure default relevance/similarity terms
# ---------------------------------------------------------------------------

#: Signature for MMR's relevance term: one score per candidate hit.
RelevanceFn = Callable[[RagHit], float]

#: Signature for MMR's diversity term: similarity between two hits.
SimilarityFn = Callable[[RagHit, RagHit], float]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def default_relevance(hit: RagHit) -> float:
    """Pure relevance term: ``hit.relevance_confidence`` (#4520) when
    present, otherwise the average of ``bm25_score``/``vector_score``
    (#4514) — always a function of sub-scores already on the hit, never a
    new call."""
    if hit.relevance_confidence is not None:
        return hit.relevance_confidence
    return (hit.bm25_score + hit.vector_score) / 2.0


def _token_set(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def default_text_similarity(a: RagHit, b: RagHit) -> float:
    """Pure diversity term: Jaccard token-set overlap of ``a.text``/
    ``b.text``. ``0.0`` when either hit's ``text`` has no tokens."""
    tokens_a, tokens_b = _token_set(a.text), _token_set(b.text)
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


# ---------------------------------------------------------------------------
# Greedy MMR selection
# ---------------------------------------------------------------------------


def mmr_select(
    hits: Sequence[RagHit],
    *,
    lambda_mult: float = DEFAULT_MMR_LAMBDA,
    top_n: int | None = None,
    relevance_fn: RelevanceFn = default_relevance,
    similarity_fn: SimilarityFn = default_text_similarity,
) -> list[RagHit]:
    """Greedily select from ``hits`` by Maximal Marginal Relevance.

    Repeatedly picks the candidate maximizing
    ``lambda_mult * relevance_fn(candidate) - (1 - lambda_mult) *
    max(similarity_fn(candidate, s) for s in already_selected)`` (``0.0``
    similarity for the first pick, nothing selected yet) until ``top_n``
    hits have been chosen or ``hits`` is exhausted.

    A pure function: no LLM call, no embedding call, no network I/O —
    deterministic given the same ``hits``/``relevance_fn``/``similarity_fn``.
    Ties are broken by ``hits``' original order (candidates are scanned in
    that order and only a strictly greater score replaces the current best).

    Args:
        hits: Candidate hits — typically
            :attr:`~relata.rag_loop.FanoutResult.merged_response`'s hits, or
            any other ``/rag/query`` result.
        lambda_mult: Relevance/diversity trade-off in ``[0.0, 1.0]``. Use
            :func:`mmr_lambda_for_purpose` to key this by ``purpose`` rather
            than hardcoding a single value (#4526's AC #3).
        top_n: Max hits to return (default: all of ``hits``, reordered).
        relevance_fn: Per-hit relevance term (default
            :func:`default_relevance`).
        similarity_fn: Pairwise diversity term (default
            :func:`default_text_similarity`).

    Returns:
        The selected hits in pick order (most-relevant-subject-to-diversity
        first).

    Raises:
        ValueError: ``lambda_mult`` is outside ``[0.0, 1.0]``.
    """
    if not 0.0 <= lambda_mult <= 1.0:
        raise ValueError(f"lambda_mult must be within [0.0, 1.0], got {lambda_mult}")

    pool = list(hits)
    limit = len(pool) if top_n is None else max(0, min(top_n, len(pool)))
    selected_idx: list[int] = []
    available = list(range(len(pool)))

    while available and len(selected_idx) < limit:
        best_i: int | None = None
        best_score: float | None = None
        for i in available:
            candidate = pool[i]
            relevance = relevance_fn(candidate)
            max_sim = max(
                (similarity_fn(candidate, pool[j]) for j in selected_idx), default=0.0
            )
            score = lambda_mult * relevance - (1.0 - lambda_mult) * max_sim
            if best_score is None or score > best_score:
                best_score = score
                best_i = i
        assert best_i is not None  # `available` is non-empty in this loop
        selected_idx.append(best_i)
        available.remove(best_i)

    return [pool[i] for i in selected_idx]


def mmr_select_for_purpose(
    hits: Sequence[RagHit],
    purpose: str | None,
    *,
    top_n: int | None = None,
    relevance_fn: RelevanceFn = default_relevance,
    similarity_fn: SimilarityFn = default_text_similarity,
) -> list[RagHit]:
    """:func:`mmr_select` with ``lambda_mult`` resolved from ``purpose`` via
    :func:`mmr_lambda_for_purpose` — the convenience entry point for callers
    who already have RelataDB's governance ``purpose`` in hand and want the
    domain-tuned default rather than picking a lambda themselves."""
    return mmr_select(
        hits,
        lambda_mult=mmr_lambda_for_purpose(purpose),
        top_n=top_n,
        relevance_fn=relevance_fn,
        similarity_fn=similarity_fn,
    )
