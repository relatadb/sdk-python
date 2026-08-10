"""The agentic loop's spine — heuristic gate, corrective retrieval grading,
and loop confidence evaluation (#4525, the RAG epic's three-tier cost
ladder).

Per ADR-0298, RelataDB runs no server-side agent loop and no LLM (ADR-013) —
this module is the SDK-side control flow that decides, after each
``POST /rag/query`` call (#4514's ADR-0299 contract), whether the hits are
good enough to synthesize from or whether another round is warranted, and it
never calls an LLM itself: every LLM-backed decision point is a caller-
supplied callable, exactly like :mod:`relata.rag_understanding`'s
``hypothesis_fn``.

**The three-tier cost ladder (verified defaults — do not change without a
test-backed reason):**

1. **Heuristic gate** (:func:`heuristic_gate`) — centroid cosine similarity
   between the query embedding and the top-K ``/rag/query`` hits' embeddings,
   <5ms, no LLM call either way. Score > :data:`HEURISTIC_PASS_THRESHOLD`
   (``0.85``) auto-passes (stop, synthesize from these hits). Score <
   :data:`HEURISTIC_RETRY_THRESHOLD` (``0.60``) auto-retries (issue another
   ``/rag/query`` call, still no LLM). Between the two thresholds, control
   proceeds to the evaluator below.
2. **Corrective retrieval grading** (:func:`grade_hits`) — one batched
   cheap-model call grades every hit ``correct``/``ambiguous``/``incorrect``
   at once (never one call per hit — that would defeat the point of a *cost*
   ladder). A low ``fraction_correct`` (below
   :data:`CORRECTIVE_FRACTION_CORRECT_FLOOR`) marks
   :attr:`CorrectiveGradingResult.needs_requery`, which
   :func:`run_agentic_loop`/:func:`arun_agentic_loop` honor by forcing a
   re-query — refined via HyDE (:func:`relata.rag_understanding.
   expand_query_hyde`) when ``hypothesis_fn`` is supplied, or a caller-
   supplied web-search fallback when ``web_search_fallback`` is supplied,
   or a plain retry otherwise.
3. **Loop confidence** (:data:`LOOP_CONFIDENCE_THRESHOLD`, ``0.85``) — once
   corrective grading runs, its ``fraction_correct`` *is* the loop's
   confidence signal (no extra LLM call to compute a separate "confidence"
   number): reaching the threshold stops iteration. :data:`MAX_ITERATIONS`
   (``5``) is a hard cap enforced unconditionally — the control-flow loop is
   bounded (``for _ in range(max_iterations)``), never a ``while True``, so
   no adversarial input (a grader that never reports high confidence, an
   embedding function that never clears the heuristic-pass bar) can make it
   spin forever.

These three thresholds only work as a set (see the epic's parameter table,
issue #4529): shipping the heuristic gate without the loop cap would leave
the "auto-retry" branch unbounded; shipping corrective grading without the
heuristic gate would spend an LLM call on every query, defeating the cost
ladder's entire purpose.

The one property every caller should be able to prove: **a query whose
heuristic-gate score clears the pass threshold on iteration 0 makes zero LLM
calls end to end** — :func:`run_agentic_loop`/:func:`arun_agentic_loop`
return before ``grader_fn``/``hypothesis_fn`` are ever invoked in that case,
and :attr:`LoopResult.llm_calls` stays ``0`` (see the regression test
suite's call-count assertions).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Sequence

if TYPE_CHECKING:
    from relata.rag import AsyncRagClient, RagClient

from relata.models import RagHit, RagQueryResponse
from relata.rag_understanding import HypothesisGenerator, expand_query_hyde

# ---------------------------------------------------------------------------
# Verified defaults — the cost ladder's thresholds (#4525's acceptance
# criteria: match these unless a test-backed reason to deviate is documented)
# ---------------------------------------------------------------------------

#: Heuristic-gate auto-pass bar. Centroid cosine strictly greater than this
#: skips the evaluator entirely — zero LLM calls.
HEURISTIC_PASS_THRESHOLD = 0.85

#: Heuristic-gate auto-retry floor. Centroid cosine strictly less than this
#: re-queries without ever reaching the evaluator either — still zero LLM
#: calls.
HEURISTIC_RETRY_THRESHOLD = 0.60

#: Loop confidence threshold. Once corrective grading's ``fraction_correct``
#: reaches this, :func:`run_agentic_loop`/:func:`arun_agentic_loop` stop
#: iterating.
LOOP_CONFIDENCE_THRESHOLD = 0.85

#: Hard cap on loop iterations, enforced unconditionally (bounded ``for``
#: loop, never a ``while True``) — no unbounded loop is possible even under
#: adversarial input.
MAX_ITERATIONS = 5

#: ``fraction_correct`` floor below which corrective grading marks
#: :attr:`CorrectiveGradingResult.needs_requery`. Not one of the epic's three
#: named thresholds (the ticket describes the *behavior* — "low
#: fractionCorrect forces a re-query" — without pinning an exact number);
#: chosen as the natural "more wrong than right" boundary and isolated in
#: its own constant so a future RAG-quality-gate finding (#4538) can retune
#: it without touching the three verified thresholds above.
CORRECTIVE_FRACTION_CORRECT_FLOOR = 0.5


# ---------------------------------------------------------------------------
# 1. Heuristic gate
# ---------------------------------------------------------------------------


class GateDecision(str, Enum):
    """Outcome of :func:`heuristic_gate`."""

    #: Score > :data:`HEURISTIC_PASS_THRESHOLD` — stop, synthesize from these
    #: hits. No LLM call.
    PASS = "pass"
    #: Score < :data:`HEURISTIC_RETRY_THRESHOLD` — issue another
    #: ``/rag/query`` call. No LLM call.
    RETRY = "retry"
    #: Between the two thresholds — proceed to corrective retrieval grading
    #: (:func:`grade_hits`), the loop's one LLM call per iteration.
    EVALUATE = "evaluate"


#: Signature for a caller-supplied embedding function — RelataDB computes no
#: embeddings itself (ADR-0298). Used only for the <5ms heuristic gate, never
#: for an LLM call.
EmbeddingFn = Callable[[str], Sequence[float]]


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _centroid(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    sums = [0.0] * dim
    for vec in vectors:
        for i, x in enumerate(vec):
            sums[i] += x
    n = float(len(vectors))
    return [s / n for s in sums]


@dataclass(frozen=True)
class HeuristicGateResult:
    """Result of :func:`heuristic_gate`."""

    #: Centroid cosine similarity between the query embedding and the top-K
    #: hit embeddings — ``0.0`` when there are no hits to compare against.
    score: float
    decision: GateDecision


def heuristic_gate(
    query_embedding: Sequence[float],
    hit_embeddings: Sequence[Sequence[float]],
) -> HeuristicGateResult:
    """Cheap pre-LLM gate: centroid cosine between ``query_embedding`` and
    the top-K ``/rag/query`` hits' embeddings, <5ms, no LLM call either way.

    Zero hits (:attr:`~relata.models.RagQueryResponse.hits` empty) scores
    ``0.0`` and decides :attr:`GateDecision.RETRY` — nothing to be
    confident about.
    """
    if not hit_embeddings:
        return HeuristicGateResult(score=0.0, decision=GateDecision.RETRY)
    score = _cosine_similarity(query_embedding, _centroid(hit_embeddings))
    if score > HEURISTIC_PASS_THRESHOLD:
        decision = GateDecision.PASS
    elif score < HEURISTIC_RETRY_THRESHOLD:
        decision = GateDecision.RETRY
    else:
        decision = GateDecision.EVALUATE
    return HeuristicGateResult(score=score, decision=decision)


# ---------------------------------------------------------------------------
# 2. Corrective retrieval grading
# ---------------------------------------------------------------------------


class HitGrade(str, Enum):
    """Per-hit grade from :func:`grade_hits`'s batched cheap-model call."""

    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


#: Signature for a caller-supplied batched grading call — RelataDB runs no
#: LLM (ADR-013). Called once per grading round with ``(query, hits)``, must
#: return exactly one :class:`HitGrade` per hit, same order — one batched
#: call, never one call per hit.
GraderFn = Callable[[str, list[RagHit]], list[HitGrade]]


@dataclass(frozen=True)
class CorrectiveGradingResult:
    """Result of :func:`grade_hits`."""

    #: One grade per hit, same order as the graded hit list.
    grades: list[HitGrade]
    #: ``count(CORRECT) / len(hits)`` — ``0.0`` when there were no hits to
    #: grade.
    fraction_correct: float

    @property
    def needs_requery(self) -> bool:
        """``True`` when :attr:`fraction_correct` is below
        :data:`CORRECTIVE_FRACTION_CORRECT_FLOOR` — the corrective-retrieval
        signal that forces a re-query (refined via HyDE, or a web-search
        fallback, or a plain retry — see :func:`run_agentic_loop`)."""
        return self.fraction_correct < CORRECTIVE_FRACTION_CORRECT_FLOOR


def grade_hits(query: str, hits: list[RagHit], *, grader_fn: GraderFn) -> CorrectiveGradingResult:
    """Run one batched corrective-retrieval grading call over ``hits``.

    Raises :class:`ValueError` if ``grader_fn`` returns a different number
    of grades than hits given — a caller bug, not a valid partial grading.
    """
    if not hits:
        return CorrectiveGradingResult(grades=[], fraction_correct=0.0)
    grades = grader_fn(query, hits)
    if len(grades) != len(hits):
        raise ValueError(
            f"grader_fn must return exactly one grade per hit: got {len(grades)} "
            f"grade(s) for {len(hits)} hit(s)"
        )
    n_correct = sum(1 for g in grades if g is HitGrade.CORRECT)
    fraction_correct = n_correct / len(hits)
    return CorrectiveGradingResult(grades=grades, fraction_correct=fraction_correct)


# ---------------------------------------------------------------------------
# 3. Loop confidence + orchestration
# ---------------------------------------------------------------------------

#: Signature for a caller-supplied web-search fallback, invoked when
#: corrective grading's :attr:`CorrectiveGradingResult.needs_requery` trips
#: and a fallback is configured. Returns a full
#: :class:`~relata.models.RagQueryResponse`-shaped result (the loop does not
#: care whether it came from ``/rag/query`` or an external search API).
WebSearchFallbackFn = Callable[[str], RagQueryResponse]


@dataclass
class LoopIteration:
    """Record of one :func:`run_agentic_loop`/:func:`arun_agentic_loop`
    iteration — the reasoning trace #4528 writes back to ``RagAnswer``/
    ``RagSource`` is built from a sequence of these."""

    query: str
    response: RagQueryResponse
    gate: HeuristicGateResult | None = None
    grading: CorrectiveGradingResult | None = None
    #: This iteration's confidence signal — the heuristic gate's score when
    #: no grading ran, otherwise the grading's ``fraction_correct``.
    confidence: float | None = None


@dataclass
class LoopResult:
    """Final result of :func:`run_agentic_loop`/:func:`arun_agentic_loop`."""

    response: RagQueryResponse
    iterations: list[LoopIteration]
    #: Total count of caller-supplied LLM-backed calls made across every
    #: iteration (``grader_fn`` + HyDE-refinement ``hypothesis_fn`` calls).
    #: ``0`` when the heuristic gate passed on iteration 0 — see the module
    #: docstring's zero-LLM-calls property.
    llm_calls: int
    #: One of ``"heuristic_pass"``, ``"confident"``, ``"web_search_fallback"``,
    #: ``"no_grader_configured"``, ``"max_iterations"``.
    stopped_reason: str


def _run_loop_iterations(
    *,
    query_fn: Callable[[str], RagQueryResponse],
    query: str,
    embedding_fn: EmbeddingFn,
    grader_fn: GraderFn | None,
    hypothesis_fn: HypothesisGenerator | None,
    web_search_fallback: WebSearchFallbackFn | None,
    max_iterations: int,
) -> LoopResult:
    """Shared control flow between the sync and async entry points —
    ``query_fn`` is the only side-effecting call and is provided by the
    caller (sync ``rag_client.query`` vs. an already-awaited async result),
    so this function itself makes no I/O decisions.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")

    iterations: list[LoopIteration] = []
    llm_calls = 0
    current_query = query

    for i in range(max_iterations):
        is_last = i == max_iterations - 1
        response = query_fn(current_query)
        record = LoopIteration(query=current_query, response=response)
        iterations.append(record)

        gate = heuristic_gate(
            embedding_fn(current_query),
            [embedding_fn(hit.text) for hit in response.hits],
        )
        record.gate = gate

        if gate.decision is GateDecision.PASS:
            record.confidence = gate.score
            return LoopResult(response, iterations, llm_calls, "heuristic_pass")

        if gate.decision is GateDecision.RETRY:
            record.confidence = gate.score
            if is_last:
                return LoopResult(response, iterations, llm_calls, "max_iterations")
            current_query = query
            continue

        # GateDecision.EVALUATE — proceed to corrective retrieval grading.
        if grader_fn is None:
            record.confidence = gate.score
            return LoopResult(response, iterations, llm_calls, "no_grader_configured")

        grading = grade_hits(current_query, response.hits, grader_fn=grader_fn)
        llm_calls += 1
        record.grading = grading
        record.confidence = grading.fraction_correct

        if grading.fraction_correct >= LOOP_CONFIDENCE_THRESHOLD:
            return LoopResult(response, iterations, llm_calls, "confident")

        if is_last:
            return LoopResult(response, iterations, llm_calls, "max_iterations")

        if grading.needs_requery and web_search_fallback is not None:
            fallback_response = web_search_fallback(current_query)
            iterations.append(LoopIteration(query=current_query, response=fallback_response))
            return LoopResult(fallback_response, iterations, llm_calls, "web_search_fallback")

        if grading.needs_requery and hypothesis_fn is not None:
            current_query = expand_query_hyde(query, hypothesis_fn=hypothesis_fn)
            llm_calls += 1
        else:
            current_query = query

    # Unreachable: every branch above returns once `is_last` is True, and
    # `range(max_iterations)`'s final index always has `is_last == True`.
    raise AssertionError("run_agentic_loop: exhausted max_iterations without returning")


def run_agentic_loop(
    rag_client: RagClient,
    query: str,
    type: str,  # noqa: A002 — matches RagClient.query's wire-facing param name
    *,
    purpose: str | None = None,
    embedding_fn: EmbeddingFn,
    grader_fn: GraderFn | None = None,
    hypothesis_fn: HypothesisGenerator | None = None,
    web_search_fallback: WebSearchFallbackFn | None = None,
    max_iterations: int = MAX_ITERATIONS,
    **rag_kwargs: Any,
) -> LoopResult:
    """Run the three-tier cost-ladder loop: heuristic gate -> (optionally)
    corrective retrieval grading -> loop-confidence stop, up to
    :data:`MAX_ITERATIONS` (or ``max_iterations``, capped the same way).

    Args:
        rag_client: A :class:`~relata.rag.RagClient` (sync).
        query: The raw user query.
        type: Object type to search (e.g. ``"DocumentChunk"``).
        purpose: PURPOSE token, passed through to every underlying call.
        embedding_fn: Caller-supplied embedding function for the heuristic
            gate (:func:`heuristic_gate`) — not an LLM call, RelataDB
            computes no embeddings itself (ADR-0298).
        grader_fn: Caller-supplied batched corrective-retrieval grader
            (:func:`grade_hits`). When ``None``, a query landing in the
            heuristic gate's :attr:`GateDecision.EVALUATE` band stops
            immediately with ``stopped_reason="no_grader_configured"``
            rather than silently skipping grading.
        hypothesis_fn: Caller-supplied HyDE hypothesis generator
            (:func:`relata.rag_understanding.expand_query_hyde`), used only
            to refine the query when corrective grading's
            ``needs_requery`` trips.
        web_search_fallback: Caller-supplied web-search fallback, tried
            before HyDE refinement when ``needs_requery`` trips.
        max_iterations: Hard cap on loop iterations (default
            :data:`MAX_ITERATIONS`). Enforced unconditionally — the loop
            cannot spin forever regardless of what ``grader_fn``/
            ``embedding_fn`` return.
        **rag_kwargs: Forwarded to :meth:`~relata.rag.RagClient.query` on
            every ``/rag/query`` call (``top_k``, ``rerank``,
            ``search_mode``, ``embedding_slot``, ``filters``, ``as_of``,
            ``expand_window``, ``graph_hops``).

    Returns:
        :class:`LoopResult` — the final response, the full iteration
        history, the total LLM-call count (:attr:`LoopResult.llm_calls`,
        ``0`` when the heuristic gate passed on iteration 0), and why the
        loop stopped.
    """

    def query_fn(text: str) -> RagQueryResponse:
        return rag_client.query(text, type, purpose=purpose, **rag_kwargs)

    return _run_loop_iterations(
        query_fn=query_fn,
        query=query,
        embedding_fn=embedding_fn,
        grader_fn=grader_fn,
        hypothesis_fn=hypothesis_fn,
        web_search_fallback=web_search_fallback,
        max_iterations=max_iterations,
    )


async def arun_agentic_loop(
    rag_client: AsyncRagClient,
    query: str,
    type: str,  # noqa: A002
    *,
    purpose: str | None = None,
    embedding_fn: EmbeddingFn,
    grader_fn: GraderFn | None = None,
    hypothesis_fn: HypothesisGenerator | None = None,
    web_search_fallback: WebSearchFallbackFn | None = None,
    max_iterations: int = MAX_ITERATIONS,
    **rag_kwargs: Any,
) -> LoopResult:
    """Async variant of :func:`run_agentic_loop` — issues each iteration's
    ``/rag/query`` call via :class:`~relata.rag.AsyncRagClient`.
    ``embedding_fn``/``grader_fn``/``hypothesis_fn``/``web_search_fallback``
    stay synchronous callables, mirroring
    :func:`relata.rag_understanding.asmart_rag_query`'s treatment of
    ``hypothesis_fn``.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")

    iterations: list[LoopIteration] = []
    llm_calls = 0
    current_query = query

    for i in range(max_iterations):
        is_last = i == max_iterations - 1
        response = await rag_client.query(current_query, type, purpose=purpose, **rag_kwargs)
        record = LoopIteration(query=current_query, response=response)
        iterations.append(record)

        gate = heuristic_gate(
            embedding_fn(current_query),
            [embedding_fn(hit.text) for hit in response.hits],
        )
        record.gate = gate

        if gate.decision is GateDecision.PASS:
            record.confidence = gate.score
            return LoopResult(response, iterations, llm_calls, "heuristic_pass")

        if gate.decision is GateDecision.RETRY:
            record.confidence = gate.score
            if is_last:
                return LoopResult(response, iterations, llm_calls, "max_iterations")
            current_query = query
            continue

        if grader_fn is None:
            record.confidence = gate.score
            return LoopResult(response, iterations, llm_calls, "no_grader_configured")

        grading = grade_hits(current_query, response.hits, grader_fn=grader_fn)
        llm_calls += 1
        record.grading = grading
        record.confidence = grading.fraction_correct

        if grading.fraction_correct >= LOOP_CONFIDENCE_THRESHOLD:
            return LoopResult(response, iterations, llm_calls, "confident")

        if is_last:
            return LoopResult(response, iterations, llm_calls, "max_iterations")

        if grading.needs_requery and web_search_fallback is not None:
            fallback_response = web_search_fallback(current_query)
            iterations.append(LoopIteration(query=current_query, response=fallback_response))
            return LoopResult(fallback_response, iterations, llm_calls, "web_search_fallback")

        if grading.needs_requery and hypothesis_fn is not None:
            current_query = expand_query_hyde(query, hypothesis_fn=hypothesis_fn)
            llm_calls += 1
        else:
            current_query = query

    raise AssertionError("arun_agentic_loop: exhausted max_iterations without returning")
