"""Reasoning-trace write-back — #4528, the final SDK-side stage of the RAG
epic's agentic loop.

**The gap this closes.** `/rag/query` (#4514) already returns governed,
citable rows, and the write path this ticket calls
(`rag_store_answer`/`rag_store_elements`, `crates/relata-cli/src/serve/mcp/
doc_writes.rs`) already turns a final answer into an auditable `RagAnswer` +
`RagSource` row pair. But an agentic loop's *decisions* — why it re-queried
(#4525's heuristic gate + corrective grading), what it discarded (a hit
graded incorrect/ambiguous, a #4526 fan-out strategy excluded as noise), and
how its confidence moved iteration to iteration — lived entirely in the SDK,
never reaching RelataDB's audit log or provenance chain. Today you could
prove *what was read*; you could not prove *how the answer was reached*.
This module is purely the bridge: it takes the real objects a completed
loop already produced (:class:`~relata.rag_loop.LoopResult` from #4525,
:class:`~relata.rag_loop.FanoutResult` from #4526,
:class:`~relata.synthesis.SynthesisResult` from #4527) and writes the full
trace back via the existing `rag_store_answer` MCP tool — no new write
path, two small additive fields on it (`iteration_trace`, `evidence_gaps`,
both JSON-encoded text; see the handler's #4528 doc comment).

RelataDB has no server-side agent loop (ADR-013): this module makes no LLM
call and issues no `/rag/query` call of its own — it is pure bookkeeping
over results the caller already produced.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from relata.mcp import AsyncMcpClient, McpClient

from relata.rag_loop import FanoutResult, HitGrade, LoopResult
from relata.synthesis import Citation, SynthesisResult


def _iteration_trace(loop_result: LoopResult) -> list[dict[str, Any]]:
    """One entry per loop iteration: query text, this iteration's confidence
    signal (:attr:`~relata.rag_loop.LoopIteration.confidence` — the
    heuristic gate's score when no grading ran, else corrective grading's
    ``fraction_correct``), and, when available, the raw gate/grading
    signals behind it. This is #4528's "per-iteration confidence"
    acceptance criterion."""
    trace: list[dict[str, Any]] = []
    for i, iteration in enumerate(loop_result.iterations):
        entry: dict[str, Any] = {
            "iteration": i,
            "query": iteration.query,
            "confidence": iteration.confidence,
            "hit_count": len(iteration.response.hits),
        }
        if iteration.gate is not None:
            entry["gate_decision"] = iteration.gate.decision.value
            entry["gate_score"] = iteration.gate.score
        if iteration.grading is not None:
            entry["fraction_correct"] = iteration.grading.fraction_correct
            entry["graded_hit_count"] = len(iteration.grading.grades)
        trace.append(entry)
    return trace


def _evidence_gaps(
    loop_result: LoopResult, fanout_result: FanoutResult | None
) -> list[dict[str, Any]]:
    """Discarded evidence: hits corrective retrieval grading marked
    ``ambiguous``/``incorrect`` in any iteration, plus — when a #4526
    sub-agent fan-out ran — every strategy it excluded as noise below
    ``LOW_CONFIDENCE_FLOOR``. This is #4528's "discarded evidence"
    acceptance criterion; it stays empty (not absent — an empty list, same
    as a loop with nothing to discard) when neither source has anything."""
    gaps: list[dict[str, Any]] = []
    for i, iteration in enumerate(loop_result.iterations):
        if iteration.grading is None:
            continue
        for hit, grade in zip(iteration.response.hits, iteration.grading.grades):
            if grade is not HitGrade.CORRECT:
                gaps.append(
                    {
                        "iteration": i,
                        "chunk_id": hit.chunk_id,
                        "grade": grade.value,
                        "reason": "corrective_grading",
                    }
                )
    if fanout_result is not None:
        for excluded in fanout_result.excluded:
            gaps.append(
                {
                    "strategy": excluded.strategy.name,
                    "confidence": excluded.confidence,
                    "reason": "fanout_low_confidence",
                }
            )
    return gaps


def _final_confidence(loop_result: LoopResult) -> float:
    """The loop's final confidence signal — the last iteration's
    :attr:`~relata.rag_loop.LoopIteration.confidence`, or ``0.0`` for the
    degenerate case of a loop result with no iterations recorded."""
    if loop_result.iterations:
        confidence = loop_result.iterations[-1].confidence
        if confidence is not None:
            return confidence
    return 0.0


def _faithfulness_score(synthesis_result: SynthesisResult) -> float:
    """``1 - (unsupported sentence count / total)``, matching the
    ``RagAnswer.faithfulness`` field's documented 0-1 scale — or the
    server's own ``-1`` "not computed" sentinel when there were no
    sentences to score (see `doc_writes.rs`'s ``faith`` default)."""
    total = len(synthesis_result.sentences)
    if total == 0:
        return -1.0
    return 1.0 - (synthesis_result.unsupported_count / total)


def _source_confidence(citation: Citation, loop_result: LoopResult) -> float:
    """Look up ``citation.chunk_id``'s ``relevance_confidence`` (#4520) from
    whichever iteration's hits it came from — searched newest iteration
    first, since a re-query's hit set is the more relevant provenance for a
    repeated chunk id. ``0.0`` when the id is not found in any iteration
    (should not happen for a citation resolved from a real hit, but this
    function never raises on it) or the field is unset."""
    for iteration in reversed(loop_result.iterations):
        for hit in iteration.response.hits:
            if hit.chunk_id == citation.chunk_id:
                return hit.relevance_confidence or 0.0
    return 0.0


def _sources_payload(
    synthesis_result: SynthesisResult, loop_result: LoopResult
) -> list[dict[str, Any]]:
    """Build the `rag_store_answer` ``sources`` array from the synthesis
    result's deduplicated, by-construction-real citations (#4527) —
    never from raw LLM output, so a fabricated source cannot reach this
    payload any more than it can reach :attr:`SynthesisResult.citations`
    itself."""
    return [
        {
            "id": citation.chunk_id,
            "source": citation.report_id,
            "score": _source_confidence(citation, loop_result),
            "page": citation.page_start,
            "sectionPath": citation.section_path,
        }
        for citation in synthesis_result.citations
    ]


def build_reasoning_trace(
    query: str,
    loop_result: LoopResult,
    synthesis_result: SynthesisResult,
    *,
    fanout_result: FanoutResult | None = None,
    case_id: str | None = None,
    purpose: str | None = None,
    duration_ms: int | None = None,
) -> dict[str, Any]:
    """Build the full ``rag_store_answer`` argument map from a completed
    #4525 loop + #4527 synthesis result (and, when a #4526 fan-out ran, its
    :class:`~relata.rag_loop.FanoutResult`): iteration count, per-iteration
    confidence, discarded evidence, and final citations — not just the
    final answer text.

    Exposed separately from :func:`store_reasoning_trace` so a caller can
    inspect or log the exact payload before it goes over the wire.
    """
    args: dict[str, Any] = {
        "question": query,
        "answer": synthesis_result.answer,
        "confidence": _final_confidence(loop_result),
        "num_iterations": len(loop_result.iterations),
        "faithfulness": _faithfulness_score(synthesis_result),
        "sources": _sources_payload(synthesis_result, loop_result),
        "iteration_trace": _iteration_trace(loop_result),
        "evidence_gaps": _evidence_gaps(loop_result, fanout_result),
    }
    if case_id:
        args["case_id"] = case_id
    if purpose:
        args["purpose"] = purpose
    if duration_ms is not None:
        args["duration_ms"] = duration_ms
    return args


def store_reasoning_trace(
    mcp_client: McpClient,
    query: str,
    loop_result: LoopResult,
    synthesis_result: SynthesisResult,
    *,
    fanout_result: FanoutResult | None = None,
    case_id: str | None = None,
    purpose: str | None = None,
    duration_ms: int | None = None,
) -> dict[str, Any]:
    """Write the full reasoning trace back via the existing `rag_store_answer`
    MCP tool (`crates/relata-cli/src/serve/mcp/doc_writes.rs`) — the SDK-side
    half of #4528. Call this at the end of the loop (#4525/#4526's call site,
    after #4527's :func:`~relata.synthesis.synthesize`), not mocked: the
    written row carries the real iteration/confidence/evidence trail this
    specific run produced.

    Returns the unwrapped MCP result dict — e.g. ``{"stored": "RagAnswer",
    "source_rows": N, "has_iteration_trace": true, "evidence_gap_count": M,
    ...}``.
    """
    args = build_reasoning_trace(
        query,
        loop_result,
        synthesis_result,
        fanout_result=fanout_result,
        case_id=case_id,
        purpose=purpose,
        duration_ms=duration_ms,
    )
    return mcp_client.call_tool("rag_store_answer", args)


async def astore_reasoning_trace(
    mcp_client: AsyncMcpClient,
    query: str,
    loop_result: LoopResult,
    synthesis_result: SynthesisResult,
    *,
    fanout_result: FanoutResult | None = None,
    case_id: str | None = None,
    purpose: str | None = None,
    duration_ms: int | None = None,
) -> dict[str, Any]:
    """Async variant of :func:`store_reasoning_trace`."""
    args = build_reasoning_trace(
        query,
        loop_result,
        synthesis_result,
        fanout_result=fanout_result,
        case_id=case_id,
        purpose=purpose,
        duration_ms=duration_ms,
    )
    return await mcp_client.call_tool("rag_store_answer", args)
