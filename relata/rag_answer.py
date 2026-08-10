"""The composed end-to-end RAG answer pipeline — the missing piece found
during a post-epic alignment audit of #4529.

**The gap this closes.** Every RAG-epic SDK stage shipped correctly and is
independently tested — dispatch (:mod:`relata.rag_understanding`, #4524),
the cost-ladder loop + fan-out (:mod:`relata.rag_loop`, #4525/#4526),
citation+faithfulness synthesis (:mod:`relata.synthesis`, #4527), and
reasoning-trace write-back (:mod:`relata.rag_trace`, #4528) — but nothing
called them *together*. :func:`relata.rag_understanding.smart_rag_query`
does dispatch and a single retrieval pass; it never runs the loop, never
synthesizes, never writes a trace back. This module is that composition:
one call that gates, retrieves (with iteration or fan-out), synthesizes a
cited and faithfulness-checked answer, and records the full reasoning trace
as governed rows — the epic's own validation criterion ("an end-to-end
query through the SDK loop produces a cited, faithfulness-checked answer
with its reasoning trace recorded as governed rows").

**Design note — why this doesn't call :func:`smart_rag_query` directly.**
``smart_rag_query`` bundles its front-door gate (content-safety + SQL-shape
routing) together with a single-pass retrieval call it makes itself. This
module needs the *gate* but not that single retrieval pass — retrieval here
is handled by :func:`~relata.rag_loop.run_agentic_loop` (iteration) or
:func:`~relata.rag_loop.run_subagent_fanout` (breadth), neither of which
:func:`smart_rag_query` can delegate to. So this module calls the same gate
primitives :func:`smart_rag_query` calls internally
(:func:`~relata.rag_understanding.check_content_safety`,
:func:`~relata.rag_understanding.classify_query_shape` + the ``route_*``
functions) directly, once, up front — not a second, divergent gate
implementation, the identical functions, just not wrapped in
``smart_rag_query``'s own retrieval call.

**Design note — content-safety/SQL-routing runs once, not per loop
iteration.** A corrective-grading re-query is a HyDE-refined variant of the
*same already-vetted query text* (see :mod:`relata.rag_loop`'s
``needs_requery`` handling), not new untrusted input — so gating it again
per iteration would be redundant, not safer.

**Design note — fan-out and the loop are alternate retrieval strategies,
not layered.** When ``fanout_strategies`` is given, :func:`run_subagent_fanout`
provides the one and only retrieval pass (breadth: several parallel
strategies, deterministically merged) and the cost-ladder loop's iteration
machinery does not run — its result is wrapped as a single-iteration
:class:`~relata.rag_loop.LoopResult` (``stopped_reason="fanout_complete"``)
purely so :func:`~relata.rag_trace.store_reasoning_trace` has the shape it
expects; no iteration/re-query happens in this path. Omit
``fanout_strategies`` for the loop's iteration/re-query behavior (depth)
instead.

**Known limitation, not solved here.** A :attr:`~relata.models.RagQueryResponse.clarification`
(#4534, entity disambiguation) surfacing mid-loop is not specially handled
— the loop's zero-hits heuristic (:func:`~relata.rag_loop.heuristic_gate`
scores empty hits ``0.0`` -> ``RETRY``) means it will retry to
``max_iterations`` and give up, a bounded and honest failure rather than a
silent wrong answer, but not a resolution. Resolving a clarification is a
separate, multi-turn caller decision by design (#4534's own
``resume_with_selection`` flow) — not something one call should
auto-resolve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    from relata.mcp import AsyncMcpClient, McpClient
    from relata.rag import AsyncRagClient, RagClient

from relata.models import RagQueryResponse
from relata.rag_loop import (
    MAX_ITERATIONS,
    EmbeddingFn,
    FanoutResult,
    GraderFn,
    LoopIteration,
    LoopResult,
    SubAgentStrategy,
    WebSearchFallbackFn,
    arun_agentic_loop,
    arun_subagent_fanout,
    run_agentic_loop,
    run_subagent_fanout,
)
from relata.rag_trace import astore_reasoning_trace, store_reasoning_trace
from relata.rag_understanding import (
    SQL_ROUTABLE_SHAPES,
    HypothesisGenerator,
    QueryShape,
    check_content_safety,
    classify_query_shape,
    route_aggregation_query,
    route_attribute_filter_query,
    route_boolean_query,
    route_negation_query,
    route_ranking_query,
)
from relata.synthesis import EntailmentFn, LlmFn, SynthesisResult, synthesize


@dataclass
class RagAnswerResult:
    """Final result of :func:`run_rag_answer`/:func:`arun_rag_answer`.

    ``loop_result``/``fanout_result``/``synthesis_result``/``trace`` are
    ``None`` when the query was refused or SQL-routed before any retrieval
    happened — check ``response.refused``/``response.sql_result`` first.
    """

    query: str
    response: RagQueryResponse
    loop_result: LoopResult | None
    fanout_result: FanoutResult | None
    synthesis_result: SynthesisResult | None
    trace: dict[str, Any] | None


def _dispatch_sql_shape(
    client: Any,
    query: str,
    type: str,  # noqa: A002
    shape: QueryShape,
    *,
    purpose: str | None,
    structured_field_map: Mapping[str, str] | None,
    structured_known_fields: Any,
    attribute_field_map: Mapping[str, str] | None,
    attribute_known_fields: Any,
) -> Any:
    """Mirrors :func:`relata.rag_understanding.smart_rag_query`'s SQL-shape
    dispatch if/elif chain exactly — same functions, same field-map
    contract, not a reimplementation."""
    if shape is QueryShape.ATTRIBUTE_FILTER:
        return route_attribute_filter_query(
            client, query, type,
            purpose=purpose, field_map=attribute_field_map, known_fields=attribute_known_fields,
        )
    if shape is QueryShape.AGGREGATION:
        return route_aggregation_query(
            client, query, type,
            purpose=purpose, field_map=structured_field_map, known_fields=structured_known_fields,
        )
    if shape is QueryShape.NEGATION:
        return route_negation_query(
            client, query, type,
            purpose=purpose, field_map=structured_field_map, known_fields=structured_known_fields,
        )
    if shape is QueryShape.BOOLEAN:
        return route_boolean_query(
            client, query, type,
            purpose=purpose, field_map=structured_field_map, known_fields=structured_known_fields,
        )
    return route_ranking_query(  # QueryShape.RANKING
        client, query, type,
        purpose=purpose, field_map=structured_field_map, known_fields=structured_known_fields,
    )


def _fanout_as_loop_result(fanout_result: FanoutResult, query: str) -> LoopResult:
    """Wrap a :class:`~relata.rag_loop.FanoutResult` as a single-iteration
    :class:`~relata.rag_loop.LoopResult` purely so
    :func:`~relata.rag_trace.store_reasoning_trace` (which requires a
    ``LoopResult``) has the shape it expects. No iteration/re-query
    happened — this is bookkeeping, not a claim that the loop ran."""
    return LoopResult(
        response=fanout_result.merged_response,
        iterations=[
            LoopIteration(
                query=query,
                response=fanout_result.merged_response,
                confidence=fanout_result.winner.confidence,
            )
        ],
        llm_calls=0,
        stopped_reason="fanout_complete",
    )


def run_rag_answer(
    rag_client: RagClient,
    mcp_client: McpClient,
    query: str,
    type: str,  # noqa: A002 — matches RagClient.query's wire-facing param name
    *,
    llm: LlmFn,
    embedding_fn: EmbeddingFn,
    purpose: str | None = None,
    grader_fn: GraderFn | None = None,
    hypothesis_fn: HypothesisGenerator | None = None,
    web_search_fallback: WebSearchFallbackFn | None = None,
    max_iterations: int = MAX_ITERATIONS,
    fanout_strategies: Sequence[SubAgentStrategy] | None = None,
    faithfulness_check: bool = True,
    faithfulness_llm: LlmFn | None = None,
    entailment_fn: EntailmentFn | None = None,
    content_safety_patterns: Mapping[str, str] | None = None,
    structured_field_map: Mapping[str, str] | None = None,
    structured_known_fields: Any = None,
    attribute_field_map: Mapping[str, str] | None = None,
    attribute_known_fields: Any = None,
    case_id: str | None = None,
    duration_ms: int | None = None,
    **rag_kwargs: Any,
) -> RagAnswerResult:
    """Run the full RAG-epic pipeline end to end: gate -> retrieve
    (iterate or fan out) -> synthesize + cite + faithfulness-check ->
    write the reasoning trace back as governed rows.

    Returns immediately after the gate, with ``loop_result``/
    ``fanout_result``/``synthesis_result``/``trace`` all ``None``, when the
    query was refused (:attr:`~relata.models.RagQueryResponse.refused`) or
    answered via SQL instead of retrieval
    (:attr:`~relata.models.RagQueryResponse.sql_result`) — see the module
    docstring for why gating happens once, not per iteration.
    """
    refusal = check_content_safety(query, patterns=content_safety_patterns)
    if refusal is not None:
        response = RagQueryResponse(hits=[], refused=refusal)
        return RagAnswerResult(query, response, None, None, None, None)

    shape = classify_query_shape(query)
    if shape in SQL_ROUTABLE_SHAPES:
        client = rag_client._client  # noqa: SLF001 — mirrors smart_rag_query's own access
        sql_result = _dispatch_sql_shape(
            client, query, type, shape,
            purpose=purpose,
            structured_field_map=structured_field_map,
            structured_known_fields=structured_known_fields,
            attribute_field_map=attribute_field_map,
            attribute_known_fields=attribute_known_fields,
        )
        if sql_result is not None:
            response = RagQueryResponse(hits=[], sql_result=sql_result)
            return RagAnswerResult(query, response, None, None, None, None)
        # Shape was SQL-routable but no predicate could be built (unknown
        # vocabulary/fields) — fall through to retrieval, flagged low-confidence
        # on the eventual response, exactly as smart_rag_query does.

    fanout_result: FanoutResult | None = None
    if fanout_strategies is not None:
        fanout_result = run_subagent_fanout(
            rag_client, query, type, fanout_strategies, purpose=purpose, **rag_kwargs
        )
        loop_result = _fanout_as_loop_result(fanout_result, query)
    else:
        loop_result = run_agentic_loop(
            rag_client, query, type,
            purpose=purpose,
            embedding_fn=embedding_fn,
            grader_fn=grader_fn,
            hypothesis_fn=hypothesis_fn,
            web_search_fallback=web_search_fallback,
            max_iterations=max_iterations,
            **rag_kwargs,
        )
    final_response = loop_result.response
    if shape in SQL_ROUTABLE_SHAPES:
        final_response.low_confidence = True
        final_response.low_confidence_reason = (
            f"{shape.value}-shaped query could not be routed to SQL "
            f"(no matching canonical field on {type!r}); fell back to retrieval "
            f"(#4536/#4535)"
        )

    synthesis_result = synthesize(
        query, final_response,
        llm=llm,
        faithfulness_check=faithfulness_check,
        faithfulness_llm=faithfulness_llm,
        entailment_fn=entailment_fn,
    )

    trace = store_reasoning_trace(
        mcp_client, query, loop_result, synthesis_result,
        fanout_result=fanout_result, case_id=case_id, purpose=purpose, duration_ms=duration_ms,
    )

    return RagAnswerResult(query, final_response, loop_result, fanout_result, synthesis_result, trace)


async def arun_rag_answer(
    rag_client: AsyncRagClient,
    mcp_client: AsyncMcpClient,
    query: str,
    type: str,  # noqa: A002
    *,
    llm: LlmFn,
    embedding_fn: EmbeddingFn,
    purpose: str | None = None,
    grader_fn: GraderFn | None = None,
    hypothesis_fn: HypothesisGenerator | None = None,
    web_search_fallback: WebSearchFallbackFn | None = None,
    max_iterations: int = MAX_ITERATIONS,
    fanout_strategies: Sequence[SubAgentStrategy] | None = None,
    faithfulness_check: bool = True,
    faithfulness_llm: LlmFn | None = None,
    entailment_fn: EntailmentFn | None = None,
    content_safety_patterns: Mapping[str, str] | None = None,
    structured_field_map: Mapping[str, str] | None = None,
    structured_known_fields: Any = None,
    attribute_field_map: Mapping[str, str] | None = None,
    attribute_known_fields: Any = None,
    case_id: str | None = None,
    duration_ms: int | None = None,
    **rag_kwargs: Any,
) -> RagAnswerResult:
    """Async variant of :func:`run_rag_answer`. ``llm``/``embedding_fn``/
    ``grader_fn``/``hypothesis_fn``/``web_search_fallback``/``entailment_fn``
    stay synchronous callables, mirroring every other async entry point in
    this SDK's RAG modules."""
    refusal = check_content_safety(query, patterns=content_safety_patterns)
    if refusal is not None:
        response = RagQueryResponse(hits=[], refused=refusal)
        return RagAnswerResult(query, response, None, None, None, None)

    shape = classify_query_shape(query)
    if shape in SQL_ROUTABLE_SHAPES:
        client = rag_client._client  # noqa: SLF001
        sql_result = _dispatch_sql_shape(
            client, query, type, shape,
            purpose=purpose,
            structured_field_map=structured_field_map,
            structured_known_fields=structured_known_fields,
            attribute_field_map=attribute_field_map,
            attribute_known_fields=attribute_known_fields,
        )
        if sql_result is not None:
            response = RagQueryResponse(hits=[], sql_result=sql_result)
            return RagAnswerResult(query, response, None, None, None, None)

    fanout_result: FanoutResult | None = None
    if fanout_strategies is not None:
        fanout_result = await arun_subagent_fanout(
            rag_client, query, type, fanout_strategies, purpose=purpose, **rag_kwargs
        )
        loop_result = _fanout_as_loop_result(fanout_result, query)
    else:
        loop_result = await arun_agentic_loop(
            rag_client, query, type,
            purpose=purpose,
            embedding_fn=embedding_fn,
            grader_fn=grader_fn,
            hypothesis_fn=hypothesis_fn,
            web_search_fallback=web_search_fallback,
            max_iterations=max_iterations,
            **rag_kwargs,
        )
    final_response = loop_result.response
    if shape in SQL_ROUTABLE_SHAPES:
        final_response.low_confidence = True
        final_response.low_confidence_reason = (
            f"{shape.value}-shaped query could not be routed to SQL "
            f"(no matching canonical field on {type!r}); fell back to retrieval "
            f"(#4536/#4535)"
        )

    synthesis_result = synthesize(
        query, final_response,
        llm=llm,
        faithfulness_check=faithfulness_check,
        faithfulness_llm=faithfulness_llm,
        entailment_fn=entailment_fn,
    )

    trace = await astore_reasoning_trace(
        mcp_client, query, loop_result, synthesis_result,
        fanout_result=fanout_result, case_id=case_id, purpose=purpose, duration_ms=duration_ms,
    )

    return RagAnswerResult(query, final_response, loop_result, fanout_result, synthesis_result, trace)
