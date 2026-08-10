"""Citation-injected synthesis + post-synthesis faithfulness scoring —
highest-priority SDK ticket in the RAG epic (#4527).

Mechanism (mirrors the toolkit's confirmed-live pattern studied for this
epic — **not** the platform system's dead `services/rag/` code, which is
never referenced or ported here):

1. **Inline citation injection** — the synthesis LLM call is prompted to
   attach a ``[chunk_id]`` marker to each claim *as it writes it* (not
   appended afterward as a separate pass), grounded in the citation-grade
   fields ``/rag/query`` (#4514, ADR-0299) already returns on every
   :class:`~relata.models.RagHit`: ``chunk_id``, ``section_path``,
   ``page_start``/``page_end``. Every marker in the raw completion is
   resolved against the actual hit set the answer was synthesized from —
   a marker that doesn't match a real ``chunk_id`` is stripped and never
   surfaced as a :class:`Citation`, so a fabricated citation cannot reach
   the caller **by construction**.
2. **Post-synthesis faithfulness pass** — a second, independent LLM call
   splits the synthesized answer into sentences and entailment-checks each
   one against the chunk(s) it cites (or the full retrieved set, for an
   uncited sentence). A sentence that fails entailment is marked with
   ``unsupported_marker`` (default ``"[unsupported]"``) rather than
   silently returned as fact. This runs **by default**
   (``faithfulness_check=True``), not as an opt-in.

RelataDB has no server-side agent loop (ADR-013) — orchestration, including
both LLM calls above, lives in the SDK and is supplied by the caller via
``llm``/``faithfulness_llm``/``entailment_fn``. This module owns only the
prompting, citation resolution, sentence splitting, and faithfulness
bookkeeping around those calls; it is provider-agnostic by design.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

from pydantic import BaseModel, Field

from relata.models import RagHit, RagQueryResponse

#: A single free-text completion call: prompt in, raw text out. The SDK is
#: intentionally provider-agnostic (ADR-013) — callers wire in
#: OpenAI/Anthropic/local models etc.
LlmFn = Callable[[str], str]

#: Sentence-level entailment check: ``(sentence, evidence_chunk_texts) ->
#: supported``. Defaults to an LLM-backed check built from ``llm`` (or
#: ``faithfulness_llm`` when given) — see :func:`synthesize`.
EntailmentFn = Callable[[str, list[str]], bool]

#: Default marker appended to a sentence that fails the faithfulness check.
DEFAULT_UNSUPPORTED_MARKER = "[unsupported]"

_CITATION_RE = re.compile(r"\[([A-Za-z0-9_\-:.]+)\]")

# Good-enough sentence splitter for synthesis output: split on
# sentence-terminal punctuation followed by whitespace and an
# uppercase/digit/citation-marker start. Not a full NLP tokenizer — this
# module never claims linguistic precision, only enough granularity for
# per-claim faithfulness marking.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[])")


class Citation(BaseModel):
    """A citation that traced back to a real retrieved chunk.

    Only ever constructed from an actual :class:`~relata.models.RagHit` in
    the response a :func:`synthesize` call was given — never from an
    unresolved ``[marker]`` in raw LLM output. This is what makes a
    fabricated citation impossible by construction (#4527 acceptance
    criterion).
    """

    chunk_id: str
    report_id: str
    section_path: list[str] = Field(default_factory=list)
    page_start: int
    page_end: int


class SynthesizedSentence(BaseModel):
    """One sentence of the synthesized answer, with its resolved citations
    and faithfulness verdict."""

    text: str = Field(..., description="Rendered sentence, incl. unsupported_marker if flagged")
    citations: list[Citation] = Field(default_factory=list)
    supported: bool = Field(
        True, description="False when the post-synthesis faithfulness pass rejected this sentence"
    )


class SynthesisResult(BaseModel):
    """Result of :func:`synthesize` — a citation-injected answer plus
    per-sentence faithfulness verdicts."""

    answer: str = Field(..., description="Full rendered answer, unsupported sentences marked")
    sentences: list[SynthesizedSentence] = Field(default_factory=list)
    citations: list[Citation] = Field(
        default_factory=list, description="Deduplicated citations across the whole answer"
    )
    unsupported_count: int = Field(0, description="Number of sentences that failed entailment")

    @property
    def has_unsupported_claims(self) -> bool:
        """``True`` when at least one sentence failed the faithfulness pass."""
        return self.unsupported_count > 0


def _hit_index(hits: Iterable[RagHit]) -> dict[str, RagHit]:
    return {hit.chunk_id: hit for hit in hits}


def build_synthesis_prompt(query: str, hits: list[RagHit]) -> str:
    """Build the synthesis LLM prompt: retrieved chunks + inline-citation
    instructions grounded in #4514's citation-grade fields.

    Mirrors the toolkit's live pattern — citations are requested *inline*,
    attached to each claim as the model writes it, not appended afterward
    as a separate pass.
    """
    context_blocks = []
    for hit in hits:
        section = " > ".join(hit.section_path) if hit.section_path else "n/a"
        context_blocks.append(
            f"[{hit.chunk_id}] (section {section}, p.{hit.page_start}-{hit.page_end})\n{hit.text}"
        )
    context = "\n\n".join(context_blocks)
    example_id = hits[0].chunk_id if hits else "chunk-id"
    return (
        "Answer the question using ONLY the evidence chunks below. Every "
        "factual claim MUST be immediately followed by the bracketed chunk "
        "id it came from, e.g. 'RelataDB fuses BM25 and vector retrieval "
        f"natively [{example_id}]'. Never invent a chunk id that is not "
        "listed below. If the evidence does not answer the question, say so "
        "plainly instead of guessing.\n\n"
        f"Question: {query}\n\nEvidence:\n{context}\n\nAnswer:"
    )


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def _resolve_citations_in_sentence(
    sentence: str, hit_index: dict[str, RagHit]
) -> tuple[str, list[Citation]]:
    """Rewrite every ``[marker]`` in ``sentence``: a marker matching a real
    ``chunk_id`` becomes a resolved :class:`Citation`; anything else is
    stripped from the visible text so it can never masquerade as one.
    """
    citations: list[Citation] = []

    def _replace(match: re.Match[str]) -> str:
        token = match.group(1)
        hit = hit_index.get(token)
        if hit is None:
            return ""
        citations.append(
            Citation(
                chunk_id=hit.chunk_id,
                report_id=hit.report_id,
                section_path=hit.section_path,
                page_start=hit.page_start,
                page_end=hit.page_end,
            )
        )
        return f"[{hit.chunk_id}]"

    resolved = _CITATION_RE.sub(_replace, sentence)
    resolved = re.sub(r"\s+", " ", resolved).strip()
    return resolved, citations


def _default_entailment_fn(llm: LlmFn) -> EntailmentFn:
    """Build an :data:`EntailmentFn` that asks ``llm`` a yes/no entailment
    question — the default second LLM call for the faithfulness pass."""

    def check(sentence: str, evidence: list[str]) -> bool:
        if not evidence:
            # Nothing to check the claim against at all: it cannot be
            # entailed by an empty evidence set, so it is unsupported by
            # definition rather than trivially "passing".
            return False
        evidence_block = "\n\n".join(f"- {e}" for e in evidence)
        prompt = (
            "Evidence:\n"
            f"{evidence_block}\n\n"
            f"Claim: {sentence}\n\n"
            "Does the evidence above support this claim? Answer with a "
            "single word, YES or NO."
        )
        raw = llm(prompt).strip().lower()
        return raw.startswith("yes")

    return check


def synthesize(
    query: str,
    response: RagQueryResponse,
    *,
    llm: LlmFn,
    faithfulness_check: bool = True,
    faithfulness_llm: LlmFn | None = None,
    entailment_fn: EntailmentFn | None = None,
    unsupported_marker: str = DEFAULT_UNSUPPORTED_MARKER,
) -> SynthesisResult:
    """Synthesize a governed, citation-injected answer from a
    ``/rag/query`` response (#4514), then faithfulness-check it (#4527).

    Args:
        query: The original question being answered.
        response: A :class:`~relata.models.RagQueryResponse` — typically
            ``RagClient.query(...)``'s return value.
        llm: Text-completion callable used for the synthesis pass. Called
            once with the citation-injection prompt built by
            :func:`build_synthesis_prompt`.
        faithfulness_check: Runs the post-synthesis entailment pass by
            default — matches the toolkit's live pattern; this is not an
            opt-in flag (#4527 acceptance criterion).
        faithfulness_llm: Completion callable for the entailment pass;
            defaults to ``llm`` when unset (same model, a different
            prompt — still a second, independent call per sentence).
        entailment_fn: Override the entailment check entirely (e.g. to use
            a purpose-built classifier instead of an LLM prompt). Takes
            precedence over ``faithfulness_llm``/``llm`` when given.
        unsupported_marker: Suffix appended to a sentence that fails the
            faithfulness check.

    Returns:
        :class:`SynthesisResult` — the citation-injected, faithfulness-
        annotated answer plus per-sentence and deduplicated aggregate
        citation lists.
    """
    hits = response.hits
    hit_index = _hit_index(hits)
    prompt = build_synthesis_prompt(query, hits)
    raw_answer = llm(prompt)

    check_fn = entailment_fn
    if check_fn is None:
        check_fn = _default_entailment_fn(faithfulness_llm or llm)

    sentences: list[SynthesizedSentence] = []
    rendered_sentences: list[str] = []
    seen_chunk_ids: set[str] = set()
    deduped_citations: list[Citation] = []
    unsupported = 0

    for raw_sentence in _split_sentences(raw_answer):
        text, citations = _resolve_citations_in_sentence(raw_sentence, hit_index)
        if not text:
            continue

        supported = True
        if faithfulness_check:
            evidence = [hit_index[c.chunk_id].text for c in citations] or [h.text for h in hits]
            supported = check_fn(text, evidence)
            if not supported:
                unsupported += 1
                text = f"{text} {unsupported_marker}"

        sentences.append(SynthesizedSentence(text=text, citations=citations, supported=supported))
        rendered_sentences.append(text)
        for citation in citations:
            if citation.chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(citation.chunk_id)
                deduped_citations.append(citation)

    return SynthesisResult(
        answer=" ".join(rendered_sentences),
        sentences=sentences,
        citations=deduped_citations,
        unsupported_count=unsupported,
    )
