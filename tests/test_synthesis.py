"""Tests for citation injection + post-synthesis faithfulness scoring —
RAG epic highest-priority SDK ticket (#4527, wraps #4514's ADR-0299
citation-grade fields and consumes #4523's typed ``/rag/query`` client).

Built and tested against the frozen ``/rag/query`` contract with a mocked
HTTP transport (``httpx.MockTransport``, same approach #4523's fix-agent
used) — no live server, and no live LLM provider: every LLM call in these
tests is a plain Python stub, since RelataDB has no server-side agent loop
(ADR-013) and this module is intentionally provider-agnostic.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from relata import RelataClient
from relata._http import AsyncHttpTransport, HttpTransport
from relata.models import RagQueryResponse
from relata.rag import RagClient
from relata.synthesis import (
    Citation,
    build_synthesis_prompt,
    synthesize,
)

BASE = "http://localhost:9090"

_HIT_SUPPORTED = {
    "bm25_score": 4.2,
    "vector_score": 0.83,
    "rerank_score": None,
    "chunk_id": "chunk-1",
    "report_id": "doc-1",
    "text": "RelataDB fuses BM25 and vector retrieval natively via RRF.",
    "section_path": ["3", "3.2"],
    "page_start": 5,
    "page_end": 6,
    "prev_chunk_id": None,
    "next_chunk_id": "chunk-2",
    "entity_ids": ["ent-1"],
}
_HIT_OTHER = {
    "bm25_score": 3.1,
    "vector_score": 0.61,
    "rerank_score": None,
    "chunk_id": "chunk-2",
    "report_id": "doc-1",
    "text": "The fused ranking uses reciprocal rank fusion (RRF) across channels.",
    "section_path": ["3", "3.3"],
    "page_start": 6,
    "page_end": 7,
    "prev_chunk_id": "chunk-1",
    "next_chunk_id": None,
    "entity_ids": [],
}


_UNSET = object()


def _response(hits: list[dict[str, Any]] | None | object = _UNSET) -> RagQueryResponse:
    """Build a :class:`RagQueryResponse`. No argument -> the two default
    hits; ``None`` or ``[]`` explicitly -> genuinely zero hits."""
    if hits is _UNSET:
        hits = [_HIT_SUPPORTED, _HIT_OTHER]
    elif hits is None:
        hits = []
    return RagQueryResponse.model_validate({"hits": hits})


def _mock_client(handler: Any) -> RelataClient:
    client = RelataClient(BASE, bearer_token="tok")
    mock = httpx.MockTransport(handler)
    extra = client._extra_headers  # type: ignore[attr-defined]
    client._RelataClient__sync_transport = HttpTransport(  # type: ignore[attr-defined]
        BASE, client._bearer_token, client._timeout, transport=mock, extra_headers=extra
    )
    client._RelataClient__async_transport = AsyncHttpTransport(  # type: ignore[attr-defined]
        BASE, client._bearer_token, client._timeout, transport=mock, extra_headers=extra
    )
    return client


# ── build_synthesis_prompt — inline citation instructions ──────────────────


def test_prompt_includes_citation_grade_fields_for_every_hit():
    response = _response()
    prompt = build_synthesis_prompt("How does RelataDB rank hybrid results?", response.hits)
    assert "[chunk-1]" in prompt
    assert "[chunk-2]" in prompt
    assert "3 > 3.2" in prompt  # section_path breadcrumb
    assert "p.5-6" in prompt
    assert "RelataDB fuses BM25 and vector retrieval natively via RRF." in prompt
    assert "Never invent a chunk id" in prompt


# ── synthesize() — inline citation injection, no fabrication possible ──────


def test_synthesize_resolves_real_citations_inline():
    """Citations attached during synthesis (not appended afterward) and
    every one traces to a real chunk_id/section_path/page from the
    response that produced it (#4527 AC1)."""

    def llm(_prompt: str) -> str:
        return "RelataDB fuses BM25 and vector retrieval natively [chunk-1]."

    def always_supported(_sentence: str, _evidence: list[str]) -> bool:
        return True

    result = synthesize(
        "How does RelataDB rank hybrid results?",
        _response(),
        llm=llm,
        entailment_fn=always_supported,
    )

    assert len(result.citations) == 1
    citation = result.citations[0]
    assert isinstance(citation, Citation)
    assert citation.chunk_id == "chunk-1"
    assert citation.report_id == "doc-1"
    assert citation.section_path == ["3", "3.2"]
    assert citation.page_start == 5
    assert citation.page_end == 6
    assert "[chunk-1]" in result.answer
    assert result.sentences[0].citations[0].chunk_id == "chunk-1"


def test_synthesize_strips_fabricated_citation_by_construction():
    """A bracket marker that does not match a real chunk_id can never
    surface as a Citation — dropped from the visible answer instead of
    rendering as a fake source (#4527 AC1: 'no fabricated citations
    possible by construction')."""

    def llm(_prompt: str) -> str:
        return "RelataDB invented this fact [chunk-does-not-exist]."

    def always_supported(_sentence: str, _evidence: list[str]) -> bool:
        return True

    result = synthesize("q", _response(), llm=llm, entailment_fn=always_supported)

    assert result.citations == []
    assert "chunk-does-not-exist" not in result.answer
    assert result.sentences[0].citations == []


def test_synthesize_dedupes_citations_across_sentences():
    def llm(_prompt: str) -> str:
        return (
            "RelataDB fuses BM25 and vector retrieval natively [chunk-1]. "
            "It uses RRF to combine channel scores [chunk-1]."
        )

    def always_supported(_sentence: str, _evidence: list[str]) -> bool:
        return True

    result = synthesize("q", _response(), llm=llm, entailment_fn=always_supported)

    assert [c.chunk_id for c in result.citations] == ["chunk-1"]
    assert len(result.sentences) == 2


# ── faithfulness pass — on by default, marks unsupported claims ────────────


def test_faithfulness_check_runs_by_default():
    """AC3: faithfulness checking is on by default, not opt-in."""
    calls: list[str] = []

    def llm(_prompt: str) -> str:
        return "RelataDB fuses BM25 and vector retrieval natively [chunk-1]."

    def entailment_fn(sentence: str, _evidence: list[str]) -> bool:
        calls.append(sentence)
        return True

    synthesize("q", _response(), llm=llm, entailment_fn=entailment_fn)

    # entailment_fn was invoked without the caller passing
    # faithfulness_check=True explicitly.
    assert len(calls) == 1


def test_synthesize_marks_deliberately_injected_unsupported_claim():
    """AC2: a deliberately injected unsupported claim gets marked, not
    silently returned as fact."""

    def llm(_prompt: str) -> str:
        return (
            "RelataDB fuses BM25 and vector retrieval natively [chunk-1]. "
            "RelataDB was founded on the moon in 1969 [chunk-2]."
        )

    def entailment_fn(sentence: str, evidence: list[str]) -> bool:
        # Only the fabricated "founded on the moon" claim fails entailment
        # against its cited evidence — deliberately injected by this test
        # fixture to prove faithfulness marking actually happens.
        return "moon" not in sentence.lower()

    result = synthesize("q", _response(), llm=llm, entailment_fn=entailment_fn)

    assert result.unsupported_count == 1
    assert result.has_unsupported_claims is True
    supported_sentence, unsupported_sentence = result.sentences
    assert supported_sentence.supported is True
    assert "[unsupported]" not in supported_sentence.text
    assert unsupported_sentence.supported is False
    assert unsupported_sentence.text.endswith("[unsupported]")
    assert "[unsupported]" in result.answer
    # The unsupported sentence still carries its (real) citation — marking
    # it doesn't erase which chunk it claimed to come from.
    assert unsupported_sentence.citations[0].chunk_id == "chunk-2"


def test_synthesize_custom_unsupported_marker():
    def llm(_prompt: str) -> str:
        return "A totally fabricated claim with no citation at all."

    def never_supported(_sentence: str, _evidence: list[str]) -> bool:
        return False

    result = synthesize(
        "q",
        _response(),
        llm=llm,
        entailment_fn=never_supported,
        unsupported_marker="[NEEDS VERIFICATION]",
    )
    assert result.answer.endswith("[NEEDS VERIFICATION]")


def test_faithfulness_check_can_be_disabled():
    def llm(_prompt: str) -> str:
        return "Some claim [chunk-1]."

    def never_called(_sentence: str, _evidence: list[str]) -> bool:
        raise AssertionError("entailment_fn must not be called when faithfulness_check=False")

    result = synthesize(
        "q", _response(), llm=llm, entailment_fn=never_called, faithfulness_check=False
    )
    assert result.unsupported_count == 0
    assert all(s.supported for s in result.sentences)


def test_default_entailment_fn_makes_a_second_independent_llm_call():
    """Mechanism point 2: faithfulness is a second, independent LLM call —
    not a heuristic reuse of the synthesis output."""
    prompts: list[str] = []

    def llm(prompt: str) -> str:
        prompts.append(prompt)
        if "Claim:" in prompt:
            return "NO"
        return "RelataDB fuses BM25 and vector retrieval natively [chunk-1]."

    result = synthesize("q", _response(), llm=llm)

    assert len(prompts) == 2  # one synthesis call, one entailment call
    assert "Claim:" in prompts[1]
    assert "Evidence:" in prompts[1]
    assert result.unsupported_count == 1


def test_uncited_sentence_with_no_evidence_is_unsupported_by_default():
    def llm(_prompt: str) -> str:
        return "This sentence cites nothing."

    def entailment_fn(_sentence: str, evidence: list[str]) -> bool:
        # No citation resolved -> evidence falls back to every retrieved
        # hit's text (non-empty here), so this exercises the "has some
        # evidence but still fails" path rather than the empty-evidence
        # short-circuit covered by test_default_entailment_fn's "NO" case.
        return len(evidence) == 0

    result = synthesize("q", _response(), llm=llm, entailment_fn=entailment_fn)
    assert result.unsupported_count == 1


# ── end-to-end: RagClient (mocked transport) -> synthesize() ───────────────


def test_synthesize_end_to_end_from_mocked_rag_client():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/rag/query"
        json.loads(req.content)  # sanity: valid JSON body
        return httpx.Response(200, json={"hits": [_HIT_SUPPORTED, _HIT_OTHER]})

    client = _mock_client(handler)
    response = RagClient.from_client(client).query(
        "How does RelataDB rank hybrid results?", "DocumentChunk", purpose="research"
    )

    def llm(_prompt: str) -> str:
        return "RelataDB fuses BM25 and vector retrieval natively [chunk-1]."

    def entailment_fn(_sentence: str, _evidence: list[str]) -> bool:
        return True

    result = synthesize(
        "How does RelataDB rank hybrid results?", response, llm=llm, entailment_fn=entailment_fn
    )

    assert result.citations[0].chunk_id == "chunk-1"
    assert result.unsupported_count == 0


@pytest.mark.parametrize("hits", [[], None])
def test_synthesize_with_no_hits_produces_no_citations(hits):
    def llm(_prompt: str) -> str:
        return "I don't have enough information to answer that."

    def entailment_fn(_sentence: str, evidence: list[str]) -> bool:
        return len(evidence) == 0

    result = synthesize("q", _response(hits), llm=llm, entailment_fn=entailment_fn)
    assert result.citations == []
