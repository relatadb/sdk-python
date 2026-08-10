"""Tests for MMR diversity selection (#4526).

Pure-function tests over `relata.models.RagHit` — no client, no mocked
transport needed, mirroring `test_rag_understanding.py`'s treatment of
`rrf_merge`/`rrf_k_for_fanout` (also pure functions over already-retrieved
`RagQueryResponse` data).
"""

from __future__ import annotations

from typing import Any

import pytest

from relata.models import RagHit
from relata.rag_rank import (
    DEFAULT_MMR_LAMBDA,
    MMR_LAMBDA_BY_PURPOSE,
    default_relevance,
    default_text_similarity,
    mmr_lambda_for_purpose,
    mmr_select,
    mmr_select_for_purpose,
)


def _hit(chunk_id: str, **overrides: Any) -> RagHit:
    base: dict[str, Any] = {
        "bm25_score": 1.0,
        "vector_score": 1.0,
        "rerank_score": None,
        "chunk_id": chunk_id,
        "report_id": "doc-1",
        "text": f"text for {chunk_id}",
        "section_path": [],
        "page_start": 1,
        "page_end": 1,
        "prev_chunk_id": None,
        "next_chunk_id": None,
        "entity_ids": [],
        "relevance_confidence": None,
    }
    base.update(overrides)
    return RagHit.model_validate(base)


# ── domain-configurable lambda (AC #3) ──────────────────────────────────────


def test_verified_lambda_defaults_by_domain():
    assert MMR_LAMBDA_BY_PURPOSE["legal"] == 0.7
    assert MMR_LAMBDA_BY_PURPOSE["clinical"] == 0.7
    assert MMR_LAMBDA_BY_PURPOSE["medical"] == 0.7
    assert MMR_LAMBDA_BY_PURPOSE["finance"] == 0.65
    assert MMR_LAMBDA_BY_PURPOSE["research"] == 0.6
    assert MMR_LAMBDA_BY_PURPOSE["general"] == 0.5
    assert MMR_LAMBDA_BY_PURPOSE["code"] == 0.4
    assert DEFAULT_MMR_LAMBDA == 0.5


@pytest.mark.parametrize(
    "purpose,expected",
    [
        ("legal", 0.7),
        ("Legal", 0.7),  # case-insensitive
        ("CLINICAL", 0.7),
        ("medical", 0.7),
        ("finance", 0.65),
        ("research", 0.6),
        ("general", 0.5),
        ("code", 0.4),
    ],
)
def test_mmr_lambda_for_purpose_known_domains(purpose: str, expected: float):
    assert mmr_lambda_for_purpose(purpose) == expected


def test_mmr_lambda_for_purpose_unknown_falls_back_to_default():
    assert mmr_lambda_for_purpose("marketing") == DEFAULT_MMR_LAMBDA


def test_mmr_lambda_for_purpose_none_falls_back_to_default():
    assert mmr_lambda_for_purpose(None) == DEFAULT_MMR_LAMBDA


# ── default relevance/similarity are pure functions over sub-scores ────────


def test_default_relevance_prefers_relevance_confidence_when_present():
    hit = _hit("c1", relevance_confidence=0.42, bm25_score=1.0, vector_score=1.0)
    assert default_relevance(hit) == pytest.approx(0.42)


def test_default_relevance_falls_back_to_score_average():
    hit = _hit("c1", relevance_confidence=None, bm25_score=0.8, vector_score=0.4)
    assert default_relevance(hit) == pytest.approx(0.6)


def test_default_text_similarity_identical_text_is_one():
    a = _hit("a", text="quarterly revenue rose sharply")
    b = _hit("b", text="quarterly revenue rose sharply")
    assert default_text_similarity(a, b) == pytest.approx(1.0)


def test_default_text_similarity_disjoint_text_is_zero():
    a = _hit("a", text="quarterly revenue")
    b = _hit("b", text="unrelated topic entirely")
    assert default_text_similarity(a, b) == pytest.approx(0.0)


def test_default_text_similarity_partial_overlap():
    a = _hit("a", text="alpha beta gamma")
    b = _hit("b", text="alpha beta delta")
    # tokens: {alpha,beta,gamma} vs {alpha,beta,delta} -> intersection 2, union 4
    assert default_text_similarity(a, b) == pytest.approx(0.5)


# ── mmr_select — pure, deterministic, no LLM/embedding/network call ────────


def test_mmr_select_lambda_1_is_pure_relevance_ranking():
    hits = [
        _hit("low", relevance_confidence=0.2, text="alpha"),
        _hit("high", relevance_confidence=0.9, text="alpha"),
        _hit("mid", relevance_confidence=0.5, text="alpha"),
    ]
    selected = mmr_select(hits, lambda_mult=1.0)
    assert [h.chunk_id for h in selected] == ["high", "mid", "low"]


def test_mmr_select_lambda_0_maximizes_diversity_only():
    # All three equally relevant; "dup" is a near-duplicate of "first".
    hits = [
        _hit("first", relevance_confidence=0.9, text="alpha beta gamma"),
        _hit("dup", relevance_confidence=0.9, text="alpha beta gamma"),
        _hit("diverse", relevance_confidence=0.9, text="totally unrelated content"),
    ]
    selected = mmr_select(hits, lambda_mult=0.0, top_n=2)
    assert selected[0].chunk_id == "first"  # first pick: no selection yet, ties broken by order
    assert selected[1].chunk_id == "diverse"  # second pick avoids the near-duplicate


def test_mmr_select_respects_top_n():
    hits = [_hit(f"c{i}", relevance_confidence=1.0 - i * 0.1) for i in range(5)]
    selected = mmr_select(hits, lambda_mult=1.0, top_n=2)
    assert len(selected) == 2
    assert [h.chunk_id for h in selected] == ["c0", "c1"]


def test_mmr_select_top_n_larger_than_pool_returns_everything():
    hits = [_hit("a"), _hit("b")]
    selected = mmr_select(hits, top_n=10)
    assert len(selected) == 2


def test_mmr_select_empty_input_returns_empty():
    assert mmr_select([]) == []


def test_mmr_select_rejects_out_of_range_lambda():
    with pytest.raises(ValueError):
        mmr_select([_hit("a")], lambda_mult=1.5)
    with pytest.raises(ValueError):
        mmr_select([_hit("a")], lambda_mult=-0.1)


def test_mmr_select_is_deterministic_across_repeated_calls():
    hits = [
        _hit("a", relevance_confidence=0.8, text="one two three"),
        _hit("b", relevance_confidence=0.7, text="two three four"),
        _hit("c", relevance_confidence=0.6, text="five six seven"),
    ]
    first = [h.chunk_id for h in mmr_select(hits, lambda_mult=0.5)]
    second = [h.chunk_id for h in mmr_select(hits, lambda_mult=0.5)]
    assert first == second


def test_mmr_select_accepts_custom_relevance_and_similarity_fns():
    hits = [_hit("a"), _hit("b"), _hit("c")]
    calls: list[str] = []

    def relevance_fn(hit: RagHit) -> float:
        calls.append(hit.chunk_id)
        return {"a": 0.1, "b": 0.9, "c": 0.5}[hit.chunk_id]

    def similarity_fn(x: RagHit, y: RagHit) -> float:
        return 0.0

    selected = mmr_select(hits, lambda_mult=1.0, relevance_fn=relevance_fn, similarity_fn=similarity_fn)
    assert [h.chunk_id for h in selected] == ["b", "c", "a"]
    assert set(calls) == {"a", "b", "c"}  # every candidate's relevance was evaluated


# ── mmr_select_for_purpose — the purpose-keyed convenience entry point ─────


def test_mmr_select_for_purpose_uses_domain_lambda():
    hits = [
        _hit("low", relevance_confidence=0.2, text="alpha"),
        _hit("high", relevance_confidence=0.9, text="alpha"),
    ]
    # "legal" -> lambda=0.7, still relevance-dominated for this simple case.
    selected = mmr_select_for_purpose(hits, "legal")
    assert selected[0].chunk_id == "high"


def test_mmr_select_for_purpose_none_uses_general_default():
    hits = [_hit("a", relevance_confidence=0.9), _hit("b", relevance_confidence=0.1)]
    selected = mmr_select_for_purpose(hits, None)
    assert selected[0].chunk_id == "a"
