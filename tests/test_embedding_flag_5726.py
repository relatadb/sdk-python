"""#5726: search / RAG / recall results expose the query-embedding flag and warn on fallback."""

from __future__ import annotations

import logging

import pytest

from relata import EmbeddingFlag, QueryResult, RagQueryResponse

FALLBACK = {
    "mode": "lexical-fallback",
    "semantic": False,
    "model": "cpu-deterministic:v1",
    "reason": "no endpoint",
}


def test_query_result_reads_flag_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="relata"):
        result = QueryResult.model_validate({"rows": 0, "data": [], "embedding": FALLBACK})
    assert isinstance(result.embedding, EmbeddingFlag)
    assert result.embedding.mode == "lexical-fallback"
    assert result.embedding.semantic is False
    assert result.embedding.reason == "no endpoint"
    assert "lexical fallback" in caplog.text


def test_rag_response_semantic_flag_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    flag = {"mode": "semantic", "semantic": True, "model": "e5:1", "reason": None}
    with caplog.at_level(logging.WARNING, logger="relata"):
        resp = RagQueryResponse.model_validate({"hits": [], "embedding": flag})
    assert resp.embedding is not None and resp.embedding.semantic is True
    assert "lexical fallback" not in caplog.text


def test_rag_response_model_mismatch_warns(caplog: pytest.LogCaptureFixture) -> None:
    flag = dict(FALLBACK, reason="model mismatch")
    with caplog.at_level(logging.WARNING, logger="relata"):
        resp = RagQueryResponse.model_validate({"hits": [], "embedding": flag})
    assert resp.embedding is not None and resp.embedding.reason == "model mismatch"
    assert "model mismatch" in caplog.text
