"""Tests for structural table-of-contents navigation (#4542).

Built against `RelataClient.query()` (the generic governed `/query` surface)
via `httpx.MockTransport` — no live server required, same approach
#4524's `test_rag_understanding.py` uses.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from relata import RelataClient
from relata._http import AsyncHttpTransport, HttpTransport
from relata.exceptions import PurposeError
from relata.structural_navigation import (
    StructureNode,
    fetch_child_nodes,
    fetch_root_node,
    lexical_child_selector,
    navigate_structural_tree,
)

BASE = "http://localhost:9090"


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


def _query_result(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"data": rows, "query_id": "qid-1", "elapsed_ms": 1}


def _queued_client(responses: list[list[dict[str, Any]]]) -> tuple[RelataClient, list[str]]:
    """A client whose `/query` calls return `responses` in call order.
    Returns the client plus the list of `sql` strings sent, for assertions."""
    sent_sql: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        payload = json.loads(req.content)
        sent_sql.append(payload["sql"])
        rows = responses[len(sent_sql) - 1]
        return httpx.Response(200, json=_query_result(rows))

    return _mock_client(handler), sent_sql


_ROOT_ROW = {
    "node_id": "doc_1::",
    "report_id": "doc_1",
    "parent_id": None,
    "title": "filing.pdf",
    "depth": 0,
    "page_start": 1,
    "page_end": 10,
    "summary": None,
    "child_count": 1,
    "leaf_chunk_ids": "[]",
}

_PART1_ROW = {
    "node_id": "doc_1::Part I",
    "report_id": "doc_1",
    "parent_id": "doc_1::",
    "title": "Part I",
    "depth": 1,
    "page_start": 1,
    "page_end": 4,
    "summary": "Liquidity and termination clauses.",
    "child_count": 0,
    "leaf_chunk_ids": '["ch_1", "ch_2"]',
}

_UNRELATED_ROW = {
    "node_id": "doc_1::Part II",
    "report_id": "doc_1",
    "parent_id": "doc_1::",
    "title": "Part II",
    "depth": 1,
    "page_start": 5,
    "page_end": 8,
    "summary": "Renewal options.",
    "child_count": 0,
    "leaf_chunk_ids": '["ch_3"]',
}

_LEAF_CHUNK_HIT_ROW = {
    "chunk_id": "ch_1",
    "report_id": "doc_1",
    "text_body": "Liquidity terms are defined here.",
    "section_path": ["Part I"],
    "page_start": 1,
    "page_end": 1,
    "prev_chunk_id": None,
    "next_chunk_id": "ch_2",
    "canonical_entity_id": "e_1",
    "_bm25_score": 2.1,
    "_vector_score": 0.9,
}


# ── StructureNode.from_row ──────────────────────────────────────────────────


def test_structure_node_from_row_parses_leaf_chunk_ids_json_string():
    node = StructureNode.from_row(_PART1_ROW)
    assert node.node_id == "doc_1::Part I"
    assert node.parent_id == "doc_1::"
    assert node.title == "Part I"
    assert node.leaf_chunk_ids == ["ch_1", "ch_2"]
    assert node.is_leaf  # child_count == 0


def test_structure_node_from_row_handles_missing_and_malformed_leaf_ids():
    row = dict(_ROOT_ROW)
    row["leaf_chunk_ids"] = "not json"
    node = StructureNode.from_row(row)
    assert node.leaf_chunk_ids == []
    assert not node.is_leaf  # child_count == 1


# ── fetch_root_node / fetch_child_nodes ─────────────────────────────────────


def test_fetch_root_node_queries_parent_id_is_null():
    client, sent_sql = _queued_client([[_ROOT_ROW]])
    node = fetch_root_node(client, report_id="doc_1", purpose="research")
    assert node is not None
    assert node.node_id == "doc_1::"
    assert "parent_id IS NULL" in sent_sql[0]
    assert "report_id = 'doc_1'" in sent_sql[0]


def test_fetch_root_node_returns_none_when_no_tree():
    client, _ = _queued_client([[]])
    assert fetch_root_node(client, report_id="doc_missing", purpose="research") is None


def test_fetch_child_nodes_scopes_by_parent_id():
    client, sent_sql = _queued_client([[_PART1_ROW, _UNRELATED_ROW]])
    children = fetch_child_nodes(
        client, report_id="doc_1", parent_id="doc_1::", purpose="research"
    )
    assert [c.title for c in children] == ["Part I", "Part II"]
    assert "parent_id = 'doc_1::'" in sent_sql[0]


# ── lexical_child_selector ───────────────────────────────────────────────────


def test_lexical_child_selector_picks_best_overlap():
    children = [StructureNode.from_row(_PART1_ROW), StructureNode.from_row(_UNRELATED_ROW)]
    chosen = lexical_child_selector("What are the liquidity termination clauses?", children)
    assert chosen is not None
    assert chosen.title == "Part I"


def test_lexical_child_selector_returns_none_on_no_overlap():
    children = [StructureNode.from_row(_PART1_ROW), StructureNode.from_row(_UNRELATED_ROW)]
    assert lexical_child_selector("zzz qqq nonsense", children) is None


# ── navigate_structural_tree — the multi-hop descent ────────────────────────


def test_navigate_structural_tree_descends_and_ranks_leaf_chunks():
    client, sent_sql = _queued_client(
        [
            [_ROOT_ROW],  # 1. root
            [_PART1_ROW, _UNRELATED_ROW],  # 2. root's children
            [],  # 3. Part I's children — none, it's a leaf
            [_LEAF_CHUNK_HIT_ROW],  # 4. ranked RAG_RETRIEVE over ch_1/ch_2
        ]
    )
    response = navigate_structural_tree(
        client,
        report_id="doc_1",
        question="What are the liquidity termination clauses?",
        type="DocumentChunk",
        purpose="research",
    )
    assert len(sent_sql) == 4
    assert "parent_id IS NULL" in sent_sql[0]
    assert "parent_id = 'doc_1::'" in sent_sql[1]
    assert "parent_id = 'doc_1::Part I'" in sent_sql[2]
    assert "RAG_RETRIEVE FROM DocumentChunk" in sent_sql[3]
    assert "chunk_id IN ('ch_1', 'ch_2')" in sent_sql[3]

    assert len(response.hits) == 1
    hit = response.hits[0]
    assert hit.chunk_id == "ch_1"
    assert hit.text == "Liquidity terms are defined here."
    assert hit.bm25_score == 2.1
    assert hit.vector_score == 0.9
    assert hit.entity_ids == ["e_1"]
    assert hit.rerank_score is None


def test_navigate_structural_tree_returns_empty_when_no_tree():
    client, sent_sql = _queued_client([[]])
    response = navigate_structural_tree(
        client,
        report_id="doc_flat",
        question="anything",
        type="DocumentChunk",
        purpose="research",
    )
    assert response.hits == []
    assert len(sent_sql) == 1  # stopped after the root fetch found nothing


def test_navigate_structural_tree_stops_when_selector_declines():
    # Root has children, but the (custom) selector never picks one — descent
    # stops at the root, and the root itself has no leaf_chunk_ids.
    client, sent_sql = _queued_client(
        [
            [_ROOT_ROW],
            [_PART1_ROW, _UNRELATED_ROW],
        ]
    )
    response = navigate_structural_tree(
        client,
        report_id="doc_1",
        question="totally unrelated query text",
        type="DocumentChunk",
        purpose="research",
        select_child=lambda _q, _children: None,
    )
    assert response.hits == []
    assert len(sent_sql) == 2  # root + one children fetch, no leaf-ranking call


def test_navigate_structural_tree_requires_purpose():
    client, _ = _queued_client([[_ROOT_ROW]])
    client._default_purpose = None  # type: ignore[attr-defined]
    with pytest.raises(PurposeError):
        navigate_structural_tree(
            client, report_id="doc_1", question="q", type="DocumentChunk"
        )
