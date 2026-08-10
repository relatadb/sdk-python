"""Structural table-of-contents navigation — RAG epic SDK-side agentic
retrieval strategy (#4542).

The substrate half of #4542 (``crates/relata-storage/src/document_ingest.rs``
``document_structure_batch``) derives a ``DocumentStructureNode`` tree from
each document's ``section_path``/``page_start``/``page_end`` — one node per
distinct heading-breadcrumb prefix, plus a synthetic root, each carrying a
cheap LLM-free one-line ``summary`` and its direct ``leaf_chunk_ids``.
``POST /rag/query``'s ``search_mode: structural`` (#4514/#4542,
``crates/relata-cli/src/serve/query/rag_query.rs``) is a single-shot,
one-hop approximation of tree navigation: it lexically matches node
title/summary, then ranks within the matched node(s)' own leaf chunks.

This module is the **true multi-hop agent** the ticket's design section
describes: starting at the document root, [`navigate_structural_tree`]
repeatedly fetches the current node's children ([`fetch_child_nodes`]),
asks a ``select_child`` strategy which subtree to descend into, and stops
either when a level has no children (a leaf) or the strategy declines to
descend further — then ranks that leaf's own ``leaf_chunk_ids`` via a
governed ``RAG_RETRIEVE`` call, exactly like every other ``/rag/query``
mode (same ACL/cell-masking enforcement — no bypass, since this still goes
through :meth:`~relata.client.RelataClient.query`, the same governed `/query`
surface `/rag/query` itself delegates to server-side).

Per ADR-0298, RelataDB has no server-side agent loop — the reasoning that
picks which child to descend into is a caller-supplied ``select_child``
callable, never called by RelataDB itself. [`lexical_child_selector`] is a
zero-dependency, LLM-free default (word-overlap between the question and
each candidate's title/summary) so this module is usable standalone; an
application wanting real judgement supplies its own LLM-backed selector
with the same signature.

Deliberately Python-only, following #4524's ``rag_understanding`` precedent
(``sdks/COVERAGE.md`` records this) — ``RagClient`` itself stays a thin
pass-through on the TypeScript/Go/Rust surfaces with no equivalent module;
``scripts/check_sdk_parity.py``'s domain-module gate only flags a module
present in exactly two of the three published SDKs, so a Python-only module
needs no stub elsewhere.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from relata.client import RelataClient

from relata.exceptions import PurposeError
from relata.models import RagHit, RagQueryResponse
from relata.rag import DEFAULT_TOP_K

#: Hard ceiling on tree descent depth — a runaway `select_child` (e.g. one
#: that always returns a child even on a cyclic/malformed tree) cannot loop
#: forever; #4542's node ids are acyclic by construction (a strict
#: prefix-of-`section_path` tree), so this should never actually bind.
DEFAULT_MAX_DEPTH = 12


def _sql_string_literal(s: str) -> str:
    """Escape ``s`` as a single-quoted SQL string literal — mirrors the
    server's own ``json_to_sql_literal`` for the `String` case
    (``crates/relata-cli/src/serve/query.rs``)."""
    return "'" + s.replace("'", "''") + "'"


@dataclass(frozen=True)
class StructureNode:
    """One `DocumentStructureNode` row (#4542), as returned by a plain
    ``SELECT * FROM DocumentStructureNode`` via
    :meth:`~relata.client.RelataClient.query`."""

    node_id: str
    report_id: str
    parent_id: str | None
    title: str
    depth: int
    page_start: int | None
    page_end: int | None
    summary: str | None
    child_count: int
    leaf_chunk_ids: list[str] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        """``True`` when this node has no children (``child_count == 0``)."""
        return self.child_count == 0

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> StructureNode:
        """Build a :class:`StructureNode` from one ``/query`` result row.

        ``leaf_chunk_ids`` is stored server-side as a JSON array string
        (``crates/relata-storage/src/document_ingest.rs`` — no native array
        `CanonicalValue` variant); this parses it back into a real list,
        defaulting to ``[]`` on anything malformed rather than raising, so a
        node with no directly-attached chunks (an internal/structural-only
        node) never crashes tree descent.
        """
        raw_ids = row.get("leaf_chunk_ids")
        leaf_chunk_ids: list[str]
        if isinstance(raw_ids, list):
            leaf_chunk_ids = [str(x) for x in raw_ids]
        elif isinstance(raw_ids, str) and raw_ids:
            try:
                parsed = json.loads(raw_ids)
            except json.JSONDecodeError:
                parsed = []
            leaf_chunk_ids = [str(x) for x in parsed] if isinstance(parsed, list) else []
        else:
            leaf_chunk_ids = []
        return cls(
            node_id=str(row.get("node_id", "")),
            report_id=str(row.get("report_id", "")),
            parent_id=row.get("parent_id"),
            title=str(row.get("title", "")),
            depth=int(row.get("depth") or 0),
            page_start=row.get("page_start"),
            page_end=row.get("page_end"),
            summary=row.get("summary"),
            child_count=int(row.get("child_count") or 0),
            leaf_chunk_ids=leaf_chunk_ids,
        )


#: A caller-supplied reasoning strategy: given the question and the current
#: level's candidate children, return the child to descend into, or `None`
#: to stop descending (the current node is judged good enough already).
#: RelataDB never calls an LLM itself (ADR-013) — a real agentic strategy is
#: always supplied by the application; see :func:`lexical_child_selector`
#: for the zero-dependency default.
ChildSelector = Callable[[str, "list[StructureNode]"], "StructureNode | None"]


def fetch_child_nodes(
    client: RelataClient,
    *,
    report_id: str,
    parent_id: str | None,
    purpose: str | None = None,
) -> list[StructureNode]:
    """Fetch the direct children of `parent_id` (or the synthetic root's
    single row when `parent_id` is `None`) for `report_id`'s
    `DocumentStructureNode` tree, via a plain governed `SELECT` — the same
    `/query` surface `/rag/query` itself delegates to server-side, so
    ACL/cell-masking apply identically."""
    parent_clause = (
        "parent_id IS NULL"
        if parent_id is None
        else f"parent_id = {_sql_string_literal(parent_id)}"
    )
    sql = (
        f"SELECT * FROM DocumentStructureNode WHERE report_id = "
        f"{_sql_string_literal(report_id)} AND {parent_clause}"
    )
    result = client.query(sql, purpose=purpose)
    return [StructureNode.from_row(row) for row in result.rows]


def fetch_root_node(
    client: RelataClient,
    *,
    report_id: str,
    purpose: str | None = None,
) -> StructureNode | None:
    """Fetch `report_id`'s synthetic tree root (`parent_id IS NULL`), or
    `None` when the document has no `DocumentStructureNode` tree at all —
    e.g. it was ingested with no `section_path` anywhere
    (`document_structure_batch` returns no batch in that case)."""
    roots = fetch_child_nodes(client, report_id=report_id, parent_id=None, purpose=purpose)
    return roots[0] if roots else None


#: Word-boundary tokenizer for :func:`lexical_child_selector` — lowercase
#: alphanumeric runs, no stemming/stopwords (kept deliberately simple: this
#: is a zero-dependency fallback, not a search-quality claim).
_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def lexical_child_selector(question: str, children: list[StructureNode]) -> StructureNode | None:
    """Zero-dependency, LLM-free default :data:`ChildSelector`: picks the
    child whose `title`/`summary` shares the most question words (simple
    token-overlap count), or `None` when no child shares even one word with
    the question — in which case :func:`navigate_structural_tree` stops
    descending at the current node rather than picking an unrelated child.

    Not a substitute for real reasoning over the child summaries (an LLM-
    backed selector will pick better subtrees on paraphrased questions);
    this exists so the module is usable standalone and as the documented
    reference implementation of the `ChildSelector` contract.
    """
    q_tokens = _tokenize(question)
    if not q_tokens:
        return None
    best: StructureNode | None = None
    best_overlap = 0
    for child in children:
        candidate_tokens = _tokenize(f"{child.title} {child.summary or ''}")
        overlap = len(q_tokens & candidate_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best = child
    return best


def _row_to_hit(row: dict[str, Any], *, rerank_requested: bool) -> RagHit:
    """Translate one raw `RAG_RETRIEVE` result row into a :class:`RagHit`,
    mirroring the server's own `translate_hit`
    (`crates/relata-cli/src/serve/query/rag_query.rs`) field-for-field:
    `text_body` -> `text`, `canonical_entity_id` -> `entity_ids` (normalised
    to a list; the stored field accepts either a bare string or an array)."""
    entity_ids_raw = row.get("canonical_entity_id")
    entity_ids: list[str]
    if isinstance(entity_ids_raw, list):
        entity_ids = [str(x) for x in entity_ids_raw]
    elif isinstance(entity_ids_raw, str) and entity_ids_raw:
        entity_ids = [entity_ids_raw]
    else:
        entity_ids = []
    return RagHit(
        bm25_score=float(row.get("_bm25_score") or 0.0),
        vector_score=float(row.get("_vector_score") or 0.0),
        rerank_score=row.get("_rerank_score") if rerank_requested else None,
        chunk_id=str(row.get("chunk_id", "")),
        report_id=str(row.get("report_id", "")),
        text=str(row.get("text_body", "")),
        section_path=list(row.get("section_path") or []),
        page_start=int(row.get("page_start") or 0),
        page_end=int(row.get("page_end") or 0),
        prev_chunk_id=row.get("prev_chunk_id"),
        next_chunk_id=row.get("next_chunk_id"),
        entity_ids=entity_ids,
    )


def _fetch_ranked_leaf_chunks(
    client: RelataClient,
    *,
    type: str,  # noqa: A002 — matches the wire field name verbatim (#4514)
    question: str,
    chunk_ids: list[str],
    purpose: str,
    top_k: int,
    rerank: bool,
) -> RagQueryResponse:
    """Rank `chunk_ids` (a leaf node's own `leaf_chunk_ids`) against
    `question` via a governed `RAG_RETRIEVE`, restricted with `WHERE
    chunk_id IN (...)` — same grammar/ordering as the server's
    `search_mode: structural` second-hop query."""
    in_list = ", ".join(_sql_string_literal(cid) for cid in chunk_ids)
    sql = (
        f"PURPOSE {_sql_string_literal(purpose)} RAG_RETRIEVE FROM {type} "
        f"QUERY {_sql_string_literal(question)} TOP_K {int(top_k)}"
    )
    if rerank:
        sql += " RERANK"
    sql += " WEIGHTS 0.33 0.34 0.33"
    sql += f" WHERE chunk_id IN ({in_list})"
    result = client.query(sql, purpose=purpose)
    hits = [_row_to_hit(row, rerank_requested=rerank) for row in result.rows]
    return RagQueryResponse(hits=hits)


def navigate_structural_tree(
    client: RelataClient,
    *,
    report_id: str,
    question: str,
    type: str,  # noqa: A002
    select_child: ChildSelector = lexical_child_selector,
    purpose: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    rerank: bool = False,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> RagQueryResponse:
    """Agentically navigate `report_id`'s `DocumentStructureNode` tree
    (#4542): descend from the synthetic root, one level per
    :func:`fetch_child_nodes` call, letting `select_child` reason over each
    level's candidate titles/summaries — until a level has no children (a
    leaf) or `select_child` returns `None` (stop here) — then rank that
    node's own `leaf_chunk_ids` against `question`.

    This is the true multi-hop agent loop #4542's design describes,
    complementing (not replacing) `/rag/query`'s single-shot
    `search_mode: structural` (#4514) — `select_child` is where an
    application plugs in real reasoning (e.g. an LLM call over the
    candidates' `summary` fields); :func:`lexical_child_selector` is the
    LLM-free default.

    Every fetch (`fetch_root_node`/`fetch_child_nodes`/the final ranked
    `RAG_RETRIEVE`) goes through :meth:`~relata.client.RelataClient.query`
    — the same governed `/query` surface — so ACL/cell-masking apply
    identically at every level; there is no bypass path.

    Returns an empty :class:`~relata.models.RagQueryResponse` when
    `report_id` has no `DocumentStructureNode` tree at all, or the node
    reached has no `leaf_chunk_ids` — never raises for "nothing to
    navigate," matching every other `search_mode`'s "genuinely nothing
    matched" shape.

    Raises:
        :class:`~relata.exceptions.PurposeError`: No purpose set (falls
            back to `client`'s default purpose when `purpose=None`, same
            contract as :meth:`~relata.rag.RagClient.query`).
    """
    effective_purpose = purpose or getattr(client, "_default_purpose", None)
    if not effective_purpose:
        raise PurposeError(
            "navigate_structural_tree requires a purpose. Pass purpose= to the "
            "call or set a default on the RelataClient."
        )

    current = fetch_root_node(client, report_id=report_id, purpose=effective_purpose)
    if current is None:
        return RagQueryResponse(hits=[])

    depth = 0
    while depth < max_depth:
        children = fetch_child_nodes(
            client, report_id=report_id, parent_id=current.node_id, purpose=effective_purpose
        )
        if not children:
            break
        chosen = select_child(question, children)
        if chosen is None:
            break
        current = chosen
        depth += 1

    if not current.leaf_chunk_ids:
        return RagQueryResponse(hits=[])

    return _fetch_ranked_leaf_chunks(
        client,
        type=type,
        question=question,
        chunk_ids=current.leaf_chunk_ids,
        purpose=effective_purpose,
        top_k=top_k,
        rerank=rerank,
    )
