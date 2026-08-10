"""Tests for :mod:`relata.coref` — session-scoped coreference resolution
(RAG epic, #4530).

Uses httpx's ``MockTransport`` against a tiny in-process fake ``MemoryItem``
store that reproduces the real server's session-exact-match scoping
(`crates/relata-cli/src/serve/mcp/memory.rs`'s `session_filter`/`matches_agency`
logic) closely enough to prove the four acceptance criteria without a live
server: two-turn resolution, bounded one-row state, zero-cost single-turn
calls, and cross-session isolation.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from relata import AsyncMemory, CorefResolver, Memory
from relata.coref import AsyncCorefResolver, _has_unresolved_pronoun, subject_from_hit
from relata.models import RagHit

BASE = "http://localhost:9090"


def _mcp(result: dict[str, Any]) -> dict[str, Any]:
    """Wrap a result the way the server's mcp_ok() does."""
    return {"content": [{"type": "text", "text": json.dumps(result)}], "isError": False}


class _FakeMemoryServer:
    """Minimal in-process fake reproducing the ``/memory/*`` contract this
    module depends on: ``remember`` inserts, ``consolidate`` supersedes (pop
    old id, insert new — mirroring the server's bi-temporal retract+re-insert),
    and ``recall`` filters by exact ``session_id`` match, same as
    `mcp_tool_recall`'s `session_filter` branch."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, str]] = {}
        self.remember_calls = 0
        self.consolidate_calls = 0
        self.recall_calls = 0
        self._next_id = 0

    def _new_id(self) -> str:
        self._next_id += 1
        return f"mem-{self._next_id}"

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/memory/remember":
            self.remember_calls += 1
            body = json.loads(request.content)
            mid = self._new_id()
            self.rows[mid] = {"content": body["content"], "session_id": body["session_id"]}
            return httpx.Response(200, json=_mcp({"id": mid, "session_id": body["session_id"]}))
        if path == "/memory/consolidate":
            self.consolidate_calls += 1
            body = json.loads(request.content)
            old = self.rows.pop(body["id"], None)
            new_id = self._new_id()
            session_id = old["session_id"] if old else ""
            self.rows[new_id] = {"content": body["content"], "session_id": session_id}
            return httpx.Response(200, json=_mcp({"new_id": new_id}))
        if path == "/memory/recall":
            self.recall_calls += 1
            sid = request.url.params.get("session_id", "")
            rows = [
                {"id": mid, "content": r["content"], "session_id": r["session_id"]}
                for mid, r in self.rows.items()
                if r["session_id"] == sid
            ]
            return httpx.Response(200, json=_mcp({"rows": rows}))
        raise AssertionError(f"unexpected path: {path}")


def _resolver() -> tuple[CorefResolver, _FakeMemoryServer]:
    server = _FakeMemoryServer()
    memory = Memory(BASE, purpose="agent-notes", transport=httpx.MockTransport(server.handler))
    return CorefResolver(memory), server


# ── the acceptance-criteria regression test ────────────────────────────────


def test_two_turn_sequence_resolves_pronoun_to_prior_subject() -> None:
    """"What is India's capital?" -> "Where is it located?" resolves "it" -> "India"."""
    coref, _server = _resolver()

    coref.remember_subject("session-1", "India")
    resolved = coref.resolve("Where is it located?", "session-1")

    assert resolved == "Where is India located?"


# ── acceptance: exactly one row, never a growing list ───────────────────────


def test_state_is_exactly_one_row_and_never_grows_with_turn_count() -> None:
    coref, server = _resolver()

    coref.remember_subject("session-1", "India")
    assert server.remember_calls == 1
    assert len(server.rows) == 1

    coref.resolve("Where is it located?", "session-1")
    coref.remember_subject("session-1", "Paris")  # turn 2 supersedes turn 1
    assert server.remember_calls == 1  # no second `add`
    assert server.consolidate_calls == 1  # superseded via consolidate
    assert len(server.rows) == 1  # still exactly one live row

    coref.remember_subject("session-1", "Berlin")  # turn 3 supersedes turn 2
    assert server.remember_calls == 1
    assert server.consolidate_calls == 2
    assert len(server.rows) == 1

    resolved = coref.resolve("Where is it now?", "session-1")
    assert resolved == "Where is Berlin now?"


# ── acceptance: single-turn callers pay no extra cost ───────────────────────


def test_no_session_id_makes_zero_network_calls() -> None:
    coref, server = _resolver()

    resolved = coref.resolve("Where is it located?", "")

    assert resolved == "Where is it located?"
    assert server.recall_calls == 0
    assert server.remember_calls == 0


def test_query_without_unresolved_pronoun_makes_zero_network_calls() -> None:
    coref, server = _resolver()

    resolved = coref.resolve("What is the capital of France?", "session-1")

    assert resolved == "What is the capital of France?"
    assert server.recall_calls == 0


def test_pronoun_with_no_prior_subject_returns_query_unchanged() -> None:
    """A genuinely single-turn caller whose session never had a subject
    remembered gets back the original query — not a mangled rewrite — after
    exactly one bounded (top_k=1) recall lookup."""
    coref, server = _resolver()

    resolved = coref.resolve("Where is it located?", "session-never-seen")

    assert resolved == "Where is it located?"
    assert server.recall_calls == 1


def test_local_antecedent_suppresses_the_trip() -> None:
    """A pronoun with a local antecedent in the same sentence needs no
    cross-turn state — no network call at all."""
    coref, server = _resolver()
    coref.remember_subject("session-1", "India")
    recall_calls_after_remember = server.recall_calls

    query = "What did Marie Curie discover and how did it change physics?"
    resolved = coref.resolve(query, "session-1")

    assert resolved == query
    assert server.recall_calls == recall_calls_after_remember  # resolve() made no new call


# ── acceptance: cross-session/cross-tenant isolation ────────────────────────


def test_cross_session_isolation() -> None:
    coref, _server = _resolver()

    coref.remember_subject("session-a", "India")
    resolved = coref.resolve("Where is it located?", "session-b")

    # session-b never stored a subject, so it must not see session-a's.
    assert resolved == "Where is it located?"


# ── subject_from_hit / anaphora-check unit coverage ─────────────────────────


def _rag_hit(**overrides: Any) -> RagHit:
    base = {
        "bm25_score": 1.0,
        "vector_score": 1.0,
        "rerank_score": None,
        "chunk_id": "chunk-1",
        "report_id": "doc-1",
        "text": "text",
        "section_path": [],
        "page_start": 1,
        "page_end": 1,
        "prev_chunk_id": None,
        "next_chunk_id": None,
        "entity_ids": [],
    }
    base.update(overrides)
    return RagHit.model_validate(base)


def test_subject_from_hit_uses_first_entity_id() -> None:
    hit = _rag_hit(entity_ids=["India", "New Delhi"])
    assert subject_from_hit(hit) == "India"


def test_subject_from_hit_returns_none_without_entity_ids() -> None:
    assert subject_from_hit(_rag_hit(entity_ids=[])) is None


def test_has_unresolved_pronoun() -> None:
    assert _has_unresolved_pronoun("Where is it located?") is True
    assert _has_unresolved_pronoun("What is the capital of France?") is False
    assert (
        _has_unresolved_pronoun("What did Marie Curie discover and how did it change physics?")
        is False
    )


# ── async mirror ─────────────────────────────────────────────────────────


async def test_async_two_turn_sequence_resolves_pronoun() -> None:
    server = _FakeMemoryServer()
    memory = AsyncMemory(
        BASE, purpose="agent-notes", transport=httpx.MockTransport(server.handler)
    )
    coref = AsyncCorefResolver(memory)

    await coref.remember_subject("session-1", "India")
    resolved = await coref.resolve("Where is it located?", "session-1")

    assert resolved == "Where is India located?"
    assert server.remember_calls == 1
    assert server.consolidate_calls == 0


async def test_async_no_session_id_makes_zero_network_calls() -> None:
    server = _FakeMemoryServer()
    memory = AsyncMemory(
        BASE, purpose="agent-notes", transport=httpx.MockTransport(server.handler)
    )
    coref = AsyncCorefResolver(memory)

    resolved = await coref.resolve("Where is it located?", "")

    assert resolved == "Where is it located?"
    assert server.recall_calls == 0
