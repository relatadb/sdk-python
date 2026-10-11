from __future__ import annotations

import pytest
import httpx

from relata.natural_language import (
    aquery_in_natural_language,
    asummarise_memory,
    query_in_natural_language,
    summarise_memory,
)


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def query_schema(self, *, purpose: str) -> dict[str, object]:
        self.calls.append(("schema", purpose))
        return {"object_types": ["Person"], "link_types": ["Knows"]}

    def query(self, sql: str, *, purpose: str) -> dict[str, object]:
        self.calls.append(("query", sql))
        return {
            "rows": [{"name": "Ada", "citation": {"chunk_id": "chunk-1", "report_id": "doc-1"}}],
            "purpose": purpose,
        }

    async def aquery_schema(self, *, purpose: str) -> dict[str, object]:
        return self.query_schema(purpose=purpose)

    async def aquery(self, sql: str, *, purpose: str) -> dict[str, object]:
        return self.query(sql, purpose=purpose)


def test_query_in_natural_language_uses_schema_and_governed_query() -> None:
    client = FakeClient()
    prompts: list[str] = []

    def complete(prompt: str, max_tokens: int) -> str:
        prompts.append(prompt)
        assert max_tokens == 512
        return "SELECT * FROM Person"

    result = query_in_natural_language(
        client, "find a person", purpose="analytics", complete=complete
    )
    assert "Visible object types: Person" in prompts[0]
    assert result["sql"] == "SELECT * FROM Person LIMIT 100"
    assert client.calls == [
        ("schema", "analytics"),
        ("query", "SELECT * FROM Person LIMIT 100"),
    ]


@pytest.mark.asyncio
async def test_async_query_in_natural_language_calls_async_transport() -> None:
    client = FakeClient()
    result = await aquery_in_natural_language(
        client,
        "find a person",
        purpose="analytics",
        complete=lambda _prompt, _budget: "SELECT * FROM Person LIMIT 1",
    )
    assert result["sql"] == "SELECT * FROM Person LIMIT 1"


def test_read_only_query_rejects_write_and_large_limit() -> None:
    client = FakeClient()
    for sql in ("DELETE FROM Person", "SELECT * FROM Person LIMIT 101"):
        with pytest.raises(ValueError):
            query_in_natural_language(
                client, "find a person", purpose="analytics", complete=lambda *_: sql
            )
    assert not any(call[0] == "query" for call in client.calls)


def test_answer_callback_receives_governed_citation_rows() -> None:
    client = FakeClient()
    seen: list[dict[str, object]] = []

    def answer(_question: str, result: dict[str, object], max_tokens: int) -> dict[str, str]:
        assert max_tokens == 512
        rows = result["rows"]
        assert isinstance(rows, list)
        seen.append(rows[0]["citation"])
        return {"text": "Ada is supported by chunk-1", "model_id": "answer-model"}

    result = query_in_natural_language(
        client,
        "find a person",
        purpose="analytics",
        complete=lambda _prompt, _budget: {"text": "SELECT * FROM Person", "model_id": "query-model"},
        answer=answer,
    )
    assert seen == [{"chunk_id": "chunk-1", "report_id": "doc-1"}]
    assert result["answer"] == "Ada is supported by chunk-1"
    assert result["answer_model_id"] == "answer-model"
    assert result["model_id"] == "query-model"


def test_provider_and_governed_query_errors_propagate() -> None:
    client = FakeClient()
    with pytest.raises(RuntimeError, match="provider unavailable"):
        query_in_natural_language(
            client,
            "find a person",
            purpose="analytics",
            complete=lambda *_: (_ for _ in ()).throw(RuntimeError("provider unavailable")),
        )
    assert not any(call[0] == "query" for call in client.calls)

    class TimedOutClient(FakeClient):
        def query(self, sql: str, *, purpose: str) -> dict[str, object]:
            self.calls.append(("query", sql))
            raise httpx.ReadTimeout("configured client request timeout")

    with pytest.raises(httpx.ReadTimeout, match="configured client request timeout"):
        query_in_natural_language(
            TimedOutClient(),
            "find a person",
            purpose="analytics",
            complete=lambda *_: "SELECT * FROM Person",
        )


class FakeMcp:
    def __init__(self) -> None:
        self.stored: dict[str, object] = {}

    def recall(self, query: str, **kwargs: object) -> dict[str, object]:
        assert query == "recent notes"
        assert kwargs["session_id"] == "session-1"
        return {"rows": [{"id": "m1", "content": "note"}]}

    def remember(self, content: str, **kwargs: object) -> dict[str, object]:
        self.stored = {"content": content, **kwargs}
        return {"stored": "MemoryItem", "id": "summary-1"}


class FakeAsyncMcp:
    async def recall(self, query: str, **kwargs: object) -> dict[str, object]:
        assert query == "recent notes"
        return {"rows": [{"id": "m1", "content": "note"}]}

    async def remember(self, content: str, **kwargs: object) -> dict[str, object]:
        return {"stored": "MemoryItem", "id": "summary-1", "content": content}


def test_summarise_memory_persists_via_remember_with_source_provenance() -> None:
    mcp = FakeMcp()
    result = summarise_memory(
        mcp,
        "recent notes",
        purpose="analytics",
        session_id="session-1",
        complete=lambda prompt, _budget: "Summary of note" if "m1" in prompt else "No source",
    )
    assert result["source_ids"] == ["m1"]
    assert mcp.stored["session_id"] == "session-1"
    assert mcp.stored["fields"] == {"source_ids": ["m1"], "summary_source": "recall"}


@pytest.mark.asyncio
async def test_async_summarise_memory_uses_async_callbacks() -> None:
    mcp = FakeAsyncMcp()
    result = await asummarise_memory(
        mcp,
        "recent notes",
        purpose="analytics",
        session_id="session-1",
        complete=lambda _prompt, _budget: "Summary of note",
    )
    assert result["summary"] == "Summary of note"
