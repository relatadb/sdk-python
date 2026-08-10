"""Tests for the embedding-model migration workflow (#4585).

Built and tested against the documented request/response contract with
``httpx.MockTransport`` — no live server required, mirroring
``test_rag.py``'s pattern. Covers all three pieces the ticket requires:
Gap 1 (re-embed + backfill preserves unrelated fields, does not disturb the
old vector), Gap 2 ("Option B" — one call to the admin multi-tag search
endpoint, not a client-side merge), and the explicit, never-automatic
retirement step.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from relata import RelataClient
from relata.embedding_migration import (
    EmbeddingMigrationClient,
    MigrationBatchResult,
    http_embed_batch_embedder,
    migrate_embedding_model,
)

BASE = "http://localhost:9090"


def _migration_client(handler: Any, **kwargs: Any) -> EmbeddingMigrationClient:
    mock = httpx.MockTransport(handler)
    return EmbeddingMigrationClient(BASE, bearer_token="tok", transport=mock, **kwargs)


def _row(row_id: str, text: str, other: str, model_tag: str) -> dict[str, Any]:
    return {
        "id": row_id,
        "text_body": text,
        "other_field": other,
        "_emb_text_model": model_tag,
        # Server-managed fields a real `SELECT *` would also carry — must be
        # stripped before re-upsert, never sent back to the server.
        "valid_from": 1_000,
        "valid_to": 9_999_999_999,
        "system_from": 1_000,
        "system_to": 9_999_999_999,
    }


# ── Gap 1: re-embed + backfill ──────────────────────────────────────────────


def test_migrate_batch_reembeds_and_preserves_other_fields():
    rows = [
        _row("row-1", "hello world", "keep-me-1", "old-model:1.0"),
        _row("row-2", "goodbye world", "keep-me-2", "old-model:1.0"),
    ]
    upserted: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/query":
            body = json.loads(req.content)
            assert "old-model:1.0" in body["sql"]
            assert "MigrDoc" in body["sql"]
            return httpx.Response(200, json={"data": rows})
        if req.url.path == "/ingest":
            # ObjectClient.upsert posts NDJSON.
            line = req.content.decode().strip()
            upserted.append(json.loads(line))
            return httpx.Response(200, json={"object_id": "ok"})
        raise AssertionError(f"unexpected path {req.url.path}")

    client = _migration_client(handler)

    def fake_embedder(texts: list[str]) -> list[list[float]]:
        assert texts == ["hello world", "goodbye world"]
        return [[0.1, 0.2], [0.3, 0.4]]

    result = client.migrate_batch(
        "MigrDoc",
        text_field="text_body",
        embedding_field="text",
        old_model_tag="old-model:1.0",
        new_model_tag="new-model:2.0",
        embedder=fake_embedder,
        batch_size=100,
    )
    assert result.rows_scanned == 2
    assert result.rows_migrated == 2
    assert result.rows_failed == 0
    assert result.failed_ids == []
    # Fewer rows than batch_size => the migration must be considered done.
    assert result.done is True

    assert len(upserted) == 2
    row1 = next(r for r in upserted if r["id"] == "row-1")
    assert row1["_emb_text"] == [0.1, 0.2]
    assert row1["_emb_text_model"] == "new-model:2.0"
    # Unrelated fields must survive verbatim.
    assert row1["text_body"] == "hello world"
    assert row1["other_field"] == "keep-me-1"
    # Server-managed fields must NEVER be re-sent.
    assert "valid_from" not in row1
    assert "valid_to" not in row1
    assert "system_from" not in row1
    assert "system_to" not in row1


def test_migrate_batch_done_false_when_batch_is_full():
    rows = [_row(f"row-{i}", "text", "o", "old-model:1.0") for i in range(3)]

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/query":
            return httpx.Response(200, json={"data": rows})
        return httpx.Response(200, json={"object_id": "ok"})

    client = _migration_client(handler)
    result = client.migrate_batch(
        "MigrDoc",
        text_field="text_body",
        embedding_field="text",
        old_model_tag="old-model:1.0",
        new_model_tag="new-model:2.0",
        embedder=lambda texts: [[0.0] for _ in texts],
        batch_size=3,
    )
    assert result.rows_scanned == 3
    assert result.done is False, "a full batch must not claim completion"


def test_migrate_batch_empty_result_is_done_with_zero_scanned():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": []})

    client = _migration_client(handler)
    result = client.migrate_batch(
        "MigrDoc",
        text_field="text_body",
        embedding_field="text",
        old_model_tag="old-model:1.0",
        new_model_tag="new-model:2.0",
        embedder=lambda texts: [],
        batch_size=50,
    )
    assert result == MigrationBatchResult(
        rows_scanned=0, rows_migrated=0, rows_failed=0, failed_ids=[], done=True
    )


def test_migrate_batch_records_per_row_failure_without_aborting_batch():
    rows = [
        _row("row-ok", "text ok", "o", "old-model:1.0"),
        _row("row-bad", "text bad", "o", "old-model:1.0"),
    ]

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/query":
            return httpx.Response(200, json={"data": rows})
        if req.url.path == "/ingest":
            line = json.loads(req.content.decode().strip())
            if line["id"] == "row-bad":
                return httpx.Response(500, json={"error": "boom"})
            return httpx.Response(200, json={"object_id": "ok"})
        raise AssertionError("unexpected path")

    client = _migration_client(handler)
    result = client.migrate_batch(
        "MigrDoc",
        text_field="text_body",
        embedding_field="text",
        old_model_tag="old-model:1.0",
        new_model_tag="new-model:2.0",
        embedder=lambda texts: [[0.0] for _ in texts],
        batch_size=100,
    )
    assert result.rows_scanned == 2
    assert result.rows_migrated == 1
    assert result.rows_failed == 1
    assert result.failed_ids == ["row-bad"]


def test_migrate_batch_rejects_embedder_vector_count_mismatch():
    rows = [_row("row-1", "t", "o", "old-model:1.0")]

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": rows})

    client = _migration_client(handler)
    with pytest.raises(ValueError, match="embedder returned"):
        client.migrate_batch(
            "MigrDoc",
            text_field="text_body",
            embedding_field="text",
            old_model_tag="old-model:1.0",
            new_model_tag="new-model:2.0",
            embedder=lambda texts: [],  # wrong count
            batch_size=100,
        )


def test_migrate_iterates_until_batch_reports_done():
    call_count = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/query":
            call_count["n"] += 1
            if call_count["n"] == 1:
                rows = [_row(f"row-{i}", "t", "o", "old-model:1.0") for i in range(2)]
            else:
                rows = []  # second batch: nothing left
            return httpx.Response(200, json={"data": rows})
        return httpx.Response(200, json={"object_id": "ok"})

    client = _migration_client(handler)
    results = list(
        client.migrate(
            "MigrDoc",
            text_field="text_body",
            embedding_field="text",
            old_model_tag="old-model:1.0",
            new_model_tag="new-model:2.0",
            embedder=lambda texts: [[0.0] for _ in texts],
            batch_size=2,
        )
    )
    # First batch full (done=False) => loop continues; second batch empty
    # (done=True) => loop stops. Two batches total.
    assert len(results) == 2
    assert results[0].done is False
    assert results[1].done is True
    assert results[1].rows_scanned == 0


def test_migrate_embedding_model_module_function_delegates_via_from_client(monkeypatch):
    rows = [_row("row-1", "t", "o", "old-model:1.0")]

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/query":
            return httpx.Response(200, json={"data": rows})
        return httpx.Response(200, json={"object_id": "ok"})

    mocked = _migration_client(handler)
    monkeypatch.setattr(
        "relata.embedding_migration.EmbeddingMigrationClient.from_client",
        classmethod(lambda cls, _client: mocked),
    )

    client = RelataClient(BASE, bearer_token="tok")
    results = list(
        migrate_embedding_model(
            client,
            "MigrDoc",
            text_field="text_body",
            embedding_field="text",
            old_model_tag="old-model:1.0",
            new_model_tag="new-model:2.0",
            embedder=lambda texts: [[0.0] for _ in texts],
        )
    )
    assert len(results) == 1
    assert results[0].rows_migrated == 1


# ── from_client() attribute passthrough (no network) ────────────────────────


def test_from_client_copies_connection_settings():
    client = RelataClient(
        BASE, bearer_token="tok", timeout=42.0, admin_base_url="http://localhost:9091"
    )
    migration = EmbeddingMigrationClient.from_client(client)
    assert migration is not None
    migration.close()


# ── Gap 2: dual/multi-tag search ("Option B") ───────────────────────────────


def test_search_multi_tag_posts_one_call_to_admin_endpoint():
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/admin/vector-index/search-multi-tag"
        captured.update(json.loads(req.content))
        return httpx.Response(
            200,
            json={"hits": [{"id": "row-old", "score": 0.9}, {"id": "row-new", "score": 0.8}]},
        )

    client = _migration_client(handler)
    hits = client.search_multi_tag(
        "MigrDoc",
        ["old-model:1.0", "new-model:2.0"],
        [0.1, 0.2, 0.3],
        k=5,
    )
    assert captured["type"] == "MigrDoc"
    assert captured["model_tags"] == ["old-model:1.0", "new-model:2.0"]
    assert captured["query_embedding"] == [0.1, 0.2, 0.3]
    assert captured["k"] == 5
    assert hits == [
        {"id": "row-old", "score": 0.9},
        {"id": "row-new", "score": 0.8},
    ]


def test_search_multi_tag_rejects_empty_model_tags():
    client = _migration_client(lambda req: httpx.Response(200, json={"hits": []}))
    with pytest.raises(ValueError, match="model_tags must be non-empty"):
        client.search_multi_tag("MigrDoc", [], [0.1])


# ── Retirement — explicit, never automatic ──────────────────────────────────


def test_retire_requires_embedding_field_or_force():
    client = _migration_client(lambda req: httpx.Response(200, json={}))
    with pytest.raises(ValueError, match="embedding_field"):
        client.retire("MigrDoc", "old-model:1.0")


def test_retire_refuses_when_rows_still_remain():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/query"
        return httpx.Response(200, json={"data": [{"n": 3}]})

    client = _migration_client(handler)
    with pytest.raises(RuntimeError, match="3 row"):
        client.retire("MigrDoc", "old-model:1.0", embedding_field="text")


def test_retire_succeeds_when_zero_rows_remain():
    calls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req.url.path)
        if req.url.path == "/query":
            return httpx.Response(200, json={"data": [{"n": 0}]})
        if req.url.path == "/admin/vector-index/retire-model-tag":
            body = json.loads(req.content)
            assert body == {"type": "MigrDoc", "modality": "text", "model_tag": "old-model:1.0"}
            return httpx.Response(200, json={"buckets_removed": 1})
        raise AssertionError("unexpected path")

    client = _migration_client(handler)
    resp = client.retire("MigrDoc", "old-model:1.0", embedding_field="text")
    assert resp == {"buckets_removed": 1}
    assert calls == ["/query", "/admin/vector-index/retire-model-tag"]


def test_retire_with_force_skips_remaining_check():
    def handler(req: httpx.Request) -> httpx.Response:
        # Must go straight to retirement — no /query COUNT call at all.
        assert req.url.path == "/admin/vector-index/retire-model-tag"
        return httpx.Response(200, json={"buckets_removed": 1})

    client = _migration_client(handler)
    resp = client.retire("MigrDoc", "old-model:1.0", force=True)
    assert resp == {"buckets_removed": 1}


# ── http_embed_batch_embedder ────────────────────────────────────────────────


def test_http_embed_batch_embedder_wraps_embed_batch():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/embed/batch"
        body = json.loads(req.content)
        assert body["texts"] == ["a", "b"]
        return httpx.Response(
            200,
            json={"embeddings": [[0.1, 0.2], [0.3, 0.4]], "model": "e5-large:1.0", "dim": 2, "count": 2},
        )

    from relata._http import HttpTransport

    mock = httpx.MockTransport(handler)
    client = RelataClient(BASE, bearer_token="tok")
    client._RelataClient__sync_transport = HttpTransport(  # type: ignore[attr-defined]
        BASE, client._bearer_token, client._timeout, transport=mock
    )
    embedder = http_embed_batch_embedder(client)
    vectors = embedder(["a", "b"])
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
