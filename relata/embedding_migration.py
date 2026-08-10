"""Embedding-model migration workflow (#4585, ADR-0298 SDK-side orchestration).

RelataDB's storage layer already supports multiple embedding models
coexisting side by side for the same field — vectors bucket on
``(object_type, modality, model_tag, tenant_id)`` (``VectorIndexKey``,
``crates/relata-storage/src/store/mod.rs``), so re-embedding a row under a
new model never disturbs the old model's vector for that row. What was
missing is a *packaged workflow* that actually drives a migration: scan the
corpus, re-embed via a caller-supplied embedder, write the new vector back,
search across both models during the cutover window, and retire the old
model's vectors once complete. This module is that workflow, per ADR-0298's
substrate/orchestration split (RelataDB provides the primitives; the SDK
owns the multi-step process).

Three pieces, matching the ticket's two gaps + retirement step:

1. **Re-embed + backfill** (Gap 1) — :meth:`EmbeddingMigrationClient.migrate`
   pages rows still tagged ``old_model_tag``, re-embeds each via a
   caller-supplied ``embedder`` callable (RelataDB's own ``/embed/batch``,
   wrapped by :func:`http_embed_batch_embedder`, or any external embedder),
   and writes the new vector back tagged ``new_model_tag`` — via the
   ordinary ``/query``/``/ingest`` data-plane surface, no admin auth needed.
2. **Dual/multi-tag search** (Gap 2, "Option B") —
   :meth:`EmbeddingMigrationClient.search_multi_tag` calls the new
   admin-gated ``POST /admin/vector-index/search-multi-tag`` endpoint, which
   wraps ``relata_storage::ObjectStore::vector_search_multi_tag_for_tenant``:
   ONE server-side call scans every named ``model_tag``'s bucket and returns
   a SINGLE fused, ranked result. This is a real storage/query-layer
   primitive, not this SDK issuing N single-tag searches and merging
   client-side — the exact design decision the ticket records. A caller
   mid-migration is never blind to not-yet-migrated content, closing the
   #4497/#4512 "model-tag mismatch = silent zero results" failure class for
   the *other* tag's rows during the migration window.
3. **Retirement** — :meth:`EmbeddingMigrationClient.retire` calls
   ``POST /admin/vector-index/retire-model-tag``, dropping the old model's
   entire vector bucket. **Never automatic**: `retire()` refuses unless the
   caller either supplies `embedding_field` (so it can verify zero rows
   remain on `old_model_tag` itself, via
   :meth:`EmbeddingMigrationClient.remaining_on_old_tag`) or passes
   `force=True` to explicitly bypass that check.

Both Gap-2 search and retirement require admin auth (they call
``/admin/vector-index/*`` routes, mounted only on the loopbound admin
control-plane listener per ADR-0261) — construct with `admin_base_url`/an
admin bearer token when the deployment splits the admin listener; unset
reuses `base_url`, correct for local/free-profile dev where the split is
not enforced. Gap-1 re-embed/backfill needs only ordinary data-plane auth.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from relata._http import HttpTransport
from relata.objects import ObjectClient

if TYPE_CHECKING:
    import httpx

    from relata.client import RelataClient

#: Callable signature every `embedder=` argument in this module must satisfy:
#: a batch of source texts in, exactly one embedding vector out per text, in
#: the same order.
Embedder = Callable[[list[str]], list[list[float]]]

#: Row fields `ObjectClient.get()`/`POST /query` return that are
#: server-derived, not caller-owned data — re-sending them verbatim as part
#: of an upsert's `fields` payload would either no-op or conflict with the
#: server's own bookkeeping. Stripped before every re-embed write-back.
_SYSTEM_FIELDS = frozenset(
    {"id", "valid_from", "valid_to", "system_from", "system_to"}
)

_TYPE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FIELD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(name: str, *, kind: str) -> str:
    """SQL-injection guard (mirrors `relata.query._validate_sql_identifier`)
    for the object_type/field names this module interpolates into SQL."""
    if not _TYPE_RE.match(name) or not _FIELD_RE.match(name):
        raise ValueError(f"Invalid {kind}: {name!r} — must match [A-Za-z_][A-Za-z0-9_]*")
    return name


def _sql_string_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def http_embed_batch_embedder(client: RelataClient, *, model: str | None = None) -> Embedder:
    """Wrap RelataDB's own `POST /embed/batch` as an :data:`Embedder`
    callable, for callers who want the server's configured embedder
    (`RELATA_ACCEL_ENDPOINT` sidecar, or the built-in CPU lexical default)
    to drive re-embedding, rather than supplying an external one.

    RelataDB does not decide which model to use (ADR-0298) — `model=None`
    (the default) uses whatever the server is configured with; pass an
    explicit `model` to request a specific one if the server's embedder
    supports it.
    """
    from relata.vectors import VectorClient

    vectors = VectorClient.from_client(client)

    def _embed(texts: list[str]) -> list[list[float]]:
        resp = vectors.embed_batch(texts, model=model)
        return [list(v) for v in resp.get("embeddings", [])]

    return _embed


@dataclass
class MigrationBatchResult:
    """One batch's outcome, yielded by
    :meth:`EmbeddingMigrationClient.migrate` — lets a caller log/checkpoint/
    rate-limit a long-running migration instead of blocking on one call."""

    #: Rows read from the `old_model_tag` bucket this batch.
    rows_scanned: int
    #: Rows successfully re-embedded and written back under `new_model_tag`.
    rows_migrated: int
    #: Rows that failed to re-embed or write back (see `failed_ids`).
    rows_failed: int
    #: `id` of every row that failed this batch (empty on full success).
    failed_ids: list[str] = field(default_factory=list)
    #: `True` when this batch found FEWER than `batch_size` remaining rows —
    #: the signal that the corpus is (very likely) fully migrated. A caller
    #: driving :meth:`EmbeddingMigrationClient.migrate` to completion should
    #: still call :meth:`EmbeddingMigrationClient.remaining_on_old_tag`
    #: before retiring (races with concurrent writers are possible; `done`
    #: here is a strong hint, not the authoritative zero-remaining check).
    done: bool = False


class EmbeddingMigrationClient:
    """Synchronous embedding-model migration client (#4585).

    Construct directly, or via :meth:`from_client` to reuse an existing
    :class:`~relata.client.RelataClient`'s connection settings.
    """

    def __init__(
        self,
        base_url: str,
        *,
        bearer_token: str | None = None,
        timeout: float = 120.0,
        extra_headers: dict[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
        admin_base_url: str | None = None,
        purpose: str | None = None,
    ) -> None:
        # #2321 (ADR-0261): `/admin/vector-index/*` is mounted only on the
        # loopbound admin control-plane listener — `HttpTransport` routes
        # those paths to `admin_base_url` automatically when supplied,
        # `/query`/`/ingest` (Gap 1) keep going to `base_url` unchanged.
        self._t = HttpTransport(
            base_url,
            bearer_token,
            timeout,
            transport=transport,
            extra_headers=extra_headers,
            admin_base_url=admin_base_url,
        )
        self._objects = ObjectClient(
            base_url,
            bearer_token=bearer_token,
            timeout=timeout,
            extra_headers=extra_headers,
            transport=transport,
            purpose=purpose,
        )
        self._purpose = purpose

    @classmethod
    def from_client(cls, client: RelataClient) -> EmbeddingMigrationClient:
        return cls(
            client._base_url,  # noqa: SLF001
            bearer_token=client._bearer_token,  # noqa: SLF001
            timeout=client._timeout,  # noqa: SLF001
            extra_headers=client._extra_headers,  # noqa: SLF001
            admin_base_url=client._admin_base_url,  # noqa: SLF001
            purpose=client._default_purpose,  # noqa: SLF001
        )

    # ── Gap 1: re-embed + backfill ──────────────────────────────────────

    def _rows_on_tag(
        self,
        object_type: str,
        embedding_field: str,
        model_tag: str,
        *,
        limit: int,
        purpose: str | None,
    ) -> list[dict[str, Any]]:
        safe_type = _validate_identifier(object_type, kind="object_type")
        safe_field = _validate_identifier(embedding_field, kind="embedding_field")
        model_col = f"_emb_{safe_field}_model"
        sql = (
            f"SELECT * FROM {safe_type} WHERE {model_col} = "
            f"{_sql_string_literal(model_tag)} LIMIT {int(limit)}"
        )
        payload: dict[str, Any] = {"sql": sql}
        eff_purpose = purpose or self._purpose
        if eff_purpose:
            payload["purpose"] = eff_purpose
        data = self._t.post("/query", payload)
        raw = data.get("data") or data.get("rows") or []
        return [dict(r) for r in raw if isinstance(r, dict)]

    def remaining_on_old_tag(
        self,
        object_type: str,
        embedding_field: str,
        old_model_tag: str,
        *,
        purpose: str | None = None,
    ) -> int:
        """Count of rows still tagged `old_model_tag` on `embedding_field` —
        the "confirmed complete" check :meth:`retire` runs before dropping
        the old model's bucket (unless the caller passes `force=True`)."""
        safe_type = _validate_identifier(object_type, kind="object_type")
        safe_field = _validate_identifier(embedding_field, kind="embedding_field")
        model_col = f"_emb_{safe_field}_model"
        sql = (
            f"SELECT COUNT(*) AS n FROM {safe_type} WHERE {model_col} = "
            f"{_sql_string_literal(old_model_tag)}"
        )
        payload: dict[str, Any] = {"sql": sql}
        eff_purpose = purpose or self._purpose
        if eff_purpose:
            payload["purpose"] = eff_purpose
        data = self._t.post("/query", payload)
        raw = data.get("data") or data.get("rows") or []
        if not raw or not isinstance(raw[0], dict):
            return 0
        try:
            return int(raw[0].get("n", 0) or 0)
        except (TypeError, ValueError):
            return 0

    def migrate_batch(
        self,
        object_type: str,
        *,
        text_field: str,
        embedding_field: str,
        old_model_tag: str,
        new_model_tag: str,
        embedder: Embedder,
        batch_size: int = 100,
        purpose: str | None = None,
    ) -> MigrationBatchResult:
        """Re-embed and backfill ONE batch (up to `batch_size` rows) still
        tagged `old_model_tag` on `embedding_field`. Each row's
        `text_field` is passed to `embedder`; the resulting vector is
        written back as `_emb_{embedding_field}` tagged `new_model_tag` via
        an ordinary upsert — the row's OLD vector (still tagged
        `old_model_tag`) is left completely untouched (RelataDB's
        `VectorIndexKey` bucketing already guarantees this; this call does
        not read or write the old bucket at all).

        Every other field on the row is preserved verbatim (fetched via
        `POST /query`, server-managed fields like `id`/timestamps stripped,
        then re-sent alongside the new embedding fields) — a partial upsert
        that dropped unrelated fields would silently corrupt the row.

        Call repeatedly via :meth:`migrate` to walk a large corpus
        incrementally instead of blocking on one call.
        """
        rows = self._rows_on_tag(
            object_type, embedding_field, old_model_tag, limit=batch_size, purpose=purpose
        )
        if not rows:
            return MigrationBatchResult(rows_scanned=0, rows_migrated=0, rows_failed=0, done=True)

        texts = [str(r.get(text_field, "")) for r in rows]
        vectors = embedder(texts)
        if len(vectors) != len(rows):
            raise ValueError(
                f"embedder returned {len(vectors)} vector(s) for {len(rows)} row(s) — "
                "must return exactly one vector per input text, in order"
            )

        migrated = 0
        failed_ids: list[str] = []
        emb_field_key = f"_emb_{embedding_field}"
        emb_model_key = f"_emb_{embedding_field}_model"
        for row, vec in zip(rows, vectors, strict=True):
            row_id = str(row.get("id", "") or "")
            if not row_id:
                failed_ids.append("")
                continue
            fields = {k: v for k, v in row.items() if k not in _SYSTEM_FIELDS}
            fields[emb_field_key] = list(vec)
            fields[emb_model_key] = new_model_tag
            try:
                self._objects.upsert(object_type, row_id, fields, purpose=purpose)
                migrated += 1
            except Exception:  # noqa: BLE001 — one bad row must not abort the batch
                failed_ids.append(row_id)

        return MigrationBatchResult(
            rows_scanned=len(rows),
            rows_migrated=migrated,
            rows_failed=len(failed_ids),
            failed_ids=failed_ids,
            done=len(rows) < batch_size,
        )

    def migrate(
        self,
        object_type: str,
        *,
        text_field: str,
        embedding_field: str,
        old_model_tag: str,
        new_model_tag: str,
        embedder: Embedder,
        batch_size: int = 100,
        max_batches: int | None = None,
        purpose: str | None = None,
    ) -> Iterator[MigrationBatchResult]:
        """Walk the whole corpus batch-by-batch, yielding one
        :class:`MigrationBatchResult` per batch. A caller can inspect/log
        each result, back off on repeated `rows_failed`, or stop early —
        this never blocks on the whole corpus in one call. Stops when a
        batch finds fewer than `batch_size` remaining rows, or after
        `max_batches` (whichever comes first).
        """
        batches = 0
        while max_batches is None or batches < max_batches:
            result = self.migrate_batch(
                object_type,
                text_field=text_field,
                embedding_field=embedding_field,
                old_model_tag=old_model_tag,
                new_model_tag=new_model_tag,
                embedder=embedder,
                batch_size=batch_size,
                purpose=purpose,
            )
            yield result
            batches += 1
            if result.done:
                return

    # ── Gap 2: dual/multi-tag search ("Option B") ───────────────────────

    def search_multi_tag(
        self,
        object_type: str,
        model_tags: list[str],
        query_embedding: list[float],
        *,
        modality: str = "text",
        tenant: str | None = None,
        k: int = 10,
    ) -> list[dict[str, Any]]:
        """Dual/multi-tag ANN search during the migration window — ONE
        server-side call (`POST /admin/vector-index/search-multi-tag`)
        scans every named tag's bucket and returns a single fused, ranked
        result. Not a client-side merge of N single-tag searches — the
        server's `ObjectStore::vector_search_multi_tag_for_tenant` does the
        fusion (#4585). Returns `[{"id": ..., "score": ...}, ...]`.
        """
        if not model_tags:
            raise ValueError("model_tags must be non-empty")
        payload: dict[str, Any] = {
            "type": object_type,
            "modality": modality,
            "model_tags": list(model_tags),
            "query_embedding": list(query_embedding),
            "k": k,
        }
        if tenant is not None:
            payload["tenant"] = tenant
        resp = self._t.post("/admin/vector-index/search-multi-tag", payload)
        return [dict(h) for h in resp.get("hits", [])]

    # ── Retirement — explicit, operator-triggered only ──────────────────

    def retire(
        self,
        object_type: str,
        old_model_tag: str,
        *,
        modality: str = "text",
        tenant: str | None = None,
        embedding_field: str | None = None,
        force: bool = False,
        purpose: str | None = None,
    ) -> dict[str, Any]:
        """The explicit, operator-triggered retirement step
        (`POST /admin/vector-index/retire-model-tag`) — drops
        `old_model_tag`'s ENTIRE vector-index bucket. **Irreversible.**

        Never automatic: refuses with `RuntimeError`/`ValueError` unless
        either (a) `embedding_field` is supplied, in which case this first
        calls :meth:`remaining_on_old_tag` and refuses if any row is still
        tagged with `old_model_tag`, or (b) `force=True`, which explicitly
        bypasses that check (e.g. the caller already verified completion
        out-of-band). At least one of the two must be given.
        """
        if not force:
            if embedding_field is None:
                raise ValueError(
                    "retire() requires embedding_field (to verify zero rows remain on "
                    "old_model_tag) or force=True to explicitly bypass that check"
                )
            remaining = self.remaining_on_old_tag(
                object_type, embedding_field, old_model_tag, purpose=purpose
            )
            if remaining > 0:
                raise RuntimeError(
                    f"refusing to retire {old_model_tag!r}: {remaining} row(s) on "
                    f"{object_type!r} are still tagged with it (pass force=True to override)"
                )
        payload: dict[str, Any] = {
            "type": object_type,
            "modality": modality,
            "model_tag": old_model_tag,
        }
        if tenant is not None:
            payload["tenant"] = tenant
        return dict(self._t.post("/admin/vector-index/retire-model-tag", payload))

    def close(self) -> None:
        self._t.close()
        self._objects.close()

    def __enter__(self) -> EmbeddingMigrationClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def migrate_embedding_model(
    client: RelataClient,
    object_type: str,
    *,
    text_field: str,
    embedding_field: str,
    old_model_tag: str,
    new_model_tag: str,
    embedder: Embedder,
    batch_size: int = 100,
    max_batches: int | None = None,
    purpose: str | None = None,
) -> Iterator[MigrationBatchResult]:
    """Module-level convenience wrapper: re-embed + backfill `object_type`
    from `old_model_tag` to `new_model_tag`, one batch at a time, reusing
    `client`'s connection settings (see
    :meth:`EmbeddingMigrationClient.from_client`).

    ```python
    from relata import RelataClient
    from relata.embedding_migration import (
        EmbeddingMigrationClient,
        http_embed_batch_embedder,
        migrate_embedding_model,
    )

    client = RelataClient("http://localhost:9090", bearer_token="...")
    embedder = http_embed_batch_embedder(client, model="e5-large:2.0")

    for progress in migrate_embedding_model(
        client,
        "DocumentChunk",
        text_field="text_body",
        embedding_field="text",
        old_model_tag="e5-large:1.0",
        new_model_tag="e5-large:2.0",
        embedder=embedder,
    ):
        print(progress)

    # Dual-search during the cutover window (see search_multi_tag's own
    # doc for why this is one server-side call, not two merged locally):
    migration = EmbeddingMigrationClient.from_client(client)
    hits = migration.search_multi_tag(
        "DocumentChunk",
        ["e5-large:1.0", "e5-large:2.0"],
        query_embedding=embedder(["some query"])[0],
    )

    # Once remaining_on_old_tag(...) == 0, retire explicitly:
    migration.retire(
        "DocumentChunk", "e5-large:1.0", embedding_field="text"
    )
    ```
    """
    migration = EmbeddingMigrationClient.from_client(client)
    yield from migration.migrate(
        object_type,
        text_field=text_field,
        embedding_field=embedding_field,
        old_model_tag=old_model_tag,
        new_model_tag=new_model_tag,
        embedder=embedder,
        batch_size=batch_size,
        max_batches=max_batches,
        purpose=purpose,
    )
