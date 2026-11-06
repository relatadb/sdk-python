# Relata Python SDK

Python SDK for the [Relata DB](https://relatadb.dev) governed temporal data engine.

- **Repo:** [github.com/relatadb/sdk-python](https://github.com/relatadb/sdk-python)
- **Docs:** [relatadb.dev](https://relatadb.dev) · [Issues](https://github.com/relatadb/sdk-python/issues)
- **By:** [ZySec AI](https://zysec.ai) — Frontier Sovereign Intelligence · hello@zysec.ai

For the v3.0 platform boundary and SDK migration plan, see the
[SDK migration guide](../README-v3-migration.md).

**Server compatibility.** SDK 3.0.x pairs with a Relata DB 3.0.x server and
calls its `/v1` routes. Earlier SDK releases (2.x, including the 2.6.2 on
the registries as of 2026-11-06) call the unversioned routes and get 404 on
every data-plane call against a 3.0 server; the server has no compatibility
alias. Keep the SDK and server on the same major.minor version. See
[version pairing](../README-v3-migration.md#server-and-sdk-version-pairing).
Python is one of the three supported consumer SDKs (Python, TypeScript, Go).
Other languages use the HTTP API and its OpenAPI reference at `GET /openapi.json`.

The v3 SDK accepts prepared detection rules through `GovernanceClient.create_rule`
or MCP `create_rule`. Raw Sigma YAML parsing/import is no longer an SDK or MCP
server method. Convert YAML in OpenETL (the former `relata-connectors import-sigma`
was removed), or convert externally and submit the prepared
`name`, `condition_sql`, and `target_type` JSON fields through `/v1/rules`
(`POST /v1/rules/sigma` answers `410 sigma-import-removed`; see the [Sigma target contract](../../docs/src/openetl-handoff/sigma-target-contract.md)).

Relata DB is a Rust-native, ontology-driven engine built for bi-temporal, governed knowledge workloads: link analysis, identity resolution, full-text and vector search, access-scoped restricted data, and provenance-tracked analytics.

## Requirements

- Python 3.11+
- `httpx >= 0.27`
- `pydantic >= 2.6`

## Installation

```bash
pip install relata-sdk
# or with uv
uv add relata-sdk
```

## Quick start

```python
from relata import RelataClient

with RelataClient("http://localhost:9090", purpose="analytics") as client:
    result = client.query("SELECT * FROM Person WHERE name LIKE 'Ahmed%' LIMIT 10")
    for row in result:
        print(row)
```

## Flagship retrieval surface — `namespace().query()` (#1991)

For the search / RAG persona, the SDK ships a typed JSON door that compiles to
a governed SQL plan — **no SQL on the client side**. `text` compiles to a BM25
`rank_by` directive; `filters` narrow the set server-side; and `.write()` is
schemaless (auto-creates the type with inferred fields via `POST /v1/ingest/auto`
— no DDL).

```python
from relata import RelataClient

with RelataClient("http://localhost:9090", purpose="rag") as client:
    docs = client.namespace("Document")
    docs.write([{"id": "d1", "title": "Knowledge graphs 101", "body": "..."}])
    res = docs.query(
        text="graph retrieval",
        match_column="title",
        filters=[{"field": "status", "op": "eq", "value": "published"}],
        limit=10,
    )
    for row in res:
        print(row["title"])
```

`rank_by` (explicit) accepts the server's array forms
(`["bm25","title","..."]`, `["vector","ann","..."]`); `filters` take
`{field, op, value}` (ops `eq/ne/gt/gte/lt/lte/like/ilike/in/between`) or an
equality-dict shorthand. Async: `AsyncNamespace` / `await docs.query(...)`. HTTP
compression is off by default (pass `compress=True` to re-enable for slow
links); one connection pool is reused across every namespace.

## Streaming query completion

`StreamingClient.from_client(client).query_rows(sql)` uses `POST /v1/query/stream`
and yields data rows without the protocol metadata. Exhaust the iterator before
treating the result as complete: a missing completion frame, truncation,
transport failure, or mismatched row count raises `IncompleteQueryStreamError`.
Its `query_id` and `rows_received` identify the partial answer; rows already
yielded are not replayed automatically. Closing an iterator intentionally stops
consumption without claiming completion. The existing `max_response_bytes`
option bounds each NDJSON frame, rather than the total stream size.

Async callers await `AsyncStreamingClient.from_client(client).query_rows(sql)`
to obtain an async iterator with the same completion contract. Both streaming
client classes are imported from `relata.streaming`.

## Cypher

RelataDB auto-detects Cypher: any query starting with `MATCH` is translated to SQL
before execution. The SDK is language-agnostic — send the string as you would SQL:

```python
result = client.query("MATCH (n:Person {id: 'p1'}) RETURN *")
# → SELECT * FROM Person WHERE id = 'p1'
```

## GraphQL

`graphql()` (ADR-220) translates a GraphQL SELECT subset to SQL and runs it
through the governed query path. Returns `data`; raises `RelataError` on a
non-empty `errors` envelope.

```python
data = client.graphql(
    "{ Person(where: { age: { _gt: 30 } }, limit: 10) { id name age } }",
)
# data == [{"id": "p1", "name": "Ada", "age": 36}, ...]

# With variables + an explicit operation name:
data = client.graphql(
    "query Q($n: Int!) { Person(where: { age: { _gt: $n } }) { name } }",
    variables={"n": 30},
    operation_name="Q",
)

# Async variant:
# data = await client.agraphql("{ __schema { types { name } } }")
```

Supported: `MATCH` / `OPTIONAL MATCH`, `WHERE`, `RETURN`, `UNION` / `UNION ALL`
(#378), and `CALL traverse.*` / `CALL gds.*` procedures (#377). `CREATE` / `MERGE`
writes route through the governed write door. See the
[SQL reference](https://www.relatadb.dev/docs/reference/sql).

## Hybrid search — `VectorClient.hybrid_search()`

`VectorClient` wraps the server's `HYBRID_SEARCH` operator (ADR-175): supply
`query_text` (BM25 leg), `query_embedding` + `embedding_slot` (vector leg), or
both — when both are present the server fuses the two rankings via
reciprocal rank fusion. This is Relata's genuine edge over a plain vector DB
or a plain full-text engine.

```python
from relata import RelataClient
from relata.vectors import VectorClient

with RelataClient("http://localhost:9090", purpose="rag") as client:
    vectors = VectorClient.from_client(client)

    query_embedding = vectors.embed("graph retrieval")["embedding"]

    hits = vectors.hybrid_search(
        "Document",
        query_text="graph retrieval",
        query_embedding=query_embedding,
        embedding_slot="_emb_text",
        k=10,
    )
    for row in hits:
        print(row["title"], row.get("_score"))
```

See `examples/hybrid_search.py` for a full runnable walkthrough (embed →
ingest → BM25-only vs. vector-only vs. fused search). The higher-level
`namespace().query(rank_by=["vector", "ann", "..."])` door (above) reaches the
same fused path when both a `text` and a vector `rank_by` are supplied;
`VectorClient.hybrid_search()` is the typed entry point when you want direct
control over `embedding_slot` and both legs at once.

## Agent memory in three lines

For agent-memory workloads, `Memory` is a Mem0-style one-liner surface over the
governed `/v1/memory/*` verbs — governance (purpose + ACL) stays on by default:

```python
from relata import Memory

with Memory("http://localhost:9090", purpose="agent-notes") as m:
    mem_id = m.add("Alice prefers dark mode")     # store
    hits = m.search("ui preferences", top_k=5)    # recall (confidence × recency × relevance)

    # ADR-145 retrieval-quality operators (#2674): bound recall by token
    # budget and confidence so one call can't blow an agent's context window.
    # `search_detailed` also surfaces the read-only `recall_cost_tokens`/
    # `cancelled` response fields.
    detailed = m.search_detailed("ui preferences", min_confidence=0.5, budget_tokens=200)

    for frame in m.recall_stream("ui preferences"):   # SSE: one memory per frame
        print(frame)

    # Versioned agent procedures (kebab-case /v1 routes).
    m.remember_procedure("agent-1", "onboarding", "Greet, then verify identity")
    m.recall_procedure("agent-1", name="onboarding")

    m.forget(mem_id)                              # governed retract, not a hard delete
```

`search` and `search_detailed` call `POST /v1/memory/recall` (the one recall
route); `recall_stream` is the SSE form (`GET /v1/memory/recall/stream`).

See `examples/memory_quickstart.py`. For async apps, `AsyncMemory` has the same
surface: `async with AsyncMemory(...) as m: await m.add(...); await m.search(...)`.

## Agent-framework adapters

Drop-in, governed memory backends for the major agent frameworks — each maps the
framework's memory/storage interface onto Relata, and (with one exception) none
imports its framework (so they load with or without it installed):

| Framework | Import | Shape |
|---|---|---|
| LangChain | `relata_adapters.langchain.RelataMemory` | `BaseMemory` |
| LlamaIndex | `relata_adapters.llamaindex.RelataMemory` | `BaseMemory` |
| CrewAI | `relata_adapters.crewai.RelataStorage` | `Storage` |
| AutoGen | `relata_adapters.autogen.RelataMemory` | `Memory` (async) |
| LangGraph* | `relata_langgraph.RelataCheckpointer` / `AsyncRelataCheckpointer` | `BaseCheckpointSaver` |

\* LangGraph is the exception: it gates checkpoint behavior behind
`isinstance(checkpointer, BaseCheckpointSaver)`, so `relata_langgraph` is a
real subclass with a real (optional-extra) dependency on `langgraph` — see
`pip install "relata-sdk[langgraph]"` and
[the LangGraph checkpointer guide](https://www.relatadb.dev/docs/guides/langgraph-checkpointer).

```python
from relata_adapters.langchain import RelataMemory

mem = RelataMemory(base_url="http://localhost:9090", purpose="agent")
# chain = ConversationChain(llm=..., memory=mem)
```

The adapters use the semantic-memory model (store + relevance recall), the same
paradigm as Mem0. They wrap the `Memory` client, so governance stays on.

## Core concepts

### Every query must declare a purpose

Relata enforces a mandatory `purpose` on every query.  The purpose must be a string registered in the tenant's `PurposeRegistry` (e.g. `"analytics"`, `"audit"`, `"analysis"`).  Queries without a purpose are rejected at the wire.

Set a default on the client:

```python
client = RelataClient("http://localhost:9090", purpose="analytics")
```

Or override per-call:

```python
result = client.query("SELECT COUNT(*) FROM Person", purpose="audit")
```

### Bearer token authentication

When the server is configured with `RELATA_BEARER_TOKEN`, pass the token to the client:

```python
client = RelataClient(
    "https://relata.example.com",
    bearer_token="your-token",
    purpose="analytics",
)
```

> **What identity does my SDK client use by default?**
>
> Without a bearer token, every request runs as the **`api-user`** principal —
> a built-in identity with read access to all domain types and write access
> in open (lite) mode. Audit entries, quota charges, and ACL decisions all
> attribute to `api-user`.
>
> MCP requests (`McpClient`) run as **`mcp-client`** — a different principal
> with separate audit and quota accounting.
>
> For multi-tenant production: set a bearer token + `tenant=` on every
> request. See `docs/src/end-users/defaults.md` for the full default-behavior
> reference.

### Fluent query builder

```python
result = (
    client.select("Person")
    .where("nationality = 'IN'")
    .where("age > 25")
    .as_of("2025-01-01")          # bi-temporal point-in-time
    .with_provenance()             # include PROV-O columns
    .order_by("name ASC")
    .limit(20)
    .execute()
)
```

### Relata SQL extensions

| Extension | Example |
|---|---|
| Bi-temporal | `SELECT * FROM Person AS OF '2024-06-01'` |
| Provenance | `SELECT * FROM Person WITH PROVENANCE` |
| Graph traversal | `FROM PATHS_BETWEEN('person-123', 'org-456', max_hops => 4)` |
| Identity lookup | `FROM LOOKUP_IDENTITY('+919876543210')` |
| Hybrid search | `SELECT *, HYBRID_SCORE('terror finance') AS score FROM Document` |

### QueryResult is iterable

```python
result = client.query("SELECT * FROM Person LIMIT 10")

# Iterate over rows
for row in result:
    print(row["name"])

# Access by index
first = result.rows[0]

# Check if empty
if not result:
    print("No results")

# Row count
print(f"{len(result)} rows in {result.processing_time_ms} ms")
```

### Async support

```python
import asyncio
from relata import RelataClient

async def main():
    async with RelataClient("http://localhost:9090", purpose="analytics") as client:
        result = await client.aquery("SELECT * FROM Person LIMIT 5")
        for row in result:
            print(row)

asyncio.run(main())
```

### HTTP Arrow IPC

For columnar results, `RelataClient.query_arrow()` returns a `pyarrow.Table`
from the HTTP `POST /v1/query/arrow` endpoint. Install `pyarrow` to use this
helper. `StreamingClient.query_arrow_raw()` exposes the same HTTP response as
byte chunks when the application wants to provide its own Arrow reader. The
raw chunks can be joined before passing them to an Arrow IPC stream reader.

```python
from relata import RelataClient, StreamingClient

with RelataClient("http://localhost:9090", purpose="analytics") as client:
    table = client.query_arrow("SELECT * FROM Person LIMIT 1000")
    print(table.num_rows)

    raw_bytes = b"".join(
        StreamingClient.from_client(client).query_arrow_raw(
            "SELECT * FROM Person LIMIT 1000"
        )
    )
    print(f"Received {len(raw_bytes)} Arrow IPC bytes")
```

### Error handling

```python
from relata.exceptions import (
    PurposeError,   # missing or unregistered purpose
    QuotaError,     # per-principal cost quota exhausted
    AuthError,      # invalid or missing bearer token
    ConnectionError, # server unreachable
    RelataError,    # base class — catch all SDK errors
)

try:
    result = client.query("SELECT * FROM Person")
except PurposeError as exc:
    print(f"Fix: set purpose= on client or per query. {exc}")
except QuotaError as exc:
    print(f"Quota exhausted — reduce query scope. {exc}")
except AuthError as exc:
    print(f"Auth failed — check bearer_token. {exc}")
except ConnectionError as exc:
    print(f"Cannot reach server. {exc}")
```

## API reference

### `RelataClient`

```python
RelataClient(
    base_url: str,
    *,
    bearer_token: str | None = None,
    purpose: str | None = None,
    timeout: float = 30.0,
    tenant: str | None = None,        # X-Relata-Tenant-Id (multi-tenant)
    acting_as: str | None = None,     # X-Acting-As (delegation)
    delegated_by: str | None = None,  # X-Delegated-By
    headers: dict[str, str] | None = None,  # arbitrary header overlay
    max_retries: int = 0,             # retry on connection errors
    retry_backoff_secs: float = 0.5,  # exponential backoff base
)
```

| Method | Returns | Description |
|---|---|---|
| `query(sql, *, purpose)` | `QueryResult` | Execute a SQL query |
| `health()` | `HealthResponse` | Check node health |
| `status()` | `StatusResponse` | Node status + quota |
| `stats()` | `Stats` | Engine-wide counts (records/states/snapshot_rows/log_leaves/tokens) |
| `version()` | `VersionInfo` | Build info (version, commit, profile, schema_version) |
| `ready()` | `ReadyReport` | 9-condition readiness report |
| `audit_count()` | `AuditCountResponse` | Audit log entry count + chain validity |
| `ingest_document(objects, *, purpose=None)` / `aingest_document(...)` | `IngestDocumentResponse` | Queue prepared mixed-type rows with `POST /v1/ingest/bulk` (`queued: true`) and return one process-local receipt |
| `ingest_document_status(task_id)` / `aingest_document_status(...)` | `IngestDocumentTaskStatus` | Poll `GET /v1/ingest/tasks/:task_id` for terminal complete/partial/failed/uncertain/incomplete state and confirmed written/dropped row counts |
| `select(*cols)` | `QueryBuilder` | Start a fluent query |
| `close()` | — | Close sync connections |
| `aquery(...)` | `QueryResult` | Async query |
| `ahealth()` ... `astats()` ... `aready()` etc. | — | Async mirrors of every method above |
| `aclose()` | — | Close async connections |

`ingest_document` accepts application-prepared governed rows (such as a
`DocumentSource`, linked `DocumentChunk` rows, and optional evidence rows with
`_type` and stable `_pk` fields). It does not parse raw files or datagrep
extractor JSONL. Parsing extractor output into an `objects` array is OpenETL's job
(the former `relata-connectors prepare-document` adapter was removed). Queue
admission is process-local and not crash-durable; poll status for confirmed
write/drop outcomes. `code` and `derived_error` can be present while the task
is still pending; after all primary batches are accounted for, its status is
`incomplete`. If the primary outcome is uncertain, `uncertain` takes
precedence. A recorded derived-write failure has `retryable: false` and no
completion watermark. The separate MCP `ingest_document(source, text, ...)`
tool remains a direct source/text application workflow.

### v1.1 typed clients (`from_client(client)`)

Each module inherits the parent client's auth, tenant, and purpose context:

| Module | Class | Surface |
|---|---|---|
| `relata.governance` | `GovernanceClient` | Rules, retention (holds + WORM), breakglass, alerts, DSAR, GDPR `erase`/`deletion_status`/`verify_receipt` |
| `relata.mcp` | `McpClient` | 22 typed MCP tool wrappers + generic `call_tool` |
| `relata.a2a` | `A2AClient` | A2A tasks + LangGraph checkpoints + agent card |
| `relata.audit` | `AuditClient` | Audit `count` + filtered `log` (`/v1/audit/count`, `/v1/audit/log`) |
| `relata.identity` | `IdentityClient` | Identity label/uncertainty + lookup tables + ERASE SUBJECT |
| `relata.objects` | `ObjectClient` | Typed upsert + batch + point-lookup (`get`) + delete via `/v1/ingest?object_type=` |
| `relata.ingest` | `IngestClient` | Bulk NDJSON + CSV + media task status; it does not upload raw media bytes |
| `relata.files` | `FilesClient` / `AsyncFilesClient` | Buffered governed Files `put`/`get`/`head`/`list`/`delete` for opaque bytes, plus `media_content` (`GET /v1/media/{row_id}/content`) and representation metadata, plus resumable chunked `upload` (one chunk in memory, offset-based resume, SHA-256 verified; #5690) |
| `relata.vectors` | `VectorClient` | KNN + hybrid search + similar-to (SQL-backed) |
| `relata.system` | `SystemClient` | Generic jobs and notification rules (webhooks are a rule channel: set `webhook_url`/`webhook_headers` on the rule); no feed broker routing |
| `relata.feed_broker` | `FeedBrokerClient` / `AsyncFeedBrokerClient` | Optional RIFN broker health, channels, signed-entry publishing at an explicit broker URL |
| `relata.streaming` | `StreamingClient` | NDJSON row streams + SSE consumers (watch/alerts) + Arrow IPC |
| `relata.tenants` | `TenantAdminClient` | Caller's own tenant record (`me`); tenant admin is Enterprise-only |
| `relata.tokens` | `TokenClient` | Tenant self-service tokens: `list_self`, `self_audit`, `rotate_self`, `revoke_self` (`/v1/auth/tokens/self`) |

Each has an async mirror (`Async*`).

The former `get_domain_summary` MCP route is not a core tool in v3.0. To build
the same application-facing inventory, call the generic, tenant-scoped
`mcp.list_entity_types()` and pass its result to
`relata.summarize_domain("financial", inventory)`. The helper is local profile
metadata; it performs no extra query or authorization check.

```python
from relata import McpClient, RelataClient, summarize_domain

with RelataClient("http://localhost:9090", purpose="analytics") as db:
    with McpClient.from_client(db) as mcp:
        inventory = mcp.list_entity_types()
        summary = summarize_domain("financial", inventory)
        print(summary)
```

RIFN is a separate optional broker service, not a route on the Relata
database listener. Construct `FeedBrokerClient` or `AsyncFeedBrokerClient`
with the broker's own URL and credentials. `publish()` sends a complete,
caller-signed feed entry and the matching `X-Publisher-Did`; the SDK does not
create or sign entries. `SystemClient.from_client()` does not configure a
broker client.

`POST /v1/ingest/media` was removed (#5942), together with
`IngestClient.media_status()` and `McpClient.ingest_media()`; poll OpenETL work
items with `MediaWorkClient.get()`. The separate new `FilesClient` and
`AsyncFilesClient` wrap `/v1/files/content` for opaque file bytes (including `HEAD` and `GET /v1/files` listing); they do not
start media processing or document extraction.

### Governed Files clients — v3.0 candidate, SDK 3.0.0

For the v3.0 candidate (engine source `57e11676c`), version-aligned SDK source
`8924d6ce8` (TypeScript test-only fixture correction `5212045`) builds cleanly;
TypeScript passes 515/515, Python 882/882, and Go tests pass in two packages.
The five-lane Files fixture passes in 10.020s. A separate prepared-document
matrix passes in 37.110s across raw HTTP and official TypeScript, Python sync
and async, and Go queries. It checks synthetic DataGrep and Unstructured JSON
citations against an independent HTTP anchor, plus bounded ingestion receipts,
auth/tenant behavior and a persistent single-node restart. Citation comparison
waits for stable index visibility: TypeScript and Python required two
before/after stabilization attempts for DataGrep; this does not promise
immediate parity. See the [Files result](../../scripts/qualification/followup_pr5786_sdk_files_common_live.json)
and [prepared result](../../scripts/qualification/followup_pr5786_sdk_files_prepared_live.json).
These source-bound results do not claim package publication, PDF/OCR/provider
extraction, replica or throughput qualification, interrupted-task resumption,
or server-side cancellation. Prepared receipts describe historical local task
outcomes; volatile profiles remain process-local.

`FilesClient.from_client(client)` and `AsyncFilesClient.from_client(client)`
borrow the parent transport and tenant/auth context. `put(key, data,
bucket=None, content_type="application/octet-stream")` returns current-file
metadata; `get(key, bucket=None)` returns buffered `data`, `etag`,
`content_type`, and byte `size`; `delete(key, bucket=None)` acknowledges an
exact-key deletion. An omitted bucket uses the
authenticated tenant's default. The server accepts uploads up to 128 MiB;
downloads use the configured response cap (64 MiB by default). PUT and DELETE
are not automatically retried, so callers must reconcile uncertain results.
These clients do not expose immutable versions, conditional writes, range
requests, or physical/global-copy deletion guarantees.
Cancelling an async call or expiring the HTTP timeout closes/cancels the client
request where supported; it does not promise that server work already accepted
by the engine is aborted or rolled back. See
the [HTTP Files contract](../../docs/src/specs/files-api.md) for server
behavior. These clients wrap `/v1/files/content` and the resumable `/v1/files/uploads`
session (`start_upload`, `upload_chunk`, `upload_status`, `complete_upload`,
`abort_upload`, and the resuming `upload` helper), plus `head`, `list` and
`media_content`.

**Admin-listener reachability (`admin_base_url`, #2321):** on a hardened
deployment, `/admin/*` is mounted only
on the loopbound admin control-plane listener (`RELATA_ADMIN_BIND`, default
`127.0.0.1:9091` — ADR-0261), never the main data-plane `base_url`. Pass
`admin_base_url=` to `RelataClient` (inherited automatically by
`BackupClient.from_client(...)`) or directly to
`BackupClient` to reach those routes. Leave it unset (the
default) on the free profile, where the admin listener isn't split from the
data plane. A bare 404 from an admin-only route with no error
detail is rewritten into a hint pointing at this option instead of a
confusing plain 404.

### v1.1 transport hardening (#79)

- **RFC 7807 problem+json parsing** — every error carries `code`, `type_url`, `retryable`, `request_id`.
- **Typed exceptions** — `ForbiddenError` (403), `NotFoundError` (404), `ConflictError` (409), `ValidationError` (422), `RateLimitedError` (429, with `retry_after`).
- **Retry** — `RelataClient(..., max_retries=3, retry_backoff_secs=0.5)`.
- **X-Request-ID** — auto-generated per request; caller-supplied IDs respected; server's response ID stamped on exceptions.

### Custom object types

RelataDB is ontology-governed — unknown types are rejected on both ingest and
read. Register custom types at runtime via `POST /v1/types` (persisted across
restart):

```python
# Register a custom type — use the register_type() wrapper (below), not a raw
# POST: the field is "properties" (each {"name", "required", "state_machine"}),
# not "fields"/"type" — the server silently ignores unrecognized keys, so a
# hand-rolled body that gets this wrong registers a type with NO declared
# properties rather than erroring.
client.register_type(
    "AgentTask",
    properties=[
        {"name": "task_id", "required": True},
        {"name": "status", "required": True},
    ],
)

# Now ingest + query work
client._sync.post("/v1/ingest?object_type=AgentTask", {"task_id": "t-1", "status": "done"})
```

`register_type()` also accepts `graph_triggers`, `computed_columns`, and `bm25_params` — see its docstring for the full shape of each.

For ACL access in strict mode, grant via env var:
```bash
RELATA_ACL_GRANT=AgentTask:read+write
```

Unknown types return `400` on both ingest and read — fail-closed by design.

### v1.1 QueryBuilder extensions (#76)

| Method | Description |
|---|---|
| `.limit(n, after="cursor")` | Keyset pagination (`LIMIT n AFTER 'cursor'`) |

The builder also validates identifiers (`from_()`, `select()`) and refuses
dangerous tokens in `where()` (SQL-injection stopgap). For user-supplied
values use `.where_param("id = ?", value)` — `?` placeholders are replaced
with safely-typed literals (str → single-quoted + escaped, int/float → numeric).

### Agent-framework adapters

| Framework | Import | Shape |
|---|---|---|
| LangChain | `relata_adapters.langchain.RelataMemory` | `BaseMemory` |
| LlamaIndex | `relata_adapters.llamaindex.RelataMemory` | `BaseMemory` |
| CrewAI | `relata_adapters.crewai.RelataStorage` | `Storage` |
| AutoGen v0.2 | `relata_adapters.autogen.RelataMemory` | `Memory` (async) |
| LangGraph | `relata_langgraph.RelataCheckpointer` / `AsyncRelataCheckpointer` | `BaseCheckpointSaver` |
| AG2 (v0.4+) | `relata_adapters.ag2.RelataAG2Memory` | `MemoryProtocol` |
| Pydantic-AI | `relata_adapters.pydantic_ai.RelataMemoryBackend` | memory backend |
| smolagents | `relata_adapters.smolagents.RelataTool` | tool callable |
| Auto-detect | `relata_adapters.registry.get_memory_adapter()` | picks the right class |

### `QueryBuilder`

| Method | Description |
|---|---|
| `.select(*cols)` | Columns or table name shorthand |
| `.from_(table)` | FROM table |
| `.where(condition)` | Add WHERE predicate (multiple calls → AND) |
| `.where_param(condition, *values)` | Safe WHERE with `?` placeholder interpolation (str/int/float) |
| `.as_of(timestamp)` | Bi-temporal AS OF |
| `.with_provenance()` | Append WITH PROVENANCE |
| `.purpose(p)` | Override purpose for this query |
| `.limit(n)` | LIMIT |
| `.offset(n)` | OFFSET |
| `.order_by(*cols)` | ORDER BY |
| `.paths_between(src, dst, max_hops)` | PATHS_BETWEEN graph operator |
| `.lookup_identity(value)` | LOOKUP_IDENTITY operator |
| `.raw(sql)` | Raw SQL escape hatch |
| `.sql()` | Return assembled SQL without executing |
| `.execute()` | Execute (sync) |
| `.aexecute()` | Execute (async) |

### Response models

| Model | Fields |
|---|---|
| `QueryResult` | `rows`, `query_id`, `processing_time_ms`, `elapsed_ms` (alias, removed v1.6.0), `row_count` |
| `HealthResponse` | `status`, `profile`, `node_id` |
| `StatusResponse` | `profile`, `role`, `query_quota` |
| `AuditCountResponse` | `entries`, `chain_valid` |
| `VersionInfo` | `version`, `commit`, `profile`, `schema_version`, `features` |
| `Stats` | `records`, `states`, `snapshot_rows`, `log_leaves`, `tokens` |
| `ReadyReport` | `is_ready`, `status`, `reason`, `detail` |

## Examples

See the `examples/` directory:

| File | Demonstrates |
|---|---|
| `basic_query.py` | Getting started, health check, quota check |
| `analytics.py` | Full analytics workflow: identity → cases → graph → provenance |
| `hybrid_search.py` | `VectorClient.hybrid_search()` — embed, ingest, BM25-only vs. vector-only vs. fused (RRF) search |
| `graph_traversal.py` | PATHS_BETWEEN, temporal network shift, hub detection |
| `audit.py` | Chain verification, audit summary, anomaly detection |

## Deployment

`RELATA_PROFILE` accepts only `free` (the default); `server`, `cluster` and `lite` are rejected at startup (#5809). Posture follows configuration, and the SDK works identically in each:

| Use case | Start command |
|---|---|
| Local / single process | `relata serve` (loopback) |
| Single-node production | `RELATA_HTTP_BIND=0.0.0.0:9090 RELATA_BEARER_TOKEN=<secret> RELATA_ADMIN_TOKEN=<secret> relata serve` |

## Testing with an ephemeral server

`tests/conftest_ephemeral.py` provides a session-scoped pytest fixture that
starts a real `relata serve` process on a random port and tears it down at the
end of the test session.

```python
# In any test file:
from tests.conftest_ephemeral import relata_server

def test_health(relata_server):
    import httpx
    r = httpx.get(f"{relata_server['base_url']}/health",
                  headers={"Authorization": f"Bearer {relata_server['token']}"})
    assert r.status_code == 200
```

Set `RELATA_BIN` to the binary path (default: `relata` on `$PATH`) and
`RELATA_TEST_TOKEN` to override the bearer token (default: `relata-test`).

## License

AGPL-3.0-only.  See the root `LICENSE` file.
