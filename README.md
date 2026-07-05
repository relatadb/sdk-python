# Relata Python SDK

Python SDK for the [Relata](https://relatadb.dev) enterprise-grade data engine.

Relata is a Rust-native, ontology-driven engine built for bi-temporal, governed knowledge workloads: link analysis, identity resolution, full-text and vector search, access-scoped restricted data, and provenance-tracked analytics.

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

## Agent memory in three lines

For agent-memory workloads, `Memory` is a Mem0-style one-liner surface over the
governed `/memory/*` verbs — governance (purpose + ACL) stays on by default:

```python
from relata import Memory

with Memory("http://localhost:8080", purpose="agent-notes") as m:
    mem_id = m.add("Alice prefers dark mode")     # store
    hits = m.search("ui preferences", top_k=5)    # recall (confidence × recency × relevance)
    m.forget(mem_id)                              # governed retract, not a hard delete
```

See `examples/memory_quickstart.py`. For async apps, `AsyncMemory` has the same
surface: `async with AsyncMemory(...) as m: await m.add(...); await m.search(...)`.

## Agent-framework adapters

Drop-in, governed memory backends for the major agent frameworks — each maps the
framework's memory/storage interface onto Relata, and none imports its framework
(so they load with or without it installed):

| Framework | Import | Shape |
|---|---|---|
| LangChain | `relata_adapters.langchain.RelataMemory` | `BaseMemory` |
| LlamaIndex | `relata_adapters.llamaindex.RelataMemory` | `BaseMemory` |
| CrewAI | `relata_adapters.crewai.RelataStorage` | `Storage` |
| AutoGen | `relata_adapters.autogen.RelataMemory` | `Memory` (async) |
| LangGraph | `relata_langgraph.RelataCheckpointer` | checkpointer |

```python
from relata_adapters.langchain import RelataMemory

mem = RelataMemory(base_url="http://localhost:8080", purpose="agent")
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
| Face search | `WHERE MATCH_FACE(probe_embedding, stored_embedding, 0.92)` |
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
print(f"{len(result)} rows in {result.elapsed_ms} ms")
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
)
```

| Method | Returns | Description |
|---|---|---|
| `query(sql, *, purpose)` | `QueryResult` | Execute a SQL query |
| `health()` | `HealthResponse` | Check node health |
| `status()` | `StatusResponse` | Node status + quota |
| `audit_count()` | `AuditCountResponse` | Audit log entry count + chain validity |
| `cluster_nodes()` | `list[ClusterNode]` | List cluster nodes |
| `select(*cols)` | `QueryBuilder` | Start a fluent query |
| `close()` | — | Close sync connections |
| `aquery(...)` | `QueryResult` | Async query |
| `ahealth()` | `HealthResponse` | Async health check |
| `astatus()` | `StatusResponse` | Async status |
| `aaudit_count()` | `AuditCountResponse` | Async audit count |
| `acluster_nodes()` | `list[ClusterNode]` | Async cluster nodes |
| `aclose()` | — | Close async connections |

### `QueryBuilder`

| Method | Description |
|---|---|
| `.select(*cols)` | Columns or table name shorthand |
| `.from_(table)` | FROM table |
| `.where(condition)` | Add WHERE predicate (multiple calls → AND) |
| `.as_of(timestamp)` | Bi-temporal AS OF |
| `.with_provenance()` | Append WITH PROVENANCE |
| `.purpose(p)` | Override purpose for this query |
| `.limit(n)` | LIMIT |
| `.offset(n)` | OFFSET |
| `.order_by(*cols)` | ORDER BY |
| `.paths_between(src, dst, max_hops)` | PATHS_BETWEEN graph operator |
| `.match_face(probe, candidate, threshold)` | MATCH_FACE operator |
| `.lookup_identity(value)` | LOOKUP_IDENTITY operator |
| `.hybrid_score(text)` | HYBRID_SCORE ranking |
| `.raw(sql)` | Raw SQL escape hatch |
| `.sql()` | Return assembled SQL without executing |
| `.execute()` | Execute (sync) |
| `.aexecute()` | Execute (async) |

### Response models

| Model | Fields |
|---|---|
| `QueryResult` | `rows`, `query_id`, `elapsed_ms`, `row_count` |
| `HealthResponse` | `status`, `profile`, `node_id` |
| `StatusResponse` | `profile`, `role`, `query_quota` |
| `AuditCountResponse` | `entries`, `chain_valid` |
| `ClusterNode` | `node_id`, `role`, `url` |

## Examples

See the `examples/` directory:

| File | Demonstrates |
|---|---|
| `basic_query.py` | Getting started, health check, quota check |
| `analytics.py` | Full analytics workflow: identity → cases → graph → provenance |
| `face_search.py` | MATCH_FACE operator, threshold comparison, CCTV temporal search |
| `graph_traversal.py` | PATHS_BETWEEN, temporal network shift, hub detection |
| `audit.py` | Chain verification, audit summary, anomaly detection |

## Deployment profiles

Relata ships three deployment profiles.  The SDK works identically across all three:

| Profile | Use case | Binary |
|---|---|---|
| `lite` | Embedded / single-process | `RELATA_PROFILE=lite relata serve` |
| `server` | Single-node production | `RELATA_PROFILE=server relata serve` |
| `cluster` | Multi-node distributed | `RELATA_PROFILE=cluster relata serve` |

## License

AGPL-3.0-only.  See the root `LICENSE` file.
