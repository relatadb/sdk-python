"""SQL-injection regression tests for the operator + vector helpers (#3211).

The audit found that every language mirror built SQL by interpolating
caller-controlled identifiers (``object_type``, ``embedding_slot``) and values
into the governed ``/query`` body with no validation. The Python fix:

- Identifiers are validated against ``^[A-Za-z_][A-Za-z0-9_]*$`` and rejected
  client-side (``ValueError``) before any HTTP call is made.
- ``metric`` is allowlisted to ``{cosine, l2, dotproduct}``.
- Free-form values (query text, reference ids, identity values, erase
  subject/reason) are routed through the server-side parameterized path
  (``$N`` binding) instead of string interpolation.

Each test feeds the classic injection payloads (``'``, ``;``, ``--``, ``)``)
and asserts either a client-side rejection or a bound parameter — never a
raw interpolation into the outgoing SQL.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from relata import RelataClient
from relata._http import AsyncHttpTransport, HttpTransport

BASE = "http://localhost:9090"

Handler = Callable[[httpx.Request], httpx.Response]

# Classic injection payloads an attacker would try to break out of an
# identifier or value context.
PAYLOADS = [
    "Person'",
    "Person; DROP TABLE Person",
    "Person--",
    "Person)",
    "Person WHERE 1=1; --",
    "x' OR '1'='1",
]


class _CallTracker:
    """MockTransport handler that records every request body + counts calls."""

    def __init__(self) -> None:
        self.calls = 0
        self.bodies: list[dict[str, object]] = []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.calls += 1
        try:
            self.bodies.append(json.loads(req.content))
        except Exception:
            self.bodies.append({})
        return httpx.Response(200, json={"rows": [], "query_id": "q1", "elapsed_ms": 1})


def _client(tracker: _CallTracker) -> RelataClient:
    client = RelataClient(BASE, purpose="analytics")
    mock = httpx.MockTransport(tracker)
    client._RelataClient__sync_transport = HttpTransport(  # type: ignore[attr-defined]
        BASE, None, 30.0, transport=mock, extra_headers=None
    )
    client._RelataClient__async_transport = AsyncHttpTransport(  # type: ignore[attr-defined]
        BASE, None, 30.0, transport=mock, extra_headers=None
    )
    return client


# ---------------------------------------------------------------------------
# Identifier validation — knn_search / hybrid_search / similar_to
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("payload", PAYLOADS)
def test_knn_search_rejects_malicious_object_type(payload: str) -> None:
    from relata.vectors import VectorClient

    tracker = _CallTracker()
    v = VectorClient.from_client(_client(tracker))
    with pytest.raises(ValueError, match="object_type"):
        v.knn_search(payload, "_embedding", [0.1, 0.2], k=3)
    assert tracker.calls == 0  # rejected before any HTTP call


@pytest.mark.parametrize("payload", PAYLOADS)
def test_knn_search_rejects_malicious_embedding_slot(payload: str) -> None:
    from relata.vectors import VectorClient

    tracker = _CallTracker()
    v = VectorClient.from_client(_client(tracker))
    with pytest.raises(ValueError, match="embedding_slot"):
        v.knn_search("Person", payload, [0.1, 0.2], k=3)
    assert tracker.calls == 0


@pytest.mark.parametrize("payload", PAYLOADS)
def test_hybrid_search_rejects_malicious_object_type(payload: str) -> None:
    from relata.vectors import VectorClient

    tracker = _CallTracker()
    v = VectorClient.from_client(_client(tracker))
    with pytest.raises(ValueError, match="object_type"):
        v.hybrid_search(payload, query_text="alice", k=3)
    assert tracker.calls == 0


@pytest.mark.parametrize("payload", PAYLOADS)
def test_similar_to_rejects_malicious_object_type(payload: str) -> None:
    from relata.vectors import VectorClient

    tracker = _CallTracker()
    v = VectorClient.from_client(_client(tracker))
    with pytest.raises(ValueError, match="object_type"):
        v.similar_to(payload, "ref-1", k=3)
    assert tracker.calls == 0


@pytest.mark.parametrize("payload", PAYLOADS)
def test_hybrid_search_sql_rejects_metric_injection(payload: str) -> None:
    from relata.vectors import _hybrid_search_sql

    with pytest.raises(ValueError, match="metric"):
        _hybrid_search_sql("Person", "alice", k=3, metric=payload)


def test_hybrid_search_allows_known_metrics() -> None:
    from relata.vectors import _hybrid_search_sql

    for metric in ("cosine", "l2", "dotproduct"):
        sql, params = _hybrid_search_sql("Person", "alice", k=3, metric=metric)
        assert f"METRIC {metric}" in sql
        assert params == ["alice"]


# ---------------------------------------------------------------------------
# Value binding — values go through $N, never interpolation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("payload", PAYLOADS)
def test_hybrid_search_binds_query_text(payload: str) -> None:
    from relata.vectors import VectorClient

    tracker = _CallTracker()
    v = VectorClient.from_client(_client(tracker))
    v.hybrid_search("Person", query_text=payload, k=3)
    body = tracker.bodies[0]
    # The value is a bound parameter, never present raw in the SQL text.
    assert body["sql"] == "HYBRID_SEARCH FROM Person QUERY $1 LIMIT 3"
    assert body["params"] == [payload]
    assert payload not in body["sql"]


@pytest.mark.parametrize("payload", PAYLOADS)
def test_similar_to_binds_reference_id(payload: str) -> None:
    from relata.vectors import VectorClient

    tracker = _CallTracker()
    v = VectorClient.from_client(_client(tracker))
    v.similar_to("Person", payload, k=3)
    body = tracker.bodies[0]
    assert body["sql"] == "SELECT * FROM SIMILAR TO Person WHERE id = $1 LIMIT 3"
    assert body["params"] == [payload]
    assert payload not in body["sql"]


@pytest.mark.parametrize("payload", PAYLOADS)
def test_resolve_identity_binds_value(payload: str) -> None:
    tracker = _CallTracker()
    client = _client(tracker)
    client.resolve_identity(payload)
    body = tracker.bodies[0]
    assert body["sql"] == "RESOLVE_IDENTITY($1)"
    assert body["params"] == [payload]
    assert payload not in body["sql"]


@pytest.mark.parametrize("payload", PAYLOADS)
def test_erase_subject_binds_values(payload: str) -> None:
    tracker = _CallTracker()
    client = _client(tracker)
    client.erase_subject(payload, reason=payload)
    body = tracker.bodies[0]
    assert body["sql"] == "ERASE SUBJECT $1 REASON $2 CERTIFY"
    assert body["params"] == [payload, payload]
    assert payload not in body["sql"]


@pytest.mark.parametrize("payload", PAYLOADS)
def test_fuse_and_split_bind_values(payload: str) -> None:
    tracker = _CallTracker()
    client = _client(tracker)
    client.fuse_identities(payload, "b")
    client.split_identities("a", payload)
    assert tracker.bodies[0]["sql"] == "FUSE_IDENTITIES($1, $2)"
    assert tracker.bodies[0]["params"] == [payload, "b"]
    assert tracker.bodies[1]["sql"] == "SPLIT_IDENTITIES($1, $2)"
    assert tracker.bodies[1]["params"] == ["a", payload]


@pytest.mark.parametrize("payload", PAYLOADS)
def test_identity_erase_subject_binds_values(payload: str) -> None:
    from relata.identity import IdentityClient

    tracker = _CallTracker()
    c = IdentityClient(BASE, transport=httpx.MockTransport(tracker))
    c.erase_subject(payload, payload, certify=True, purpose="gdpr")
    body = tracker.bodies[0]
    assert body["sql"] == "ERASE SUBJECT $1 REASON $2 CERTIFY"
    assert body["params"] == [payload, payload]


# ---------------------------------------------------------------------------
# watch_stream — identifier + since-cursor validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("payload", PAYLOADS)
def test_watch_stream_rejects_malicious_type(payload: str) -> None:
    from relata.streaming import StreamingClient

    tracker = _CallTracker()
    stream = StreamingClient.from_client(_client(tracker))
    with pytest.raises(ValueError, match="object type"):
        stream.watch_stream(payload)
    assert tracker.calls == 0


@pytest.mark.parametrize("payload", ["1; DROP TABLE x", "1--", "1)", "1' OR '1'='1"])
def test_watch_stream_rejects_malicious_since(payload: str) -> None:
    from relata.streaming import StreamingClient

    tracker = _CallTracker()
    stream = StreamingClient.from_client(_client(tracker))
    with pytest.raises(ValueError, match="since"):
        stream.watch_stream("Person", since=payload)
    assert tracker.calls == 0


def test_watch_stream_accepts_numeric_since() -> None:
    from relata.streaming import _watch_stream_sql

    assert _watch_stream_sql("Person", "12345") == (
        "SELECT * FROM Person WHERE system_from > 12345"
    )
    assert _watch_stream_sql("Person", None) == "SELECT * FROM Person"
