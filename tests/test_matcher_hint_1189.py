"""Matcher-hint injection tests for the VF3 subgraph matcher (#1189).

The server accepts a ``/*+ matcher=auto|vf3|bfs */`` hint comment in Cypher
query text to select the subgraph matcher. The SDK exposes this as a typed
``matcher`` kwarg on the query methods; these tests assert the outgoing query
text carries the hint (and that ``None`` / ``"auto"`` leave the text alone).
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from relata import RelataClient
from relata._http import AsyncHttpTransport, HttpTransport
from relata.client import _inject_matcher_hint

BASE = "http://localhost:9090"

Handler = Callable[[httpx.Request], httpx.Response]

CYPHER = (
    "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) RETURN a, b, c"
)


class _CallTracker:
    """MockTransport handler that records every request body."""

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
# _inject_matcher_hint — the shared helper
# ---------------------------------------------------------------------------


def test_inject_none_and_auto_leave_sql_untouched() -> None:
    assert _inject_matcher_hint(CYPHER, None) == CYPHER
    assert _inject_matcher_hint(CYPHER, "auto") == CYPHER


def test_inject_vf3_prepends_hint() -> None:
    assert _inject_matcher_hint(CYPHER, "vf3") == f"/*+ matcher=vf3 */ {CYPHER}"


def test_inject_bfs_prepends_hint() -> None:
    assert _inject_matcher_hint(CYPHER, "bfs") == f"/*+ matcher=bfs */ {CYPHER}"


def test_inject_rejects_unknown_matcher() -> None:
    with pytest.raises(ValueError, match="matcher"):
        _inject_matcher_hint(CYPHER, "gremlin")


# ---------------------------------------------------------------------------
# Wire-level — the hint reaches the outgoing /query body
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "matcher,hint",
    [("vf3", "/*+ matcher=vf3 */"), ("bfs", "/*+ matcher=bfs */")],
)
def test_query_sends_hint(matcher: str, hint: str) -> None:
    tracker = _CallTracker()
    _client(tracker).query(CYPHER, matcher=matcher)
    assert tracker.bodies[0]["sql"] == f"{hint} {CYPHER}"


@pytest.mark.parametrize("matcher", [None, "auto"])
def test_query_sends_no_hint_for_auto(matcher: str | None) -> None:
    tracker = _CallTracker()
    _client(tracker).query(CYPHER, matcher=matcher)
    assert tracker.bodies[0]["sql"] == CYPHER


def test_query_rejects_unknown_matcher_before_any_http_call() -> None:
    tracker = _CallTracker()
    with pytest.raises(ValueError, match="matcher"):
        _client(tracker).query(CYPHER, matcher="nope")
    assert tracker.calls == 0


def test_query_params_sends_hint() -> None:
    tracker = _CallTracker()
    _client(tracker).query_params(CYPHER, [], matcher="vf3")
    assert tracker.bodies[0]["sql"] == f"/*+ matcher=vf3 */ {CYPHER}"


async def test_aquery_sends_hint() -> None:
    tracker = _CallTracker()
    await _client(tracker).aquery(CYPHER, matcher="bfs")
    assert tracker.bodies[0]["sql"] == f"/*+ matcher=bfs */ {CYPHER}"


async def test_aquery_params_sends_hint() -> None:
    tracker = _CallTracker()
    await _client(tracker).aquery_params(CYPHER, [], matcher="vf3")
    assert tracker.bodies[0]["sql"] == f"/*+ matcher=vf3 */ {CYPHER}"
