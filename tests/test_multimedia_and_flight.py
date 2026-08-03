"""Tests for the multimedia ops (#2251) + Arrow Flight client (#2253).

Uses ``httpx.MockTransport`` for the SQL-door multimedia ops (no live server)
and pure unit tests for the Flight ticket/endpoint builders. The live Flight
RPC needs a running server + ``pyarrow`` so it is not exercised here.
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


def _wrap(handler: Handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def _mocked_client(handler: Handler) -> RelataClient:
    client = RelataClient(BASE, purpose="analytics")
    client._RelataClient__sync_transport = (  # type: ignore[attr-defined]
        HttpTransport(BASE, None, 30.0, transport=_wrap(handler), extra_headers=None)
    )
    return client


# ---------------------------------------------------------------------------
# #2251 — SQL builder shapes
# ---------------------------------------------------------------------------


def test_face_search_sql_from_list() -> None:
    from relata.client import _face_search_sql

    sql = _face_search_sql("gallery-1", [0.1, 0.2, 0.3], k=5, threshold=0.6)
    assert sql == (
        "SELECT * FROM FACE_SEARCH('0.1,0.2,0.3', 'gallery-1', "
        "K => 5, THRESHOLD => 0.6)"
    )


def test_face_search_sql_from_string() -> None:
    from relata.client import _face_search_sql

    sql = _face_search_sql("g", "1.0,0.0", k=10, threshold=0.7)
    assert "FACE_SEARCH('1.0,0.0', 'g', K => 10, THRESHOLD => 0.7)" in sql


def test_match_pdq_sql_shape() -> None:
    from relata.client import _match_pdq_sql

    sql = _match_pdq_sql("ncmec", "ffff", threshold=0.9)
    assert sql == "SELECT * FROM MATCH_PDQ('ffff', 'ncmec', THRESHOLD => 0.9)"


def test_multimedia_sql_quotes_single_quotes() -> None:
    """A gallery / corpus id with an apostrophe must be SQL-escaped (#76)."""
    from relata.client import _face_search_sql, _match_pdq_sql

    assert _face_search_sql("o'reilly", [0.1], k=1, threshold=0.5) == (
        "SELECT * FROM FACE_SEARCH('0.1', 'o''reilly', K => 1, THRESHOLD => 0.5)"
    )
    assert _match_pdq_sql("o'reilly", "ff", threshold=0.5) == (
        "SELECT * FROM MATCH_PDQ('ff', 'o''reilly', THRESHOLD => 0.5)"
    )


def test_similar_image_sql_bare_form() -> None:
    """No threshold/index: mirrors the parser's bare-form default (#2840)."""
    from relata.client import _similar_image_sql

    sql = _similar_image_sql("media-42", threshold=None, index=None)
    assert sql == "SELECT * FROM SIMILAR_IMAGE('media-42')"


def test_similar_image_sql_with_threshold_and_index() -> None:
    from relata.client import _similar_image_sql

    sql = _similar_image_sql("media-42", threshold=0.6, index="ncmec")
    assert sql == (
        "SELECT * FROM SIMILAR_IMAGE('media-42', THRESHOLD => 0.6, INDEX => 'ncmec')"
    )


def test_similar_image_sql_threshold_only() -> None:
    from relata.client import _similar_image_sql

    sql = _similar_image_sql("media-42", threshold=0.6, index=None)
    assert sql == "SELECT * FROM SIMILAR_IMAGE('media-42', THRESHOLD => 0.6)"


def test_similar_image_sql_quotes_single_quotes() -> None:
    from relata.client import _similar_image_sql

    sql = _similar_image_sql("o'reilly", threshold=0.5, index="o'index")
    assert sql == (
        "SELECT * FROM SIMILAR_IMAGE('o''reilly', THRESHOLD => 0.5, INDEX => 'o''index')"
    )


# ---------------------------------------------------------------------------
# #2251 — HTTP dispatch through /query
# ---------------------------------------------------------------------------


def test_face_search_posts_face_search_operator() -> None:
    seen: list[dict[str, object]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/query"
        body = json.loads(req.content)
        seen.append(body)
        return httpx.Response(
            200,
            json={
                "rows": [{"entity_id": "e-1", "score": 0.98}],
                "query_id": "q1",
                "elapsed_ms": 2,
            },
        )

    client = _mocked_client(handler)
    result = client.face_search("gallery-1", [0.1, 0.2], k=5, threshold=0.6)
    assert "SELECT * FROM FACE_SEARCH(" in str(seen[0]["sql"])
    assert "K => 5" in str(seen[0]["sql"])
    assert "THRESHOLD => 0.6" in str(seen[0]["sql"])
    assert seen[0]["purpose"] == "analytics"
    assert result.rows == [{"entity_id": "e-1", "score": 0.98}]


def test_match_pdq_posts_match_pdq_operator() -> None:
    seen: list[dict[str, object]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        seen.append(body)
        return httpx.Response(
            200,
            json={"rows": [{"entity_id": "e-9"}], "query_id": "q2", "elapsed_ms": 1},
        )

    client = _mocked_client(handler)
    result = client.match_pdq("ncmec", "ffff", threshold=0.95, purpose="investigation")
    assert str(seen[0]["sql"]) == (
        "SELECT * FROM MATCH_PDQ('ffff', 'ncmec', THRESHOLD => 0.95)"
    )
    assert seen[0]["purpose"] == "investigation"
    assert result.rows[0]["entity_id"] == "e-9"


def test_similar_image_posts_similar_image_operator() -> None:
    seen: list[dict[str, object]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/query"
        body = json.loads(req.content)
        seen.append(body)
        return httpx.Response(
            200,
            json={"rows": [{"entity_id": "e-7", "score": 0.91}], "query_id": "q3", "elapsed_ms": 1},
        )

    client = _mocked_client(handler)
    result = client.similar_image(
        "media-42", threshold=0.6, index="ncmec", purpose="investigation"
    )
    assert str(seen[0]["sql"]) == (
        "SELECT * FROM SIMILAR_IMAGE('media-42', THRESHOLD => 0.6, INDEX => 'ncmec')"
    )
    assert seen[0]["purpose"] == "investigation"
    assert result.rows[0]["entity_id"] == "e-7"


@pytest.mark.asyncio
async def test_aface_search_uses_same_sql() -> None:
    client = RelataClient(BASE, purpose="analytics")
    client._RelataClient__async_transport = (  # type: ignore[attr-defined]
        AsyncHttpTransport(BASE, None, 30.0, transport=_wrap(_async_face_handler))
    )
    result = await client.aface_search("g1", [0.1, 0.2], k=3)
    assert result.rows == [{"entity_id": "x"}]

@pytest.mark.asyncio
async def test_asimilar_image_uses_same_sql() -> None:
    client = RelataClient(BASE, purpose="analytics")
    client._RelataClient__async_transport = (  # type: ignore[attr-defined]
        AsyncHttpTransport(BASE, None, 30.0, transport=_wrap(_async_similar_image_handler))
    )
    result = await client.asimilar_image("media-42", threshold=0.6, index="ncmec")
    assert result.rows == [{"entity_id": "y"}]


def _async_similar_image_handler(req: httpx.Request) -> httpx.Response:
    body = json.loads(req.content)
    assert body["sql"] == (
        "SELECT * FROM SIMILAR_IMAGE('media-42', THRESHOLD => 0.6, INDEX => 'ncmec')"
    )
    return httpx.Response(
        200, json={"rows": [{"entity_id": "y"}], "query_id": "q", "elapsed_ms": 0}
    )


def _async_face_handler(req: httpx.Request) -> httpx.Response:
    body = json.loads(req.content)
    assert "FACE_SEARCH(" in body["sql"]
    return httpx.Response(
        200, json={"rows": [{"entity_id": "x"}], "query_id": "q", "elapsed_ms": 0}
    )


# ---------------------------------------------------------------------------
# #2253 — Flight ticket + endpoint builders (pure unit tests; live do_get
# needs a running Flight door + pyarrow).
# ---------------------------------------------------------------------------


def test_flight_ticket_injects_purpose_comment() -> None:
    from relata.client import _flight_ticket

    assert _flight_ticket("SELECT 1", None) == "SELECT 1"
    assert _flight_ticket("SELECT 1", "analytics") == "/* PURPOSE 'analytics' */ SELECT 1"


def test_flight_ticket_escapes_purpose_quote() -> None:
    from relata.client import _flight_ticket

    # The server scans for /* PURPOSE ' up to the next '; an embedded quote
    # must be doubled so the purpose is not truncated.
    assert _flight_ticket("SELECT 1", "o'reilly") == "/* PURPOSE 'o''reilly' */ SELECT 1"


def test_flight_endpoint_derives_from_base_url() -> None:
    from relata.client import _flight_endpoint_from

    assert _flight_endpoint_from("http://localhost:9090", None) == "grpc://localhost:8815"
    assert _flight_endpoint_from("https://relata.example.com", None) == (
        "grpc://relata.example.com:8815"
    )


def test_flight_endpoint_explicit_override_wins() -> None:
    from relata.client import _flight_endpoint_from

    assert _flight_endpoint_from("http://localhost:9090", "grpc://host:9999") == (
        "grpc://host:9999"
    )


# ---------------------------------------------------------------------------
# #3213 — Flight call metadata threads tenant/acting-as/delegated-by.
# Pure unit tests (no pyarrow): the Flight door must receive the same scope
# the HTTP door does, or a multi-tenant cluster silently falls back to the
# default tenant.
# ---------------------------------------------------------------------------


def test_flight_metadata_threads_tenant_scope() -> None:
    from relata import RelataClient
    from relata.flight import flight_call_headers

    client = RelataClient(
        BASE,
        bearer_token="tok",
        tenant="acme",
        acting_as="agent-1",
        delegated_by="root-org",
        headers={"X-Verified-Principal": "svc-a"},
    )
    headers = dict(flight_call_headers(client, None))
    assert headers[b"authorization"] == b"Bearer tok"
    assert headers[b"x-relata-tenant-id"] == b"acme"
    assert headers[b"x-acting-as"] == b"agent-1"
    assert headers[b"x-delegated-by"] == b"root-org"
    assert headers[b"x-verified-principal"] == b"svc-a"
    # Per-request correlation id is always present.
    assert b"x-request-id" in headers
    assert headers[b"x-request-id"]


def test_flight_metadata_bearer_override_wins() -> None:
    from relata import RelataClient
    from relata.flight import flight_call_headers

    client = RelataClient(BASE, bearer_token="tok", tenant="acme")
    headers = dict(flight_call_headers(client, "override-tok"))
    assert headers[b"authorization"] == b"Bearer override-tok"
    assert headers[b"x-relata-tenant-id"] == b"acme"


def test_flight_metadata_unauthenticated_still_threads_tenant() -> None:
    from relata import RelataClient
    from relata.flight import flight_call_headers

    client = RelataClient(BASE, tenant="acme")
    headers = dict(flight_call_headers(client, None))
    assert b"authorization" not in headers
    assert headers[b"x-relata-tenant-id"] == b"acme"
    assert b"x-request-id" in headers


def test_flight_metadata_caller_request_id_wins() -> None:
    from relata import RelataClient
    from relata.flight import flight_call_headers

    client = RelataClient(BASE, headers={"X-Request-ID": "pinned-rid"})
    headers = dict(flight_call_headers(client, None))
    assert headers[b"x-request-id"] == b"pinned-rid"


def test_query_flight_requires_pyarrow(monkeypatch: pytest.MonkeyPatch) -> None:
    """If pyarrow is not importable the method raises ImportError, not a crash."""
    import builtins

    from relata import RelataClient

    real_import = builtins.__import__

    def _block(name: str, *args: object) -> object:
        if name.startswith("pyarrow"):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args)  # type: ignore[misc]

    monkeypatch.setattr(builtins, "__import__", _block)
    client = RelataClient(BASE, purpose="analytics")
    with pytest.raises(ImportError):
        client.query_flight("SELECT 1")
