"""Tests for multimedia operators over native HTTP using mock transports."""

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


def test_multimedia_sql_never_wraps_operator_in_select_from() -> None:
    """#5497: the engine only parses the bare top-level OPERATOR(...) form.

    A ``SELECT * FROM`` wrapper parses the operator name as a table
    identifier and fails on the trailing ``(`` — every builder here must
    emit the bare form (mirrors the TS SDK's #5496 regression test).
    """
    from relata.client import _similar_image_sql

    assert "SELECT" not in _similar_image_sql("m", threshold=None, index=None)


def test_similar_image_sql_bare_form() -> None:
    """No threshold/index: mirrors the parser's bare-form default (#2840)."""
    from relata.client import _similar_image_sql

    sql = _similar_image_sql("media-42", threshold=None, index=None)
    assert sql == "SIMILAR_IMAGE('media-42')"


def test_similar_image_sql_with_threshold_and_index() -> None:
    from relata.client import _similar_image_sql

    sql = _similar_image_sql("media-42", threshold=0.6, index="ncmec")
    assert sql == (
        "SIMILAR_IMAGE('media-42', THRESHOLD => 0.6, INDEX => 'ncmec')"
    )


def test_similar_image_sql_threshold_only() -> None:
    from relata.client import _similar_image_sql

    sql = _similar_image_sql("media-42", threshold=0.6, index=None)
    assert sql == "SIMILAR_IMAGE('media-42', THRESHOLD => 0.6)"


def test_similar_image_sql_quotes_single_quotes() -> None:
    from relata.client import _similar_image_sql

    sql = _similar_image_sql("o'reilly", threshold=0.5, index="o'index")
    assert sql == (
        "SIMILAR_IMAGE('o''reilly', THRESHOLD => 0.5, INDEX => 'o''index')"
    )


# ---------------------------------------------------------------------------
# #2251 — HTTP dispatch through /v1/query
# ---------------------------------------------------------------------------


def test_similar_image_posts_similar_image_operator() -> None:
    seen: list[dict[str, object]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/v1/query"
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
        "SIMILAR_IMAGE('media-42', THRESHOLD => 0.6, INDEX => 'ncmec')"
    )
    assert seen[0]["purpose"] == "investigation"
    assert result.rows[0]["entity_id"] == "e-7"


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
        "SIMILAR_IMAGE('media-42', THRESHOLD => 0.6, INDEX => 'ncmec')"
    )
    return httpx.Response(
        200, json={"rows": [{"entity_id": "y"}], "query_id": "q", "elapsed_ms": 0}
    )


# ---------------------------------------------------------------------------
# #2253 — Flight ticket + endpoint builders (pure unit tests; live do_get
# needs a running Flight door + pyarrow).
# ---------------------------------------------------------------------------










# ---------------------------------------------------------------------------
# #3213 — Flight call metadata threads tenant/acting-as/delegated-by.
# Pure unit tests (no pyarrow): the Flight door must receive the same scope
# the HTTP door does, or a multi-tenant deployment silently falls back to the
# default tenant.
# ---------------------------------------------------------------------------
