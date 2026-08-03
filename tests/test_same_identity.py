"""Tests for ``RelataClient.same_identity`` (#2246).

Verifies the SQL emitted to ``POST /query`` (``SAME_IDENTITY($1, $2)`` with
the values bound via the server-side parameterized path — #3211) and the
boolean decode of the ``match`` verdict across the true / false / empty
cases. Uses ``httpx.MockTransport`` — no live server required.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx

from relata import RelataClient
from relata._http import AsyncHttpTransport, HttpTransport

BASE = "http://localhost:9090"


def _mock_client(handler: Callable[[httpx.Request], httpx.Response]) -> RelataClient:
    client = RelataClient(BASE, bearer_token="tok")
    mock = httpx.MockTransport(handler)
    extra = client._extra_headers  # type: ignore[attr-defined]
    client._RelataClient__sync_transport = HttpTransport(  # type: ignore[attr-defined]
        BASE, client._bearer_token, client._timeout, transport=mock, extra_headers=extra
    )
    client._RelataClient__async_transport = AsyncHttpTransport(  # type: ignore[attr-defined]
        BASE, client._bearer_token, client._timeout, transport=mock, extra_headers=extra
    )
    return client


def test_same_identity_true_decodes_match_and_emits_sql() -> None:
    captured: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured.update(json.loads(req.content))
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "match": True,
                        "confidence": 0.9,
                        "evidence": "direct_link(same_person)",
                    }
                ],
                "columns": ["match", "confidence", "evidence"],
            },
        )

    client = _mock_client(handler)
    assert client.same_identity("+111", "a@b.com") is True
    assert captured["sql"] == "SAME_IDENTITY($1, $2)"
    assert captured["params"] == ["+111", "a@b.com"]
    assert captured["purpose"] == "analytics"


def test_same_identity_false_and_empty_return_false() -> None:
    def false_handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"match": False, "confidence": 0.0, "evidence": "no_evidence"}
                ]
            },
        )

    assert _mock_client(false_handler).same_identity("x", "y") is False

    def empty_handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": []})

    assert _mock_client(empty_handler).same_identity("x", "y") is False


def test_same_identity_binds_single_quotes_as_params() -> None:
    captured: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured.update(json.loads(req.content))
        return httpx.Response(200, json={"data": [{"match": False}]})

    _mock_client(handler).same_identity("o'brien", "x'y")
    # Values are bound as $1/$2 — never interpolated, so no escaping needed (#3211).
    assert captured["sql"] == "SAME_IDENTITY($1, $2)"
    assert captured["params"] == ["o'brien", "x'y"]
