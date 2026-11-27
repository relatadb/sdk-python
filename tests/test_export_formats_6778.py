"""``export_data`` decodes ``/v1/export`` by the response Content-Type (#6778):
csv comes back as text under ``content`` instead of ``{}`` or a JSONDecodeError."""

from __future__ import annotations

import httpx
import pytest

from relata import RelataClient
from relata._http import AsyncHttpTransport, HttpTransport, _decode_body
from relata.exceptions import RelataError

BASE = "http://localhost:9090"
CSV = "_pk,name,email\np1,Ada,ada@example.com\np2,Bob,bob@example.com\n"


def _respond(status: int, body: bytes, content_type: str) -> httpx.MockTransport:
    return httpx.MockTransport(
        lambda req: httpx.Response(status, content=body, headers={"content-type": content_type})
    )


def _sync(transport: httpx.MockTransport) -> RelataClient:
    client = RelataClient(BASE, purpose="analytics")
    client._RelataClient__sync_transport = HttpTransport(  # type: ignore[attr-defined]
        BASE, None, 30.0, transport=transport, extra_headers=None
    )
    return client


def _async(transport: httpx.MockTransport) -> RelataClient:
    client = RelataClient(BASE, purpose="analytics")
    client._RelataClient__async_transport = AsyncHttpTransport(  # type: ignore[attr-defined]
        BASE, None, 30.0, transport=transport, extra_headers=None
    )
    return client


def test_csv_with_rows_returns_text() -> None:
    out = _sync(_respond(200, CSV.encode(), "text/csv")).export_data("Person", format="csv")
    assert out == {"type": "Person", "format": "csv", "content": CSV}


def test_csv_for_empty_type_returns_empty_text() -> None:
    out = _sync(_respond(200, b"", "text/csv")).export_data("Empty", format="csv")
    assert out == {"type": "Empty", "format": "csv", "content": ""}


def test_json_export_still_parsed() -> None:
    body = b'{"type":"Person","format":"json","rows":1,"data":[{"_pk":"p1"}]}'
    out = _sync(_respond(200, body, "application/json")).export_data("Person")
    assert out == {"type": "Person", "format": "json", "rows": 1, "data": [{"_pk": "p1"}]}


def test_stix_export_parsed_as_json() -> None:
    body = b'{"type":"bundle","objects":[]}'
    out = _sync(_respond(200, body, "application/stix+json")).export_data("P", format="stix")
    assert out == {"type": "bundle", "objects": []}


def test_jsonl_export_returns_text() -> None:
    body = '{"_pk":"p1"}\n'
    out = _sync(_respond(200, body.encode(), "application/x-ndjson")).export_data(
        "Person", format="jsonl"
    )
    assert out["content"] == body


def test_export_error_still_raises() -> None:
    body = b'{"type":"about:blank","title":"Bad Request","status":400,"detail":"unknown type"}'
    with pytest.raises(RelataError):
        _sync(_respond(400, body, "application/problem+json")).export_data("Nope", format="csv")


async def test_async_csv_with_rows_and_empty_type() -> None:
    rows = await _async(_respond(200, CSV.encode(), "text/csv")).aexport_data(
        "Person", format="csv"
    )
    assert rows == {"type": "Person", "format": "csv", "content": CSV}
    empty = await _async(_respond(200, b"", "text/csv")).aexport_data("Empty", format="csv")
    assert empty == {"type": "Empty", "format": "csv", "content": ""}
    parsed = await _async(_respond(200, b'{"rows":0,"data":[]}', "application/json")).aexport_data(
        "Empty"
    )
    assert parsed == {"rows": 0, "data": []}


@pytest.mark.parametrize(
    ("content_type", "raw", "want"),
    [
        ("application/json", b'{"a":1}', {"a": 1}),
        ("", b"", {}),
        ("text/csv; charset=latin-1", "é\n".encode("latin-1"), "é\n"),
        ("application/octet-stream", b"\x00\x01", b"\x00\x01"),
    ],
)
def test_decode_body_by_media_type(content_type: str, raw: bytes, want: object) -> None:
    assert _decode_body(content_type, raw) == want
