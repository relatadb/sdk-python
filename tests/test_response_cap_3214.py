"""Bearer-zeroize + response-cap regression tests (#3214).

Parity with Rust #3195 / #3198. Two properties:

1. The bearer token is not reachable via any public accessor or ``repr``, and
   ``close()`` clears it (drops the SDK's reference and strips the
   ``Authorization`` header from any still-open transport).
2. Response buffering is capped (``max_response_bytes``, default ~64 MiB). A
   mocked body larger than the cap raises the typed
   :class:`~relata.exceptions.ResponseTooLargeError` instead of being fully
   buffered — a malicious/buggy server cannot OOM the client.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from relata import RelataClient, ResponseTooLargeError
from relata._http import DEFAULT_MAX_RESPONSE_BYTES, AsyncHttpTransport, HttpTransport

BASE = "http://localhost:9090"

Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: Handler, **kwargs: object) -> RelataClient:
    client = RelataClient(BASE, bearer_token="s3cr3t", purpose="analytics", **kwargs)  # type: ignore[arg-type]
    client._RelataClient__sync_transport = HttpTransport(  # type: ignore[attr-defined]
        BASE,
        "s3cr3t",
        30.0,
        transport=httpx.MockTransport(handler),
        max_response_bytes=client._max_response_bytes,  # type: ignore[attr-defined]
    )
    return client


# ---------------------------------------------------------------------------
# 1. Bearer not reachable via public accessor / repr; close() clears it
# ---------------------------------------------------------------------------


def test_no_public_bearer_accessor() -> None:
    client = RelataClient(BASE, bearer_token="s3cr3t")
    # No public attribute exposes the token.
    assert not hasattr(client, "bearer_token")
    assert not hasattr(client, "token")
    # The private holder has it (used only to build transports).
    assert client._bearer_token == "s3cr3t"  # type: ignore[attr-defined]


def test_repr_does_not_leak_bearer() -> None:
    client = RelataClient(BASE, bearer_token="s3cr3t")
    assert "s3cr3t" not in repr(client)


def test_close_clears_bearer() -> None:
    client = RelataClient(BASE, bearer_token="s3cr3t")
    _ = client._sync  # force transport creation
    client.close()
    assert client._bearer_token is None  # type: ignore[attr-defined]


async def test_aclose_clears_bearer() -> None:
    client = RelataClient(BASE, bearer_token="s3cr3t")
    await client.aclose()
    assert client._bearer_token is None  # type: ignore[attr-defined]


def test_close_strips_auth_header_from_open_async_transport() -> None:
    client = RelataClient(BASE, bearer_token="s3cr3t")
    async_t = client._async  # open the async transport, keep it open
    assert "Authorization" in async_t._client.headers
    client.close()  # sync close must strip the async transport's bearer too
    assert "Authorization" not in async_t._client.headers


def test_s3_client_bearer_private_and_close_clears() -> None:
    from relata.s3 import S3Client

    s3 = S3Client(BASE, bearer_token="s3cr3t")
    assert not hasattr(s3, "bearer_token")
    assert s3._bearer_token == "s3cr3t"
    s3.close()
    assert s3._bearer_token is None


def test_langgraph_checkpointer_token_private_and_close_clears() -> None:
    pytest.importorskip("langgraph")
    from relata_langgraph import RelataCheckpointer

    cp = RelataCheckpointer(BASE, "s3cr3t")
    assert not hasattr(cp, "token")
    assert cp._token == "s3cr3t"
    cp.close()
    assert cp._token == ""


# ---------------------------------------------------------------------------
# 2. Response cap — oversized body raises typed ResponseTooLargeError
# ---------------------------------------------------------------------------


def _big_body_handler(size: int) -> Handler:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * size)

    return handler


def test_default_cap_is_64mib() -> None:
    assert DEFAULT_MAX_RESPONSE_BYTES == 64 * 1024 * 1024


def test_oversized_body_raises_typed_error_on_read() -> None:
    """A body with no Content-Length is capped on the bytes actually read."""
    body = b"x" * 2048

    def handler(req: httpx.Request) -> httpx.Response:
        resp = httpx.Response(200, content=body)
        # Hide the size — the cap must be enforced while reading, not from
        # an advertised header.
        del resp.headers["content-length"]
        return resp

    client = _client(handler, max_response_bytes=1024)
    with pytest.raises(ResponseTooLargeError) as exc:
        client.query("SELECT 1")
    assert exc.value.max_bytes == 1024
    assert exc.value.content_length is None


def test_oversized_body_rejected_via_content_length() -> None:
    """An advertised Content-Length over the cap is rejected before buffering."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"x" * 4096,
            headers={"content-length": "4096", "content-type": "application/json"},
        )

    client = _client(handler, max_response_bytes=1024)
    with pytest.raises(ResponseTooLargeError) as exc:
        client.query("SELECT 1")
    assert exc.value.content_length == 4096


def test_body_within_cap_still_parses() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"rows": [{"id": 1}], "query_id": "q"})

    client = _client(handler, max_response_bytes=1024)
    result = client.query("SELECT 1")
    assert result.rows == [{"id": 1}]


def test_oversized_error_body_also_capped() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"e" * 4096)

    client = _client(handler, max_response_bytes=1024)
    with pytest.raises(ResponseTooLargeError):
        client.query("SELECT 1")


async def test_async_oversized_body_raises_typed_error() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 4096)

    client = RelataClient(BASE, bearer_token="s3cr3t", purpose="analytics", max_response_bytes=1024)
    client._RelataClient__async_transport = AsyncHttpTransport(  # type: ignore[attr-defined]
        BASE,
        "s3cr3t",
        30.0,
        transport=httpx.MockTransport(handler),
        max_response_bytes=1024,
    )
    with pytest.raises(ResponseTooLargeError):
        await client.aquery("SELECT 1")


def test_ingest_raw_path_respects_cap() -> None:
    """The NDJSON/CSV ingest doors route through the same capped transport."""
    from relata.ingest import IngestClient

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 4096)

    transport = HttpTransport(
        BASE, None, 30.0, transport=httpx.MockTransport(handler), max_response_bytes=1024
    )
    c = IngestClient(BASE)
    c._t = transport
    with pytest.raises(ResponseTooLargeError):
        c.bulk("Person", [{"id": 1}])
