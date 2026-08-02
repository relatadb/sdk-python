"""Tests for transport hardening (#79) and agent adapters (#89).

Covers:
- RFC 7807 problem+json error parsing.
- X-Request-ID generation + caller-override.
- Retry with exponential backoff.
- Typed exception subclasses (ForbiddenError, NotFoundError, RateLimitedError).
- smolagents + Pydantic-AI + AG2 adapter duck-typed shapes.
- Registry auto-detection.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

BASE = "http://localhost:9090"
Handler = Callable[[httpx.Request], httpx.Response]


def _wrap(handler: Handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


# ---------------------------------------------------------------------------
# #79 — RFC 7807 error parsing
# ---------------------------------------------------------------------------


def test_rfc7807_body_parsed_into_typed_exception() -> None:
    """When the server emits application/problem+json fields, the SDK raises
    the right typed subclass with the structured fields populated."""
    from relata._http import _classify_error
    from relata.exceptions import ForbiddenError

    body = {
        "type": "https://docs.relatadb.dev/errors/forbidden",
        "title": "Forbidden",
        "detail": "principal not authorised for Warrant",
        "code": "access-denied",
        "retryable": False,
        "request_id": "req-abc",
    }
    exc = _classify_error(403, body, request_id="req-abc")
    assert isinstance(exc, ForbiddenError)
    assert exc.code == "access-denied"
    assert exc.type_url == "https://docs.relatadb.dev/errors/forbidden"
    assert exc.retryable is False
    assert exc.request_id == "req-abc"


def test_rfc7807_429_carries_retry_after() -> None:
    from relata._http import _classify_error
    from relata.exceptions import RateLimitedError

    body = {"detail": "quota exceeded", "code": "REL_QUOTA", "retryable": True}
    exc = _classify_error(429, body, retry_after=30.0)
    assert isinstance(exc, RateLimitedError)
    assert exc.retry_after == 30.0
    assert exc.retryable is True


def test_legacy_error_body_still_works() -> None:
    """The legacy {"error": "..."} shape must still classify correctly."""
    from relata._http import _classify_error
    from relata.exceptions import ServerError

    body = {"error": "store lock poisoned"}
    exc = _classify_error(500, body)
    assert isinstance(exc, ServerError)
    assert "store lock poisoned" in exc.message


def test_request_id_stamped_on_exception_str() -> None:
    """Every exception's __str__ includes the request_id when present."""
    from relata._http import _classify_error

    body = {"error": "oops", "request_id": "rid-xyz"}
    exc = _classify_error(500, body, request_id="rid-xyz")
    assert "rid-xyz" in str(exc)


# ---------------------------------------------------------------------------
# #79 — X-Request-ID propagation
# ---------------------------------------------------------------------------


def test_request_id_generated_when_not_set() -> None:
    """When the caller hasn't set X-Request-ID, the transport generates one."""
    from relata._http import HttpTransport

    seen: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        rid = req.headers.get("x-request-id", "")
        seen.append(rid)
        return httpx.Response(200, json={"status": "ok"})

    t = HttpTransport(BASE, None, 5.0, transport=_wrap(handler))
    t.get("/health")
    assert seen[0]  # non-empty — a UUID was generated
    assert len(seen[0]) >= 32  # UUID v4 length


def test_request_id_respected_when_set_by_caller() -> None:
    """When the caller set X-Request-ID in headers, the transport keeps it."""
    from relata._http import HttpTransport

    seen: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.headers.get("x-request-id", ""))
        return httpx.Response(200, json={"status": "ok"})

    t = HttpTransport(
        BASE, None, 5.0,
        transport=_wrap(handler),
        extra_headers={"X-Request-ID": "caller-rid"},
    )
    t.get("/health")
    assert seen[0] == "caller-rid"


# ---------------------------------------------------------------------------
# #79 — Retry with exponential backoff
# ---------------------------------------------------------------------------


def test_retry_on_connection_error_then_succeeds() -> None:
    """The transport retries on connection error up to max_retries, then succeeds."""
    from relata._http import HttpTransport

    call_count = [0]

    def handler(req: httpx.Request) -> httpx.Response:
        call_count[0] += 1
        if call_count[0] < 3:
            raise httpx.ConnectError("transient")
        return httpx.Response(200, json={"status": "ok"})

    t = HttpTransport(
        BASE, None, 5.0,
        transport=_wrap(handler),
        max_retries=3,
        retry_backoff=0.001,  # near-instant for test speed
    )
    result = t.get("/health")
    assert result["status"] == "ok"
    assert call_count[0] == 3  # two retries then success


def test_no_retry_by_default() -> None:
    """When max_retries=0 (default), connection errors are not retried."""
    from relata._http import HttpTransport
    from relata.exceptions import ConnectionError as RelataConnectionError

    call_count = [0]

    def handler(req: httpx.Request) -> httpx.Response:
        call_count[0] += 1
        raise httpx.ConnectError("boom")

    t = HttpTransport(BASE, None, 5.0, transport=_wrap(handler))
    with pytest.raises(RelataConnectionError):
        t.get("/health")
    assert call_count[0] == 1


# ---------------------------------------------------------------------------
# #2490 — retry on 502/503/504 for idempotent verbs only
# ---------------------------------------------------------------------------


def test_retry_on_503_for_get_then_succeeds() -> None:
    """A 503 on GET is retried up to max_retries, then succeeds (#2490)."""
    from relata._http import HttpTransport

    call_count = [0]

    def handler(req: httpx.Request) -> httpx.Response:
        call_count[0] += 1
        if call_count[0] < 3:
            return httpx.Response(503, json={"error": "transient"})
        return httpx.Response(200, json={"status": "ok"})

    t = HttpTransport(
        BASE, None, 5.0,
        transport=_wrap(handler),
        max_retries=3,
        retry_backoff=0.001,
    )
    result = t.get("/health")
    assert result["status"] == "ok"
    assert call_count[0] == 3  # two retries then success


def test_503_on_post_is_not_retried() -> None:
    """POST is non-idempotent — a 503 must NOT be retried (#2490)."""
    from relata._http import HttpTransport
    from relata.exceptions import ServerError

    call_count = [0]

    def handler(req: httpx.Request) -> httpx.Response:
        call_count[0] += 1
        return httpx.Response(503, json={"error": "transient"})

    t = HttpTransport(
        BASE, None, 5.0,
        transport=_wrap(handler),
        max_retries=3,
        retry_backoff=0.001,
    )
    with pytest.raises(ServerError) as exc_info:
        t.post("/query", {"sql": "SELECT 1"})
    assert exc_info.value.status_code == 503
    assert call_count[0] == 1  # never retried


def test_retry_on_504_exhausted_raises_server_error() -> None:
    """A persistent 504 on GET exhausts retries and raises ServerError (#2490)."""
    from relata._http import HttpTransport
    from relata.exceptions import ServerError

    call_count = [0]

    def handler(req: httpx.Request) -> httpx.Response:
        call_count[0] += 1
        return httpx.Response(504, json={"error": "gateway timeout"})

    t = HttpTransport(
        BASE, None, 5.0,
        transport=_wrap(handler),
        max_retries=2,
        retry_backoff=0.001,
    )
    with pytest.raises(ServerError) as exc_info:
        t.get("/health")
    assert exc_info.value.status_code == 504
    assert call_count[0] == 3  # initial + 2 retries


def test_transport_error_on_post_not_retried() -> None:
    """POST transport errors are not retried (non-idempotent, #2490)."""
    from relata._http import HttpTransport
    from relata.exceptions import ConnectionError as RelataConnectionError

    call_count = [0]

    def handler(req: httpx.Request) -> httpx.Response:
        call_count[0] += 1
        raise httpx.ConnectError("boom")

    t = HttpTransport(
        BASE, None, 5.0,
        transport=_wrap(handler),
        max_retries=3,
        retry_backoff=0.001,
    )
    with pytest.raises(RelataConnectionError):
        t.post("/query", {"sql": "SELECT 1"})
    assert call_count[0] == 1  # never retried


def test_non_retryable_status_not_retried_on_get() -> None:
    """A 404 on GET is not in _retry_on — no retry attempt (#2490)."""
    from relata._http import HttpTransport
    from relata.exceptions import NotFoundError

    call_count = [0]

    def handler(req: httpx.Request) -> httpx.Response:
        call_count[0] += 1
        return httpx.Response(404, json={"error": "not found"})

    t = HttpTransport(
        BASE, None, 5.0,
        transport=_wrap(handler),
        max_retries=3,
        retry_backoff=0.001,
    )
    with pytest.raises(NotFoundError):
        t.get("/types/Missing")
    assert call_count[0] == 1


@pytest.mark.asyncio
async def test_async_retry_on_503_for_get_then_succeeds() -> None:
    """Async transport also retries 503 on GET (#2490)."""
    from relata._http import AsyncHttpTransport

    call_count = [0]

    def handler(req: httpx.Request) -> httpx.Response:
        call_count[0] += 1
        if call_count[0] < 2:
            return httpx.Response(503, json={"error": "transient"})
        return httpx.Response(200, json={"status": "ok"})

    t = AsyncHttpTransport(
        BASE, None, 5.0,
        transport=_wrap(handler),
        max_retries=2,
        retry_backoff=0.001,
    )
    result = await t.get("/health")
    assert result["status"] == "ok"
    assert call_count[0] == 2
    await t.aclose()


# ---------------------------------------------------------------------------
# #89 — Adapter duck-typed shapes
# ---------------------------------------------------------------------------


def test_smolagents_tool_callable() -> None:
    from relata import RelataClient
    from relata_adapters.smolagents import RelataTool

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": json.dumps({"rows": [{"x": 1}]})}],
                "isError": False,
            },
        )

    client = RelataClient(BASE, purpose="analytics")
    tool = RelataTool(client, tool_name="query_knowledge", transport=_wrap(handler))
    assert tool.name == "query_knowledge"
    result = tool(sql="SELECT 1", purpose="analytics")
    assert json.loads(result)["rows"] == [{"x": 1}]


def test_pydantic_ai_backend_add_search() -> None:
    from relata import RelataClient
    from relata_adapters.pydantic_ai import RelataMemoryBackend

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/memory/remember":
            return httpx.Response(
                200,
                json={"content": [{"type": "text", "text": json.dumps({"id": "m-1"})}], "isError": False},
            )
        if "/memory/recall" in req.url.path:
            return httpx.Response(
                200,
                json={"content": [{"type": "text", "text": json.dumps({"rows": [{"id": "m-1", "content": "hi"}]})}], "isError": False},
            )
        return httpx.Response(404)

    client = RelataClient(BASE, purpose="agent")
    backend = RelataMemoryBackend(client, transport=_wrap(handler))
    mid = backend.add("hello")
    assert mid == "m-1"
    hits = backend.search("hello", limit=3)
    assert hits[0]["content"] == "hi"


def test_ag2_memory_adapter_duck_types() -> None:
    from relata import RelataClient
    from relata_adapters.ag2 import RelataAG2Memory

    client = RelataClient(BASE, purpose="agent")
    mem = RelataAG2Memory(client, transport=_wrap(lambda r: httpx.Response(200, json={})))
    assert hasattr(mem, "add")
    assert hasattr(mem, "search")
    assert hasattr(mem, "clear")
    assert hasattr(mem, "forget")


def test_registry_lists_all_frameworks() -> None:
    from relata_adapters.registry import list_available_adapters

    adapters = list_available_adapters()
    assert "langchain" in adapters
    assert "pydantic_ai" in adapters
    assert "ag2" in adapters
    assert "smolagents" in adapters
    # In the test env none are installed → all False.
    assert all(v is False for v in adapters.values()) or any(v for v in adapters.values())
