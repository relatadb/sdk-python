"""#3214 follow-up: AuditClient.export_pdf caps the response body.

The audit PDF export used to read the raw httpx response via ``.content``
(unbounded) — a malicious/buggy server returning a multi-GB "PDF" would OOM
the Python SDK. It now streams the POST and reads through the transport's
``_read_capped``, so an oversized body raises ResponseTooLargeError instead of
being buffered. Parity with the TS ObjectClient fix and the Rust/Go caps.
"""

from __future__ import annotations

import httpx
import pytest

from relata.audit import AuditClient
from relata.exceptions import ResponseTooLargeError

BASE = "http://localhost:9090"


def test_export_pdf_caps_oversized_body() -> None:
    big = b"x" * (1 << 20)  # 1 MiB

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=big, headers={"content-type": "application/pdf"}
        )

    client = AuditClient(BASE, bearer_token="tok", transport=httpx.MockTransport(handler))
    # Force a tiny cap so the 1 MiB body trips it without allocating a 64 MiB fixture.
    client._t._max_response_bytes = 1024  # type: ignore[attr-defined]
    with pytest.raises(ResponseTooLargeError):
        client.export_pdf("case-1")


def test_export_pdf_within_cap_returns_bytes() -> None:
    payload = b"%PDF-1.4 short"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=payload, headers={"content-type": "application/pdf"}
        )

    client = AuditClient(BASE, bearer_token="tok", transport=httpx.MockTransport(handler))
    client._t._max_response_bytes = 4096  # type: ignore[attr-defined]
    assert client.export_pdf("case-1") == payload


@pytest.mark.asyncio
async def test_export_pdf_caps_oversized_body_async() -> None:
    big = b"y" * (1 << 20)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=big, headers={"content-type": "application/pdf"}
        )

    from relata.audit import AsyncAuditClient

    client = AsyncAuditClient(
        BASE, bearer_token="tok", transport=httpx.MockTransport(handler)
    )
    client._t._max_response_bytes = 1024  # type: ignore[attr-defined]
    with pytest.raises(ResponseTooLargeError):
        await client.export_pdf("case-1")
