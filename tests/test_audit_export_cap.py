"""Generic capped raw transport remains after retiring vertical PDF export."""
import httpx
import pytest
from relata._http import AsyncHttpTransport, HttpTransport
from relata.exceptions import ResponseTooLargeError

BASE = "http://localhost:9090"

def test_raw_response_cap_refuses_oversize() -> None:
    transport = HttpTransport(BASE, "tok", 1.0, max_response_bytes=1024,
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"x" * 2048)))
    try:
        with transport._client.stream("GET", "/objects/Blob/1") as response:
            with pytest.raises(ResponseTooLargeError):
                transport._read_capped(response)
    finally:
        transport.close()

def test_raw_response_within_cap_retains_exact_bytes() -> None:
    payload=b"binary\x00content"
    transport=HttpTransport(BASE,"tok",1.0,max_response_bytes=1024,
        transport=httpx.MockTransport(lambda _: httpx.Response(200,content=payload)))
    try:
        with transport._client.stream("GET", "/objects/Blob/1") as response:
            assert transport._read_capped(response)==payload
    finally:
        transport.close()

@pytest.mark.asyncio
async def test_async_raw_cap_refuses_oversize() -> None:
    transport=AsyncHttpTransport(BASE,"tok",1.0,max_response_bytes=1024,
        transport=httpx.MockTransport(lambda _: httpx.Response(200,content=b"x"*2048)))
    try:
        async with transport._client.stream("GET", "/objects/Blob/1") as response:
            with pytest.raises(ResponseTooLargeError):
                await transport._read_capped(response)
    finally:
        await transport.aclose()
