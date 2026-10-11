from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from relata import AsyncFeedBrokerClient, FeedBrokerClient


ENTRY = {
    "entry_id": "entry-1",
    "channel": "osint-ioc",
    "publisher_id": "did:web:feeds.example.test",
    "published_at": "2026-10-11T00:00:00Z",
    "valid_from": "2026-10-11T00:00:00Z",
    "tlp": "WHITE",
    "confidence": 90,
    "payload_kind": "object_assertion",
    "payload": {"indicator": "example.test"},
    "publisher_signature": "ed25519-abcd",
}


def test_sync_feed_broker_uses_explicit_url_and_publish_contract() -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        if req.url.path == "/feed/channels":
            return httpx.Response(200, json=[{"path": "relata-feed/osint-ioc", "min_tlp": "TLP:WHITE"}])
        return httpx.Response(200, json={"ok": True})

    client = FeedBrokerClient(
        "http://broker.test:9190", bearer_token="broker-token",
        transport=httpx.MockTransport(handler),
    )
    client.health()
    assert client.channels()[0]["path"] == "relata-feed/osint-ioc"
    client.publish(ENTRY["publisher_id"], ENTRY)
    assert [r.url for r in seen] == [
        httpx.URL("http://broker.test:9190/feed/health"),
        httpx.URL("http://broker.test:9190/feed/channels"),
        httpx.URL("http://broker.test:9190/feed/publish"),
    ]
    assert seen[2].headers["x-publisher-did"] == ENTRY["publisher_id"]
    assert seen[2].headers["authorization"] == "Bearer broker-token"
    assert json.loads(seen[2].content) == ENTRY


def test_async_feed_broker_uses_separate_broker_url_and_header() -> None:
    seen: list[httpx.Request] = []

    async def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json={"ok": True})

    async def run() -> None:
        async with AsyncFeedBrokerClient(
            "http://broker.test", transport=httpx.MockTransport(handler)
        ) as client:
            await client.publish(ENTRY["publisher_id"], ENTRY)

    asyncio.run(run())
    assert seen[0].url == httpx.URL("http://broker.test/feed/publish")
    assert seen[0].headers["x-publisher-did"] == ENTRY["publisher_id"]
    assert json.loads(seen[0].content) == ENTRY


def test_feed_publish_requires_publisher_did() -> None:
    client = FeedBrokerClient("http://broker.test", transport=httpx.MockTransport(lambda _: httpx.Response(200)))
    with pytest.raises(ValueError, match="publisher_did"):
        client.publish(" ", ENTRY)
