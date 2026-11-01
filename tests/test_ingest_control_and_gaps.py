"""Routes and bodies for the ingest control plane, media qualification, memory
association batch and Files folders -- SDK parity with the server routes."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from relata import (
    AsyncFilesClient,
    AsyncIngestControlClient,
    FilesClient,
    IngestControlClient,
    Memory,
    RelataError,
)
from relata.memory import AsyncMemory

BASE = "http://localhost:9090"


def _rec(
    seen: list[tuple[str, str, str, Any]], status: int = 200, body: object = None
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(req: httpx.Request) -> httpx.Response:
        payload = json.loads(req.content) if req.content else None
        seen.append(
            (req.method, req.url.raw_path.decode().split("?")[0], req.url.query.decode(), payload)
        )
        if status == 204:
            return httpx.Response(204)
        return httpx.Response(status, json=body if body is not None else {})

    return handler


def test_ingest_control_routes_and_bodies() -> None:
    seen: list[tuple[str, str, str, Any]] = []
    c = IngestControlClient(BASE, transport=httpx.MockTransport(_rec(seen)))
    c.checkpoint("kafka/orders 1")
    c.list_contracts()
    c.get_contract("orders")
    c.put_contract("orders", {"version": 2, "fields": []})
    c.contract_quality("orders")
    c.list_quarantine(dataset="orders", after_id=5, limit=50)
    c.list_quarantine()
    assert seen == [
        ("GET", "/v1/ingest/checkpoints/kafka%2Forders%201", "", None),
        ("GET", "/v1/ingest/contracts", "", None),
        ("GET", "/v1/ingest/contracts/orders", "", None),
        ("PUT", "/v1/ingest/contracts/orders", "", {"version": 2, "fields": []}),
        ("GET", "/v1/ingest/contracts/orders/quality", "", None),
        ("GET", "/v1/ingest/quarantine", "dataset=orders&after_id=5&limit=50", None),
        ("GET", "/v1/ingest/quarantine", "", None),
    ]


def test_quarantine_delete_expects_204() -> None:
    seen: list[tuple[str, str, str, Any]] = []
    c = IngestControlClient(BASE, transport=httpx.MockTransport(_rec(seen, 204)))
    assert c.delete_quarantine(9) is None  # type: ignore[func-returns-value]
    assert seen == [("DELETE", "/v1/ingest/quarantine/9", "", None)]
    missing = IngestControlClient(
        BASE, transport=httpx.MockTransport(_rec([], 404, {"title": "nf", "status": 404}))
    )
    with pytest.raises(RelataError):
        missing.delete_quarantine(1)


def test_ingest_control_async() -> None:
    seen: list[tuple[str, str, str, Any]] = []

    async def run() -> None:
        c = AsyncIngestControlClient(BASE, transport=httpx.MockTransport(_rec(seen)))
        await c.checkpoint("s")
        await c.put_contract("d", {"version": 1})
        await c.list_quarantine(limit=1)
        await c.close()

    asyncio.run(run())
    assert [(s[0], s[1]) for s in seen] == [
        ("GET", "/v1/ingest/checkpoints/s"),
        ("PUT", "/v1/ingest/contracts/d"),
        ("GET", "/v1/ingest/quarantine"),
    ]


def test_async_quarantine_delete() -> None:
    seen: list[tuple[str, str, str, Any]] = []

    async def run() -> None:
        c = AsyncIngestControlClient(BASE, transport=httpx.MockTransport(_rec(seen, 204)))
        await c.delete_quarantine(3)
        await c.close()

    asyncio.run(run())
    assert seen == [("DELETE", "/v1/ingest/quarantine/3", "", None)]


def test_memory_associate_batch() -> None:
    seen: list[tuple[str, str, str, Any]] = []
    m = Memory(BASE, purpose="notes", transport=httpx.MockTransport(_rec(seen, body={"stored": 2})))
    out = m.associate_batch(
        [
            {"from_id": "a", "to_id": "b", "relation": "causes"},
            {"from_id": "c", "to_id": "d", "purpose": "x"},
        ]
    )
    assert out == {"stored": 2}
    assert seen == [
        (
            "POST",
            "/v1/memory/associate/batch",
            "",
            {
                "items": [
                    {"purpose": "notes", "from_id": "a", "to_id": "b", "relation": "causes"},
                    {"purpose": "x", "from_id": "c", "to_id": "d"},
                ]
            },
        )
    ]
    with pytest.raises(ValueError):
        m.associate_batch([{"from_id": "a"}])


def test_async_memory_associate_batch() -> None:
    seen: list[tuple[str, str, str, Any]] = []

    async def run() -> None:
        m = AsyncMemory(BASE, purpose="notes", transport=httpx.MockTransport(_rec(seen)))
        await m.associate_batch([{"from_id": "a", "to_id": "b"}])
        await m.close()

    asyncio.run(run())
    assert seen[0][1] == "/v1/memory/associate/batch"
    assert seen[0][3] == {"items": [{"purpose": "notes", "from_id": "a", "to_id": "b"}]}


_FOLDER = dict(
    bucket="t", key="a/", row_id=None, size=0, etag='"e"', content_type="application/x-directory",
    modified_ns=1, is_folder=True, download_url="/v1/files/content?key=a%2F",
)


def test_files_create_folder() -> None:
    seen: list[tuple[str, str, str, Any]] = []
    c = FilesClient(BASE, transport=httpx.MockTransport(_rec(seen, 201, _FOLDER)))
    meta = c.create_folder("a", bucket="t")
    assert meta.is_folder and meta.key == "a/"
    assert seen == [("POST", "/v1/files/folders", "", {"key": "a", "bucket": "t"})]

    async def run() -> None:
        ac = AsyncFilesClient(BASE, transport=httpx.MockTransport(_rec(seen, 201, _FOLDER)))
        await ac.create_folder("b")

    asyncio.run(run())
    assert seen[1] == ("POST", "/v1/files/folders", "", {"key": "b"})
