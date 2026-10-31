"""R2 (/v1 restructure) route-surface tests: the cheap-gap wrappers plus the
investigation / evidence / assertion / ontology wrappers, sync and async.

No live server: every test drives ``httpx.MockTransport``.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

BASE = "http://localhost:9090"


def _recorder(
    status: int = 200, body: Any = None, *, headers: dict[str, str] | None = None
) -> tuple[list[tuple[str, str, str, Any]], httpx.MockTransport]:
    seen: list[tuple[str, str, str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        payload = json.loads(req.content) if req.content else None
        seen.append((req.method, req.url.path, req.url.query.decode(), payload))
        if isinstance(body, (bytes, bytearray)):
            return httpx.Response(status, content=bytes(body), headers=headers)
        return httpx.Response(status, json=body if body is not None else {}, headers=headers)

    return seen, httpx.MockTransport(handler)


def _mcp(result: dict[str, Any]) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(result)}], "isError": False}


# -- media work requeue ------------------------------------------------------


def test_media_work_requeue_posts_to_item() -> None:
    from relata.media_work import MediaWorkClient

    seen, tr = _recorder(200, {"state": "queued"})
    out = MediaWorkClient(BASE, transport=tr).requeue("w/1")
    assert out == {"state": "queued"}
    assert seen == [("POST", "/v1/media/work/w/1/requeue", "", {})]


async def test_async_media_work_requeue_mirrors_sync() -> None:
    from relata.media_work import AsyncMediaWorkClient

    seen, tr = _recorder(200, {"state": "queued"})
    async with AsyncMediaWorkClient(BASE, transport=tr) as c:
        assert (await c.requeue("w1"))["state"] == "queued"
    assert seen == [("POST", "/v1/media/work/w1/requeue", "", {})]


# -- files: HEAD / list / media content --------------------------------------


def test_files_head_returns_headers_view_and_none_on_404() -> None:
    from relata.files import FileHead, FilesClient

    seen, tr = _recorder(200, b"", headers={"etag": '"e1"', "content-type": "text/plain", "content-length": "12"})
    head = FilesClient(BASE, transport=tr).head("a/b.txt", bucket="b1")
    assert head == FileHead(etag='"e1"', content_type="text/plain", size=12)
    assert seen[0][:2] == ("HEAD", "/v1/files/content")
    assert "key=a%2Fb.txt" in seen[0][2] and "bucket=b1" in seen[0][2]

    _, tr404 = _recorder(404, {"title": "Not Found", "status": 404, "detail": "no such file"})
    assert FilesClient(BASE, transport=tr404).head("missing") is None


def test_files_list_sends_prefix_cursor_limit() -> None:
    from relata.files import FilesClient

    seen, tr = _recorder(200, {"files": [], "next_cursor": None})
    out = FilesClient(BASE, transport=tr).list(prefix="logs/", cursor="c1", limit=5)
    assert out == {"files": [], "next_cursor": None}
    assert seen[0][0:2] == ("GET", "/v1/files")
    assert "prefix=logs%2F" in seen[0][2] and "cursor=c1" in seen[0][2] and "limit=5" in seen[0][2]


def test_files_media_content_returns_bytes() -> None:
    from relata.files import FilesClient

    seen, tr = _recorder(200, b"\x00\x01", headers={"content-type": "image/png", "etag": '"m"'})
    content = FilesClient(BASE, transport=tr).media_content("row/1")
    assert content.data == b"\x00\x01"
    assert content.content_type == "image/png"
    assert seen[0][:2] == ("GET", "/v1/media/row/1/content")


async def test_async_files_head_list_media_mirror_sync() -> None:
    from relata.files import AsyncFilesClient

    seen, tr = _recorder(200, {"files": []}, headers={"content-length": "3"})
    async with AsyncFilesClient(BASE, transport=tr) as c:
        assert (await c.list(limit=1)) == {"files": []}
        assert (await c.head("k")).size == 3
        await c.media_content("r1")
    assert [s[:2] for s in seen] == [
        ("GET", "/v1/files"),
        ("HEAD", "/v1/files/content"),
        ("GET", "/v1/media/r1/content"),
    ]


# -- memory: recall stream + procedures --------------------------------------


def test_memory_recall_stream_yields_sse_frames() -> None:
    from relata import Memory

    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        sse = b'data: {"id": "m1"}\n\ndata: {"id": "m2"}\n\ndata: {"type": "done"}\n\n'
        return httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"})

    with Memory(BASE, purpose="agent", transport=httpx.MockTransport(handler)) as m:
        frames = list(m.recall_stream("ui", top_k=2, min_confidence=0.5))
    assert [f.get("id") for f in frames[:2]] == ["m1", "m2"]
    assert seen[0].method == "GET" and seen[0].url.path == "/v1/memory/recall/stream"
    params = dict(seen[0].url.params)
    assert params["q"] == "ui" and params["purpose"] == "agent" and params["min_confidence"] == "0.5"


async def test_async_memory_recall_stream_mirrors_sync() -> None:
    from relata import AsyncMemory

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/v1/memory/recall/stream"
        return httpx.Response(
            200, content=b'data: {"id": "m1"}\n\n', headers={"content-type": "text/event-stream"}
        )

    async with AsyncMemory(BASE, purpose="agent", transport=httpx.MockTransport(handler)) as m:
        frames = [f async for f in m.recall_stream("ui")]
    assert frames == [{"id": "m1"}]


def test_memory_procedures_use_kebab_case_routes() -> None:
    from relata import Memory

    seen, tr = _recorder(200, _mcp({"ok": True}))
    with Memory(BASE, purpose="agent", transport=tr) as m:
        assert m.remember_procedure("a1", "onboard", "do x") == {"ok": True}
        assert m.recall_procedure("a1", name="onboard", all_versions=True, limit=3) == {"ok": True}
    assert seen[0] == (
        "POST",
        "/v1/memory/remember-procedure",
        "",
        {"agent_id": "a1", "name": "onboard", "instruction_text": "do x", "purpose": "agent"},
    )
    assert seen[1][:2] == ("GET", "/v1/memory/recall-procedure")
    q = dict(httpx.QueryParams(seen[1][2]))
    assert q == {
        "agent_id": "a1",
        "purpose": "agent",
        "name": "onboard",
        "all_versions": "true",
        "limit": "3",
    }


async def test_async_memory_procedures_mirror_sync() -> None:
    from relata import AsyncMemory

    seen, tr = _recorder(200, _mcp({"ok": True}))
    async with AsyncMemory(BASE, purpose="agent", transport=tr) as m:
        await m.remember_procedure("a1", "p", "t")
        await m.recall_procedure("a1")
    assert [s[:2] for s in seen] == [
        ("POST", "/v1/memory/remember-procedure"),
        ("GET", "/v1/memory/recall-procedure"),
    ]


# -- GDPR --------------------------------------------------------------------


async def test_async_gdpr_erase_deletion_verify() -> None:
    from relata.governance import AsyncGovernanceClient

    seen, tr = _recorder(200, {"ok": True})
    async with AsyncGovernanceClient(BASE, transport=tr) as g:
        await g.erase("s1", reason="r", tenant_id="t", physical=True)
        await g.deletion_status("op1")
        await g.verify_receipt({"receipt_id": "r1"})
    assert seen == [
        ("POST", "/v1/gdpr/erase", "", {"subject_id": "s1", "reason": "r", "tenant_id": "t", "physical": True}),
        ("GET", "/v1/gdpr/deletions/op1", "", None),
        ("POST", "/v1/gdpr/verify-receipt", "", {"receipt_id": "r1"}),
    ]


# -- investigation / evidence / assertions / ontology ------------------------


def test_alert_evidence_pivot_timeline_routes() -> None:
    from relata.investigation import InvestigationClient

    seen, tr = _recorder(200, {})
    c = InvestigationClient(BASE, transport=tr)
    c.alert_evidence("a1", max_events=5)
    c.alert_timeline("a1")
    c.pivot("a1", field="user", value="bob")
    assert [s[:3] for s in seen] == [
        ("GET", "/v1/alerts/a1/evidence", "max_events=5"),
        ("GET", "/v1/alerts/a1/timeline", ""),
        ("POST", "/v1/alerts/a1/pivot", ""),
    ]
    assert seen[2][3] == {"field": "user", "value": "bob"}


async def test_async_alert_evidence_pivot_timeline_routes() -> None:
    from relata.investigation import AsyncInvestigationClient

    seen, tr = _recorder(200, {})
    async with AsyncInvestigationClient(BASE, transport=tr) as c:
        await c.alert_evidence("a1")
        await c.alert_timeline("a1")
        await c.pivot("a1")
    assert [s[:2] for s in seen] == [
        ("GET", "/v1/alerts/a1/evidence"),
        ("GET", "/v1/alerts/a1/timeline"),
        ("POST", "/v1/alerts/a1/pivot"),
    ]


def test_evidence_replay_route() -> None:
    from relata.evidence import EvidenceClient

    seen, tr = _recorder(200, {"citations": []})
    EvidenceClient(BASE, transport=tr).replay("evd_1", include_content=True)
    assert seen == [("POST", "/v1/evidence/evd_1/replay", "", {"include_content": True})]


async def test_async_evidence_replay_route() -> None:
    from relata.evidence import AsyncEvidenceClient

    seen, tr = _recorder(200, {"citations": []})
    async with AsyncEvidenceClient(BASE, transport=tr) as c:
        await c.replay("evd_1")
    assert seen[0][:2] == ("POST", "/v1/evidence/evd_1/replay")


def test_assertions_evidence_graph_route() -> None:
    from relata.assertions import AssertionsClient

    seen, tr = _recorder(200, {})
    AssertionsClient(BASE, transport=tr).evidence_graph("asr_a", depth=2, limit=10)
    assert seen[0][:2] == ("GET", "/v1/assertions/asr_a/evidence-graph")
    assert dict(httpx.QueryParams(seen[0][2])) == {"depth": "2", "limit": "10"}


async def test_async_assertions_evidence_graph_route() -> None:
    from relata.assertions import AsyncAssertionsClient

    seen, tr = _recorder(200, {})
    async with AsyncAssertionsClient(BASE, transport=tr) as c:
        await c.evidence_graph("asr_a")
    assert seen[0][:2] == ("GET", "/v1/assertions/asr_a/evidence-graph")


def test_ontology_compare_route() -> None:
    from relata.ontology_definitions import OntologyDefinitionsClient

    seen, tr = _recorder(200, {})
    OntologyDefinitionsClient(BASE, transport=tr).compare("crm", from_version=1, to="draft")
    assert seen[0][:2] == ("GET", "/v1/ontology/definitions/crm/compare")
    assert dict(httpx.QueryParams(seen[0][2])) == {"from": "1", "to": "draft"}


async def test_async_ontology_compare_route() -> None:
    from relata.ontology_definitions import AsyncOntologyDefinitionsClient

    seen, tr = _recorder(200, {})
    async with AsyncOntologyDefinitionsClient(BASE, transport=tr) as c:
        await c.compare("crm")
    assert seen[0][:2] == ("GET", "/v1/ontology/definitions/crm/compare")


# -- hard-break removals -----------------------------------------------------


def test_removed_wrappers_are_gone() -> None:
    import relata
    from relata.client import RelataClient
    from relata.namespace import AsyncNamespace, Namespace
    from relata.system import AsyncSystemClient, SystemClient
    from relata.vectors import AsyncVectorClient, VectorClient

    removed_client = (
        "list_modules", "register_webhook", "list_webhooks", "delete_webhook",
        "cluster_topology", "cluster_rebalance", "cluster_drain", "cluster_nodes", "acluster_nodes",
    )
    for name in removed_client:
        assert not hasattr(RelataClient, name), name
    for cls in (SystemClient, AsyncSystemClient):
        assert not hasattr(cls, "list_pipelines") and not hasattr(cls, "define_pipeline")
    for cls in (VectorClient, AsyncVectorClient):
        for name in ("embed_image", "embed_face", "embed_audio", "embed_video"):
            assert not hasattr(cls, name)
        assert hasattr(cls, "embed") and hasattr(cls, "embed_batch")
    for cls in (Namespace, AsyncNamespace):
        assert not hasattr(cls, "branch_from")
    for name in ("LogClient", "AsyncLogClient", "ClusterNode"):
        assert not hasattr(relata, name)
