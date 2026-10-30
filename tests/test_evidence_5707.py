"""#5707 evidence client: wire shapes, path quoting and header helper."""

from __future__ import annotations

import json

import httpx

from relata.evidence import EvidenceClient, evidence_headers

BASE = "http://localhost:9090"


def test_headers_record_and_bound_retention() -> None:
    assert evidence_headers() == {"X-Relata-Record-Evidence": "true"}
    assert evidence_headers(retention_days=7) == {
        "X-Relata-Record-Evidence": "true",
        "X-Relata-Evidence-Retention-Days": "7",
    }


def test_get_list_and_replay_hit_the_documented_routes() -> None:
    seen: list[tuple[str, str, str, object]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content) if req.content else None
        seen.append((req.method, req.url.raw_path.decode().split("?")[0], req.url.query.decode(), body))
        return httpx.Response(200, json={"items": [], "citations": [{"outcome": "verified"}]})

    c = EvidenceClient(BASE, transport=httpx.MockTransport(handler))
    c.list(limit=5)
    c.get("evd_a/b")
    out = c.replay("evd_1", include_content=True)
    c.replay("evd_1")
    assert out["citations"][0]["outcome"] == "verified"
    assert seen == [
        ("GET", "/evidence", "limit=5", None),
        ("GET", "/evidence/evd_a%2Fb", "", None),
        ("POST", "/evidence/evd_1/replay", "", {"include_content": True}),
        ("POST", "/evidence/evd_1/replay", "", {}),
    ]
