"""Namespace.delete_all must use the tombstone ingest path, not /v1/ingest/auto (which had no
tombstone semantics: rows stayed live and a stray `{id}` row was appended)."""

from relata._http import HttpTransport
from relata.client import RelataClient
from relata.models import QueryResult


def test_delete_all_posts_ndjson_tombstones(monkeypatch) -> None:
    calls: list = []
    monkeypatch.setattr(HttpTransport, "post_raw",
                        lambda self, path, content, **kw: calls.append((path, content)) or {})
    monkeypatch.setattr(RelataClient, "query",
                        lambda self, sql, **kw: QueryResult(rows=[{"id": "a"}, {"id": "b"}]))
    out = RelataClient("http://x", bearer_token="t", purpose="p").namespace("Docs").delete_all()
    assert out == {"deleted": 2}
    path, body = calls[0]
    assert path.startswith("/v1/ingest?") and "object_type=Docs" in path
    assert body.splitlines() == ['{"id": "a", "_deleted": true}', '{"id": "b", "_deleted": true}']
