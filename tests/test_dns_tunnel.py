from __future__ import annotations

from relata import RelataClient
from relata.dns_tunnel import analyze_dns_tunnel_rows
from relata.models import QueryResult


def test_dns_tunnel_preserves_legacy_fields_thresholds_and_precedence() -> None:
    rows = [
        {"query_name": name, "qname": "ignored.example"}
        for name in (
            "z9q4f8k2m7x1c3v6b0n5h2j8l4p7.exfil.bad.",
            "a1s2d3f4g5h6j7k8l9z0x1c2v3b4.exfil.bad",
            "qwe8rty4uio0pas2dfg6hjk1lzx9.exfil.bad",
            "m4n8b2v6c0x5z3a7s1d9f5g2h6j8.exfil.bad",
            "p3o7i1u5y9t2r6e0w4q8z3x7c1v5.exfil.bad",
            "l9k3j7h1g5f9d3s7a1q5w9e3r7t1.exfil.bad",
        )
    ]
    rows += [{"query_name": "www.cdn.example"} for _ in range(20)]
    rows.append({"query_name": None, "qname": "should-not-fall-through.example"})
    result = analyze_dns_tunnel_rows(rows, min_entropy=3.0, min_queries=5)
    assert len(result) == 1
    assert list(result[0]) == ["domain", "query_count", "avg_entropy", "max_label_len", "confidence"]
    assert result[0]["domain"] == "exfil.bad"
    assert result[0]["query_count"] == 6
    assert result[0]["max_label_len"] == 28


def test_dns_tunnel_client_paginates_and_reports_budget_cursor() -> None:
    client = RelataClient("http://localhost:9090", purpose="security")
    client.query_schema = lambda *, purpose=None: {"purpose": purpose, "object_types": ["DnsLog", "Person"]}  # type: ignore[method-assign]
    calls: list[str] = []

    def query(sql: str, *, purpose: str | None = None) -> QueryResult:
        calls.append(sql)
        if "AFTER" not in sql:
            return QueryResult.model_validate(
                {
                    "rows": 2,
                    "data": [{"query_name": "a.example.com"}, {"query_name": "b.example.com"}],
                    "next_cursor": "100:row-id",
                }
            )
        return QueryResult.model_validate(
            {"rows": 1, "data": [{"query_name": "c.example.com"}], "next_cursor": "200:row-id"}
        )

    client.query = query  # type: ignore[method-assign]
    try:
        result = client.dns_tunnel_detect(min_entropy=0.0, min_queries=1, page_size=2, max_rows=3)
        assert result["complete"] is False
        assert result["incomplete_reason"] == "max_rows_reached"
        assert result["continuation"] == {"DnsLog": "200:row-id", "remaining_types": "DnsLog"}
        assert result["rows_scanned"] == 3
        assert result["types_scanned"] == ["DnsLog"]
        assert len(calls) == 2
        assert all("WITH CACHE BYPASS" in sql for sql in calls)
        assert "AFTER '100:row-id'" in calls[1]
    finally:
        client.close()


def test_dns_tunnel_client_refuses_server_truncation_as_complete() -> None:
    client = RelataClient("http://localhost:9090", purpose="security")
    client.query_schema = lambda *, purpose=None: {"purpose": purpose, "object_types": ["DnsLog"]}  # type: ignore[method-assign]
    client.query = lambda sql, *, purpose=None: QueryResult.model_validate(  # type: ignore[method-assign]
        {"rows": 1, "data": [{"query_name": "a.example.com"}], "truncated": True}
    )
    try:
        result = client.dns_tunnel_detect(page_size=2, max_rows=3)
        assert result["complete"] is False
        assert result["incomplete_reason"] == "server_truncated_page_without_safe_resume_cursor"
        assert result["continuation"] == {"remaining_types": "DnsLog"}
    finally:
        client.close()


def test_dns_tunnel_client_refuses_partial_shard_page_as_complete() -> None:
    client = RelataClient("http://localhost:9090", purpose="security")
    client.query_schema = lambda *, purpose=None: {"purpose": purpose, "object_types": ["DnsLog"]}  # type: ignore[method-assign]
    client.query = lambda sql, *, purpose=None: QueryResult.model_validate(  # type: ignore[method-assign]
        {"rows": 1, "data": [{"query_name": "a.example.com"}], "partial": True}
    )
    try:
        result = client.dns_tunnel_detect(page_size=2, max_rows=3)
        assert result["complete"] is False
        assert result["incomplete_reason"] == "server_returned_partial_shard_result"
    finally:
        client.close()


def test_query_result_records_explicit_exhausted_cursor_metadata() -> None:
    result = QueryResult.model_validate({"rows": 0, "data": [], "next_cursor": None})
    assert result.has_next_cursor_metadata
    assert result.next_cursor is None
