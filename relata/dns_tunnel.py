"""Application-side DNS tunnel triage over governed, paginated query results."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from math import log2
from typing import Any

DNS_TUNNEL_TYPES = ("DnsLog", "DnsQuery", "DnsEvent")
DNS_TUNNEL_COLUMNS = ["domain", "query_count", "avg_entropy", "max_label_len", "confidence"]
DNS_TUNNEL_PAGE_SIZE = 1_000
DNS_TUNNEL_MAX_ROWS = 100_000


def _entropy(value: str) -> float:
    if not value:
        return 0.0
    counts: dict[str, int] = defaultdict(int)
    for char in value:
        counts[char] += 1
    length = len(value)
    return sum(-(count / length) * log2(count / length) for count in counts.values())


def _query_name(row: Mapping[str, Any]) -> str | None:
    for field in ("query_name", "qname", "domain", "name"):
        if field in row:
            value = row[field]
            return value if isinstance(value, str) else None
    return None


def analyze_dns_tunnel_rows(
    rows: Iterable[Mapping[str, Any]], *, min_entropy: float, min_queries: int
) -> list[dict[str, Any]]:
    """Return legacy-compatible finding rows from already-authorized DNS rows."""
    acc: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = _query_name(row)
        if value is None:
            continue
        qname = value.rstrip(".").lower()
        if not qname:
            continue
        labels = [label for label in qname.split(".") if label]
        if not labels:
            continue
        domain = ".".join(labels[-2:]) if len(labels) >= 2 else labels[0]
        sub_labels = labels[:-2] if len(labels) > 2 else []
        item = acc.setdefault(
            domain,
            {"entropy_sum": 0.0, "label_count": 0, "query_count": 0, "max_label_len": 0},
        )
        item["query_count"] += 1
        for label in sub_labels:
            item["entropy_sum"] += _entropy(label)
            item["label_count"] += 1
            item["max_label_len"] = max(item["max_label_len"], len(label.encode("utf-8")))

    result: list[dict[str, Any]] = []
    for domain, item in acc.items():
        if item["query_count"] < min_queries:
            continue
        avg_entropy = (
            item["entropy_sum"] / item["label_count"] if item["label_count"] else 0.0
        )
        if avg_entropy < min_entropy:
            continue
        entropy_score = max(0.0, min(0.6, (avg_entropy - min_entropy) * 0.25))
        len_score = max(0.0, min(0.4, (item["max_label_len"] - 20.0) / 43.0))
        confidence = max(0.0, min(1.0, 0.4 + entropy_score + len_score))
        result.append(
            {
                "domain": domain,
                "query_count": item["query_count"],
                "avg_entropy": avg_entropy,
                "max_label_len": item["max_label_len"],
                "confidence": confidence,
            }
        )
    result.sort(key=lambda row: (-row["confidence"], row["domain"]))
    return result


def dns_tunnel_result(
    rows: Iterable[Mapping[str, Any]],
    *,
    min_entropy: float,
    min_queries: int,
    rows_scanned: int,
    types_scanned: list[str],
    complete: bool,
    continuation: dict[str, str],
    incomplete_reason: str | None = None,
) -> dict[str, Any]:
    """Format the historical five columns plus explicit scan completeness."""
    return {
        "columns": DNS_TUNNEL_COLUMNS.copy(),
        "rows": analyze_dns_tunnel_rows(rows, min_entropy=min_entropy, min_queries=min_queries),
        "rows_scanned": rows_scanned,
        "types_scanned": list(types_scanned),
        "complete": complete,
        "continuation": continuation,
        "incomplete_reason": incomplete_reason,
    }
