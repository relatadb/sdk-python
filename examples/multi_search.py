"""
Multi-search example — federated multi-query search in one round-trip (#967).

`POST /multi-search` lets you query N types (or N query strings) in a single
HTTP call. The server runs each query in parallel and returns a combined
result envelope with `processing_time_ms` for the whole batch.

This is the right tool when an agent needs hits across multiple entity types
(e.g. "find this phone number in Person, Phone, and CallRecord") without
fanning out N round-trips.

Run:
    RELATA_TOKEN=secret NEEDLE=alice python -m examples.multi_search
"""

from __future__ import annotations

import os

from relata import RelataClient


def main() -> None:
    url = os.getenv("RELATA_URL", "http://localhost:9090")
    token = os.getenv("RELATA_TOKEN")
    needle = os.getenv("NEEDLE", "alice")

    with RelataClient(url, bearer_token=token, purpose="analytics") as relata:
        # ── 1. Fan out across 3 types in one call ──────────────────────────
        print(f"=== 1. Multi-search across 3 types for '{needle}' ===")
        resp = relata.multi_search(
            [
                {"query": needle, "type": "Person", "limit": 5},
                {"query": needle, "type": "EmailAddress", "limit": 5},
                {"query": needle, "type": "PhoneCall", "limit": 5},
            ]
        )
        print_multi(resp)

        # ── 2. Same type, different query strings ──────────────────────────
        print("\n=== 2. Multi-search: same type, different queries ===")
        resp = relata.multi_search(
            [
                {"query": "alice", "type": "Person", "limit": 3},
                {"query": "bob", "type": "Person", "limit": 3},
                {"query": "carol", "type": "Person", "limit": 3},
            ]
        )
        print_multi(resp)

        # ── 3. With matching_strategy + typo_tolerance ─────────────────────
        print("\n=== 3. With matching_strategy='all' + typo_tolerance ===")
        resp = relata.multi_search(
            [
                {
                    "query": needle,
                    "type": "Person",
                    "limit": 5,
                    "matching_strategy": "all",
                    "typo_tolerance": {"enabled": True, "min_token_size": 4},
                },
            ]
        )
        print_multi(resp)


def print_multi(resp: dict[str, object]) -> None:
    results = resp.get("results", []) if isinstance(resp, dict) else []
    elapsed = resp.get("processing_time_ms", 0) if isinstance(resp, dict) else 0
    print(f"  {len(results)} sub-query result set(s) in {elapsed} ms")
    for i, sub in enumerate(results):
        if isinstance(sub, dict):
            hits = len(sub.get("hits", []))
            total = sub.get("estimatedTotalHits", sub.get("total", 0))
            print(f"    [{i}] hits returned={hits}, estimated total={total}")


if __name__ == "__main__":
    main()
