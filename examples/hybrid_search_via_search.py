"""
Hybrid search via the flagship `client.search()` entry point (#2672).

`client.search()` — the most discoverable search method in the SDK — is
**BM25-only by default**. It only sends `query`, `type`, and the other
non-vector params; the server's `search_handler` builds a plain
`SELECT * FROM {type} WHERE MATCH(*, '{query}')` statement unless the
request body also carries `metric` and/or `weights`
(`crates/relata-cli/src/serve.rs::search_handler`).

To actually get BM25 + vector reciprocal-rank fusion out of `client.search()`,
pass `metric` and/or `weights` — either one alone is enough to switch the
request onto the server's `HYBRID_SEARCH` SQL path instead of plain BM25.

This is distinct from (and simpler than) `VectorClient.hybrid_search()`,
which builds a `HYBRID_SEARCH(...)` SQL statement directly and additionally
supports a raw query embedding vector. Use whichever surface fits: this one
if you're already calling `client.search()` and just want the vector channel
switched on; `VectorClient.hybrid_search()` if you need to pass a raw
embedding rather than free text.

Run:
    RELATA_TOKEN=secret NEEDLE="alice smith" python -m examples.hybrid_search_via_search
"""

from __future__ import annotations

import os

from relata import RelataClient


def main() -> None:
    url = os.getenv("RELATA_URL", "http://localhost:9090")
    token = os.getenv("RELATA_TOKEN")
    needle = os.getenv("NEEDLE", "alice smith")

    with RelataClient(url, bearer_token=token, purpose="analytics") as relata:
        # ── 1. Plain BM25 — the server never touches the vector channel ───
        print("=== 1. client.search() with no metric/weights: BM25-only ===")
        bm25_only = relata.search(needle, "Person", limit=5)
        print(f"  {len(bm25_only.hits)} hit(s), pure BM25/FTS ranking")

        # ── 2. Real hybrid fusion through the SAME method ─────────────────
        # Setting `metric` and/or `weights` is what flips search_handler onto
        # `PURPOSE 'search' HYBRID_SEARCH FROM Person QUERY '...' METRIC ...
        # WEIGHTS ...` instead of the plain MATCH() query (#1330/#1338).
        print("\n=== 2. client.search() WITH metric+weights: real HYBRID_SEARCH ===")
        hybrid = relata.search(
            needle,
            "Person",
            limit=5,
            metric="cosine",
            # [graph, bm25, vector] fusion weights — here we split BM25 and
            # vector evenly and ignore the graph channel.
            weights=[0.0, 0.5, 0.5],
        )
        print(f"  {len(hybrid.hits)} hit(s), fused BM25 + vector (RRF) ranking")
        for hit in hybrid.hits:
            print(f"    score={hit.score:.4f} fields={hit.fields}")


if __name__ == "__main__":
    main()
