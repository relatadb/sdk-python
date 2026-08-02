"""Hybrid BM25 + vector search — the flagship retrieval path (#88, ADR-175).

Emits ``HYBRID_SEARCH FROM <type> QUERY '<text>' LIMIT <k>`` fused via
reciprocal rank fusion (graph + BM25 + vector). The server embeds the query
text server-side; caller-supplied embeddings are not accepted by the
``/query`` SQL surface (there is no vector-literal grammar).

Run a local server first:

    cargo run -p relata-cli -- serve

then:

    python examples/hybrid_search_quickstart.py
"""

from relata import RelataClient, VectorClient

with RelataClient("http://localhost:9090", purpose="analytics") as client:
    vectors = VectorClient.from_client(client)

    # 1. Plain hybrid search.
    for hit in vectors.hybrid_search("Document", query_text="graph database", k=5):
        print(f"  {hit.get('score')}  {hit.get('id')}")

    # 2. Tuned: re-rank via the sidecar cross-encoder (#611), cosine vector
    #    channel (#1330), and per-query fusion weights [graph, bm25, vector]
    #    (#1338).
    tuned = vectors.hybrid_search(
        "Document",
        query_text="graph database",
        k=5,
        rerank=True,
        metric="cosine",
        weights=[0.3, 0.5, 0.2],
    )
    print(f"tuned hits: {len(tuned)}")
