"""
Hybrid search example — BM25 + vector fusion via ``VectorClient.hybrid_search()`` (#2678).

``HYBRID_SEARCH`` is the operator that makes Relata more than "yet another
vector DB" or "yet another BM25 engine": supply a ``query_text`` (BM25 leg),
a ``query_embedding`` (vector leg), or both, and — when both are present —
the server fuses the two rankings via reciprocal rank fusion (ADR-175).

This walkthrough:

1. Embeds a few short texts with the built-in embedder (``POST /embed``,
   #1172) and writes them into a ``Document`` namespace with the embedding
   pre-computed in the ``_emb_text`` slot (the caller-supplied convention,
   see ``docs/src/end-users/search.md``).
2. Runs ``hybrid_search`` three ways — BM25-only, vector-only, and fused —
   so the effect of fusion is visible side by side.

Run:
    RELATA_TOKEN=secret python -m examples.hybrid_search
"""

from __future__ import annotations

import os

from relata import RelataClient
from relata.vectors import VectorClient

DOCS = [
    {
        "id": "doc-1",
        "title": "Knowledge graphs 101",
        "body": "An introduction to entities, edges, and graph retrieval.",
    },
    {
        "id": "doc-2",
        "title": "Vector search at scale",
        "body": "HNSW indexes and approximate nearest neighbour retrieval.",
    },
    {
        "id": "doc-3",
        "title": "Bi-temporal databases",
        "body": "Tracking valid time and system time for auditable history.",
    },
]


def main() -> None:
    url = os.getenv("RELATA_URL", "http://localhost:9090")
    token = os.getenv("RELATA_TOKEN")

    with RelataClient(url, bearer_token=token, purpose="rag") as relata:
        vectors = VectorClient.from_client(relata)

        # ── 1. Embed each doc's body and ingest with the embedding attached ─
        print("=== 1. Embed + ingest 3 Documents (schemaless write) ===")
        rows = []
        for doc in DOCS:
            embedding = vectors.embed(doc["body"])["embedding"]
            rows.append({**doc, "_emb_text": embedding})
        relata.namespace("Document").write(rows)
        print(f"  wrote {len(rows)} rows with pre-computed _emb_text embeddings")

        # ── 2. BM25-only leg (query_text, no embedding) ─────────────────────
        print("\n=== 2. BM25-only: query_text='graph retrieval' ===")
        for row in vectors.hybrid_search("Document", query_text="graph retrieval", k=5):
            print(f"  score={row.get('_score', 0):.4f}  {row.get('title')}")

        # ── 3. Vector-only leg (query_embedding, no text) ───────────────────
        print("\n=== 3. Vector-only: nearest neighbour to doc-2's embedding ===")
        query_embedding = vectors.embed("approximate nearest neighbour search")["embedding"]
        for row in vectors.hybrid_search(
            "Document",
            query_embedding=query_embedding,
            embedding_slot="_emb_text",
            k=5,
        ):
            print(f"  score={row.get('_score', 0):.4f}  {row.get('title')}")

        # ── 4. Fused: both legs — reciprocal rank fusion (ADR-175) ──────────
        print("\n=== 4. Fused: query_text + query_embedding together ===")
        fused_embedding = vectors.embed("graph retrieval with vectors")["embedding"]
        for row in vectors.hybrid_search(
            "Document",
            query_text="graph retrieval",
            query_embedding=fused_embedding,
            embedding_slot="_emb_text",
            k=5,
        ):
            print(f"  fused score={row.get('_score', 0):.4f}  {row.get('title')}")


if __name__ == "__main__":
    main()
