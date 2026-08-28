"""
Hybrid search example — BM25 + vector fusion via ``VectorClient.hybrid_search()`` (#2678).

``HYBRID_SEARCH`` is the operator that makes Relata more than "yet another
vector DB" or "yet another BM25 engine": supply a ``query_text`` and the
server embeds it and fuses the BM25 + vector rankings via reciprocal rank
fusion (ADR-175, #4491). A caller-supplied query embedding is a *separate*
capability — ``VectorClient.knn_search()``, the pure-vector, no-BM25-leg
path (the ``/query`` SQL surface has no vector-literal grammar, so
``hybrid_search()`` itself only ever accepts ``query_text``).

This walkthrough:

1. Embeds a few short texts with the built-in embedder (``POST /embed``,
   #1172) and writes them into a ``Document`` namespace with the embedding
   pre-computed in the ``_emb_text`` slot (the caller-supplied convention,
   see ``docs/src/end-users/search.md``).
2. Runs three retrieval shapes side by side — BM25-only (``hybrid_search``
   with no ``_emb_text`` involvement), vector-only (``knn_search`` against a
   caller-supplied embedding), and fused BM25 + server-embedded-vector
   (``hybrid_search``, #4491) — so the effect of fusion is visible.

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

        # ── 3. Vector-only leg (caller-supplied embedding, no BM25 text) ────
        # `VectorClient.hybrid_search()` requires `query_text` (the `/query`
        # SQL surface has no vector-literal grammar — see its docstring), so
        # a caller-supplied-embedding, vector-only query goes through
        # `knn_search()` instead, not `hybrid_search()`.
        print("\n=== 3. Vector-only: nearest neighbour to doc-2's embedding ===")
        query_embedding = vectors.embed("approximate nearest neighbour search")["embedding"]
        for row in vectors.knn_search(
            "Document",
            "_emb_text",
            query_embedding,
            k=5,
        ):
            print(f"  score={row.get('_score', 0):.4f}  {row.get('title')}")

        # ── 4. Fused: BM25 + vector — reciprocal rank fusion (ADR-175) ──────
        # `hybrid_search()` fuses BM25 with the vector channel by embedding
        # `query_text` server-side (#4491) — it does not accept a
        # caller-supplied `query_embedding`.
        print("\n=== 4. Fused: query_text='graph retrieval with vectors' ===")
        for row in vectors.hybrid_search(
            "Document",
            query_text="graph retrieval with vectors",
            k=5,
        ):
            print(f"  fused score={row.get('_score', 0):.4f}  {row.get('title')}")


if __name__ == "__main__":
    main()
