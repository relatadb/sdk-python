"""
Ingest example — batch ingest + CSV ingest via IngestClient.

Demonstrates:
- Bulk NDJSON ingest via `ingest_bulk` with the `Person` type
- CSV ingest via `ingest_csv`
- Verify the rows are queryable back

Run:
    RELATA_TOKEN=secret python -m examples.ingest
"""

from __future__ import annotations

import os

from relata import RelataClient


def main() -> None:
    url = os.getenv("RELATA_URL", "http://localhost:9090")
    token = os.getenv("RELATA_TOKEN")

    with RelataClient(url, bearer_token=token, purpose="data-load") as relata:
        # ── 1. Bulk NDJSON ingest ──────────────────────────────────────────
        print("=== 1. Bulk NDJSON ingest (3 rows) ===")
        batch = [
            {"_pk": "p1", "name": "Alice", "email": "alice@example.com", "age": 30},
            {"_pk": "p2", "name": "Bob", "email": "bob@example.com", "age": 25},
            {"_pk": "p3", "name": "Carol", "email": "carol@example.com", "age": 35},
        ]
        resp = relata.ingest_bulk("Person", batch)
        print(f"  Accepted {resp.accepted} row(s); errors: {resp.errors}")

        # ── 2. CSV ingest ──────────────────────────────────────────────────
        print("\n=== 2. CSV ingest ===")
        csv = "_pk,name,email\np4,Dave,dave@example.com\np5,Eve,eve@example.com\n"
        resp = relata.ingest_csv("Person", csv)
        print(f"  Accepted {resp.accepted} row(s); errors: {resp.errors}")

        # ── 3. Verify queryable ────────────────────────────────────────────
        print("\n=== 3. Verify rows are queryable ===")
        result = relata.query("SELECT name, age FROM Person ORDER BY name LIMIT 10")
        print(f"  {len(list(result))} row(s) returned")


if __name__ == "__main__":
    main()
