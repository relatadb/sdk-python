"""
Advanced query — filtered SELECT + aggregate + Arrow result.

Demonstrates:
- WHERE + ORDER BY + LIMIT
- COUNT(*) aggregate
- Arrow RecordBatch via `query_arrow()` (zero-copy columnar)

Run:
    RELATA_TOKEN=secret python -m examples.advanced_query
"""

from __future__ import annotations

import os

from relata import RelataClient


def main() -> None:
    url = os.getenv("RELATA_URL", "http://localhost:9090")
    token = os.getenv("RELATA_TOKEN")

    with RelataClient(url, bearer_token=token, purpose="analytics") as relata:
        # ── 1. Filtered SELECT ─────────────────────────────────────────────
        print("=== 1. Filtered SELECT ===")
        result = relata.query(
            "SELECT name, age FROM Person WHERE age > 25 ORDER BY name LIMIT 10"
        )
        rows = list(result)
        print(f"  Rows: {len(rows)}")
        for row in rows:
            print(f"    name={row.get('name')}  age={row.get('age')}")

        # ── 2. Aggregate ───────────────────────────────────────────────────
        print("\n=== 2. COUNT aggregate ===")
        result = relata.query("SELECT COUNT(*) AS total FROM Person")
        rows = list(result)
        if rows:
            print(f"  Total persons: {rows[0].get('total')}")

        # ── 3. Arrow batch result ──────────────────────────────────────────
        print("\n=== 3. Arrow RecordBatch ===")
        try:
            table = relata.query_arrow("SELECT name, age FROM Person LIMIT 5")
            print(f"  Arrow table: {table.num_rows} rows × {table.num_columns} cols")
        except Exception as exc:  # noqa: BLE001
            print(f"  Arrow path needs pyarrow installed — {exc}")


if __name__ == "__main__":
    main()
