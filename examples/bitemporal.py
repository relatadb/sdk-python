"""
Bi-temporal queries — point-in-time travel + provenance (Phase 3, ADR-117).

Relata is bi-temporal: every row carries four timestamps —
  - `valid_from` / `valid_to`       — when the fact was true in the real world
  - `system_from` / `system_to`     — when the database knew about it

Two SQL clauses operate on these:

  * `AS OF '<ts>'`          — point-in-time snapshot. Returns rows whose
                              `[valid_from, valid_to)` interval contains `<ts>`
                              AND whose `[system_from, system_to)` interval
                              also contains `<ts>` (i.e. what did we know was
                              true at that moment).
  * `WITH PROVENANCE`       — adds PROV-O columns to each row: `source_id`,
                              `collector`, `method`, `confidence`,
                              `observed_at`, `recorded_at`.

The clauses combine: `SELECT ... AS OF '2024-06-01' WITH PROVENANCE`.

Run:
    RELATA_TOKEN=secret python -m examples.bitemporal
"""

from __future__ import annotations

import os

from relata import RelataClient


def main() -> None:
    url = os.getenv("RELATA_URL", "http://localhost:9090")
    token = os.getenv("RELATA_TOKEN")

    with RelataClient(url, bearer_token=token, purpose="analytics") as relata:
        # ── 1. Current snapshot (baseline) ────────────────────────────────
        print("=== 1. Current snapshot of Person ===")
        result = relata.query("SELECT _pk, name, valid_from, valid_to FROM Person LIMIT 5")
        for row in result:
            print(f"    {row}")

        # ── 2. AS OF '<ts>' — what was true at a past point in time? ──────
        print("\n=== 2. AS OF '2024-06-01' — point-in-time snapshot ===")
        result = relata.query(
            "SELECT _pk, name FROM Person AS OF '2024-06-01' LIMIT 5"
        )
        for row in result:
            print(f"    {row}")

        # ── 3. WITH PROVENANCE — PROV-O columns ───────────────────────────
        print("\n=== 3. WITH PROVENANCE — provenance columns ===")
        result = relata.query(
            "SELECT _pk, name FROM Person WITH PROVENANCE LIMIT 5"
        )
        for row in result:
            keys = ("source_id", "collector", "method", "confidence", "observed_at", "recorded_at")
            prov = {k: row.get(k) for k in keys if k in row}
            print(f"    _pk={row.get('_pk')}  name={row.get('name')}  prov={prov}")

        # ── 4. Combined: AS OF + WITH PROVENANCE ──────────────────────────
        print("\n=== 4. AS OF + WITH PROVENANCE — full forensic snapshot ===")
        result = relata.query(
            "SELECT _pk, name, source_id FROM Person "
            "AS OF '2024-06-01' WITH PROVENANCE LIMIT 5"
        )
        for row in result:
            print(f"    {row}")

        # ── 5. Temporal shift: compare two points in time ─────────────────
        print("\n=== 5. Temporal shift: who was present at each date? ===")
        for label, ts in [
            ("2024-01-01", "2024-01-01"),
            ("2024-06-01", "2024-06-01"),
            ("2025-01-01", "2025-01-01"),
        ]:
            r = relata.query(f"SELECT COUNT(*) AS n FROM Person AS OF '{ts}'")
            n = list(r)[0].get("n", "?") if list(r) else "?"
            print(f"    {label}: {n} row(s)")

        # ── 6. Fluent builder with `.as_of()` + `.with_provenance()` ──────
        print("\n=== 6. QueryBuilder: relata.select(...).as_of(...).with_provenance() ===")
        result = (
            relata.select("Person")
            .as_of("2024-06-01")
            .with_provenance()
            .limit(5)
            .execute()
        )
        for row in result:
            print(f"    {row}")


if __name__ == "__main__":
    main()
