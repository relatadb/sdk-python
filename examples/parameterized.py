"""
Parameterized query example — server-side $N binding (#1162).

`$N` placeholders are substituted safely by the server before planning — no
string concatenation, no SQL injection. Use this for any query that takes
caller-supplied values.

In Python, `?` placeholders are auto-rewritten to `$1`, `$2`, … on the way out,
so both forms work.

Run:
    RELATA_TOKEN=secret python -m examples.parameterized
"""

from __future__ import annotations

import os

from relata import RelataClient


def main() -> None:
    url = os.getenv("RELATA_URL", "http://localhost:9090")
    token = os.getenv("RELATA_TOKEN")

    with RelataClient(url, bearer_token=token, purpose="analytics") as relata:
        # ── 1. Filter by a single value ($1) ───────────────────────────────
        print("=== 1. Single-parameter filter (age = $1) ===")
        result = relata.query_params(
            "SELECT name, age FROM Person WHERE age = $1",
            [30],
            purpose="analytics",
        )
        print_rows(result)

        # ── 2. Multiple typed parameters ───────────────────────────────────
        print("\n=== 2. Multi-type parameters (age > $1 AND city = $2) ===")
        result = relata.query_params(
            "SELECT name, age, city FROM Person WHERE age > $1 AND city = $2 ORDER BY name",
            [25, "Karachi"],
            purpose="analytics",
        )
        print_rows(result)

        # ── 3. NULL + bool + float binding ─────────────────────────────────
        print("\n=== 3. NULL / bool / float parameters ===")
        result = relata.query_params(
            "SELECT _pk FROM Sample WHERE flagged = $1 AND score >= $2 "
            "AND note IS NOT DISTINCT FROM $3",
            [True, 0.85, None],
            purpose="analytics",
        )
        print_rows(result)

        # ── 4. IN-list via repeated placeholders ───────────────────────────
        print("\n=== 4. IN-list via repeated placeholders ($1, $2, $3) ===")
        result = relata.query_params(
            "SELECT _pk FROM Person WHERE _pk IN ($1, $2, $3)",
            ["p1", "p2", "p3"],
            purpose="analytics",
        )
        print_rows(result)

        # ── 5. ?-placeholder form (Python auto-rewrites to $N) ─────────────
        print("\n=== 5. ?-placeholder form (auto-rewritten to $N) ===")
        result = relata.query_params(
            "SELECT name FROM Person WHERE age > ? AND city = ?",
            [25, "Karachi"],
            purpose="analytics",
        )
        print_rows(result)


def print_rows(result: object) -> None:
    # QueryResult behaves like an iterable of dict rows.
    rows = list(result)
    if not rows:
        print("  (empty result)")
        return
    print(f"  {len(rows)} row(s):")
    for row in rows[:10]:
        print(f"    {row}")


if __name__ == "__main__":
    main()
