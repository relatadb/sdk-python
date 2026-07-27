"""
GraphQL example — governed `/graphql` door (ADR-220).

Demonstrates the four supported query shapes plus error handling:
1. Schema introspection (`{ __schema { types { name } } }`)
2. Simple field selection (`{ Person { _pk name email } }`)
3. Filtered + limited selection with `where` + `limit` args
4. Variables bound server-side
5. Error envelope from an unsupported operation (mutation)

The server translates a GraphQL subset to SQL and runs it through the same
governed query path (auth, purpose, ACL, audit) as a normal `POST /query`.
Mutations, subscriptions, fragments, and unions are not supported — the server
returns a non-empty `errors` envelope which the SDK raises as `RelataError`.

Run:
    RELATA_TOKEN=secret python -m examples.graphql
"""

from __future__ import annotations

import json
import os

from relata import RelataClient
from relata.exceptions import RelataError


def main() -> None:
    url = os.getenv("RELATA_URL", "http://localhost:9090")
    token = os.getenv("RELATA_TOKEN")

    with RelataClient(url, bearer_token=token, purpose="analytics") as relata:
        # ── 1. Introspection: list registered types ───────────────────────
        print("=== 1. GraphQL introspection ===")
        schema = relata.graphql("{ __schema { types { name } } }")
        types_ = (schema or {}).get("__schema", {}).get("types", []) if isinstance(schema, dict) else []
        print(f"  {len(types_)} registered type(s):")
        for t in types_[:10]:
            print(f"    - {t.get('name')}")

        # ── 2. Simple field selection over a known type ───────────────────
        print("\n=== 2. Field selection: Person ===")
        persons = relata.graphql("{ Person { _pk name email } }")
        if isinstance(persons, list):
            print(f"  {len(persons)} row(s):")
            for row in persons[:5]:
                print(
                    f"    _pk={row.get('_pk', '?'):<12}  "
                    f"name={row.get('name', '?'):<20}  email={row.get('email', '?')}"
                )
        else:
            print("  (no rows — register a `Person` type and insert a row first)")

        # ── 3. Filtered selection: `limit` + `where` arguments ────────────
        print("\n=== 3. Filtered selection: limit + where ===")
        filtered = relata.graphql(
            '{ Person(limit: 3, where: { _pk: "p1" }) { _pk name } }'
        )
        print(f"  Response: {json.dumps(filtered)[:200]}")

        # ── 4. Variables bound server-side ────────────────────────────────
        print("\n=== 4. Variables: $limit ===")
        with_vars = relata.graphql(
            "query Limited($limit: Int!) { Person(limit: $limit) { _pk name } }",
            variables={"limit": 2},
            operation_name="Limited",
        )
        print(f"  Response: {json.dumps(with_vars)[:200]}")

        # ── 5. Error envelope: mutations are not supported (ADR-220) ──────
        print("\n=== 5. Mutation → expected error envelope ===")
        try:
            relata.graphql('mutation { createPerson(name: "Bob") { id } }')
            print("  (unexpected success — server may have added mutation support)")
        except RelataError as exc:
            print(f"  Got expected error: {exc}")


if __name__ == "__main__":
    main()
