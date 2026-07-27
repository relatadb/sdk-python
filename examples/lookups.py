"""
Lookup tables — CSV enrichment at query time via `/lookup/*` (#967 / #79).

Lookup tables are small, read-only dictionaries the server joins into queries
at plan time. Classic use case: enriching a country code or currency code into
a human-readable label without denormalising it into every row.

Flow:
    1. `register_lookup`  — upload name + CSV (key_column + value_columns)
    2. `list_lookups`     — confirm it landed
    3. `invoke_lookup`    — read back one value by key (same path the query
                            planner uses internally for LOOKUP() JOINs)

Run:
    RELATA_TOKEN=secret python -m examples.lookups
"""

from __future__ import annotations

import os

from relata import RelataClient
from relata.exceptions import RelataError
from relata.identity import IdentityClient


COUNTRY_CSV = """alpha2,alpha3,name
PK,PAK,Pakistan
IN,IND,India
AE,ARE,United Arab Emirates
SA,SAU,Saudi Arabia
"""


def main() -> None:
    url = os.getenv("RELATA_URL", "http://localhost:9090")
    token = os.getenv("RELATA_TOKEN")

    with RelataClient(url, bearer_token=token, purpose="analytics") as relata:
        identity = IdentityClient.from_client(relata)

        # ── 1. Register the country_alpha3 lookup ──────────────────────────
        print("=== 1. register_lookup: country_alpha3 ===")
        registered = identity.register_lookup(
            "country_alpha3",
            COUNTRY_CSV,
            key_column="alpha2",
            value_columns=["alpha3", "name"],
        )
        print(f"  Registered: {registered}")

        # ── 2. List all registered lookup tables ───────────────────────────
        print("\n=== 2. list_lookups ===")
        lookups = identity.list_lookups()
        print(f"  {len(lookups)} lookup(s) registered:")
        for l in lookups[:10]:
            print(f"    - {l.get('name')} (size={l.get('size', l.get('entries', '?'))})")

        # ── 3. Invoke a lookup by key ──────────────────────────────────────
        print("\n=== 3. invoke_lookup: country_alpha3['PK'] ===")
        val = identity.invoke_lookup("country_alpha3", "PK")
        print(f"  PK → {val}")

        # ── 4. Try a missing key ───────────────────────────────────────────
        print("\n=== 4. invoke_lookup: missing key 'ZZ' ===")
        try:
            print(f"  ZZ → {identity.invoke_lookup('country_alpha3', 'ZZ')}")
        except RelataError as exc:
            print(f"  Got expected error: {exc}")

        # ── 5. Use the lookup inside a SQL query (server-side JOIN) ────────
        print("\n=== 5. SQL JOIN via LOOKUP() function ===")
        result = relata.query(
            "SELECT _pk, passport_country, "
            "LOOKUP('country_alpha3', passport_country) AS iso3 "
            "FROM Person LIMIT 5"
        )
        for row in result:
            print(f"    {row}")


if __name__ == "__main__":
    main()
