"""
Lookup tables — CSV enrichment at query time via `/v1/lookup/*` (#967 / #79).

Lookup tables are small, read-only dictionaries the server joins into queries
at plan time. Classic use case: enriching a country code or currency code into
a human-readable label without denormalising it into every row.

Flow:
    1. `register_lookup`  — register a CSV that already exists ON THE SERVER under
                            `<data dir>/lookups` (key_column + value_columns)
    2. `list_lookups`     — confirm it landed
    3. `invoke_lookup`    — read back one row by key (`GET /v1/lookup/{name}/{key}`)

Run:
    # country_alpha3.csv: alpha2,alpha3,name / PK,PAK,Pakistan / IN,IND,India ...
    RELATA_LOOKUP_PATH=/var/lib/relata/lookups/country_alpha3.csv RELATA_TOKEN=secret \
        python -m examples.lookups
"""

from __future__ import annotations

import os

from relata import RelataClient
from relata.exceptions import RelataError
from relata.identity import IdentityClient

# The CSV must already exist on the server, under <data dir>/lookups.
LOOKUP_PATH = os.getenv("RELATA_LOOKUP_PATH", "/var/lib/relata/lookups/country_alpha3.csv")


def main() -> None:
    url = os.getenv("RELATA_URL", "http://localhost:9090")
    token = os.getenv("RELATA_TOKEN")

    with RelataClient(url, bearer_token=token, purpose="analytics") as relata:
        identity = IdentityClient.from_client(relata)

        # ── 1. Register the country_alpha3 lookup ──────────────────────────
        print("=== 1. register_lookup: country_alpha3 ===")
        registered = identity.register_lookup(
            "country_alpha3",
            LOOKUP_PATH,
            key_column="alpha2",
            value_columns=["alpha3", "name"],
        )
        print(f"  Registered: {registered}")

        # ── 2. List all registered lookup tables ───────────────────────────
        print("\n=== 2. list_lookups ===")
        lookups = identity.list_lookups()
        print(f"  {len(lookups)} lookup(s) registered:")
        for entry in lookups[:10]:
            print(f"    - {entry.get('name')}")

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
