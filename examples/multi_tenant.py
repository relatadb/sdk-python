"""
Multi-tenant — write as tenant A, verify tenant B cannot read the data.

In Relata every row is organisation-scoped. Queries from a different tenant
never see rows they do not own. This example ingests data as tenant A, then
shows that a query from tenant B (a different bearer token mapped to a
different organisation_id) returns zero rows.

The server must be running with `RELATA_TENANCY_MODE=multi` and two
admin-issued bearer tokens for two different tenants.

Run:
    RELATA_TOKEN_A=<token-a> RELATA_TOKEN_B=<token-b> python -m examples.multi_tenant
"""

from __future__ import annotations

import os

from relata import RelataClient
from relata.ingest import IngestClient


def main() -> None:
    url = os.getenv("RELATA_URL", "http://localhost:9090")
    token_a = os.getenv("RELATA_TOKEN_A", "token-a")
    token_b = os.getenv("RELATA_TOKEN_B", "token-b")

    print(f"Tenant A token : {token_a}")
    print(f"Tenant B token : {token_b}")

    # ── 1. Tenant A: ingest a secret record ───────────────────────────────
    print("\n=== Tenant A: ingest ===")
    with RelataClient(url, bearer_token=token_a, purpose="data-load") as client_a:
        ingest = IngestClient.from_client(client_a)
        receipt = ingest.bulk(
            "SecretRecord",
            [{"_pk": "s1", "secret": "tenant-a-only", "value": 42}],
        )
        print(f"  Tenant A receipt: {receipt}")

        # ── 2. Tenant A reads back its own data ───────────────────────────
        print("\n=== Tenant A: query (should see ≥1 row) ===")
        result = client_a.query("SELECT secret FROM SecretRecord LIMIT 10")
        print(f"  Tenant A sees {len(list(result))} row(s)")

    # ── 3. Tenant B queries the same type (should see 0 rows) ─────────────
    print("\n=== Tenant B: query (should see 0 rows — org isolation) ===")
    with RelataClient(url, bearer_token=token_b, purpose="analytics") as client_b:
        result = client_b.query("SELECT secret FROM SecretRecord LIMIT 10")
        print(f"  Tenant B sees {len(list(result))} row(s)")


if __name__ == "__main__":
    main()
