"""
Governance — PURPOSE declaration, audit trail, type listing.

Every Relata query declares a PURPOSE registered in the tenant's
PurposeRegistry. This example queries under two different purposes and
inspects the audit log.

Run:
    RELATA_TOKEN=secret python -m examples.governance
"""

from __future__ import annotations

import os

from relata import RelataClient
from relata.audit import AuditClient


def main() -> None:
    url = os.getenv("RELATA_URL", "http://localhost:9090")
    token = os.getenv("RELATA_TOKEN")

    with RelataClient(url, bearer_token=token) as relata:
        audit = AuditClient.from_client(relata)

        # ── 1. Query under the 'analytics' purpose ─────────────────────────
        print("=== Query under PURPOSE analytics ===")
        r1 = relata.query("SELECT * FROM Person LIMIT 3", purpose="analytics")
        print(f"  Rows returned: {len(list(r1))}")

        # ── 2. Same query under the 'audit' purpose ────────────────────────
        print("\n=== Query under PURPOSE audit ===")
        r2 = relata.query("SELECT * FROM Person LIMIT 3", purpose="audit")
        print(f"  Rows returned: {len(list(r2))}")

        # ── 3. Audit log ───────────────────────────────────────────────────
        print("\n=== Audit log ===")
        count = audit.count()
        print(f"  Audit entries: {count}")
        entries = audit.entries(limit=5)
        print(f"  Recent entries: {entries}")

        # ── 4. List registered object types ────────────────────────────────
        print("\n=== Registered object types ===")
        types = relata.list_types()
        if isinstance(types, dict):
            types = types.get("types", [])
        for t in (types or [])[:10]:
            name = t.get("name") if isinstance(t, dict) else t
            print(f"    - {name}")


if __name__ == "__main__":
    main()
