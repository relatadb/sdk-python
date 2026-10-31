"""
Tenant introspection (``GET /v1/tenants/me``).

Demonstrates the typed `TenantAdminClient`, which in the public SDK exposes only
`me()`: the caller's own tenant record. Tenant provisioning, quotas, sharing
agreements and platform administration are Enterprise features and are not part
of this SDK.

Run:
    RELATA_TOKEN=<token> python -m examples.tenant_admin
"""

from __future__ import annotations

import os

from relata import RelataClient
from relata.exceptions import RelataError
from relata.tenants import TenantAdminClient


def main() -> None:
    url = os.getenv("RELATA_URL", "http://localhost:9090")
    token = os.getenv("RELATA_TOKEN")

    with RelataClient(url, bearer_token=token, purpose="admin") as relata:
        tenants = TenantAdminClient.from_client(relata)

        print("=== me() ===")
        try:
            me = tenants.me()
            print(f"  tenant_id={me.get('tenant_id')}  tier={me.get('tier')}")
        except RelataError as exc:
            print(f"  me() failed — {exc}")


if __name__ == "__main__":
    main()
