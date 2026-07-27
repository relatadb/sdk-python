"""
Tenant admin — multi-tenant lifecycle (#967).

Demonstrates the typed `TenantAdminClient`: whoami + usage, CRUD on tenants,
tier management, suspend / reactivate, per-tenant quota, sharing agreements,
and platform-wide usage / license.

The flow requires an admin-scoped bearer token (the same token used for normal
queries will return 403 on the platform-level endpoints). Run against a server
started with `RELATA_TENANCY_MODE=multi` and an admin token.

Run:
    RELATA_TOKEN=<admin-token> python -m examples.tenant_admin
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
        admin = TenantAdminClient.from_client(relata)

        # ── 1. Whoami + usage for the calling tenant ──────────────────────
        print("=== 1. me() + usage() ===")
        try:
            me = admin.me()
            print(f"  tenant_id={me.get('tenant_id')}  tier={me.get('tier')}")
            usage = admin.usage()
            print(f"  usage: {usage}")
        except RelataError as exc:
            print(f"  admin endpoints require admin token — {exc}")
            return

        # ── 2. Create a new tenant ────────────────────────────────────────
        new_id = f"tenant-{os.urandom(4).hex()}"
        print(f"\n=== 2. create('{new_id}') ===")
        try:
            created = admin.create(new_id, tier="free")
            print(f"  {created}")
        except RelataError as exc:
            print(f"  create failed (admin permission?) — {exc}")
            new_id = None

        # ── 3. Set tier + quota ───────────────────────────────────────────
        if new_id:
            print(f"\n=== 3. set_tier('{new_id}', 'server') + set_quota(1_000_000) ===")
            try:
                print(f"  set_tier: {admin.set_tier(new_id, 'server')}")
                print(f"  set_quota: {admin.set_quota(new_id, 1_000_000)}")
            except RelataError as exc:
                print(f"  failed — {exc}")

            # ── 4. Suspend + reactivate ───────────────────────────────────
            print(f"\n=== 4. suspend('{new_id}') → reactivate('{new_id}') ===")
            try:
                print(f"  suspend: {admin.suspend(new_id)}")
                print(f"  reactivate: {admin.reactivate(new_id)}")
            except RelataError as exc:
                print(f"  failed — {exc}")

            # ── 5. Per-tenant usage + sharing agreements ──────────────────
            print(f"\n=== 5. tenant_usage('{new_id}') + list_sharing ===")
            try:
                print(f"  usage: {admin.tenant_usage(new_id)}")
                print(f"  sharing: {admin.list_sharing(new_id)}")
            except RelataError as exc:
                print(f"  failed — {exc}")

        # ── 6. Platform-wide usage + license ──────────────────────────────
        print("\n=== 6. platform_usage() + platform_license() ===")
        try:
            print(f"  platform usage: {admin.platform_usage()}")
        except RelataError as exc:
            print(f"  failed — {exc}")


if __name__ == "__main__":
    main()
