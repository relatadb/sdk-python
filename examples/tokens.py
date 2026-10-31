"""
Tenant self-service tokens (``/v1/auth/tokens/self``).

A tenant authenticates with its current bearer token and can, without any admin
involvement, list its own tokens, read their lifecycle audit trail, rotate the
token in use, and revoke one of its own tokens.

Flow:
    1. `list_self()`         -> the caller's own tenant-scoped tokens
    2. `self_audit()`        -> audit trail of the caller's token lifecycle
    3. `rotate_self()`       -> fresh secret (the old token stops working)
    4. `revoke_self(id)`     -> revoke one of the caller's own tokens

Minting tokens is an admin-listener operation (`/admin/tokens`) and is not part
of this client.

Run:
    RELATA_TOKEN=<registry-token> python -m examples.tokens
"""

from __future__ import annotations

import os

from relata import RelataClient
from relata.exceptions import RelataError
from relata.tokens import TokenClient


def main() -> None:
    url = os.getenv("RELATA_URL", "http://localhost:9090")
    token = os.getenv("RELATA_TOKEN")

    with RelataClient(url, bearer_token=token, purpose="admin") as relata:
        tokens = TokenClient.from_client(relata)

        print("=== 1. list_self() ===")
        print(f"  {tokens.list_self()}")

        print("\n=== 2. self_audit() ===")
        print(f"  {tokens.self_audit()}")

        # Rotating invalidates the token this client is using, so it is opt-in.
        if os.getenv("RELATA_ROTATE") == "1":
            print("\n=== 3. rotate_self() ===")
            try:
                print(f"  {tokens.rotate_self()}")
            except RelataError as exc:
                print(f"  (rotate requires a per-tenant registry token — {exc})")

        revoke_id = os.getenv("RELATA_REVOKE_TOKEN_ID")
        if revoke_id:
            print(f"\n=== 4. revoke_self('{revoke_id}') ===")
            print(f"  {tokens.revoke_self(revoke_id)}")


if __name__ == "__main__":
    main()
