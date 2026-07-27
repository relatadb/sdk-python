"""
Dedup-token primitive — replay defence (#84).

Relata ships a server-side dedup / uniqueness token set accessible via
`POST /tokens`, `GET /tokens/:id`, `DELETE /tokens/:id`, `GET /tokens/stats`.
The classic use case is **idempotency for replay-sensitive writes**: an upstream
system that might re-send the same request (Kafka retry, webhook storm, network
blip) attaches a unique token ID. The server atomically test-and-sets the
token; the handler proceeds only when the token was newly inserted.

Flow:
    1. `remember(token_id)` → True on first insert, False on duplicate (atomic)
    2. `check(token_id)`    → presence probe without mutation
    3. `stats()`            → dedup-set size, oldest entry, eviction policy
    4. `revoke(token_id)`   → explicit removal (admin only)

Run:
    RELATA_TOKEN=secret python -m examples.tokens
"""

from __future__ import annotations

import os

from relata import RelataClient
from relata.tokens import TokenClient


def main() -> None:
    url = os.getenv("RELATA_URL", "http://localhost:9090")
    token = os.getenv("RELATA_TOKEN")

    with RelataClient(url, bearer_token=token, purpose="admin") as relata:
        tokens = TokenClient.from_client(relata)

        # ── 1. Atomic test-and-set: the replay-defence primitive ──────────
        token_id = "replay-defence-7QT3"
        print(f"=== 1. remember('{token_id}') ===")
        first = tokens.remember(token_id, ttl_secs=3600)
        print(f"  first call:  newly_inserted={first}  (expected True)")
        second = tokens.remember(token_id, ttl_secs=3600)
        print(f"  second call: newly_inserted={second}  (expected False — duplicate)")

        # ── 2. Non-mutating presence check ────────────────────────────────
        print(f"\n=== 2. check('{token_id}') ===")
        info = tokens.check(token_id)
        print(f"  present={info.get('present')}  inserted_at_ns={info.get('inserted_at_ns')}  ttl={info.get('ttl_secs')}")

        # ── 3. Dedup-set stats ────────────────────────────────────────────
        print("\n=== 3. stats() ===")
        s = tokens.stats()
        print(f"  {s}")

        # ── 4. Explicit revoke (admin only) ───────────────────────────────
        print(f"\n=== 4. revoke('{token_id}') ===")
        try:
            revoked = tokens.revoke(token_id)
            print(f"  revoked={revoked}")
        except Exception as exc:  # noqa: BLE001
            print(f"  (revoke requires admin — {exc})")


if __name__ == "__main__":
    main()
