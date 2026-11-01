"""
Streaming example — SSE watch + audit log (#967 Tier 2b).

Two related surfaces:
  1. `StreamingClient.watch(sql)` — `/v1/watch/stream` polls for governed
     query-result notifications. The server emits hashes and explicit
     gap/failure events, not changed row contents. Reconnects use the same
     in-memory per-principal queue; clients must re-register after restart.
  2. `AuditClient.log` — `GET /v1/audit/log` is the tamper-evident audit
     log. Every governed write lands here as an entry.

The example spawns the watch consumer in a background thread, performs a
couple of writes, prints any events received, then reads the audit log to
verify the writes landed.

Run:
    RELATA_TOKEN=secret python -m examples.streaming
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

from relata import RelataClient
from relata.audit import AuditClient
from relata.ingest import IngestClient
from relata.streaming import StreamingClient


def main() -> None:
    url = os.getenv("RELATA_URL", "http://localhost:9090")
    token = os.getenv("RELATA_TOKEN")

    relata = RelataClient(url, bearer_token=token, purpose="analytics")
    streaming = StreamingClient.from_client(relata)
    audit = AuditClient.from_client(relata)
    ingest = IngestClient.from_client(relata)

    # ── 1. Background thread: consume watch events for 5 seconds ──────────
    print("=== 1. Spawn StreamingClient.watch('SELECT * FROM Person') ===")
    events: list[dict[str, Any]] = []
    stop = threading.Event()

    def watcher() -> None:
        try:
            for evt in streaming.watch("SELECT * FROM Person", purpose="analytics"):
                events.append(evt)
                print(f"    [{len(events)}] {evt}")
                if stop.is_set():
                    break
        except Exception as exc:  # noqa: BLE001 — example-wide net
            print(f"  watcher error: {exc}")

    t = threading.Thread(target=watcher, daemon=True)
    t.start()
    time.sleep(0.3)

    # ── 2. Issue writes while the watcher runs ────────────────────────────
    # `/v1/query` is read-only (#782) — writes go through the governed ingest
    # door, not SQL INSERT.
    print("\n=== 2. Issuing writes to trigger watch events ===")
    ingest.bulk("Person", [{"_pk": "watch-1", "name": "Watch Test 1"}])
    print("  inserted Person watch-1")
    time.sleep(0.5)

    stop.set()
    t.join(timeout=2.0)
    print(f"  saw {len(events)} event(s)")

    # ── 3. Alerts stream URL (for an external curl consumer) ──────────────
    print("\n=== 3. Alerts stream URL (for external consumer) ===")
    print(f"  {url}/v1/alerts/stream")

    # ── 4. Audit log: read back ───────────────────────────────────────────
    print("\n=== 4. Audit log (count + recent entries) ===")
    print(f"  count: {audit.count()}")
    recent = audit.log(limit=5)
    print(
        f"  {len(recent.get('entries', []))} recent entr(ies); "
        f"chain_valid={recent.get('chain_valid')}"
    )

    relata.close()


if __name__ == "__main__":
    main()
