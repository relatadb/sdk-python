"""
Streaming example — SSE watch + transparency log (#967 Tier 2b).

Two related real-time surfaces:
  1. `StreamingClient.watch(sql)` — `/watch/stream` pushes `RowsAppended`
     events as commits matching the SQL land. Reconnects with exponential
     backoff. This is what agent frameworks subscribe to for live updates.
  2. `LogClient` — `/log/append|leaves|head` is the tamper-evident audit
     append-log. Every governed write lands here as a leaf.

The example spawns the watch consumer in a background thread, performs a
couple of writes, prints any events received, then queries the transparency
log to verify the writes landed.

Run:
    RELATA_TOKEN=secret python -m examples.streaming
"""

from __future__ import annotations

import os
import threading
import time

from relata import RelataClient
from relata.log import LogClient
from relata.streaming import StreamingClient


def main() -> None:
    url = os.getenv("RELATA_URL", "http://localhost:9090")
    token = os.getenv("RELATA_TOKEN")

    relata = RelataClient(url, bearer_token=token, purpose="analytics")
    streaming = StreamingClient.from_client(relata)
    log = LogClient.from_client(relata)

    # ── 1. Background thread: consume watch events for 5 seconds ──────────
    print("=== 1. Spawn StreamingClient.watch('SELECT * FROM Person') ===")
    events: list[dict] = []
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
    print("\n=== 2. Issuing writes to trigger watch events ===")
    relata.query(
        "INSERT INTO Person (_pk, name) VALUES ('watch-1', 'Watch Test 1')"
    )
    print("  inserted Person watch-1")
    time.sleep(0.5)

    stop.set()
    t.join(timeout=2.0)
    print(f"  saw {len(events)} event(s)")

    # ── 3. Alerts stream URL (for an external curl consumer) ──────────────
    print("\n=== 3. Alerts stream URL (for external consumer) ===")
    print(f"  {url}/alerts/stream")

    # ── 4. Transparency log: append + read back ───────────────────────────
    print("\n=== 4. Transparency log (append + head + leaves) ===")
    leaf = log.append(b'{"action": "example_run", "note": "streaming example"}')
    print(f"  appended leaf (index, size) = {leaf}")

    head = log.head()
    print(f"  log head (tree size) = {head}")

    leaves = log.leaves()
    print(f"  {len(leaves) if hasattr(leaves, '__len__') else '?'} leaf/leaves in log")

    relata.close()


if __name__ == "__main__":
    main()
