"""
Investigation: from ingested events to an archived alert record (#5702).

The SDK version of the curl walk-through in
docs/src/guides/siem-detection-and-investigation.md ("Worked example: from ingest
to export") and docs/src/guides/investigation-archive-workflow.md:

  1. ingest a few fixed AuthEvent and Session rows
  2. create a window rule and wait for its alert
  3. fetch the alert evidence and reproduce the alert from its query
  4. read the ordered timeline
  5. pivot on the alert's user into the other readable records
  6. keep the reproduction query as a saved search and run it
  7. archive the evidence export as a file and check its SHA-256

The token needs read+write on AuthEvent, Session, Alert, SavedSearch and
S3Object. The rule fires on the next evaluation tick
(RELATA_RULE_EVAL_INTERVAL_SECS, default 30), so the script waits up to 90 s.

Run (from sdks/python):
    RELATA_URL=http://localhost:9090 RELATA_TOKEN=<data-plane token> \\
        python -m examples.investigation
"""

from __future__ import annotations

import hashlib
import json
import os
import time

from relata import RelataClient
from relata.files import FilesClient
from relata.governance import GovernanceClient
from relata.ingest import IngestClient
from relata.investigation import InvestigationClient

PURPOSE = "audit"
MIN = 60 * 1_000_000_000
T0 = 1_760_000_100 * 1_000_000_000  # fixed event time on a 5-minute boundary

AUTH_EVENTS = [
    {"_pk": "a1", "ts": T0 + 1 * MIN, "kind": "login_failed", "user": "alice"},
    {"_pk": "a2", "ts": T0 + 2 * MIN, "kind": "login_failed", "user": "alice"},
    {"_pk": "a3", "ts": T0 + 3 * MIN, "kind": "login_failed", "user": "alice"},
    {"_pk": "b1", "ts": T0 + 2 * MIN, "kind": "login_failed", "user": "bob"},
]
SESSIONS = [
    {"_pk": "s1", "ts": T0 + 4 * MIN, "user": "alice", "host": "srv-9"},
    {"_pk": "s2", "ts": T0 + 4 * MIN, "user": "bob", "host": "srv-1"},
]
RULE = {
    "name": "brute force (sdk example)",
    "target_type": "AuthEvent",
    "severity": "high",
    "window": {
        "kind": "threshold",
        "event_type": "login_failed",
        "min_count": 3,
        "window_ns": 5 * MIN,
        "allowed_lateness_ns": 0,
        "version": 1,
        "time_field": "ts",
        "group_field": "user",
        "event_type_field": "kind",
    },
}
SEARCH_NAME = "alice failed logins (sdk example)"


def wait_for_alert(gov: GovernanceClient, rule_id: str, timeout_s: float = 90.0) -> str:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for alert in gov.list_alerts(newest_first=True, limit=1000):
            if alert.get("rule_id") == rule_id:
                return str(alert["alert_id"])
        time.sleep(2)
    raise SystemExit(f"rule {rule_id} raised no alert within {timeout_s:.0f}s")


def main() -> None:
    url = os.getenv("RELATA_URL", "http://localhost:9090")
    token = os.getenv("RELATA_TOKEN")

    with RelataClient(url, bearer_token=token, purpose=PURPOSE, timeout=60.0) as relata:
        ingest = IngestClient.from_client(relata)
        gov = GovernanceClient.from_client(relata)
        inv = InvestigationClient.from_client(relata)
        files = FilesClient.from_client(relata)

        print("1. Ingest fixed events (idempotent on _pk)")
        for object_type, rows in (("AuthEvent", AUTH_EVENTS), ("Session", SESSIONS)):
            ingest.bulk(object_type, rows, on_conflict="upsert", durability="sync")
        print(f"   {len(AUTH_EVENTS)} AuthEvent, {len(SESSIONS)} Session")

        print("2. Window rule and its alert")
        rule_id = str(gov.create_rule(RULE)["rule_id"])
        alert_id = wait_for_alert(gov, rule_id)
        print(f"   rule {rule_id} -> alert {alert_id}")
        gov.update_alert(alert_id, status="acknowledged", note="sdk example triage")

        print("3. Evidence, and the alert reproduced from its query")
        evidence = inv.alert_evidence(alert_id, max_events=1000, deadline_ms=5000)
        sql = evidence["evidence"]["reproduce"]["sql"]
        print(
            f"   completeness={evidence['completeness']['status']} "
            f"events={[e['id'] for e in evidence['events']]}"
        )
        reproduced = relata.query(sql).rows
        print(f"   {sql}\n   -> {len(reproduced)} rows")

        print("4. Timeline")
        timeline = inv.alert_timeline(alert_id)
        for entry in timeline["entries"]:
            label = entry.get("id") or entry.get("to") or ""
            print(f"   {entry['at_ns']}  {entry['kind']:<13} {label}")
        print(f"   pivot entities: {timeline['entities']}")

        print("5. Authorized pivot on the alert's user")
        pivot = inv.pivot(alert_id, target_types=["Session", "AuthEvent"], limit=50)
        for result in pivot["results"]:
            ids = [row["id"] for row in result["rows"]]
            print(f"   {result['object_type']}: {ids} (truncated={result['truncated']})")
        print(f"   completeness={pivot['completeness']['status']}")

        print("6. Saved search")
        existing = [
            s for s in inv.list_saved_searches(q=SEARCH_NAME)["items"] if s["name"] == SEARCH_NAME
        ]
        if existing:
            saved = inv.update_saved_search(existing[0]["id"], SEARCH_NAME, sql, purpose=PURPOSE)
        else:
            saved = inv.create_saved_search(SEARCH_NAME, sql, purpose=PURPOSE)
        run = inv.run_saved_search(saved["id"])
        print(f"   {saved['id']} v{saved['version']} -> {len(run.get('data', []))} rows")

        print("7. Archive the evidence export and check its SHA-256")
        record = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(record).hexdigest()
        key = f"cases/{alert_id}/evidence.json"
        files.put(key, record, content_type="application/json")
        stored = files.get(key)
        if hashlib.sha256(stored.data).hexdigest() != digest:
            raise SystemExit(f"archived {key} does not match its SHA-256")
        print(f"   {key} sha256={digest} (verified on read-back)")
        print(f"   events_sha256={evidence['export']['events_sha256']}")


if __name__ == "__main__":
    main()
