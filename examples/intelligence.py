"""
Intelligence and governed-data example.

Demonstrates generic governed source retrieval plus retained operators:
- OwnershipEdge and TransactionGraph rows are raw inputs for caller-owned analysis
- CONVOY and DNS tunnel triage remain separate retained operations

Errors from governed queries and operators propagate; an error is never
reported as an empty successful result.

Sanctions data retrieval uses governed SELECTs; matching and review belong to
the application. Ownership and transaction queries retrieve source rows only; this example does not
reproduce their retired graph analyses.

Run:
    RELATA_TOKEN=secret TARGET_PARTY="Acme Holdings Ltd" python -m examples.intelligence
"""

from __future__ import annotations

import os

from relata import RelataClient


def main() -> None:
    url = os.getenv("RELATA_URL", "http://localhost:9090")
    token = os.getenv("RELATA_TOKEN")
    target = os.getenv("TARGET_PARTY", "Acme Holdings Ltd")

    with RelataClient(url, bearer_token=token, purpose="investigation") as relata:
        # ── 1. Governed source rows for application-owned sanctions analysis ─
        print("=== 1. Sanctions source data (application matching required) ===")
        people = relata.query_params(
            "SELECT id, name, dob FROM Person WHERE name = $1 LIMIT 100",
            [target],
            purpose="compliance",
        )
        entries = relata.query(
            "SELECT name, aliases, dob, list_name, program FROM SanctionEntry LIMIT 10000",
            purpose="compliance",
        )
        print(f"  fetched {len(people.rows)} Person row(s) and {len(entries.rows)} sanctions reference row(s)")
        print("  apply your approved application-owned matching, list-version, and review policy")
        if people.truncated or entries.truncated or people.next_cursor or entries.next_cursor:
            print("  source results are incomplete; do not treat them as a negative screen")

        # ── 2. Ownership source rows (retrieval only) ────────────────────
        print("\n=== 2. OwnershipEdge source rows (application analysis required) ===")
        ownership = relata.query(
            "SELECT subject, predicate, object, ownership_pct, jurisdiction FROM OwnershipEdge LIMIT 1000",
            purpose="investigation",
        )
        print(f"  fetched {len(ownership.rows)} source row(s)")
        if ownership.truncated or ownership.next_cursor:
            print("  source result is incomplete; do not treat it as a complete ownership chain")
        print("  apply your documented ownership traversal and legal review outside this SDK")

        # ── 3. Transaction source rows (retrieval only) ───────────────────
        print(f"\n=== 3. TransactionGraph source rows for {target} (retrieval only) ===")
        transactions = relata.query(
            "SELECT from_wallet, to_wallet, amount, asset, chain, is_mixer FROM TransactionGraph LIMIT 1000",
            purpose="investigation",
        )
        print(f"  fetched {len(transactions.rows)} source row(s)")
        if transactions.truncated or transactions.next_cursor:
            print("  source result is incomplete; do not treat it as a complete transaction trace")
        print("  apply your approved graph-walk and attribution policy outside this SDK")

        # ── 4. Convoy detection — co-travelling entities ──────────────────
        print("\n=== 4. CONVOY_DETECT: radius=200m, time_tol=300s, min_points=2 ===")
        convoys = relata.convoy_detect(
            radius_m=200.0, time_tol_secs=300, min_points=2, purpose="investigation"
        )
        c_rows = convoys.get("rows", []) if isinstance(convoys, dict) else []
        print(f"  {len(c_rows)} convoy(s) detected")
        for row in c_rows[:5]:
            print(
                f"    convoy_id={row.get('convoy_id', '?')}  "
                f"members={row.get('members')}  first_seen={row.get('first_seen')}"
            )

        # ── 5. DNS tunnel detection ───────────────────────────────────────
        print("\n=== 5. DNS tunnel triage ===")
        dns = relata.dns_tunnel_detect(purpose="investigation")
        d_rows = dns.get("rows", []) if isinstance(dns, dict) else []
        print(f"  {len(d_rows)} anomalous DNS pattern(s); complete={dns.get('complete')} rows_scanned={dns.get('rows_scanned')}")
        for row in d_rows[:5]:
            print(
                f"    domain={row.get('domain', '?'):<30}  entropy={row.get('avg_entropy')}  "
                f"queries={row.get('query_count')} max_label_len={row.get('max_label_len')} "
                f"confidence={row.get('confidence')}"
            )


if __name__ == "__main__":
    main()
