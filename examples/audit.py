"""
Audit chain verification example.

Demonstrates:
- Verifying the tamper-evident audit hash chain
- Querying the audit log for specific query IDs
- Checking audit coverage (entries per time window)
- Generating an audit report for a given case or review
- Detecting anomalies in audit patterns (high-frequency queries, off-hours access)

The Relata audit chain is PROV-O-compliant and hash-chained.  Every query,
ingest, schema change, and administrative action is logged.  The chain is
HSM-rooted in production deployments.

Audit integrity rules:
- ``chain_valid=True`` means every entry's hash references the prior entry's
  hash and the chain is unbroken from genesis.
- ``chain_valid=False`` must be treated as a security incident and escalated
  immediately.  Do not continue the workflow until integrity is restored.
"""

import os
import sys
from datetime import datetime, timezone

from relata import RelataClient
from relata.exceptions import AuthError, RelataError


def verify_chain_integrity(relata: RelataClient) -> bool:
    """Verify that the audit hash chain is intact.

    This is the most important audit check.  Always run this before relying
    on audit data for any court or regulatory submission.

    Args:
        relata: Bound ``RelataClient`` instance.

    Returns:
        ``True`` when the chain is valid; ``False`` when tampered.
    """
    print("=== Audit chain integrity check ===")
    audit = relata.audit_count()

    print(f"  Total audit entries : {audit.entries:,}")
    print(f"  Chain valid         : {audit.chain_valid}")

    if audit.is_tampered:
        print()
        print("  !!! CRITICAL ALERT !!!")
        print("  The audit hash chain is BROKEN.  This indicates potential")
        print("  tampering with audit records.  Do NOT submit any audit data")
        print("  derived from this deployment for court or regulatory purposes.")
        print("  Escalate to your security officer immediately.")
        return False

    print("  Chain integrity verified.  Safe to proceed.")
    return True


def audit_summary_by_window(
    relata: RelataClient,
    start: str,
    end: str,
    purpose_filter: str | None = None,
) -> None:
    """Query the audit log for entries within a time window.

    Args:
        relata: Bound ``RelataClient`` instance.
        start: ISO 8601 start of window.
        end: ISO 8601 end of window.
        purpose_filter: Optional purpose to filter (e.g. ``"analytics"``).
    """
    print(f"\n=== Audit summary [{start} -> {end}] ===")

    purpose_clause = ""
    if purpose_filter:
        purpose_clause = f"AND a.purpose = '{purpose_filter}'"

    result = relata.query(
        f"""
        SELECT
            a.purpose,
            COUNT(*) AS entry_count,
            COUNT(DISTINCT a.principal_id) AS distinct_principals,
            MIN(a.recorded_at) AS first_entry,
            MAX(a.recorded_at) AS last_entry,
            AVG(a.elapsed_ms) AS avg_elapsed_ms
        FROM AuditLog a
        WHERE a.recorded_at BETWEEN '{start}' AND '{end}'
        {purpose_clause}
        GROUP BY a.purpose
        ORDER BY entry_count DESC
        """,
        purpose="audit",
    )

    print(f"  Entries in window: (see below)  ({result.elapsed_ms} ms)")
    print()
    print(
        f"  {'Purpose':20s}  {'Entries':>8}  {'Principals':>10}  "
        f"{'Avg ms':>8}"
    )
    print("  " + "-" * 55)
    for row in result:
        print(
            f"  {row.get('purpose', '—'):20s}  "
            f"{row.get('entry_count', 0):>8}  "
            f"{row.get('distinct_principals', 0):>10}  "
            f"{row.get('avg_elapsed_ms', 0):>8.1f}"
        )


def audit_for_query_id(relata: RelataClient, query_id: str) -> None:
    """Retrieve the audit record for a specific query execution.

    Every ``QueryResult`` includes a ``query_id`` that uniquely identifies the
    execution.  Pass it here to retrieve the corresponding audit record for
    chain-of-custody documentation.

    Args:
        relata: Bound ``RelataClient`` instance.
        query_id: The ``query_id`` from a ``QueryResult``.
    """
    print(f"\n=== Audit record for query_id={query_id!r} ===")

    result = relata.query(
        f"""
        SELECT
            a.entry_id,
            a.query_id,
            a.principal_id,
            a.purpose,
            a.sql_hash,
            a.result_row_count,
            a.elapsed_ms,
            a.recorded_at,
            a.chain_hash,
            a.prev_chain_hash
        FROM AuditLog a
        WHERE a.query_id = '{query_id}'
        LIMIT 1
        WITH PROVENANCE
        """,
        purpose="audit",
    )

    if not result:
        print("  No audit record found for this query_id.")
        return

    row = result.rows[0]
    print(f"  Entry ID       : {row.get('entry_id')}")
    print(f"  Principal      : {row.get('principal_id')}")
    print(f"  Purpose        : {row.get('purpose')}")
    print(f"  SQL hash       : {row.get('sql_hash')}")
    print(f"  Result rows    : {row.get('result_row_count')}")
    print(f"  Elapsed ms     : {row.get('elapsed_ms')}")
    print(f"  Recorded at    : {row.get('recorded_at')}")
    print(f"  Chain hash     : {row.get('chain_hash')}")
    print(f"  Prev hash      : {row.get('prev_chain_hash')}")


def audit_for_case(
    relata: RelataClient,
    case_ref: str,
    start: str,
    end: str,
) -> None:
    """Pull all audit entries for a given case or review reference.

    In Relata, case-scoped queries should include the case or review reference
    in their purpose token or as a query comment so audit queries can
    reconstruct the complete trail for regulated submission.

    Args:
        relata: Bound ``RelataClient`` instance.
        case_ref: Case or review reference string (e.g. ``"CASE-2025-0042"``).
        start: ISO 8601 start of the review period.
        end: ISO 8601 end of the review period.
    """
    print(f"\n=== Full audit trail: case={case_ref!r} ===")

    result = relata.query(
        f"""
        SELECT
            a.entry_id,
            a.recorded_at,
            a.principal_id,
            a.purpose,
            a.action_type,
            a.result_row_count,
            a.elapsed_ms,
            a.chain_hash
        FROM AuditLog a
        WHERE a.case_ref = '{case_ref}'
           AND a.recorded_at BETWEEN '{start}' AND '{end}'
        ORDER BY a.recorded_at ASC
        """,
        purpose="audit",
    )

    print(f"  Audit entries for {case_ref}: {result.row_count}")
    print()

    for row in result:
        ts = row.get("recorded_at", "—")
        principal = row.get("principal_id", "—")
        action = row.get("action_type", "query")
        rows_returned = row.get("result_row_count", "—")
        elapsed = row.get("elapsed_ms", "—")
        print(
            f"  [{ts}]  {principal:20s}  {action:12s}  "
            f"rows={rows_returned}  ms={elapsed}"
        )


def detect_anomalies(relata: RelataClient, threshold_queries_per_hour: int = 500) -> None:
    """Detect anomalous query patterns in the audit log.

    Flags:
    - Principals exceeding ``threshold_queries_per_hour`` queries/hour
    - Queries executed outside business hours (before 06:00 or after 22:00 UTC)
    - Bulk data exfiltration patterns (result_row_count > 10,000)

    Args:
        relata: Bound ``RelataClient`` instance.
        threshold_queries_per_hour: Alert threshold for query frequency.
    """
    print(f"\n=== Anomaly detection (threshold={threshold_queries_per_hour} q/h) ===")

    # High-frequency principals
    high_freq = relata.query(
        f"""
        SELECT
            principal_id,
            DATE_TRUNC('hour', recorded_at) AS hour_bucket,
            COUNT(*) AS queries_in_hour
        FROM AuditLog
        WHERE recorded_at > NOW() - INTERVAL '7 days'
        GROUP BY principal_id, DATE_TRUNC('hour', recorded_at)
        HAVING COUNT(*) > {threshold_queries_per_hour}
        ORDER BY queries_in_hour DESC
        LIMIT 20
        """,
        purpose="audit",
    )
    if high_freq:
        print(f"  HIGH-FREQUENCY PRINCIPALS ({high_freq.row_count} flagged):")
        for row in high_freq:
            print(
                f"    [{row.get('hour_bucket')}]  "
                f"{row.get('principal_id'):30s}  "
                f"{row.get('queries_in_hour')} queries"
            )
    else:
        print("  No high-frequency anomalies detected.")

    # Off-hours queries
    off_hours = relata.query(
        """
        SELECT
            principal_id,
            recorded_at,
            purpose,
            result_row_count
        FROM AuditLog
        WHERE recorded_at > NOW() - INTERVAL '7 days'
          AND (EXTRACT(HOUR FROM recorded_at) < 6
               OR EXTRACT(HOUR FROM recorded_at) >= 22)
        ORDER BY recorded_at DESC
        LIMIT 20
        """,
        purpose="audit",
    )
    print(f"\n  Off-hours queries (last 7 days): {off_hours.row_count}")
    for row in off_hours:
        print(
            f"    [{row.get('recorded_at')}]  "
            f"{row.get('principal_id'):30s}  "
            f"purpose={row.get('purpose')}  rows={row.get('result_row_count')}"
        )

    # Bulk exfiltration
    bulk = relata.query(
        """
        SELECT principal_id, recorded_at, result_row_count, purpose
        FROM AuditLog
        WHERE result_row_count > 10000
          AND recorded_at > NOW() - INTERVAL '30 days'
        ORDER BY result_row_count DESC
        LIMIT 10
        """,
        purpose="audit",
    )
    print(f"\n  Bulk result sets (>10k rows, last 30 days): {bulk.row_count}")
    for row in bulk:
        print(
            f"    [{row.get('recorded_at')}]  "
            f"{row.get('principal_id'):30s}  "
            f"{row.get('result_row_count'):,} rows  "
            f"purpose={row.get('purpose')}"
        )


if __name__ == "__main__":
    RELATA_URL = os.getenv("RELATA_URL", "http://localhost:8080")
    RELATA_TOKEN = os.getenv("RELATA_TOKEN")
    CASE_REF = os.getenv("CASE_REF", "CASE-2025-0042")

    now = datetime.now(timezone.utc)
    window_start = "2025-01-01T00:00:00Z"
    window_end = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    with RelataClient(
        RELATA_URL,
        bearer_token=RELATA_TOKEN,
        purpose="audit",
        timeout=60.0,
    ) as relata:
        try:
            # 1. Always verify chain integrity first.
            chain_ok = verify_chain_integrity(relata)
            if not chain_ok:
                print("\nChain integrity failed — aborting audit queries.")
                sys.exit(1)

            # 2. Summary by time window and purpose.
            audit_summary_by_window(
                relata,
                start=window_start,
                end=window_end,
                purpose_filter="analytics",
            )

            # 3. Look up a specific query execution (replace with a real query_id).
            sample_query_id = os.getenv("SAMPLE_QUERY_ID", "qry-abc123")
            audit_for_query_id(relata, sample_query_id)

            # 4. Full audit trail for a case or review.
            audit_for_case(
                relata,
                case_ref=CASE_REF,
                start=window_start,
                end=window_end,
            )

            # 5. Anomaly detection.
            detect_anomalies(relata, threshold_queries_per_hour=200)

        except AuthError as exc:
            print(f"Auth error: {exc}")
            print("Audit endpoints require a valid bearer token.")
            sys.exit(1)
        except RelataError as exc:
            print(f"Error: {exc}")
            sys.exit(1)
