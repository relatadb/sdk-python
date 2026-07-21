"""
Basic query example — getting started with the Relata Python SDK.

Demonstrates:
- Creating a RelataClient
- Running a simple SELECT query
- Iterating over QueryResult rows
- Using the fluent QueryBuilder
- Checking server health and status
"""

from relata import RelataClient
from relata.exceptions import ConnectionError, PurposeError, QuotaError


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Create a client with a default purpose.
    # ------------------------------------------------------------------
    # The 'purpose' is mandatory for every Relata query.  Setting it on the
    # client means you don't have to repeat it on every query() call.
    with RelataClient(
        "http://localhost:9090",
        purpose="analytics",
        # bearer_token="your-token-here",  # uncomment if auth is enabled
    ) as relata:

        # ------------------------------------------------------------------
        # 2. Check server health before issuing queries.
        # ------------------------------------------------------------------
        health = relata.health()
        print(f"Server health  : {health.status}")
        print(f"Profile        : {health.profile}")
        print(f"Node ID        : {health.node_id}")
        print()

        # ------------------------------------------------------------------
        # 3. Check your remaining query quota.
        # ------------------------------------------------------------------
        status = relata.status()
        print(f"Role           : {status.role}")
        print(f"Remaining quota: {status.query_quota} cost units")
        print()

        # ------------------------------------------------------------------
        # 4. Run a simple SELECT query.
        # ------------------------------------------------------------------
        print("=== Simple SELECT ===")
        result = relata.query(
            "SELECT * FROM Person WHERE name LIKE 'Ahmed%' LIMIT 10"
        )
        print(f"Returned {result.row_count} rows in {result.elapsed_ms} ms")
        print(f"Query ID: {result.query_id}")
        print()

        # QueryResult is iterable — iterate directly over rows.
        for row in result:
            name = row.get("name", "—")
            dob = row.get("dob", "—")
            nationality = row.get("nationality", "—")
            print(f"  {name:30s}  DOB: {dob}  Nationality: {nationality}")

        print()

        # ------------------------------------------------------------------
        # 5. Use the fluent QueryBuilder for more complex queries.
        # ------------------------------------------------------------------
        print("=== Fluent QueryBuilder ===")
        result = (
            relata.select("Person")
            .where("nationality = 'IN'")
            .where("age > 25")
            .order_by("name ASC")
            .limit(5)
            .execute()
        )
        print(f"Returned {result.row_count} rows")
        for row in result:
            print(f"  {row}")

        print()

        # ------------------------------------------------------------------
        # 6. Per-query purpose override.
        # ------------------------------------------------------------------
        # The client default is 'analytics', but you can override it
        # per-call when you need a different registered purpose.
        print("=== Per-query purpose override ===")
        audit_result = relata.query(
            "SELECT COUNT(*) AS total FROM Person",
            purpose="audit",
        )
        print(f"Total persons (audit query): {audit_result.rows}")


def demonstrate_error_handling() -> None:
    """Show how to handle common SDK exceptions."""

    with RelataClient("http://localhost:9090") as relata:
        # PurposeError — no default purpose and none passed to query()
        try:
            relata.query("SELECT * FROM Person")
        except PurposeError as exc:
            print(f"PurposeError: {exc}")
            # Fix: pass purpose= to query() or set on client

        # QuotaError — quota exhausted
        try:
            relata.query(
                "SELECT * FROM Person",
                purpose="analytics",
            )
        except QuotaError as exc:
            print(f"QuotaError: {exc}")
            # Fix: reduce query scope or request a quota increase

        # ConnectionError — server not running
        try:
            bad_client = RelataClient("http://localhost:9999", purpose="analytics")
            bad_client.health()
        except ConnectionError as exc:
            print(f"ConnectionError: {exc}")


if __name__ == "__main__":
    main()
