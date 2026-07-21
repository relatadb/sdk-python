"""
Full analytics workflow example.

This example walks through a governed enterprise knowledge workflow:

  1. Search for a person by name
  2. Look up the person's full identity profile (phone, email, documents)
  3. Cross-reference against open cases or service records
  4. Trace financial and associate connections via graph traversal
  5. Pull bi-temporal history (what did we know, and when?)
  6. Retrieve the provenance chain for regulated submission
  7. Verify the audit log chain has not been tampered with

Every query declares ``purpose="analytics"`` as required by the
PurposeRegistry.  Access-scoped intercept data queries would additionally
require a ``access_scope_ref`` — see the comment in step 4.
"""

import json
from datetime import datetime, timezone

from relata import RelataClient
from relata.exceptions import AuthError, QuotaError, RelataError


def run_analytics_workflow(
    base_url: str,
    bearer_token: str | None,
    person_name: str,
    access_scope_ref: str | None = None,
) -> None:
    """Execute a full analytics workflow for a named person.

    Args:
        base_url: Relata server URL.
        bearer_token: Authentication token (required in production).
        person_name: Name or partial name of the person.
        access_scope_ref: Active access scope reference for intercept-data queries.
            Pass ``None`` to skip intercept lookups.
    """
    print("=" * 70)
    print(f"ANALYTICS WORKFLOW — person: {person_name!r}")
    print(f"Started at: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    with RelataClient(
        base_url,
        bearer_token=bearer_token,
        purpose="analytics",
        timeout=60.0,
    ) as relata:

        # --------------------------------------------------------------
        # Step 1: Locate persons matching the name pattern.
        # --------------------------------------------------------------
        print("\n[Step 1] Person search")
        persons = (
            relata.select("name", "dob", "nationality", "person_id")
            .from_("Person")
            .where(f"name ILIKE '%{person_name}%'")
            .order_by("confidence DESC")
            .limit(10)
            .execute()
        )
        print(f"  Found {persons.row_count} candidate(s) in {persons.elapsed_ms} ms")
        for p in persons:
            print(
                f"  [{p.get('person_id', '?')}] {p.get('name')}  "
                f"DOB: {p.get('dob')}  Nationality: {p.get('nationality')}"
            )

        if not persons:
            print("  No matches — workflow complete.")
            return

        # Use the top-ranked match for subsequent steps.
        primary = persons.rows[0]
        person_id: str = primary["person_id"]
        print(f"\n  Primary person: {primary.get('name')} ({person_id})")

        # --------------------------------------------------------------
        # Step 2: Identity profile — all canonical identifiers on record.
        # --------------------------------------------------------------
        print("\n[Step 2] Identity profile (IdentityIndex lookup)")
        identity = (
            relata.select("*")
            .lookup_identity(person_id)
            .with_provenance()
            .execute()
        )
        print(f"  Identity records: {identity.row_count}")
        for rec in identity:
            itype = rec.get("identity_type", "—")
            value = rec.get("canonical_value", "—")
            confidence = rec.get("confidence", "—")
            print(f"  {itype:20s}: {value:40s}  confidence={confidence}")

        # --------------------------------------------------------------
        # Step 3: Cross-reference with case/service records.
        # --------------------------------------------------------------
        print("\n[Step 3] Case cross-reference")
        cases = relata.query(
            f"""
            SELECT
                c.case_id,
                c.case_type,
                c.owner_team,
                c.opened_at,
                c.status,
                c.summary
            FROM Case c
            WHERE c.subject_person_id = '{person_id}'
               OR c.subject_name ILIKE '%{person_name}%'
            ORDER BY c.opened_at DESC
            LIMIT 20
            """
        )
        print(f"  Cases on record: {cases.row_count}")
        for case in cases:
            print(
                f"  [{case.get('case_id')}] {case.get('owner_team'):20s}  "
                f"Type: {case.get('case_type')}  "
                f"Status: {case.get('status')}"
            )

        # --------------------------------------------------------------
        # Step 4: Graph traversal — financial and associate network.
        # The PATHS_BETWEEN operator uses the Pregel BFS engine (PLL index)
        # to find connections through the bi-temporal knowledge graph.
        # --------------------------------------------------------------
        print("\n[Step 4] Associate / financial network (graph traversal)")

        # If an access scope is active, add a access_scope_ref comment so the planner
        # allows access to intercept-scope edges.  Without a valid access_scope_ref,
        # the planner rejects the query when it encounters intercept-scope data.
        scope_comment = (
            f"/* access_scope_ref={access_scope_ref} */" if access_scope_ref else ""
        )
        graph_result = relata.query(
            f"""
            {scope_comment}
            SELECT
                path_id,
                src_entity,
                dst_entity,
                edge_type,
                hop_count,
                total_path_weight
            FROM PATHS_BETWEEN('{person_id}', 'org-hawala-001', max_hops => 4)
            WHERE edge_type IN ('FINANCIAL_TRANSFER', 'CONTACTED', 'ASSOCIATE_OF', 'DIRECTOR_OF')
            ORDER BY hop_count ASC, total_path_weight DESC
            LIMIT 50
            """
        )
        print(f"  Paths found: {graph_result.row_count}")
        for path in graph_result:
            print(
                f"  [{path.get('hop_count')} hops] "
                f"{path.get('src_entity')} → {path.get('dst_entity')}  "
                f"via: {path.get('edge_type')}"
            )

        # --------------------------------------------------------------
        # Step 5: Bi-temporal history — what did we know and when?
        # AS OF queries the state of the knowledge graph at a prior point
        # in time, useful for retrospective reconstruction.
        # --------------------------------------------------------------
        print("\n[Step 5] Bi-temporal history")
        history = (
            relata.select("*")
            .from_("Person")
            .where(f"person_id = '{person_id}'")
            .as_of("2024-01-01T00:00:00Z")
            .limit(1)
            .execute()
        )
        print("  Person record as of 2024-01-01:")
        for rec in history:
            print(f"  {json.dumps(rec, default=str, indent=4)}")

        # --------------------------------------------------------------
        # Step 6: Provenance chain for regulated submission.
        # WITH PROVENANCE adds PROV-O columns to every result row,
        # enabling chain-of-custody documentation required for audit.
        # --------------------------------------------------------------
        print("\n[Step 6] Provenance chain for regulated submission")
        provenance = (
            relata.select("*")
            .from_("Person")
            .where(f"person_id = '{person_id}'")
            .with_provenance()
            .limit(1)
            .execute()
        )
        for rec in provenance:
            print(f"  Source      : {rec.get('source_id')}")
            print(f"  Collector   : {rec.get('collector')}")
            print(f"  Method      : {rec.get('method')}")
            print(f"  Confidence  : {rec.get('confidence')}")
            print(f"  Observed at : {rec.get('observed_at')}")
            print(f"  Recorded at : {rec.get('recorded_at')}")

        # --------------------------------------------------------------
        # Step 7: Audit log integrity check.
        # A broken hash chain (chain_valid=False) means potential tampering
        # and must be escalated before the workflow can proceed.
        # --------------------------------------------------------------
        print("\n[Step 7] Audit log integrity")
        audit = relata.audit_count()
        print(f"  Audit entries : {audit.entries}")
        print(f"  Chain valid   : {audit.chain_valid}")
        if audit.is_tampered:
            print("  *** WARNING: AUDIT CHAIN BROKEN — ESCALATE IMMEDIATELY ***")
        else:
            print("  Chain integrity verified — safe to proceed.")

    print("\n" + "=" * 70)
    print("Analytics workflow complete.")
    print("=" * 70)


if __name__ == "__main__":
    # In a real deployment supply the bearer token via an environment variable.
    import os

    run_analytics_workflow(
        base_url=os.getenv("RELATA_URL", "http://localhost:9090"),
        bearer_token=os.getenv("RELATA_TOKEN"),
        person_name="Ahmed",
        access_scope_ref=os.getenv("ACCESS_SCOPE_REF"),  # e.g. "SCOPE-2025-0042"
    )
