"""
Graph traversal example — PATHS_BETWEEN operator.

Demonstrates:
- Finding paths between two entities in the knowledge graph
- Adjusting ``max_hops`` for depth vs. performance trade-offs
- Filtering by edge type (call records, financial transfers, associate links)
- Combining graph traversal with bi-temporal AS OF
- Network centrality: finding "hub" entities in a relationship network

The ``PATHS_BETWEEN`` operator uses the Pregel BFS engine with the PLL (Pruned
Landmark Labelling) index to enumerate all simple paths up to ``max_hops``
edges between a source and destination entity ID.

Operator signature::

    FROM PATHS_BETWEEN(src_id TEXT, dst_id TEXT, max_hops => INTEGER)

The operator returns a virtual table with columns:
- ``path_id TEXT`` — unique ID for this path result
- ``src_entity TEXT`` — source entity ID
- ``dst_entity TEXT`` — destination entity ID
- ``edge_type TEXT`` — type of the edge (e.g. ``CALLS``, ``FINANCIAL_TRANSFER``)
- ``hop_count INTEGER`` — number of hops in this path
- ``total_path_weight FLOAT`` — sum of edge weights along the path
- ``path_json TEXT`` — JSON array of (entity_id, edge_type) tuples

Performance note: Each additional hop multiplies the search space.  Keep
``max_hops <= 4`` for interactive queries; use background ``Job`` kind for
deeper traversals (ADR-026).
"""

import json
import os

from relata import RelataClient
from relata.exceptions import QuotaError, RelataError


def direct_connection_check(
    relata: RelataClient,
    entity_a: str,
    entity_b: str,
) -> None:
    """Check if two entities are directly connected (max_hops=1).

    Args:
        relata: Bound ``RelataClient`` instance.
        entity_a: Source entity ID.
        entity_b: Destination entity ID.
    """
    print(f"\n=== Direct connection check: {entity_a} <-> {entity_b} ===")

    result = relata.query(
        f"""
        SELECT path_id, edge_type, total_path_weight
        FROM PATHS_BETWEEN('{entity_a}', '{entity_b}', max_hops => 1)
        ORDER BY total_path_weight DESC
        LIMIT 10
        """
    )
    if result:
        print(f"  Directly connected via {result.row_count} edge type(s):")
        for row in result:
            print(f"  Edge: {row.get('edge_type'):30s}  weight: {row.get('total_path_weight')}")
    else:
        print("  No direct connection found.")


def network_traversal(
    relata: RelataClient,
    entity_id: str,
    target_org_id: str,
    max_hops: int = 4,
    edge_types: tuple[str, ...] = (
        "FINANCIAL_TRANSFER",
        "CALLS",
        "ASSOCIATE_OF",
        "DIRECTOR_OF",
        "MEMBER_OF",
        "OWNS",
    ),
) -> None:
    """Find all paths between a source entity and a target organisation.

    Args:
        relata: Bound ``RelataClient`` instance.
        entity_id: Source entity (person) ID.
        target_org_id: Destination entity (organisation) ID.
        max_hops: Maximum path depth (default 4).  Keep <= 4 for interactive use.
        edge_types: Tuple of edge types to include in the traversal.
    """
    print(
        f"\n=== Network traversal: {entity_id} -> {target_org_id}  "
        f"(max_hops={max_hops}) ==="
    )

    edge_list = ", ".join(f"'{e}'" for e in edge_types)

    try:
        result = relata.query(
            f"""
            SELECT
                path_id,
                src_entity,
                dst_entity,
                edge_type,
                hop_count,
                total_path_weight,
                path_json
            FROM PATHS_BETWEEN('{entity_id}', '{target_org_id}', max_hops => {max_hops})
            WHERE edge_type IN ({edge_list})
            ORDER BY hop_count ASC, total_path_weight DESC
            LIMIT 100
            """
        )
    except QuotaError:
        print("  Quota exceeded — reduce max_hops or narrow edge_types filter.")
        return

    print(f"  Paths found: {result.row_count}  ({result.elapsed_ms} ms)")
    print()

    for row in result:
        hops = row.get("hop_count", "?")
        weight = row.get("total_path_weight", "?")
        edge = row.get("edge_type", "?")
        path_json_str = row.get("path_json")

        print(f"  [{hops} hops, weight={weight}]  via {edge}")

        if path_json_str:
            try:
                path_nodes = json.loads(path_json_str)
                segments = [
                    f"{node['entity_id']} --[{node['edge_type']}]-->"
                    for node in path_nodes[:-1]
                ]
                segments.append(path_nodes[-1]["entity_id"])
                print("    " + " ".join(segments))
            except (json.JSONDecodeError, KeyError):
                pass


def temporal_network_shift(
    relata: RelataClient,
    person_id: str,
    org_id: str,
) -> None:
    """Compare the network between two time periods to detect structural change.

    Useful for tracing when a connection was established or severed — e.g.
    whether a financial link existed before or after a particular event.

    Args:
        relata: Bound ``RelataClient`` instance.
        person_id: Source entity (person) ID.
        org_id: Destination entity (organisation) ID.
    """
    print(f"\n=== Temporal network shift: {person_id} -> {org_id} ===")

    for period_label, as_of in [
        ("Before event (2023-12-31)", "2023-12-31T23:59:59Z"),
        ("After event  (2024-06-01)", "2024-06-01T00:00:00Z"),
    ]:
        result = relata.query(
            f"""
            SELECT hop_count, edge_type, total_path_weight
            FROM PATHS_BETWEEN('{person_id}', '{org_id}', max_hops => 3)
            AS OF '{as_of}'
            ORDER BY hop_count ASC
            LIMIT 20
            """
        )
        print(f"  {period_label}: {result.row_count} paths")
        for row in result:
            print(
                f"    [{row.get('hop_count')} hops]  {row.get('edge_type')}  "
                f"weight={row.get('total_path_weight')}"
            )


def hub_detection(
    relata: RelataClient,
    seed_entity: str,
    max_hops: int = 2,
    top_n: int = 10,
) -> None:
    """Find the highest-degree intermediary (hub) entities in the local network.

    Runs PATHS_BETWEEN and counts how many times each intermediate entity
    appears across all paths, surfacing the most-connected nodes.

    Args:
        relata: Bound ``RelataClient`` instance.
        seed_entity: Entity to build the ego graph from.
        max_hops: Maximum depth for the ego-graph (default 2).
        top_n: Number of top hubs to display.
    """
    print(f"\n=== Hub detection: ego graph of {seed_entity!r} (depth={max_hops}) ===")

    result = relata.query(
        f"""
        SELECT
            intermediate_entity_id,
            COUNT(*) AS path_count,
            AVG(total_path_weight) AS avg_weight
        FROM PATHS_BETWEEN('{seed_entity}', '{seed_entity}', max_hops => {max_hops})
        CROSS JOIN UNNEST(path_intermediates) AS t(intermediate_entity_id)
        WHERE intermediate_entity_id != '{seed_entity}'
        GROUP BY intermediate_entity_id
        ORDER BY path_count DESC
        LIMIT {top_n}
        """
    )

    print(f"  Top {top_n} hub entities ({result.elapsed_ms} ms):")
    for rank, row in enumerate(result, start=1):
        entity = row.get("intermediate_entity_id", "?")
        count = row.get("path_count", "?")
        avg_w = row.get("avg_weight", "?")
        print(f"  #{rank:02d}  {entity:40s}  paths={count}  avg_weight={avg_w}")


if __name__ == "__main__":
    RELATA_URL = os.getenv("RELATA_URL", "http://localhost:8080")
    RELATA_TOKEN = os.getenv("RELATA_TOKEN")

    ENTITY_ID = os.getenv("ENTITY_ID", "person-001")
    TARGET_ORG = os.getenv("TARGET_ORG", "org-hawala-001")

    with RelataClient(
        RELATA_URL,
        bearer_token=RELATA_TOKEN,
        purpose="analytics",
        timeout=120.0,
    ) as relata:
        try:
            direct_connection_check(relata, ENTITY_ID, TARGET_ORG)
            network_traversal(relata, ENTITY_ID, TARGET_ORG, max_hops=4)
            temporal_network_shift(relata, ENTITY_ID, TARGET_ORG)
            hub_detection(relata, ENTITY_ID, max_hops=2, top_n=10)
        except RelataError as exc:
            print(f"Error: {exc}")
            raise SystemExit(1) from exc
