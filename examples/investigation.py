"""
Investigation — `PATHS_BETWEEN` operator for forensic link analysis.

Demonstrates the multi-hop path enumeration operator used in investigation
workflows: given two entity IDs, find all simple paths up to `max_hops` edges
between them in the knowledge graph.

Operator signature:

    FROM PATHS_BETWEEN(src_id TEXT, dst_id TEXT, max_hops => INTEGER)

Returns a virtual table with columns: `path_id`, `src_entity`, `dst_entity`,
`edge_type`, `hop_count`, `total_path_weight`, `path_json`.

Run:
    RELATA_TOKEN=secret FROM_ID=person-001 TO_ID=org-001 \
        python -m examples.investigation
"""

from __future__ import annotations

import json
import os

from relata import RelataClient
from relata.exceptions import QuotaError, RelataError


def main() -> None:
    url = os.getenv("RELATA_URL", "http://localhost:9090")
    token = os.getenv("RELATA_TOKEN")
    src = os.getenv("FROM_ID", "person-001")
    dst = os.getenv("TO_ID", "org-hawala-001")
    max_hops = int(os.getenv("MAX_HOPS", "4"))

    with RelataClient(url, bearer_token=token, purpose="investigation", timeout=120.0) as relata:
        # ── 1. Direct connection check (max_hops=1) ────────────────────────
        print(f"=== Direct connection: {src} <-> {dst} ===")
        r = relata.query(
            f"SELECT path_id, edge_type, total_path_weight "
            f"FROM PATHS_BETWEEN('{src}', '{dst}', max_hops => 1) "
            f"ORDER BY total_path_weight DESC LIMIT 10"
        )
        rows = list(r)
        if rows:
            print(f"  Directly connected via {len(rows)} edge(s):")
            for row in rows:
                print(f"    {row.get('edge_type'):30s}  weight={row.get('total_path_weight')}")
        else:
            print("  No direct connection found.")

        # ── 2. Multi-hop traversal ─────────────────────────────────────────
        print(f"\n=== Multi-hop traversal: {src} -> {dst} (max_hops={max_hops}) ===")
        try:
            r = relata.query(
                f"SELECT path_id, edge_type, hop_count, total_path_weight, path_json "
                f"FROM PATHS_BETWEEN('{src}', '{dst}', max_hops => {max_hops}) "
                f"ORDER BY hop_count ASC, total_path_weight DESC LIMIT 100"
            )
        except QuotaError:
            print("  Quota exceeded — reduce MAX_HOPS or narrow the edge-type filter.")
            return

        rows = list(r)
        print(f"  {len(rows)} path(s) found ({getattr(r, 'elapsed_ms', '?')} ms)")
        for row in rows[:10]:
            hops = row.get("hop_count", "?")
            edge = row.get("edge_type", "?")
            weight = row.get("total_path_weight", "?")
            print(f"    [{hops} hops, weight={weight}]  via {edge}")
            path_json = row.get("path_json")
            if path_json:
                try:
                    nodes = json.loads(path_json) if isinstance(path_json, str) else path_json
                    segments = [f"{n['entity_id']} --[{n['edge_type']}]-->" for n in nodes[:-1]]
                    segments.append(nodes[-1]["entity_id"])
                    print(f"      {' '.join(segments)}")
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass


if __name__ == "__main__":
    main()
