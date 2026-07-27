"""
Intelligence operators — chained KYC / sanctions sweep (#967).

Demonstrates the typed intelligence operators on `RelataClient`:
- SANCTIONS_SCREEN             — fuzzy-match a party against sanction lists
- BENEFICIAL_OWNERSHIP_CHAIN   — trace to ultimate beneficial owner (UBO)
- CRYPTO_TRACE                 — follow cryptocurrency fund flow
- CONVOY_DETECT                — find entities travelling together
- DNS_TUNNEL_DETECT            — exfiltration over DNS

Each maps to an intelligence TVF documented in
`docs/src/end-users/sql-grammar.md` (ADRs 109–115). Each call goes through the
governed query path — PURPOSE is audited and ACLs apply.

The example shows a realistic KYC workflow: screen a target, then if there are
hits, run the deeper operators to surface the network around them.

Run:
    RELATA_TOKEN=secret TARGET_PARTY="Acme Holdings Ltd" python -m examples.intelligence
"""

from __future__ import annotations

import os

from relata import RelataClient
from relata.exceptions import RelataError


def main() -> None:
    url = os.getenv("RELATA_URL", "http://localhost:9090")
    token = os.getenv("RELATA_TOKEN")
    target = os.getenv("TARGET_PARTY", "Acme Holdings Ltd")

    with RelataClient(url, bearer_token=token, purpose="investigation") as relata:
        # ── 1. Sanctions screening (fuzzy threshold 0.85) ─────────────────
        print(f"=== 1. SANCTIONS_SCREEN: {target} (threshold=0.85) ===")
        try:
            hits = relata.sanctions_screen(target, threshold=0.85, purpose="investigation")
        except RelataError as exc:
            print(f"  Error: {exc}")
            hits = {"rows": []}
        rows = hits.get("rows", []) if isinstance(hits, dict) else []
        print(f"  {len(rows)} potential hit(s)")
        for row in rows[:5]:
            print(
                f"    list={row.get('list', '?'):<20} score={row.get('score')}  "
                f"match={row.get('matched_name', '?')}"
            )

        # ── 2. Beneficial ownership chain — who really owns it? ───────────
        print(f"\n=== 2. BENEFICIAL_OWNERSHIP_CHAIN: {target} (max_depth=5) ===")
        try:
            chain = relata.beneficial_ownership_chain(target, max_depth=5, purpose="investigation")
        except RelataError as exc:
            print(f"  Error: {exc}")
            chain = {"rows": []}
        layers = chain.get("rows") or chain.get("layers") or []
        print(f"  {len(layers)} layer(s) to UBO:")
        for i, layer in enumerate(layers, start=1):
            share = layer.get("share", 0)
            share_pct = (share * 100.0) if isinstance(share, (int, float)) else 0.0
            print(
                f"    {i}. {layer.get('entity', '?'):<30} owns {share_pct:.2f}%  "
                f"via {layer.get('instrument', '?')}"
            )

        # ── 3. Crypto trace — follow the funds ────────────────────────────
        print(f"\n=== 3. CRYPTO_TRACE: {target} ===")
        try:
            trace = relata.crypto_trace(target, purpose="investigation")
        except RelataError as exc:
            print(f"  Error: {exc}")
            trace = {"rows": []}
        hops = trace.get("rows", []) if isinstance(trace, dict) else []
        print(f"  {len(hops)} hop(s) in fund-flow graph")
        for i, hop in enumerate(hops[:10]):
            print(
                f"    {i:>2}. {hop.get('from', '?')} → {hop.get('to', '?')}  "
                f"amount={hop.get('amount')}"
            )

        # ── 4. Convoy detection — co-travelling entities ──────────────────
        print("\n=== 4. CONVOY_DETECT: radius=200m, time_tol=300s, min_points=2 ===")
        try:
            convoys = relata.convoy_detect(
                radius_m=200.0, time_tol_secs=300, min_points=2, purpose="investigation"
            )
        except RelataError as exc:
            print(f"  Error: {exc}")
            convoys = {"rows": []}
        c_rows = convoys.get("rows", []) if isinstance(convoys, dict) else []
        print(f"  {len(c_rows)} convoy(s) detected")
        for row in c_rows[:5]:
            print(
                f"    convoy_id={row.get('convoy_id', '?')}  "
                f"members={row.get('members')}  first_seen={row.get('first_seen')}"
            )

        # ── 5. DNS tunnel detection ───────────────────────────────────────
        print(f"\n=== 5. DNS_TUNNEL_DETECT: {target} ===")
        try:
            dns = relata.dns_tunnel_detect(target, purpose="investigation")
        except RelataError as exc:
            print(f"  Error: {exc}")
            dns = {"rows": []}
        d_rows = dns.get("rows", []) if isinstance(dns, dict) else []
        print(f"  {len(d_rows)} anomalous DNS pattern(s)")
        for row in d_rows[:5]:
            print(
                f"    domain={row.get('domain', '?'):<30}  entropy={row.get('entropy')}  "
                f"bytes={row.get('bytes_exchanged')}"
            )


if __name__ == "__main__":
    main()
