"""Application-side domain inventory composition over visible type counts.

Pass the result of ``McpClient.list_entity_types()``. This module performs no
network access and does not authorize or query stored data.
"""

from __future__ import annotations

from typing import Any

_PROFILES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "financial": (
        ("BankAccount", "FinancialTransaction", "CryptoWallet", "AmlAlert", "Transaction", "WireTransferHop"),
        (
            "SELECT * FROM FinancialTransaction ORDER BY amount DESC LIMIT 20",
            "SELECT * FROM AmlAlert WHERE risk_score > 70 LIMIT 10",
            "SELECT * FROM CryptoWallet LIMIT 20",
        ),
    ),
    "telco": (
        ("MsisdnRecord", "CallRecord", "CdrRecord", "ImsiRecord", "TowerDump"),
        ("SELECT * FROM CallRecord LIMIT 20", "SELECT * FROM MsisdnRecord LIMIT 20", "SELECT * FROM TowerDump LIMIT 10"),
    ),
    "cyber": (
        ("ThreatIndicator", "MalwareRecord", "DomainRecord", "NetworkEvent", "IpAddress"),
        (
            "SELECT * FROM ThreatIndicator WHERE threat_score > 50 LIMIT 20",
            "SELECT * FROM MalwareRecord LIMIT 10",
            "SELECT * FROM NetworkEvent LIMIT 20",
        ),
    ),
    "humint": (
        ("SourceReport", "DocumentSource", "CaseFile"),
        ("SELECT * FROM DocumentSource LIMIT 10", "SELECT * FROM CaseFile LIMIT 10", "SELECT * FROM SourceReport LIMIT 10"),
    ),
    "narcotics": (("Person", "Organization", "CaseFile", "Transaction", "Vehicle"), ("SELECT * FROM Person LIMIT 20",)),
    "fara": (("Person", "Organization", "Transaction", "SanctionEntry", "PepRecord"), ("SELECT * FROM Person LIMIT 20",)),
    "maritime": (("VesselRecord", "FlightRecord", "Location"), ("SELECT * FROM Person LIMIT 20",)),
    "border": (("PassportRecord", "VisaRecord", "BorderCrossing", "Person"), ("SELECT * FROM Person LIMIT 20",)),
    "sanctions": (
        ("SanctionEntry", "WatchlistEntry", "PepRecord", "Person", "Organization"),
        (
            "SELECT * FROM SanctionEntry LIMIT 20",
            "SELECT * FROM WatchlistEntry WHERE status = 'active' LIMIT 20",
            "SELECT * FROM PepRecord LIMIT 10",
        ),
    ),
}


def summarize_domain(domain: str, inventory: dict[str, Any]) -> dict[str, Any]:
    """Compose the former MCP summary shape from a visible type inventory."""
    if domain != "all" and domain not in _PROFILES:
        valid = ", ".join((*_PROFILES, "all"))
        raise ValueError(f"unknown domain: {domain}. Valid: {valid}")
    if not isinstance(inventory, dict):
        raise TypeError("list_entity_types result must be an object with a types array")

    counts: dict[str, int] = {}
    types = inventory.get("types")
    if not isinstance(types, list):
        raise TypeError("list_entity_types result must contain a types array")
    for item in types:
        if not isinstance(item, dict) or not isinstance(item.get("type"), str):
            raise TypeError("list_entity_types inventory entries must contain a type name")
        rows = item.get("rows")
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
            raise TypeError("list_entity_types inventory rows must be a nonnegative integer")
        counts[item["type"]] = rows

    selected = _PROFILES if domain == "all" else {domain: _PROFILES[domain]}
    summaries = []
    for name, (object_types, queries) in selected.items():
        per_type = [{"type": object_type, "rows": counts.get(object_type, 0)} for object_type in object_types]
        summaries.append(
            {
                "domain": name,
                "total_rows": sum(item["rows"] for item in per_type),
                "types": per_type,
                "suggested_queries": list(queries),
            }
        )
    return {"domain": domain, "summaries": summaries}
