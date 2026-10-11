"""Optional application-side extension recommendations.

Pass the tenant-visible result of ``McpClient.list_entity_types()``. This
helper performs no network access and is not an authorization boundary.
"""

from __future__ import annotations

from typing import Any

_PROFILES: dict[str, tuple[str, ...]] = {
    "telco": ("MsisdnRecord", "CallRecord", "CdrRecord", "ImsiRecord", "TowerDump"),
    "finint": ("BankAccount", "FinancialTransaction", "CryptoWallet", "AmlAlert", "WireTransferHop"),
    "fara": ("Organization", "Transaction", "SanctionEntry", "PepRecord"),
    "cyber": ("ThreatIndicator", "MalwareRecord", "DomainRecord", "NetworkEvent"),
    "maritime": ("VesselRecord", "FlightRecord"),
    "border": ("PassportRecord", "VisaRecord", "BorderCrossing"),
}


def suggest_extensions(inventory: dict[str, Any]) -> dict[str, Any]:
    """Compose the former MCP extension result from caller-visible types.

    ``no_data`` means no profile types appear in this tenant-visible inventory;
    it says nothing about global package installation or other tenants.
    """
    if not isinstance(inventory, dict) or not isinstance(inventory.get("types"), list):
        raise TypeError("list_entity_types result must contain a types array")

    store_types: list[str] = []
    for entry in inventory["types"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("type"), str):
            raise TypeError("list_entity_types entries must contain a type name")
        rows = entry.get("rows")
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
            raise TypeError("list_entity_types rows must be nonnegative integers")
        store_types.append(entry["type"])

    visible = set(store_types)
    extensions = []
    for pack, types in _PROFILES.items():
        active = [name for name in types if name in visible]
        relevance = "no_data" if not active else "full" if len(active) == len(types) else "partial"
        extensions.append(
            {
                "pack": pack,
                "relevance": relevance,
                "active_types": active,
                "all_types": list(types),
            }
        )
    return {"extensions": extensions, "store_types": store_types}
