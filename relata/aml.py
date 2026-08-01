"""Typed decoders for AML / financial-intelligence result shapes (#2255).

The server's AML operators (``SANCTIONS_SCREEN``,
``BENEFICIAL_OWNERSHIP_CHAIN``, ``CRYPTO_TRACE``, ``WIRE_RECONSTRUCTION``,
``HAWALA_TRACE``, ``RESOLVE_IDENTITY``, and the shortest-path / Dijkstra path
operators) return generic query-result rows (dicts of column → JSON value).
This module declares typed :mod:`dataclasses` for each result shape and
provides a decoder that maps the generic rows into those dataclasses, so
callers can work with strongly-typed objects instead of untyped dicts.

Field names mirror the exact column names emitted by the server operators
(see ``crates/relata-query/src/finint_operators.rs`` and
``investigation_ops.rs``).

Scope
-----
Only the AML / financial-intelligence result shapes are typed here. All other
packs (police, evidence, supply-chain, health, cyber, …) continue to return
generic ``dict`` results — typing them is the remaining scope of #2255.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _get_str(row: dict[str, Any], key: str) -> str | None:
    """Extract a string field; stringify numbers, treat ``None``/missing as absent."""
    if key not in row or row[key] is None:
        return None
    v = row[key]
    return v if isinstance(v, str) else str(v)


def _get_float(row: dict[str, Any], key: str) -> float | None:
    if key not in row or row[key] is None:
        return None
    v = row[key]
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v))
    except (TypeError, ValueError):
        return None


def _get_int(row: dict[str, Any], key: str) -> int | None:
    if key not in row or row[key] is None:
        return None
    v = row[key]
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        return int(v)
    try:
        return int(str(v))
    except (TypeError, ValueError):
        return None


# ── SanctionsHit ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SanctionsHit:
    """A single sanctions-list match from ``SANCTIONS_SCREEN``.

    Columns: ``list_name``, ``entry_id``, ``matched_field``, ``match_score``.
    """

    list_name: str
    entry_id: str
    matched_field: str
    match_score: float


def decode_sanctions_hits(rows: list[dict[str, Any]]) -> list[SanctionsHit]:
    """Decode generic query rows into typed :class:`SanctionsHit` objects.

    Rows missing required fields are skipped.
    """
    out: list[SanctionsHit] = []
    for r in rows:
        list_name = _get_str(r, "list_name")
        entry_id = _get_str(r, "entry_id")
        matched_field = _get_str(r, "matched_field")
        if list_name is None or entry_id is None or matched_field is None:
            continue
        out.append(
            SanctionsHit(
                list_name=list_name,
                entry_id=entry_id,
                matched_field=matched_field,
                match_score=_get_float(r, "match_score") or 0.0,
            )
        )
    return out


# ── BeneficialOwner ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BeneficialOwner:
    """One hop in a beneficial-ownership chain.

    Columns: ``depth``, ``parent_id``, ``child_id``, ``link_type``,
    ``ownership_pct``.
    """

    depth: int
    parent_id: str
    child_id: str
    link_type: str
    ownership_pct: float | None = None


def decode_beneficial_owners(rows: list[dict[str, Any]]) -> list[BeneficialOwner]:
    """Decode generic query rows into typed :class:`BeneficialOwner` hops."""
    out: list[BeneficialOwner] = []
    for r in rows:
        depth = _get_int(r, "depth")
        parent_id = _get_str(r, "parent_id")
        child_id = _get_str(r, "child_id")
        link_type = _get_str(r, "link_type")
        if depth is None or parent_id is None or child_id is None or link_type is None:
            continue
        out.append(
            BeneficialOwner(
                depth=depth,
                parent_id=parent_id,
                child_id=child_id,
                link_type=link_type,
                ownership_pct=_get_float(r, "ownership_pct"),
            )
        )
    return out


# ── CryptoTraceHop ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CryptoTraceHop:
    """One hop in a cryptocurrency fund-flow trace from ``CRYPTO_TRACE``.

    Columns: ``hop``, ``from_wallet``, ``to_wallet``, ``amount``, ``asset``,
    ``chain``.
    """

    hop: int
    from_wallet: str
    to_wallet: str
    amount: float
    asset: str
    chain: str


def decode_crypto_trace(rows: list[dict[str, Any]]) -> list[CryptoTraceHop]:
    """Decode generic query rows into typed :class:`CryptoTraceHop` objects."""
    out: list[CryptoTraceHop] = []
    for r in rows:
        hop = _get_int(r, "hop")
        if hop is None:
            continue
        out.append(
            CryptoTraceHop(
                hop=hop,
                from_wallet=_get_str(r, "from_wallet") or "",
                to_wallet=_get_str(r, "to_wallet") or "",
                amount=_get_float(r, "amount") or 0.0,
                asset=_get_str(r, "asset") or "",
                chain=_get_str(r, "chain") or "",
            )
        )
    return out


# ── WireHop ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WireHop:
    """One leg of a reconstructed wire-transfer chain.

    Columns: ``step``, ``from_bank``, ``to_bank``, ``amount``, ``currency``,
    ``reference``.
    """

    step: int
    from_bank: str
    to_bank: str
    amount: float
    currency: str
    reference: str


def decode_wire_hops(rows: list[dict[str, Any]]) -> list[WireHop]:
    """Decode generic query rows into typed :class:`WireHop` objects."""
    out: list[WireHop] = []
    for r in rows:
        step = _get_int(r, "step")
        if step is None:
            continue
        out.append(
            WireHop(
                step=step,
                from_bank=_get_str(r, "from_bank") or "",
                to_bank=_get_str(r, "to_bank") or "",
                amount=_get_float(r, "amount") or 0.0,
                currency=_get_str(r, "currency") or "",
                reference=_get_str(r, "reference") or "",
            )
        )
    return out


# ── HawalaPair ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class HawalaPair:
    """A matched informal-value-transfer pair from ``HAWALA_TRACE``.

    Columns: ``sender``, ``receiver``, ``amount``, ``sender_country``,
    ``receiver_country``, ``ts``.
    """

    sender: str
    receiver: str
    amount: float
    sender_country: str
    receiver_country: str
    ts: str


def decode_hawala_pairs(rows: list[dict[str, Any]]) -> list[HawalaPair]:
    """Decode generic query rows into typed :class:`HawalaPair` objects."""
    out: list[HawalaPair] = []
    for r in rows:
        sender = _get_str(r, "sender")
        if sender is None:
            continue
        out.append(
            HawalaPair(
                sender=sender,
                receiver=_get_str(r, "receiver") or "",
                amount=_get_float(r, "amount") or 0.0,
                sender_country=_get_str(r, "sender_country") or "",
                receiver_country=_get_str(r, "receiver_country") or "",
                ts=_get_str(r, "ts") or "",
            )
        )
    return out


# ── ResolvedIdentity ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ResolvedIdentity:
    """A resolved identity member from ``RESOLVE_IDENTITY``.

    Columns: ``_type``, ``_id``, ``identity_value``, ``source``,
    ``observed_at``, ``confidence``.
    """

    id: str
    kind: str = ""
    identity_value: str = ""
    source: str = ""
    observed_at: int = 0
    confidence: float = 0.0


def decode_resolved_identities(rows: list[dict[str, Any]]) -> list[ResolvedIdentity]:
    """Decode generic query rows into typed :class:`ResolvedIdentity` objects."""
    out: list[ResolvedIdentity] = []
    for r in rows:
        ident = _get_str(r, "_id")
        if ident is None:
            continue
        out.append(
            ResolvedIdentity(
                id=ident,
                kind=_get_str(r, "_type") or "",
                identity_value=_get_str(r, "identity_value") or "",
                source=_get_str(r, "source") or "",
                observed_at=_get_int(r, "observed_at") or 0,
                confidence=_get_float(r, "confidence") or 0.0,
            )
        )
    return out


# ── PathHop ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PathHop:
    """One node on a graph path from ``SHORTEST_PATH`` / ``DIJKSTRA``.

    Columns: ``position``, ``node``, and optionally ``cost``.
    """

    position: int
    node: str
    cost: float | None = None


def decode_path_hops(rows: list[dict[str, Any]]) -> list[PathHop]:
    """Decode generic query rows into typed :class:`PathHop` objects."""
    out: list[PathHop] = []
    for r in rows:
        position = _get_int(r, "position")
        node = _get_str(r, "node")
        if position is None or node is None:
            continue
        out.append(PathHop(position=position, node=node, cost=_get_float(r, "cost")))
    return out


__all__ = [
    "SanctionsHit",
    "BeneficialOwner",
    "CryptoTraceHop",
    "WireHop",
    "HawalaPair",
    "ResolvedIdentity",
    "PathHop",
    "decode_sanctions_hits",
    "decode_beneficial_owners",
    "decode_crypto_trace",
    "decode_wire_hops",
    "decode_hawala_pairs",
    "decode_resolved_identities",
    "decode_path_hops",
]
