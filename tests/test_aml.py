"""Tests for the AML typed-decoder module (#2255)."""

from __future__ import annotations

from relata.aml import (
    decode_beneficial_owners,
    decode_crypto_trace,
    decode_hawala_pairs,
    decode_path_hops,
    decode_resolved_identities,
    decode_sanctions_hits,
    decode_wire_hops,
)


def test_decode_sanctions_hits() -> None:
    rows = [
        {
            "list_name": "OFAC",
            "entry_id": "e1",
            "matched_field": "name",
            "match_score": 0.95,
        },
        {
            "list_name": "EU",
            "entry_id": "e2",
            "matched_field": "aliases",
            "match_score": "0.8",
        },
    ]
    hits = decode_sanctions_hits(rows)
    assert len(hits) == 2
    assert hits[0].list_name == "OFAC"
    assert abs(hits[1].match_score - 0.8) < 1e-9


def test_decode_sanctions_skips_missing() -> None:
    assert decode_sanctions_hits([{"list_name": "OFAC"}]) == []


def test_decode_beneficial_owners() -> None:
    rows = [
        {
            "depth": 1,
            "parent_id": "p",
            "child_id": "c",
            "link_type": "owns",
            "ownership_pct": 51.0,
        }
    ]
    bos = decode_beneficial_owners(rows)
    assert len(bos) == 1
    assert bos[0].ownership_pct == 51.0


def test_decode_beneficial_owners_null_pct() -> None:
    rows = [
        {
            "depth": 1,
            "parent_id": "p",
            "child_id": "c",
            "link_type": "owns",
            "ownership_pct": None,
        }
    ]
    assert decode_beneficial_owners(rows)[0].ownership_pct is None


def test_decode_crypto_trace() -> None:
    rows = [
        {
            "hop": 1,
            "from_wallet": "w1",
            "to_wallet": "w2",
            "amount": 0.5,
            "asset": "BTC",
            "chain": "bitcoin",
        }
    ]
    assert decode_crypto_trace(rows)[0].to_wallet == "w2"


def test_decode_wire_hops() -> None:
    rows = [
        {
            "step": 0,
            "from_bank": "b1",
            "to_bank": "b2",
            "amount": 100,
            "currency": "USD",
            "reference": "r1",
        }
    ]
    assert decode_wire_hops(rows)[0].currency == "USD"


def test_decode_hawala_pairs() -> None:
    rows = [
        {
            "sender": "s",
            "receiver": "r",
            "amount": 200,
            "sender_country": "AE",
            "receiver_country": "IN",
            "ts": "2024-01-01",
        }
    ]
    assert decode_hawala_pairs(rows)[0].receiver_country == "IN"


def test_decode_resolved_identities() -> None:
    rows = [
        {
            "_type": "Canonical",
            "_id": "x",
            "identity_value": "x",
            "source": "relata-detect",
            "observed_at": 123,
            "confidence": 1.0,
        }
    ]
    ids = decode_resolved_identities(rows)
    assert ids[0].id == "x"
    assert ids[0].kind == "Canonical"


def test_decode_path_hops() -> None:
    rows = [{"position": 1, "node": "a", "cost": 0.0}, {"position": 2, "node": "b"}]
    hops = decode_path_hops(rows)
    assert len(hops) == 2
    assert hops[0].cost == 0.0
    assert hops[1].cost is None
