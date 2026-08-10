"""Tests for the canonical validation module (#2248 Tier-1).

Consumes a vendored copy of the SHARED test-vectors file (``tests/fixtures/canonical-vectors.json``,
mirrored from ``sdks/shared/canonical-vectors.json``) so all four SDKs
(Rust/Python/TypeScript/Go) exercise identical valid + reject cases. Vendored
rather than read from ``sdks/shared`` directly because this package is mirrored
to a standalone repo (relatadb/sdk-python) via `git subtree split`, which drops
everything outside `sdks/python`. Keep this file in sync with the shared source.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from relata.canonical import (
    CanonicalError,
    normalize_email,
    normalize_iban,
    normalize_imei,
    normalize_ipv4,
    normalize_ipv6,
    normalize_msisdn,
    normalize_phone,
    normalize_vin,
    validate_email,
    validate_iban,
    validate_ipv6,
)

VECTORS_PATH = Path(__file__).resolve().parent / "fixtures" / "canonical-vectors.json"


def _vectors() -> dict:
    return json.loads(VECTORS_PATH.read_text())


NORMALIZERS = {
    "email": normalize_email,
    "phone": normalize_phone,
    "iban": normalize_iban,
    "imei": normalize_imei,
    "vin": normalize_vin,
    "msisdn": normalize_msisdn,
    "ipv4": normalize_ipv4,
    "ipv6": normalize_ipv6,
}


@pytest.mark.parametrize("kind", sorted(NORMALIZERS))
def test_canonical_vectors(kind: str) -> None:
    vec = _vectors()[kind]
    norm = NORMALIZERS[kind]
    for case in vec["valid"]:
        got = norm(case["input"])
        assert got == case["expected"], (
            f"{kind}: normalize({case['input']!r}) "
            f"expected {case['expected']!r}, got {got!r}"
        )
    for bad in vec["reject"]:
        with pytest.raises(CanonicalError):
            norm(bad)


def test_validate_bool_matches_normalize() -> None:
    assert validate_email("a@b.co") is True
    assert validate_email("bad") is False
    assert validate_iban("DE89370400440532013000") is True
    assert validate_iban("GB82WEST12345698765433") is False
    assert validate_ipv6("::1") is True
    assert validate_ipv6("1::2::3") is False
