from __future__ import annotations

import pytest

from relata.extension_suggestions import suggest_extensions


def test_extension_suggestions_use_only_visible_type_inventory() -> None:
    result = suggest_extensions(
        {
            "types": [
                {"type": "CallRecord", "rows": 2},
                {"type": "MsisdnRecord", "rows": 0},
                {"type": "Person", "rows": 100},
            ]
        }
    )
    assert result["store_types"] == ["CallRecord", "MsisdnRecord", "Person"]
    telco = result["extensions"][0]
    assert telco == {
        "pack": "telco",
        "relevance": "partial",
        "active_types": ["MsisdnRecord", "CallRecord"],
        "all_types": ["MsisdnRecord", "CallRecord", "CdrRecord", "ImsiRecord", "TowerDump"],
    }
    assert result["extensions"][1]["relevance"] == "no_data"


def test_extension_suggestions_report_full_visibility_and_validate_input() -> None:
    all_finint = ["BankAccount", "FinancialTransaction", "CryptoWallet", "AmlAlert", "WireTransferHop"]
    result = suggest_extensions({"types": [{"type": name, "rows": 1} for name in all_finint]})
    assert result["extensions"][1]["relevance"] == "full"
    with pytest.raises(TypeError):
        suggest_extensions({})
    with pytest.raises(TypeError):
        suggest_extensions({"types": [{"type": "CallRecord", "rows": -1}]})
