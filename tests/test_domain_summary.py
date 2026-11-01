from __future__ import annotations

import pytest

from relata import summarize_domain


def test_domain_summary_composes_list_entity_types_result() -> None:
    result = summarize_domain(
        "financial",
        {
            "types": [
                {"type": "FinancialTransaction", "rows": 7},
                {"type": "BankAccount", "rows": 2},
                {"type": "Person", "rows": 1000},
            ]
        },
    )
    summary = result["summaries"][0]
    assert result["domain"] == "financial"
    assert summary["types"][:2] == [
        {"type": "BankAccount", "rows": 2},
        {"type": "FinancialTransaction", "rows": 7},
    ]
    assert summary["total_rows"] == 9
    assert len(summary["suggested_queries"]) == 3


def test_domain_summary_supports_all_and_rejects_bad_inputs() -> None:
    assert len(summarize_domain("all", {"types": []})["summaries"]) == 9
    with pytest.raises(ValueError, match="unknown domain"):
        summarize_domain("other", {"types": []})
    with pytest.raises(TypeError, match="object with a types array"):
        summarize_domain("telco", None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="types array"):
        summarize_domain("telco", {})
    with pytest.raises(TypeError, match="types array"):
        summarize_domain("telco", {"types": None})
    with pytest.raises(TypeError, match="nonnegative integer"):
        summarize_domain("telco", {"types": [{"type": "CallRecord", "rows": True}]})
