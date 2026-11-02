"""``WITH PROVENANCE`` records returned by the server must reach the caller."""

from relata.models import QueryResult

WIRE = {
    "rows": 1,
    "columns": ["v"],
    "data": [{"v": "two"}],
    "query_id": "q",
    "provenance": [
        {"source": "ingest", "method": "initial-ingest", "confidence": 1.0,
         "recorded_at": 1793659927942725000, "derived_from": None}
    ],
}


def test_provenance_is_surfaced() -> None:
    r = QueryResult.model_validate(dict(WIRE))
    assert r.rows == [{"v": "two"}]
    assert r.provenance is not None and r.provenance[0]["source"] == "ingest"


def test_provenance_absent_is_none() -> None:
    wire = {k: v for k, v in WIRE.items() if k != "provenance"}
    assert QueryResult.model_validate(wire).provenance is None
