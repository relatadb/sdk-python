"""Tests for the maritime domain-op client methods (#2247).

Verifies the generated SQL shape mirrors the server operator signatures in
``crates/relata-query/src/maritime_operators.rs``. No live server required —
the client's sync transport is replaced with a capturing fake.
"""

from __future__ import annotations

import json
from typing import Any

from relata import RelataClient

BASE = "http://localhost:9090"


class _FakeSync:
    """Captures the (path, body) posted by client methods."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def post(self, path: str, body: dict[str, Any]) -> dict[str, object]:
        self.calls.append((path, body))
        return {"rows": []}


def _client_with_fake() -> tuple[RelataClient, _FakeSync]:
    client = RelataClient(BASE, purpose="analytics")
    fake = _FakeSync()
    # Inject the fake into the lazily-created transport slot (name-mangled).
    client._RelataClient__sync_transport = fake  # type: ignore[attr-defined]
    return client, fake


def _sql(fake: _FakeSync) -> str:
    assert fake.calls, "expected at least one POST"
    path, body = fake.calls[-1]
    assert path == "/query"
    return body["sql"]


def test_vessel_track_basic() -> None:
    client, fake = _client_with_fake()
    client.vessel_track(123456789)
    assert _sql(fake) == "VESSEL_TRACK(123456789)"


def test_vessel_track_with_window() -> None:
    client, fake = _client_with_fake()
    client.vessel_track(123456789, window_secs=3600)
    sql = _sql(fake)
    assert sql.startswith("VESSEL_TRACK(123456789, WINDOW => ")
    assert str(3600 * 1_000_000_000) in sql  # converted to ns


def test_dark_fleet_detect_default() -> None:
    client, fake = _client_with_fake()
    client.dark_fleet_detect()
    assert _sql(fake) == "DARK_FLEET_DETECT()"


def test_dark_fleet_detect_with_gap() -> None:
    client, fake = _client_with_fake()
    client.dark_fleet_detect(max_gap_hours=12.0)
    assert _sql(fake) == "DARK_FLEET_DETECT(MAX_GAP_HOURS => 12.0)"


def test_vessel_to_vessel_transfer_default() -> None:
    client, fake = _client_with_fake()
    client.vessel_to_vessel_transfer()
    assert _sql(fake) == "VESSEL_TO_VESSEL_TRANSFER()"


def test_vessel_to_vessel_transfer_with_opts() -> None:
    client, fake = _client_with_fake()
    client.vessel_to_vessel_transfer(proximity_nm=0.5, time_window_minutes=30)
    sql = _sql(fake)
    assert "PROXIMITY_NM => 0.5" in sql
    assert "TIME_WINDOW_MINUTES => 30" in sql


def test_maritime_posts_expected_body_shape() -> None:
    client, fake = _client_with_fake()
    client.vessel_track(1)
    path, body = fake.calls[-1]
    assert set(body.keys()) == {"purpose", "sql"}
    assert body["purpose"] == "analytics"
    # body must be JSON-serialisable
    json.dumps(body)
