"""Tests for the optional document-usage recipe over generic event append."""

from __future__ import annotations

import asyncio

import pytest

from relata import EventAppendReceipt, EventAppendRequest, EventTarget, arecord_usage, record_usage


class _FakeClient:
    def __init__(self) -> None:
        self.sync_requests: list[EventAppendRequest] = []
        self.async_requests: list[EventAppendRequest] = []
        self.receipt = EventAppendReceipt(
            event_id="evt_test",
            object_type="DocumentUsageEvent",
            status="accepted",
            durability="local_wal_sync",
        )

    def append_event(self, request: EventAppendRequest) -> EventAppendReceipt:
        self.sync_requests.append(request)
        return self.receipt

    async def aappend_event(self, request: EventAppendRequest) -> EventAppendReceipt:
        self.async_requests.append(request)
        return self.receipt


def _target() -> EventTarget:
    return EventTarget(object_type="DocumentSource", key="report_id", value="report-42")


def test_record_usage_builds_generic_schema_event_and_preserves_false() -> None:
    client = _FakeClient()
    receipt = record_usage(
        client,  # type: ignore[arg-type]
        idempotency_key="answer-42:citation-1",
        target=_target(),
        purpose="rag-usage",
        cited=False,
        retrieved=True,
        feedback_score=0.75,
    )
    assert receipt.event_id == "evt_test"
    assert client.sync_requests == [
        EventAppendRequest(
            object_type="DocumentUsageEvent",
            idempotency_key="answer-42:citation-1",
            target=_target(),
            purpose="rag-usage",
            attributes={"cited": False, "retrieved": True, "feedback_score": 0.75},
        )
    ]


def test_arecord_usage_uses_async_generic_event_method() -> None:
    client = _FakeClient()
    receipt = asyncio.run(
        arecord_usage(
            client,  # type: ignore[arg-type]
            idempotency_key="answer-42:feedback",
            target=_target(),
            feedback_score=1.0,
        )
    )
    assert receipt.status == "accepted"
    assert client.async_requests[0].attributes == {"feedback_score": 1.0}


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1])
def test_record_usage_rejects_invalid_optional_package_scores(score: float) -> None:
    with pytest.raises(ValueError, match="feedback_score"):
        record_usage(
            _FakeClient(),  # type: ignore[arg-type]
            idempotency_key="answer-42:feedback",
            target=_target(),
            feedback_score=score,
        )


def test_record_usage_requires_key_and_observation_without_sending() -> None:
    client = _FakeClient()
    with pytest.raises(ValueError, match="idempotency_key"):
        record_usage(
            client, idempotency_key=" ", target=_target(), cited=True  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="idempotency_key"):
        record_usage(
            client,
            idempotency_key="é" * 129,
            target=_target(),
            cited=True,
        )  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one"):
        record_usage(client, idempotency_key="stable", target=_target())  # type: ignore[arg-type]
    assert client.sync_requests == []
