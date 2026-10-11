"""Optional document-usage recipe over the generic append-only event API."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from relata.models import EventAppendReceipt, EventAppendRequest, EventTarget

if TYPE_CHECKING:
    from relata.client import RelataClient


@dataclass(frozen=True)
class UsageAggregate:
    """Counters rebuilt from a complete governed event inventory.

    These summarize only the supplied post-migration events; legacy feedback
    means cannot be merged because the old row did not store its sample count.
    """

    target_type: str
    target_id: str
    event_count: int
    citation_count: int
    retrieval_count: int
    feedback_count: int
    feedback_avg: float | None
    last_cited_at: int | None


def aggregate_usage_events(
    rows: Iterable[Mapping[str, Any]], *, target: EventTarget
) -> UsageAggregate:
    """Rebuild former document counters from fetched `DocumentUsageEvent` rows.

    Pass only a complete result from a governed query scoped to the same
    tenant and purpose as the caller. Detect query truncation before calling
    this pure reducer. Exact duplicate `event_id` rows count once; conflicting
    semantic payloads under one ID fail closed. This does not merge historical
    feedback means because legacy `_feedback_n` values were not retained.
    """
    if not target.object_type.strip() or not target.value.strip():
        raise ValueError("target object_type and value are required")

    unique: dict[str, tuple[tuple[Any, ...], int, bool, bool, float | None]] = {}
    for row in rows:
        event_id = row.get("event_id")
        if not isinstance(event_id, str) or not event_id.strip():
            raise ValueError("every usage event must have an event_id")
        if row.get("target_type") != target.object_type or row.get("target_id") != target.value:
            raise ValueError("usage inventory contains a different target")
        observed_at = row.get("observed_at")
        if isinstance(observed_at, str) and observed_at.lstrip("-").isdigit():
            observed_at = int(observed_at)
        if isinstance(observed_at, bool) or not isinstance(observed_at, int):
            raise ValueError("observed_at must be an integer timestamp")
        if not -(1 << 63) <= observed_at < (1 << 63):
            raise ValueError("observed_at must fit a signed 64-bit timestamp")
        cited = row.get("cited", False)
        retrieved = row.get("retrieved", False)
        if cited is None:
            cited = False
        if retrieved is None:
            retrieved = False
        if not isinstance(cited, bool) or not isinstance(retrieved, bool):
            raise ValueError("cited and retrieved must be booleans when present")
        score = row.get("feedback_score")
        if score is not None:
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError("feedback_score must be numeric when present")
            score = float(score)
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("feedback_score must be finite and between 0 and 1")
        semantic = (target.object_type, target.value, observed_at, cited, retrieved, score)
        previous = unique.get(event_id)
        if previous is not None:
            if previous[0] != semantic:
                raise ValueError(f"conflicting usage event id {event_id}")
            continue
        unique[event_id] = (semantic, observed_at, cited, retrieved, score)

    ordered = [unique[event_id] for event_id in sorted(unique)]
    citation_count = sum(1 for _, _, cited, _, _ in ordered if cited)
    retrieval_count = sum(1 for _, _, _, retrieved, _ in ordered if retrieved)
    feedback_scores = [entry[4] for entry in ordered if entry[4] is not None]
    cited_timestamps = [entry[1] for entry in ordered if entry[2]]
    return UsageAggregate(
        target_type=target.object_type,
        target_id=target.value,
        event_count=len(unique),
        citation_count=citation_count,
        retrieval_count=retrieval_count,
        feedback_count=len(feedback_scores),
        feedback_avg=(sum(feedback_scores) / len(feedback_scores)) if feedback_scores else None,
        last_cited_at=max(cited_timestamps) if cited_timestamps else None,
    )


def _request(
    *,
    idempotency_key: str,
    target: EventTarget,
    purpose: str | None,
    cited: bool | None,
    retrieved: bool | None,
    feedback_score: float | None,
) -> EventAppendRequest:
    """Build one optional DocumentUsageEvent without inventing server state."""
    if not idempotency_key.strip() or len(idempotency_key.encode("utf-8")) > 256:
        raise ValueError("idempotency_key must contain 1 to 256 non-whitespace UTF-8 bytes")
    attributes: dict[str, bool | float] = {}
    if cited is not None:
        attributes["cited"] = cited
    if retrieved is not None:
        attributes["retrieved"] = retrieved
    if feedback_score is not None:
        if not math.isfinite(feedback_score) or not 0 <= feedback_score <= 1:
            raise ValueError("feedback_score must be finite and between 0 and 1")
        attributes["feedback_score"] = feedback_score
    if not attributes:
        raise ValueError("at least one usage observation must be supplied")
    return EventAppendRequest(
        object_type="DocumentUsageEvent",
        idempotency_key=idempotency_key,
        target=target,
        attributes=attributes,
        purpose=purpose,
    )


def record_usage(
    client: RelataClient,
    *,
    idempotency_key: str,
    target: EventTarget,
    purpose: str | None = None,
    cited: bool | None = None,
    retrieved: bool | None = None,
    feedback_score: float | None = None,
) -> EventAppendReceipt:
    """Record a document-usage observation with the generic event contract.

    Install the optional ``document-usage`` package first. The caller chooses
    a stable key and owns retries; :func:`aggregate_usage_events` provides a
    separate reducer for a complete governed inventory. ``target`` is a logical
    key/value reference, not a version-pinned snapshot; the current event
    contract has no target-revision field. The server checks tenant and target
    visibility and acknowledges only after local WAL sync. It rechecks target
    visibility on exact retries, so a deleted/hidden target returns 404 without
    appending another row. Cluster mode is currently rejected.
    """
    return client.append_event(
        _request(
            idempotency_key=idempotency_key,
            target=target,
            purpose=purpose,
            cited=cited,
            retrieved=retrieved,
            feedback_score=feedback_score,
        )
    )


async def arecord_usage(
    client: RelataClient,
    *,
    idempotency_key: str,
    target: EventTarget,
    purpose: str | None = None,
    cited: bool | None = None,
    retrieved: bool | None = None,
    feedback_score: float | None = None,
) -> EventAppendReceipt:
    """Async counterpart to :func:`record_usage`."""
    return await client.aappend_event(
        _request(
            idempotency_key=idempotency_key,
            target=target,
            purpose=purpose,
            cited=cited,
            retrieved=retrieved,
            feedback_score=feedback_score,
        )
    )
