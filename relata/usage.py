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


@dataclass(frozen=True)
class LegacyUsageBaseline:
    """Versioned snapshot of counters read before the usage-event cutover.

    ``feedback_count`` is nullable because older aggregates persisted only a
    mean. Leave it unknown unless an independent, trusted source supplies the
    original sample count.
    """

    format_version: str
    target: EventTarget
    cutover_at: int
    citation_count: int
    retrieval_count: int
    feedback_avg: float | None
    feedback_count: int | None
    last_cited_at: int | None


@dataclass(frozen=True)
class MigratedUsageAggregate:
    """A frozen legacy baseline plus separately reduced post-cutover events."""

    legacy: LegacyUsageBaseline
    post_cutover: UsageAggregate
    citation_count: int
    retrieval_count: int
    feedback_count: int | None
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


def aggregate_usage_with_baseline(
    rows: Iterable[Mapping[str, Any]],
    *,
    target: EventTarget,
    baseline: LegacyUsageBaseline,
) -> MigratedUsageAggregate:
    """Combine a frozen v1 snapshot with a complete post-cutover event set.

    The caller must stop old writers before taking the baseline. Events before
    ``cutover_at`` are rejected. If the old feedback sample count is unknown,
    the two means stay separate and combined feedback fields are ``None``.
    """
    _validate_legacy_usage_baseline(baseline)
    if baseline.target != target:
        raise ValueError("legacy usage baseline belongs to a different target")
    materialized = list(rows)
    usage = aggregate_usage_events(materialized, target=target)
    for row in materialized:
        observed_at = row.get("observed_at")
        if isinstance(observed_at, str) and observed_at.lstrip("-").isdigit():
            observed_at = int(observed_at)
        if isinstance(observed_at, bool) or not isinstance(observed_at, int):
            raise ValueError("observed_at must be an integer timestamp")
        if observed_at < baseline.cutover_at:
            raise ValueError("usage event predates the baseline cutover")

    max_count = (1 << 63) - 1
    if baseline.citation_count + usage.citation_count > max_count:
        raise ValueError("combined citation_count exceeds the signed 64-bit range")
    if baseline.retrieval_count + usage.retrieval_count > max_count:
        raise ValueError("combined retrieval_count exceeds the signed 64-bit range")

    if baseline.feedback_count is None:
        combined_feedback_count = None
        combined_feedback_avg = None
    else:
        combined_feedback_count = baseline.feedback_count + usage.feedback_count
        if combined_feedback_count > max_count:
            raise ValueError("combined feedback_count exceeds the signed 64-bit range")
        total = 0.0
        if baseline.feedback_count and baseline.feedback_avg is not None:
            total += baseline.feedback_avg * baseline.feedback_count
        if usage.feedback_count and usage.feedback_avg is not None:
            total += usage.feedback_avg * usage.feedback_count
        combined_feedback_avg = (
            total / combined_feedback_count if combined_feedback_count else None
        )

    last_cited_at = baseline.last_cited_at
    if usage.last_cited_at is not None and (
        last_cited_at is None or usage.last_cited_at > last_cited_at
    ):
        last_cited_at = usage.last_cited_at
    return MigratedUsageAggregate(
        legacy=baseline,
        post_cutover=usage,
        citation_count=baseline.citation_count + usage.citation_count,
        retrieval_count=baseline.retrieval_count + usage.retrieval_count,
        feedback_count=combined_feedback_count,
        feedback_avg=combined_feedback_avg,
        last_cited_at=last_cited_at,
    )


def _validate_legacy_usage_baseline(baseline: LegacyUsageBaseline) -> None:
    if baseline.format_version != "document-usage-baseline-v1":
        raise ValueError("unsupported legacy usage baseline format_version")
    if not isinstance(baseline.target, EventTarget) or not all(
        isinstance(value, str) and value.strip()
        for value in (baseline.target.object_type, baseline.target.key, baseline.target.value)
    ):
        raise ValueError("baseline target object_type, key, and value are required")
    if isinstance(baseline.cutover_at, bool) or not isinstance(baseline.cutover_at, int):
        raise ValueError("cutover_at must be an integer timestamp")
    if not -(1 << 63) <= baseline.cutover_at < (1 << 63):
        raise ValueError("cutover_at must fit a signed 64-bit timestamp")
    for name, value in (
        ("citation_count", baseline.citation_count),
        ("retrieval_count", baseline.retrieval_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < (1 << 63):
            raise ValueError(f"{name} must be a non-negative integer")
    if baseline.feedback_count is not None and (
        isinstance(baseline.feedback_count, bool)
        or not isinstance(baseline.feedback_count, int)
        or not 0 <= baseline.feedback_count < (1 << 63)
    ):
        raise ValueError("feedback_count must be a non-negative integer or None")
    if baseline.feedback_avg is not None and (
        isinstance(baseline.feedback_avg, bool)
        or not isinstance(baseline.feedback_avg, (int, float))
        or not math.isfinite(baseline.feedback_avg)
        or not 0 <= baseline.feedback_avg <= 1
    ):
        raise ValueError("feedback_avg must be finite and between 0 and 1")
    if baseline.feedback_count == 0 and baseline.feedback_avg is not None:
        raise ValueError("feedback_avg must be None when feedback_count is zero")
    if (
        baseline.feedback_count is not None
        and baseline.feedback_count > 0
        and baseline.feedback_avg is None
    ):
        raise ValueError("feedback_avg is required when feedback_count is positive")
    if baseline.last_cited_at is not None and (
        isinstance(baseline.last_cited_at, bool)
        or not isinstance(baseline.last_cited_at, int)
    ):
        raise ValueError("last_cited_at must be an integer timestamp or None")
    if baseline.last_cited_at is not None and not -(1 << 63) <= baseline.last_cited_at < (1 << 63):
        raise ValueError("last_cited_at must fit a signed 64-bit timestamp")


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
