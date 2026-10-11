"""Optional document-usage recipe over the generic append-only event API."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from relata.models import EventAppendReceipt, EventAppendRequest, EventTarget

if TYPE_CHECKING:
    from relata.client import RelataClient


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
    a stable key and owns retries and aggregation. ``target`` is a logical
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
