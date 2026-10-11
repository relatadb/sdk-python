"""Record a usage observation through the optional event schema package."""

import json
import os

from relata import (
    EventTarget,
    LegacyUsageBaseline,
    RelataClient,
    aggregate_usage_events,
    aggregate_usage_with_baseline,
    record_usage,
)

tenant = os.environ.get("RELATA_TENANT")
if not tenant:
    raise RuntimeError("Set RELATA_TENANT to the destination tenant ID")
target_id = os.environ.get("RELATA_USAGE_TARGET_ID", "report-42")
idempotency_key = os.environ.get("RELATA_USAGE_IDEMPOTENCY_KEY", "answer-42:citation-1")

with RelataClient(
    os.environ.get("RELATA_URL", "http://localhost:9090"),
    bearer_token=os.environ.get("RELATA_BEARER_TOKEN"),
    tenant=tenant,
) as client:
    target_literal = target_id.replace("'", "''")
    receipt = record_usage(
        client,
        # Keep stable across retries after a response is lost.
        idempotency_key=idempotency_key,
        target=EventTarget(
            object_type="DocumentSource", key="report_id", value=target_id
        ),
        purpose="rag-usage",
        cited=True,
        retrieved=True,
        feedback_score=0.9,
    )
    # Also grant read permission for DocumentUsageEvent. The bounded query
    # fails closed at its cap rather than reporting incomplete counters.
    event_limit = 10_000
    events = client.query(
        "SELECT event_id, target_type, target_id, "
        "observed_at, cited, retrieved, feedback_score "
        "FROM DocumentUsageEvent WHERE target_type = 'DocumentSource' "
        f"AND target_id = '{target_literal}' LIMIT 10000",
        purpose="rag-usage",
    )
    if events.row_count >= event_limit:
        raise RuntimeError(
            f"Usage event inventory reached the {event_limit} row cap; "
            "paginate before aggregation"
        )
    summary = aggregate_usage_events(
        events.rows,
        target=EventTarget(object_type="DocumentSource", key="report_id", value=target_id),
    )
    target = EventTarget(object_type="DocumentSource", key="report_id", value=target_id)
    # A fixed synthetic frozen snapshot demonstrates migration arithmetic;
    # replace it with the tenant-governed pre-cutover export in applications.
    baseline = LegacyUsageBaseline(
        format_version="document-usage-baseline-v1",
        target=target,
        cutover_at=0,
        citation_count=7,
        retrieval_count=11,
        feedback_avg=0.6,
        feedback_count=None,
        last_cited_at=None,
    )
    migrated = aggregate_usage_with_baseline(events.rows, target=target, baseline=baseline)
    trusted_baseline = LegacyUsageBaseline(
        **{**baseline.__dict__, "feedback_count": 2}
    )
    trusted_migrated = aggregate_usage_with_baseline(
        events.rows, target=target, baseline=trusted_baseline
    )
    try:
        aggregate_usage_with_baseline(
            events.rows,
            target=target,
            baseline=LegacyUsageBaseline(
                **{
                    **baseline.__dict__,
                    "target": EventTarget(
                        object_type="DocumentSource", key="report_id", value="other"
                    ),
                }
            ),
        )
        wrong_target_rejected = False
    except ValueError:
        wrong_target_rejected = True
    aggregate = {
        "target_type": summary.target_type,
        "target_id": summary.target_id,
        "event_count": summary.event_count,
        "citation_count": summary.citation_count,
        "retrieval_count": summary.retrieval_count,
        "feedback_count": summary.feedback_count,
        "feedback_avg": summary.feedback_avg,
        "last_cited_at_ns": (
            str(summary.last_cited_at) if summary.last_cited_at is not None else None
        ),
    }
    print(json.dumps({
        "receipt": receipt.model_dump(mode="json"),
        "summary": aggregate,
        "migration": {
            "baseline_format_version": migrated.legacy.format_version,
            "citation_count": migrated.citation_count,
            "retrieval_count": migrated.retrieval_count,
            "legacy_feedback_avg": migrated.legacy.feedback_avg,
            "post_cutover_feedback_avg": migrated.post_cutover.feedback_avg,
            "feedback_count": migrated.feedback_count,
            "feedback_avg": migrated.feedback_avg,
            "trusted_feedback_count": trusted_migrated.feedback_count,
            "trusted_feedback_avg": trusted_migrated.feedback_avg,
            "wrong_target_rejected": wrong_target_rejected,
        },
    }))
