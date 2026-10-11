"""Record a usage observation through the optional event schema package."""

import json
import os

from relata import EventTarget, RelataClient, aggregate_usage_events, record_usage

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
    print(json.dumps({"receipt": receipt.model_dump(mode="json"), "summary": aggregate}))
