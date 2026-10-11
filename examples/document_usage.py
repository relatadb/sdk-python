"""Record a usage observation through the optional event schema package."""

import os

from relata import EventTarget, RelataClient, record_usage

tenant = os.environ.get("RELATA_TENANT")
if not tenant:
    raise RuntimeError("Set RELATA_TENANT to the destination tenant ID")

with RelataClient(
    os.environ.get("RELATA_URL", "http://localhost:9090"),
    bearer_token=os.environ.get("RELATA_BEARER_TOKEN"),
    tenant=tenant,
) as client:
    receipt = record_usage(
        client,
        # Keep stable across retries after a response is lost.
        idempotency_key="answer-42:citation-1",
        target=EventTarget(
            object_type="DocumentSource", key="report_id", value="report-42"
        ),
        purpose="rag-usage",
        cited=True,
        retrieved=True,
    )
    print(receipt)
