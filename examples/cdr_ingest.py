"""Ingest telecom CDR rows through the SDK adapter (#5744).

The server has no telecom parser: ``ingest_cdr`` maps aliases such as
``a_number``/``cell_id`` to canonical ``CdrRecord`` rows client-side and posts
them to the generic ``POST /v1/ingest?object_type=CdrRecord`` contract.
Register the type once with the ``telco-cdr`` schema package.

Run:
    RELATA_BEARER_TOKEN=secret python examples/cdr_ingest.py
"""

import os
import time

from relata import RelataClient
from relata.ingest import IngestClient

rows = [
    {"a_number": "+971 50 123 4567", "b_number": "971507654321",
     "timestamp": "2026-01-01T00:00:00Z", "duration": 60, "cell_id": "CELL-042", "type": "voice"},
    {"caller": "971501111111", "callee": "971502222222",
     "start": 1735689700, "duration_seconds": 5, "site": "CELL-007", "record_type": "sms"},
    {"caller": "not-a-number"},  # rejected client-side, reported in errors
]

with RelataClient(
    os.environ.get("RELATA_URL", "http://localhost:9090"),
    bearer_token=os.environ.get("RELATA_BEARER_TOKEN"),
    purpose="investigation",
) as client:
    receipt = IngestClient.from_client(client).ingest_cdr(rows)
    print("row errors:", receipt["errors"])
    task_id = receipt.get("task_id")
    if task_id:
        # Acceptance is not durability: poll the generic task for the outcome.
        for _ in range(20):
            status = client.ingest_document_status(task_id)
            if status.status != "pending":
                break
            time.sleep(0.25)
        print(status.status, status.rows_written, "written", status.rows_dropped, "dropped")
