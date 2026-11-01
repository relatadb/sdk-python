"""Submit prepared document rows through the generic ingest contract (#5743).

Extraction and chunking (PDF/DOCX, datagrep JSONL, Unstructured, ...) happen
in an external processor or adapter. The server only accepts governed
``DocumentSource``/``DocumentChunk`` rows; it exposes no extractor envelope.

Run:
    RELATA_BEARER_TOKEN=secret python examples/document_ingest.py
"""

import os
import time

from relata import RelataClient

objects = [
    {"_type": "DocumentSource", "_pk": "rpt-1", "report_id": "rpt-1",
     "source_filename": "paper.pdf", "parser": "my-adapter", "status": "complete",
     "total_chunks": 2},
    {"_type": "DocumentChunk", "_pk": "rpt-1:c0", "chunk_id": "rpt-1:c0", "report_id": "rpt-1",
     "sequence_index": 0, "prev_chunk_id": None, "next_chunk_id": "rpt-1:c1",
     "page_start": 1, "page_end": 1, "section_path": "Abstract", "text_body": "First chunk."},
    {"_type": "DocumentChunk", "_pk": "rpt-1:c1", "chunk_id": "rpt-1:c1", "report_id": "rpt-1",
     "sequence_index": 1, "prev_chunk_id": "rpt-1:c0", "next_chunk_id": None,
     "page_start": 2, "page_end": 2, "section_path": "Methods", "text_body": "Second chunk."},
]

with RelataClient(
    os.environ.get("RELATA_URL", "http://localhost:9090"),
    bearer_token=os.environ.get("RELATA_BEARER_TOKEN"),
    purpose="rag",
) as client:
    receipt = client.ingest_document(objects)
    print("queued", receipt.rows_queued, "rejected", receipt.rows_rejected, receipt.errors)
    for _ in range(40):
        status = client.ingest_document_status(receipt.task_id)
        if status.status != "pending":
            break
        time.sleep(0.25)
    # Only "complete" means every requested row was written; partial, failed,
    # uncertain and incomplete are explicit outcomes the caller must handle.
    print(status.status, status.rows_written, "/", status.rows_total, status.drop_reasons)
