"""Decode the retained NDJSON query protocol without buffering the result."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from typing import Any

from relata.exceptions import IncompleteQueryStreamError, ResponseTooLargeError

_META_FIELDS = {
    "query_id", "started_at", "columns", "processing_time_ms", "row_count",
    "truncated", "error",
}


def _unsigned(value: Any) -> bool:  # noqa: ANN401
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and 0 <= value <= 9_007_199_254_740_991 and math.isfinite(value)
        and value == int(value)
    )


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


class QueryRowDecoder:
    """Per-stream framing/completion state; the existing response cap is per frame."""

    def __init__(self, max_frame_bytes: int) -> None:
        self._limit = max_frame_bytes
        self._buffer = bytearray()
        self._started = False
        self._terminal = False
        self.query_id: str | None = None
        self.rows_received = 0

    def failure(self, message: str) -> IncompleteQueryStreamError:
        return IncompleteQueryStreamError(
            message, query_id=self.query_id, rows_received=self.rows_received,
        )

    def feed(self, chunk: bytes) -> Iterator[dict[str, Any]]:
        start = 0
        while start < len(chunk):
            newline = chunk.find(b"\n", start)
            end = len(chunk) if newline < 0 else newline
            if self._limit > 0 and len(self._buffer) + end - start > self._limit:
                raise ResponseTooLargeError(
                    "Query stream frame exceeds max_response_bytes", max_bytes=self._limit,
                )
            self._buffer.extend(memoryview(chunk)[start:end])
            if newline < 0:
                break
            line = bytes(self._buffer).strip()
            self._buffer.clear()
            if line:
                row = self._frame(line)
                if row is not None:
                    yield row
            start = newline + 1

    def finish(self) -> Iterator[dict[str, Any]]:
        line = bytes(self._buffer).strip()
        self._buffer.clear()
        if line:
            row = self._frame(line)
            if row is not None:
                yield row
        if not self._terminal:
            raise self.failure("Query stream ended without a terminal completion frame")

    def _frame(self, line: bytes) -> dict[str, Any] | None:
        try:
            row = json.loads(line.decode("utf-8"), parse_constant=_invalid_constant)
        except (ValueError, UnicodeError) as error:
            raise self.failure("Query stream contains malformed JSON or UTF-8") from error
        if not isinstance(row, dict):
            raise self.failure("Query stream contains a non-object frame")
        if self._terminal:
            raise self.failure("Query stream contains frames after completion")
        meta = row.get("_meta")
        if len(row) == 1 and isinstance(meta, dict) and _META_FIELDS.intersection(meta):
            if not self._started:
                if (
                    not isinstance(meta.get("query_id"), str)
                    or not _unsigned(meta.get("started_at"))
                    or not isinstance(meta.get("columns"), list)
                    or not all(isinstance(column, str) for column in meta["columns"])
                ):
                    raise self.failure("Query stream has a malformed leading metadata frame")
                self.query_id = meta["query_id"]
                self._started = True
                return None
            if (
                not _unsigned(meta.get("processing_time_ms"))
                or not _unsigned(meta.get("row_count"))
                or not isinstance(meta.get("truncated"), bool)
                or ("error" in meta and not isinstance(meta["error"], str))
            ):
                raise self.failure("Query stream has a malformed terminal metadata frame")
            if meta["truncated"] or "error" in meta:
                raise self.failure("Query stream reports an incomplete answer")
            if meta["row_count"] != self.rows_received:
                raise self.failure("Query stream terminal count does not match received rows")
            self._terminal = True
            return None
        if not self._started:
            raise self.failure("Query stream is missing its leading metadata frame")
        self.rows_received += 1
        return row
