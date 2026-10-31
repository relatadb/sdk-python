"""Bulk ingest SDK (#83).

Exposes the generic row-ingest APIs. :meth:`relata.client.RelataClient.ingest_document`
sends prepared document rows as a queued mixed-type request to
``/v1/ingest/bulk``; document extraction and source-format mapping belong in an
application adapter.
"""

from __future__ import annotations

import time
import re
from datetime import date
from typing import TYPE_CHECKING, Any, AsyncIterable, Iterable
from urllib.parse import urlencode

from relata._http import AsyncHttpTransport, HttpTransport, path_segment

if TYPE_CHECKING:
    import httpx

    from relata.client import RelataClient


# ── CDR / OTLP ingest helpers (#2252) ──────────────────────────────────────────

# Canonical CDR field → accepted input aliases for the SDK domain adapter.
_CDR_HEADER_TRANSLATION = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")
_CDR_ALIASES: dict[str, tuple[str, ...]] = {
    "caller": ("caller", "caller_msisdn", "a_number", "a_msisdn"),
    "callee": ("callee", "callee_msisdn", "b_number", "b_msisdn", "called"),
    "start": ("start", "call_start", "start_time", "timestamp", "call_start_ns"),
    "duration": ("duration", "duration_secs", "duration_seconds"),
    "tower": ("tower", "tower_id", "cell_id", "site", "cell"),
    "type": ("type", "call_type", "record_type"),
}


def _cdr_value(row: dict[str, Any], aliases: tuple[str, ...]) -> Any:
    """Get the first supplied alias from an already-normalized header map."""
    return next((row[key] for key in aliases if key in row), None)


def _cdr_metadata(row: dict[str, Any], normalized: dict[str, Any]) -> None:
    """Preserve generic-ingest identity and source-file provenance fields."""
    for field in ("_pk", "id"):
        value = _cdr_value(row, (field,))
        if value is not None:
            normalized[field] = value
    source_file_id = _cdr_value(row, ("source_file_id",))
    if source_file_id is not None:
        normalized["source_file_id"] = source_file_id


def _cdr_int(value: Any, name: str, *, default: int | None = None) -> int:
    if value is None and default is not None:
        return default
    text = str(value).strip()
    try:
        return int(text, 10)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} '{text}': invalid integer") from exc


_RFC3339_CDR = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})"
    r"(?:\.(\d+))?(Z|[+-]\d{2}:\d{2})$"
)
_I64_MIN = -(1 << 63)
_I64_MAX = (1 << 63) - 1


def _cdr_timestamp(value: Any, now_ns: int) -> int:
    """Normalize legacy epoch values or exact RFC3339-with-offset timestamps."""
    if value is None:
        return now_ns
    try:
        raw = _cdr_int(value, "call_start")
    except ValueError:
        if not isinstance(value, str):
            raise
        match = _RFC3339_CDR.fullmatch(value.strip())
        if match is None:
            raise ValueError(f"call_start '{value}': expected integer epoch or RFC3339 timestamp")
        year, month, day, hour, minute, second = map(int, match.groups()[:6])
        fraction, zone = match.group(7), match.group(8)
        if fraction is not None and len(fraction) > 9:
            raise ValueError("call_start RFC3339 fraction exceeds nanosecond precision")
        if second > 59 or hour > 23 or minute > 59:
            raise ValueError(f"call_start '{value}': invalid RFC3339 time")
        try:
            days = (date(year, month, day) - date(1970, 1, 1)).days
        except ValueError as exc:
            raise ValueError(f"call_start '{value}': invalid RFC3339 date") from exc
        offset_seconds = 0
        if zone != "Z":
            if zone == "-00:00":
                raise ValueError("call_start RFC3339 -00:00 offset is unknown, not UTC")
            offset_hour, offset_minute = int(zone[1:3]), int(zone[4:6])
            if offset_hour > 23 or offset_minute > 59:
                raise ValueError(f"call_start '{value}': invalid RFC3339 offset")
            offset_seconds = (offset_hour * 60 + offset_minute) * 60
            if zone[0] == "-":
                offset_seconds = -offset_seconds
        epoch_seconds = days * 86_400 + hour * 3_600 + minute * 60 + second - offset_seconds
        nanos = epoch_seconds * 1_000_000_000 + int((fraction or "").ljust(9, "0") or "0")
        if not _I64_MIN <= nanos <= _I64_MAX:
            raise ValueError("call_start RFC3339 value is outside i64 nanosecond range")
        return nanos
    if not _I64_MIN <= raw <= _I64_MAX:
        raise ValueError(f"call_start '{raw}': value out of range for i64")
    if raw < 10_000_000_000_000:
        return max(_I64_MIN, min(_I64_MAX, raw * 1_000_000_000))
    return raw


def _normalize_cdr_rows(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Map telecom aliases to the persisted CdrRecord fields for generic ingest."""
    normalized: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    now_ns = time.time_ns()
    for row_number, row in enumerate(rows, 1):
        try:
            # Reuse one case-insensitive view for all fields and metadata.
            # Reject case-only collisions rather than choosing a value based
            # on source-language map iteration order.
            normalized_keys: dict[str, Any] = {}
            for key, value in row.items():
                # CDR aliases are ASCII identifiers; match the other SDKs by
                # folding only ASCII A-Z rather than Unicode case equivalents.
                folded = str(key).translate(_CDR_HEADER_TRANSLATION)
                if folded in normalized_keys:
                    raise ValueError(f"ambiguous case-insensitive field '{folded}'")
                normalized_keys[folded] = value
            row = normalized_keys
            caller_raw = _cdr_value(row, _CDR_ALIASES["caller"])
            callee_raw = _cdr_value(row, _CDR_ALIASES["callee"])
            caller_digits = "".join(c for c in str("" if caller_raw is None else caller_raw) if c.isascii() and c.isdigit())
            callee_digits = "".join(c for c in str("" if callee_raw is None else callee_raw) if c.isascii() and c.isdigit())
            if len(caller_digits) > 15:
                raise ValueError(f"caller_msisdn '{caller_raw}': too many digits (max 15 for E.164)")
            if len(callee_digits) > 15:
                raise ValueError(f"callee_msisdn '{callee_raw}': too many digits (max 15 for E.164)")
            caller = _cdr_int(caller_digits, "caller_msisdn")
            callee = _cdr_int(callee_digits, "callee_msisdn")
            start_raw = _cdr_value(row, _CDR_ALIASES["start"])
            start = _cdr_timestamp(start_raw, now_ns)
            duration = _cdr_int(_cdr_value(row, _CDR_ALIASES["duration"]), "duration", default=0)
            if not 0 <= duration <= 0xFFFFFFFF:
                raise ValueError(f"duration '{duration}': value out of range for u32")
            call_type_raw = _cdr_value(row, _CDR_ALIASES["type"])
            call_type = str("" if call_type_raw is None else call_type_raw).strip().lower()
            call_type = {
                "voice": "voice", "call": "voice", "0": "voice",
                "sms": "sms", "mms": "sms", "1": "sms",
                "data": "data", "gprs": "data", "2": "data",
            }.get(call_type, "unknown")
            tower = _cdr_value(row, _CDR_ALIASES["tower"])
            normalized_row = {
                "caller_msisdn": caller,
                "callee_msisdn": callee,
                "call_start_ns": start,
                "duration_secs": duration,
                "tower_id": "" if tower is None else str(tower).strip(),
                "call_type": call_type,
                "source_file_id": 0,
            }
            _cdr_metadata(row, normalized_row)
            normalized.append(normalized_row)
        except (TypeError, ValueError) as exc:
            errors.append({"row": row_number, "error": str(exc)})
    return normalized, errors


def _cdr_receipt(receipt: dict[str, Any], local_errors: list[dict[str, Any]]) -> dict[str, Any]:
    server_errors = receipt.get("errors")
    errors = list(local_errors)
    if isinstance(server_errors, list):
        errors.extend(server_errors)
    result = {**receipt, "errors": errors}
    result["rows_rejected"] = int(receipt.get("rows_rejected", 0)) + len(local_errors)
    return result


def _purpose_path(base: str, purpose: str | None) -> str:
    """Append ``?purpose=<p>`` when a purpose is set; otherwise return ``base``."""
    if purpose:
        return f"{base}?{urlencode({'purpose': purpose})}"
    return base


def _detect_headers(detect_packs: str | None) -> dict[str, str]:
    """Build the ``X-Detect-Packs`` header for a per-call SmartIngest override
    (#2258 SI-3). ``None``/empty → no override (empty dict spreads cleanly)."""
    if detect_packs:
        return {"X-Detect-Packs": detect_packs}
    return {}


def _envelope_headers(
    detect_packs: str | None,
    idempotency_key: str | None,
    durability: str | None,
    media_work_claim: int | None = None,
) -> dict[str, str]:
    """Headers for the ``/v1/ingest/bulk`` envelope: SmartIngest override,
    ``Idempotency-Key`` (#5689), ``X-Relata-Durability`` (``"sync"``) and the
    ``X-Relata-Claim`` media-work fence."""
    headers = _detect_headers(detect_packs)
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key
    if durability:
        headers["X-Relata-Durability"] = durability
    if media_work_claim is not None:
        headers["X-Relata-Claim"] = str(int(media_work_claim))
    return headers


def _envelope_payload(
    object_type: str,
    rows: list[dict[str, Any]],
    on_conflict: str | None,
    purpose: str | None,
    tenant_id: str | None,
    source_checkpoint: dict[str, Any] | None,
    contract: dict[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"objects": [{**r, "_type": object_type} for r in rows]}
    if on_conflict is not None:
        payload["on_conflict"] = on_conflict
    if purpose:
        payload["purpose"] = purpose
    if tenant_id is not None:
        payload["tenant_id"] = tenant_id
    if source_checkpoint is not None:
        payload["source_checkpoint"] = source_checkpoint
    if contract is not None:
        payload["contract"] = contract
    return payload


class IngestClient:
    """Synchronous bulk-ingest client."""

    def __init__(
        self,
        base_url: str,
        *,
        bearer_token: str | None = None,
        purpose: str | None = None,
        tenant: str | None = None,
        timeout: float = 60.0,
        extra_headers: dict[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._purpose = purpose
        self._tenant = tenant
        extra: dict[str, str] = {}
        if tenant is not None:
            extra["X-Relata-Tenant-Id"] = tenant
        if extra_headers:
            extra.update(extra_headers)
        self._t = HttpTransport(
            base_url, bearer_token, timeout, transport=transport, extra_headers=extra or None
        )

    @classmethod
    def from_client(cls, client: RelataClient) -> IngestClient:
        return cls(
            client._base_url,
            bearer_token=client._bearer_token,
            purpose=client._default_purpose,
            tenant=client._tenant,
            timeout=client._timeout,
            extra_headers=client._extra_headers,
        )

    def _params(self, object_type: str, purpose: str | None) -> str:
        eff_purpose = purpose or self._purpose
        params: dict[str, str] = {"object_type": object_type}
        if eff_purpose:
            params["purpose"] = eff_purpose
        return urlencode(params)

    def bulk(
        self,
        object_type: str,
        rows: list[dict[str, Any]],
        *,
        purpose: str | None = None,
        detect_packs: str | None = None,
        on_conflict: str | None = None,
        tenant_id: str | None = None,
        idempotency_key: str | None = None,
        durability: str | None = None,
        source_checkpoint: dict[str, Any] | None = None,
        contract: dict[str, Any] | None = None,
        media_work_claim: int | None = None,
    ) -> dict[str, Any]:
        """Bulk-ingest ``rows``. Returns the server receipt.

        Per-call controls:
        - ``detect_packs`` — SmartIngest detector-pack override (#2258 SI-3),
          sent as the ``X-Detect-Packs`` header. ``"none"`` disables SmartIngest
          for this call; a comma-separated list (e.g. ``"network,financial"``)
          selects packs. Requires the ``IngestDetectOverride`` ACL grant.
        - ``on_conflict`` — when set (``"upsert"`` | ``"skip"`` | ``"error"``),
          routes to ``POST /v1/ingest/bulk`` so conflict resolution + ``tenant_id``
          take effect.
        - ``tenant_id`` — tenant scope (only honoured on the ``/v1/ingest/bulk`` door).
        - ``idempotency_key`` — ``Idempotency-Key`` header (#5689): a retry with
          the same key and rows replays the recorded response without a second
          write; a different body is ``409 idempotency-conflict``.
        - ``durability`` — ``"sync"`` sends ``X-Relata-Durability: sync`` (WAL
          fsynced before the response).
        - ``source_checkpoint`` — ``{"source", "position", "expected_position"?}``;
          the server advances the durable checkpoint only after a successful
          synchronous write (requires ``durability="sync"``).
        - ``contract`` — ``{"dataset", "version"?}``: validate against a data
          contract (#5714); invalid rows are quarantined server-side.
        Any of ``on_conflict``, ``idempotency_key``, ``source_checkpoint`` or
        ``contract`` routes the call to ``POST /v1/ingest/bulk``.
        - ``media_work_claim`` — the ``claim_id`` from ``MediaWorkClient.claim``,
          sent as ``X-Relata-Claim``. Required when ``source_checkpoint`` names a
          ``openetl/media/<work_id>`` source (the server fences stale holders).
        """
        import json

        if (
            on_conflict is not None
            or idempotency_key is not None
            or source_checkpoint is not None
            or contract is not None
            or media_work_claim is not None
        ):
            payload = _envelope_payload(
                object_type,
                rows,
                on_conflict,
                purpose or self._purpose,
                tenant_id,
                source_checkpoint,
                contract,
            )
            return self._t.post_raw(
                "/v1/ingest/bulk",
                json.dumps(payload),
                content_type="application/json",
                extra_headers=_envelope_headers(detect_packs, idempotency_key, durability, media_work_claim),
            )
        body = "\n".join(json.dumps(r) for r in rows)
        return self._t.post_raw(
            f"/v1/ingest?{self._params(object_type, purpose)}",
            body,
            content_type="application/x-ndjson",
            extra_headers=_detect_headers(detect_packs),
        )

    def bulk_csv(
        self,
        object_type: str,
        csv_text: str,
        *,
        purpose: str | None = None,
        detect_packs: str | None = None,
    ) -> dict[str, Any]:
        """Bulk-ingest a CSV string. The server parses it server-side.

        ``detect_packs`` is a per-call SmartIngest override (``X-Detect-Packs``
        header, #2258 SI-3): ``"none"`` disables, ``"network,financial"`` selects.
        """
        return self._t.post_raw(
            f"/v1/ingest?{self._params(object_type, purpose)}",
            csv_text,
            content_type="text/csv",
            extra_headers=_detect_headers(detect_packs),
        )

    def ingest_cdr(
        self,
        rows: Iterable[dict[str, Any]],
        *,
        purpose: str | None = None,
    ) -> dict[str, Any]:
        """Normalize CDR rows and queue them through ``POST /v1/ingest``.

        The returned receipt includes SDK-side ``errors`` for invalid rows and
        the generic ingest ``task_id`` for accepted rows. Poll that task for
        writer completion; acceptance is not durability.
        """
        normalized, errors = _normalize_cdr_rows(rows)
        import json

        receipt = self._t.post_raw(
            f"/v1/ingest?{self._params('CdrRecord', purpose)}",
            json.dumps(normalized, separators=(",", ":")),
            content_type="application/json",
        )
        return _cdr_receipt(receipt, errors)

    def otlp_traces(self, payload: dict[str, Any], *, purpose: str | None = None) -> dict[str, Any]:
        """Ingest OTLP/JSON traces via ``POST /v1/traces`` (#2252)."""
        return self._t.post(_purpose_path("/v1/traces", purpose or self._purpose), payload)

    def otlp_logs(self, payload: dict[str, Any], *, purpose: str | None = None) -> dict[str, Any]:
        """Ingest OTLP/JSON logs via ``POST /v1/logs`` (#2252)."""
        return self._t.post(_purpose_path("/v1/logs", purpose or self._purpose), payload)

    def otlp_metrics(
        self, payload: dict[str, Any], *, purpose: str | None = None
    ) -> dict[str, Any]:
        """Ingest OTLP/JSON metrics via ``POST /v1/metrics`` (#2252)."""
        return self._t.post(_purpose_path("/v1/metrics", purpose or self._purpose), payload)

    def ingest_iter(
        self,
        object_type: str,
        rows: Iterable[dict[str, Any]],
        *,
        purpose: str | None = None,
        batch_size: int = 500,
    ) -> int:
        """Stream rows from an iterator into batched ``POST /v1/ingest`` calls.

        Memory is O(batch_size), not O(total_rows). Returns the number of
        input rows in batches whose ``bulk`` call returned without an SDK
        exception. ``/v1/ingest`` is queued: this is not a count of rows accepted
        by the writer or confirmed written, and this helper discards batch
        receipts/task IDs. Call :meth:`bulk` per batch and poll each task when
        completion or per-row error accounting is required. Stops when a
        ``bulk`` call raises; a successful HTTP response may still report
        row-level errors that this helper does not inspect. TypeScript throws
        on a failed request; Go returns the count from earlier completed calls
        with the error, excluding the failing batch.

        >>> rows = [{"name": "Alice"}, {"name": "Bob"}]
        >>> total = client.ingest_iter("Person", rows, purpose="onboarding", batch_size=500)
        """
        batch: list[dict[str, Any]] = []
        total = 0
        for row in rows:
            batch.append(row)
            if len(batch) >= batch_size:
                self.bulk(object_type, batch, purpose=purpose)
                total += len(batch)
                batch = []
        if batch:
            self.bulk(object_type, batch, purpose=purpose)
            total += len(batch)
        return total

    def close(self) -> None:
        self._t.close()

    def __enter__(self) -> IngestClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class AsyncIngestClient:
    """Asynchronous bulk-ingest client — see :class:`IngestClient`."""

    def __init__(
        self,
        base_url: str,
        *,
        bearer_token: str | None = None,
        purpose: str | None = None,
        tenant: str | None = None,
        timeout: float = 60.0,
        extra_headers: dict[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._purpose = purpose
        self._tenant = tenant
        extra: dict[str, str] = {}
        if tenant is not None:
            extra["X-Relata-Tenant-Id"] = tenant
        if extra_headers:
            extra.update(extra_headers)
        self._t = AsyncHttpTransport(
            base_url, bearer_token, timeout, transport=transport, extra_headers=extra or None
        )

    @classmethod
    def from_client(cls, client: RelataClient) -> AsyncIngestClient:
        return cls(
            client._base_url,
            bearer_token=client._bearer_token,
            purpose=client._default_purpose,
            tenant=client._tenant,
            timeout=client._timeout,
            extra_headers=client._extra_headers,
        )

    def _params(self, object_type: str, purpose: str | None) -> str:
        eff_purpose = purpose or self._purpose
        params: dict[str, str] = {"object_type": object_type}
        if eff_purpose:
            params["purpose"] = eff_purpose
        return urlencode(params)

    async def bulk(
        self,
        object_type: str,
        rows: list[dict[str, Any]],
        *,
        purpose: str | None = None,
        detect_packs: str | None = None,
        on_conflict: str | None = None,
        tenant_id: str | None = None,
        idempotency_key: str | None = None,
        durability: str | None = None,
        source_checkpoint: dict[str, Any] | None = None,
        contract: dict[str, Any] | None = None,
        media_work_claim: int | None = None,
    ) -> dict[str, Any]:
        """Async variant of :meth:`bulk` (per-call SmartIngest, conflict,
        idempotency, checkpoint and contract controls)."""
        import json

        if (
            on_conflict is not None
            or idempotency_key is not None
            or source_checkpoint is not None
            or contract is not None
            or media_work_claim is not None
        ):
            payload = _envelope_payload(
                object_type,
                rows,
                on_conflict,
                purpose or self._purpose,
                tenant_id,
                source_checkpoint,
                contract,
            )
            return await self._t.post_raw(
                "/v1/ingest/bulk",
                json.dumps(payload),
                content_type="application/json",
                extra_headers=_envelope_headers(detect_packs, idempotency_key, durability, media_work_claim),
            )
        body = "\n".join(json.dumps(r) for r in rows)
        return await self._t.post_raw(
            f"/v1/ingest?{self._params(object_type, purpose)}",
            body,
            content_type="application/x-ndjson",
            extra_headers=_detect_headers(detect_packs),
        )

    async def bulk_csv(
        self,
        object_type: str,
        csv_text: str,
        *,
        purpose: str | None = None,
        detect_packs: str | None = None,
    ) -> dict[str, Any]:
        """Async variant of :meth:`bulk_csv`."""
        return await self._t.post_raw(
            f"/v1/ingest?{self._params(object_type, purpose)}",
            csv_text,
            content_type="text/csv",
            extra_headers=_detect_headers(detect_packs),
        )

    async def ingest_cdr(
        self,
        rows: Iterable[dict[str, Any]],
        *,
        purpose: str | None = None,
    ) -> dict[str, Any]:
        """Async normalize CDR rows and queue them through generic ``/v1/ingest``."""
        normalized, errors = _normalize_cdr_rows(rows)
        import json

        receipt = await self._t.post_raw(
            f"/v1/ingest?{self._params('CdrRecord', purpose)}",
            json.dumps(normalized, separators=(",", ":")),
            content_type="application/json",
        )
        return _cdr_receipt(receipt, errors)

    async def otlp_traces(
        self, payload: dict[str, Any], *, purpose: str | None = None
    ) -> dict[str, Any]:
        """Async ingest OTLP/JSON traces via ``POST /v1/traces`` (#2252)."""
        return await self._t.post(_purpose_path("/v1/traces", purpose or self._purpose), payload)

    async def otlp_logs(
        self, payload: dict[str, Any], *, purpose: str | None = None
    ) -> dict[str, Any]:
        """Async ingest OTLP/JSON logs via ``POST /v1/logs`` (#2252)."""
        return await self._t.post(_purpose_path("/v1/logs", purpose or self._purpose), payload)

    async def otlp_metrics(
        self, payload: dict[str, Any], *, purpose: str | None = None
    ) -> dict[str, Any]:
        """Async ingest OTLP/JSON metrics via ``POST /v1/metrics`` (#2252)."""
        return await self._t.post(_purpose_path("/v1/metrics", purpose or self._purpose), payload)

    async def ingest_iter(
        self,
        object_type: str,
        rows: AsyncIterable[dict[str, Any]] | Iterable[dict[str, Any]],
        *,
        purpose: str | None = None,
        batch_size: int = 500,
    ) -> int:
        """Async equivalent of :meth:`IngestClient.ingest_iter`.

        Accepts either a synchronous or an asynchronous iterable/iterator of
        rows, so callers can stream from an async source (e.g. an async DB
        cursor or an async generator) as well as a plain list. Memory is
        O(batch_size), not O(total_rows). Returns the number of input rows in
        batches whose ``bulk`` call returned without an SDK exception. The
        queued ``/v1/ingest`` request is not a writer-confirmed count; this helper
        discards batch receipts/task IDs and does not inspect row-level errors
        in a successful response. Call ``bulk`` per batch and poll each task
        when completion or error accounting is required.
        """
        batch: list[dict[str, Any]] = []
        total = 0
        if hasattr(rows, "__aiter__"):
            async for row in rows:  # type: ignore[union-attr]
                batch.append(row)
                if len(batch) >= batch_size:
                    await self.bulk(object_type, batch, purpose=purpose)
                    total += len(batch)
                    batch = []
        else:
            for row in rows:  # type: ignore[union-attr]
                batch.append(row)
                if len(batch) >= batch_size:
                    await self.bulk(object_type, batch, purpose=purpose)
                    total += len(batch)
                    batch = []
        if batch:
            await self.bulk(object_type, batch, purpose=purpose)
            total += len(batch)
        return total

    async def close(self) -> None:
        await self._t.aclose()

    async def __aenter__(self) -> AsyncIngestClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()
