"""Bulk ingest SDK (#83).

Surfaces the server's ``POST /ingest?object_type=<Type>`` for NDJSON / CSV /
Arrow batches. Distinct from :class:`relata.client.RelataClient.ingest_document`
which is the datagrep-extractor envelope; this module is the general batch
path the partner contract (§4) calls for.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable
from urllib.parse import urlencode

from relata._http import AsyncHttpTransport, HttpTransport

if TYPE_CHECKING:
    import httpx

    from relata.client import RelataClient


def _handle_raw(resp: httpx.Response) -> dict[str, Any]:
    """Static-method equivalent of HttpTransport._handle for raw-response paths."""
    if resp.is_success:
        return resp.json()  # type: ignore[no-any-return]
    try:
        body: dict[str, Any] = resp.json()
    except Exception:
        body = {"error": resp.text or "empty response"}
    from relata._http import _classify_error

    raise _classify_error(resp.status_code, body)


# ── CDR / OTLP ingest helpers (#2252) ──────────────────────────────────────────

# Canonical CDR column → accepted aliases (mirrors the server's
# `CdrRecord::from_csv_row` header recognition, case-insensitive server-side).
_CDR_ALIASES: dict[str, tuple[str, ...]] = {
    "caller": ("caller", "caller_msisdn", "a_number", "a_msisdn"),
    "callee": ("callee", "callee_msisdn", "b_number", "b_msisdn", "called"),
    "start": ("start", "call_start", "start_time", "timestamp", "call_start_ns"),
    "duration": ("duration", "duration_secs", "duration_seconds"),
    "tower": ("tower", "tower_id", "cell_id", "site", "cell"),
    "type": ("type", "call_type", "record_type"),
}
_CDR_COLUMNS: tuple[str, ...] = ("caller", "callee", "start", "duration", "tower", "type")


def _cdr_field(value: object) -> str:
    """Render a CDR cell value — the server parses CSV via naive split(',')."""
    if value is None:
        return ""
    # Strip commas / CR / LF so a value can never break column alignment.
    return str(value).replace(",", "").replace("\r", " ").replace("\n", " ")


def _rows_to_cdr_csv(rows: Iterable[dict[str, Any]]) -> str:
    """Serialise typed CDR rows into the CSV shape ``POST /ingest/cdr`` expects."""
    lines = [",".join(_CDR_COLUMNS)]
    for row in rows:
        fields: list[str] = []
        for col in _CDR_COLUMNS:
            val: Any = ""
            for key in _CDR_ALIASES[col]:
                if key in row:
                    val = row[key]
                    break
            fields.append(_cdr_field(val))
        lines.append(",".join(fields))
    return "\n".join(lines)


def _purpose_path(base: str, purpose: str | None) -> str:
    """Append ``?purpose=<p>`` when a purpose is set; otherwise return ``base``."""
    if purpose:
        return f"{base}?{urlencode({'purpose': purpose})}"
    return base


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
    ) -> dict[str, Any]:
        """Bulk-ingest ``rows`` as NDJSON. Returns the server receipt."""
        import json

        body = "\n".join(json.dumps(r) for r in rows)
        resp = self._t._client.post(  # noqa: SLF001 — raw NDJSON body
            f"/ingest?{self._params(object_type, purpose)}",
            content=body,
            headers={"Content-Type": "application/x-ndjson"},
        )
        return _handle_raw(resp)

    def bulk_csv(
        self,
        object_type: str,
        csv_text: str,
        *,
        purpose: str | None = None,
    ) -> dict[str, Any]:
        """Bulk-ingest a CSV string. The server parses it server-side."""
        resp = self._t._client.post(  # noqa: SLF001
            f"/ingest?{self._params(object_type, purpose)}",
            content=csv_text,
            headers={"Content-Type": "text/csv"},
        )
        return _handle_raw(resp)

    def ingest_cdr(
        self,
        rows: Iterable[dict[str, Any]],
        *,
        purpose: str | None = None,
    ) -> dict[str, Any]:
        """Ingest CDR (call detail records) via ``POST /ingest/cdr`` (#2252).

        Each row is a dict mapping CDR fields (``caller``, ``callee``, ``start``,
        ``duration``, ``tower``, ``type``); common aliases like ``caller_msisdn``
        / ``a_number`` are accepted. Rows are serialised to the CSV shape the
        server's ``CdrBatch::from_csv`` expects. ``purpose`` is mandatory
        server-side — pass it here or via the client's default purpose.
        """
        csv_text = _rows_to_cdr_csv(rows)
        resp = self._t._client.post(  # noqa: SLF001 — raw CSV body
            _purpose_path("/ingest/cdr", purpose or self._purpose),
            content=csv_text,
            headers={"Content-Type": "text/csv"},
        )
        return _handle_raw(resp)

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
        """Stream rows from an iterator into batched ``POST /ingest`` calls.

        Memory is O(batch_size), not O(total_rows). Returns the total number
        of rows successfully ingested. Stops on the first error.

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

    def media_status(self, task_id: str) -> dict[str, Any]:
        """Poll the status of a multipart media upload (paired with #76)."""
        return self._t.get(f"/ingest/media/{task_id}")

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
    ) -> dict[str, Any]:
        import json

        body = "\n".join(json.dumps(r) for r in rows)
        resp = await self._t._client.post(  # noqa: SLF001
            f"/ingest?{self._params(object_type, purpose)}",
            content=body,
            headers={"Content-Type": "application/x-ndjson"},
        )
        return _handle_raw(resp)

    async def bulk_csv(
        self,
        object_type: str,
        csv_text: str,
        *,
        purpose: str | None = None,
    ) -> dict[str, Any]:
        resp = await self._t._client.post(  # noqa: SLF001
            f"/ingest?{self._params(object_type, purpose)}",
            content=csv_text,
            headers={"Content-Type": "text/csv"},
        )
        return _handle_raw(resp)

    async def ingest_cdr(
        self,
        rows: Iterable[dict[str, Any]],
        *,
        purpose: str | None = None,
    ) -> dict[str, Any]:
        """Async ingest CDR (call detail records) via ``POST /ingest/cdr`` (#2252)."""
        csv_text = _rows_to_cdr_csv(rows)
        resp = await self._t._client.post(  # noqa: SLF001 — raw CSV body
            _purpose_path("/ingest/cdr", purpose or self._purpose),
            content=csv_text,
            headers={"Content-Type": "text/csv"},
        )
        return _handle_raw(resp)

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

    async def media_status(self, task_id: str) -> dict[str, Any]:
        return await self._t.get(f"/ingest/media/{task_id}")

    async def close(self) -> None:
        await self._t.aclose()

    async def __aenter__(self) -> AsyncIngestClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()
