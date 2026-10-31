"""Streaming SDK (#75) — SSE consumers + NDJSON row streams + Arrow IPC.

The server ships streaming surfaces at:

- ``POST /v1/query/arrow`` → Arrow IPC stream (columnar).
- ``POST /v1/query/stream`` with ``Accept: application/x-ndjson`` → row stream.
- ``GET /v1/watch/stream`` → finite SSE polls — emits governed query-result
  notifications from the server's bounded background evaluator.
- ``GET /v1/alerts/stream`` → SSE — pushes alerts to subscribed principals.
- ``GET /v1/a2a/tasks/:id/stream`` → SSE — A2A task lifecycle events.

This module wraps each as a Python iterator (sync) or async iterator (async),
with disconnect/reconnect backoff for SSE.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from relata._http import path_segment
from relata.query import _validate_sql_identifier
from relata._query_stream import QueryRowDecoder
from relata.exceptions import IncompleteQueryStreamError, ResponseTooLargeError

if TYPE_CHECKING:
    from relata.client import RelataClient

#: ``watch_stream(since=)`` is a decimal ``system_from`` ns cursor interpolated
#: into the watch SQL — restrict it to ASCII digits (#3211).
_SINCE_RE = re.compile(r"^[0-9]+$")


def _validate_since(since: str) -> str:
    """Return ``since`` unchanged if it is a decimal ns timestamp, else raise
    ``ValueError`` (#3211)."""
    if not _SINCE_RE.match(since):
        raise ValueError(
            f"Invalid since cursor: {since!r}. Must be a decimal system_from ns "
            "timestamp (digits only). This is a SQL-injection defence — see #3211."
        )
    return since


def _watch_stream_sql(type: str, since: str | None) -> str:  # noqa: A002
    """Build the ``watch_stream`` SQL with validated identifier/cursor (#3211)."""
    sql = f"SELECT * FROM {_validate_sql_identifier(type, kind='object type')}"
    if since:
        sql += f" WHERE system_from > {_validate_since(since)}"
    return sql


def _classify_capped_error(transport: Any, resp: Any) -> None:
    """Materialise a streaming error response through ``transport``'s response
    cap (#3214) and raise the classified error. Reading via
    ``transport._read_capped`` means a malicious server can't OOM the SDK with
    a huge "error" body on a streaming query — the read aborts past
    ``max_response_bytes`` instead of the old unbounded ``resp.read()``."""
    raw = transport._read_capped(resp)
    _raise_stream_http_error(resp, raw)


def _graph_changes_path(type: str | None, tenant_id: str | None) -> str:  # noqa: A002
    """Build ``/v1/graph/changes`` with only the supplied filters."""
    from urllib.parse import urlencode

    params = {k: v for k, v in (("type", type), ("tenant_id", tenant_id)) if v}
    return "/v1/graph/changes" + (f"?{urlencode(params)}" if params else "")


def _raise_stream_http_error(resp: Any, raw: bytes) -> None:  # noqa: ANN401
    from relata._http import _classify_error, _parse_retry_after

    text = raw.decode("utf-8", errors="replace")
    try:
        body: dict[str, Any] = json.loads(raw)
        if not isinstance(body, dict):
            raise ValueError("error response is not an object")
    except (ValueError, UnicodeError):
        body = {"error": text}
    raise _classify_error(
        resp.status_code, body, request_id=resp.headers.get("x-request-id"),
        retry_after=_parse_retry_after(resp.headers.get("retry-after")),
        content_type=resp.headers.get("content-type", ""), raw_text=text,
    )


async def _aclassify_capped_error(transport: Any, resp: Any) -> None:
    """Async twin of :func:`_classify_capped_error`."""
    raw = await transport._read_capped(resp)
    _raise_stream_http_error(resp, raw)


class StreamingClient:
    """Synchronous streaming client — row streams + SSE consumers."""

    def __init__(self, client: RelataClient) -> None:
        self._client = client

    @classmethod
    def from_client(cls, client: RelataClient) -> StreamingClient:
        return cls(client)

    def _purpose(self, purpose: str | None) -> str:
        eff = purpose or self._client._default_purpose  # noqa: SLF001
        if not eff:
            from relata.exceptions import PurposeError

            raise PurposeError("Streaming queries require a purpose.")
        return eff

    # ------------------------------------------------------------------
    # Row streaming (NDJSON)
    # ------------------------------------------------------------------

    def query_rows(
        self,
        sql: str,
        *,
        purpose: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream query results as NDJSON rows. Yields one ``dict`` per row.

        Metadata is not yielded as rows. Exhaust the iterator to verify the
        terminal count; an incomplete answer raises ``IncompleteQueryStreamError``
        after any partial rows, without automatic retry. ``max_response_bytes``
        bounds each NDJSON frame, rather than the whole result stream.
        """
        eff_purpose = self._purpose(purpose)
        # Use the sync transport's underlying httpx client for streaming.
        sync = self._client._sync  # noqa: SLF001
        with sync._client.stream(  # noqa: SLF001
            "POST",
            "/v1/query/stream",
            headers={"Accept": "application/x-ndjson"},
            json={"purpose": eff_purpose, "sql": sql},
        ) as resp:
            if not resp.is_success:
                # Materialise the error body for classification.
                _classify_capped_error(sync, resp)
            decoder = QueryRowDecoder(sync._max_response_bytes)  # noqa: SLF001
            try:
                for chunk in resp.iter_bytes():
                    yield from decoder.feed(chunk)
                yield from decoder.finish()
            except (IncompleteQueryStreamError, ResponseTooLargeError):
                raise
            except Exception as error:
                raise decoder.failure("Query stream transport failed before verified completion") from error

    # ------------------------------------------------------------------
    # Arrow IPC stream
    # ------------------------------------------------------------------

    def query_arrow_raw(
        self,
        sql: str,
        *,
        purpose: str | None = None,
    ) -> Iterator[bytes]:
        """Stream the raw Arrow IPC byte stream from ``POST /v1/query/arrow``.

        Yields raw bytes chunks. Wrap with ``pyarrow.ipc.open_stream`` for
        typed record-batch consumption::

            import pyarrow.ipc as ipc
            chunks = client.streaming.query_arrow_raw(sql, purpose="x")
            # Consume the first chunk to open the stream reader, then iterate.
        """
        eff_purpose = self._purpose(purpose)
        sync = self._client._sync  # noqa: SLF001
        with sync._client.stream(  # noqa: SLF001
            "POST",
            "/v1/query/arrow",
            json={"purpose": eff_purpose, "sql": sql},
        ) as resp:
            if not resp.is_success:
                _classify_capped_error(sync, resp)
            for chunk in resp.iter_bytes():
                if chunk:
                    yield chunk

    # ------------------------------------------------------------------
    # SSE consumers (watch / alerts / a2a)
    # ------------------------------------------------------------------

    def _sse_stream(
        self,
        path: str,
        *,
        reconnect: bool = True,
        max_retries: int = 5,
        base_backoff: float = 1.0,
    ) -> Iterator[dict[str, Any]]:
        """Generic SSE consumer with optional exponential-backoff reconnect.

        Parses the ``event: <type>\\ndata: <json>`` framing per the SSE spec.
        Yields one parsed event dict per server-sent event.
        """
        sync = self._client._sync  # noqa: SLF001
        retries = 0
        while True:
            try:
                with sync._client.stream("GET", path) as resp:  # noqa: SLF001
                    if not resp.is_success:
                        _classify_capped_error(sync, resp)
                    event_type: str | None = None
                    data_lines: list[str] = []
                    for line in resp.iter_lines():
                        if line.startswith("event:"):
                            event_type = line[6:].strip()
                        elif line.startswith("data:"):
                            data_lines.append(line[5:].strip())
                        elif line == "" and data_lines:
                            # Event boundary — emit.
                            raw = "\n".join(data_lines)
                            try:
                                payload = json.loads(raw)
                            except json.JSONDecodeError:
                                payload = {"raw": raw}
                            if event_type:
                                payload["event"] = event_type
                            yield payload
                            event_type = None
                            data_lines = []
                    # Stream ended without error — if reconnect is on, retry.
                    if not reconnect:
                        return
                    retries += 1
                    if retries > max_retries:
                        return
                    time.sleep(base_backoff * (2 ** min(retries - 1, 5)))
            except (ConnectionError, OSError):
                if not reconnect or retries >= max_retries:
                    return
                retries += 1
                time.sleep(base_backoff * (2 ** min(retries - 1, 5)))

    def watch(
        self,
        sql: str,
        *,
        purpose: str,
        reconnect: bool = True,
    ) -> Iterator[dict[str, Any]]:
        """SSE poll consumer for ``GET /v1/watch/stream``.

        The server emits ``result_changed`` hashes, ``gap``, and
        ``evaluation_failed`` notifications; it does not stream changed row
        contents. Reconnects poll the same consumptive queue for this
        authenticated principal, so concurrent polls with the same principal
        share delivery. Registrations are in-memory and must be re-established
        after a server restart."""
        from urllib.parse import urlencode

        params = urlencode({"sql": sql, "purpose": purpose})
        yield from self._sse_stream(f"/v1/watch/stream?{params}", reconnect=reconnect)

    def watch_stream(
        self,
        type: str,  # noqa: A002
        *,
        since: str | None = None,
        reconnect: bool = True,
    ) -> Iterator[dict[str, Any]]:
        """Convenience wrapper building a SELECT for ``watch(sql, purpose)``.

        The current server emits query-result notifications, not row-level
        ``RowsAppended`` changes. ``since`` adds a ``system_from`` predicate
        to the generated SELECT; it is not a durable resume cursor and cannot
        resume a watch after a gap or server restart.

        Args:
            type: Object type to watch (e.g. ``"HmDispute"``).
            since: Optional system_from cursor; only events after this timestamp.
            reconnect: Reconnect on disconnect with exponential back-off.
        """
        from urllib.parse import urlencode

        sql = _watch_stream_sql(type, since)
        params = urlencode({"sql": sql, "purpose": "watch"})
        return self._sse_stream(f"/v1/watch/stream?{params}", reconnect=reconnect)

    def alerts(self, *, reconnect: bool = True) -> Iterator[dict[str, Any]]:
        """SSE consumer for ``GET /v1/alerts/stream``. Yields alert events
        addressed to the caller's principal / tenant."""
        yield from self._sse_stream("/v1/alerts/stream", reconnect=reconnect)

    def graph_changes(
        self,
        *,
        type: str | None = None,  # noqa: A002
        tenant_id: str | None = None,
        reconnect: bool = True,
    ) -> Iterator[dict[str, Any]]:
        """SSE consumer for ``GET /v1/graph/changes`` (real-time graph change stream).

        ``type`` narrows to one object type; ``tenant_id`` must match the
        caller's authenticated tenant when supplied.
        """
        yield from self._sse_stream(_graph_changes_path(type, tenant_id), reconnect=reconnect)

    def a2a_task_stream(self, task_id: str, *, reconnect: bool = False) -> Iterator[dict[str, Any]]:
        """SSE consumer for ``GET /v1/a2a/tasks/:id/stream`` (A2A task lifecycle events).

        A task stream is finite, so ``reconnect`` defaults to ``False``.
        """
        yield from self._sse_stream(
            f"/v1/a2a/tasks/{path_segment(task_id)}/stream", reconnect=reconnect
        )


class AsyncStreamingClient:
    """Asynchronous streaming client — async iterators for every surface."""

    def __init__(self, client: RelataClient) -> None:
        self._client = client

    @classmethod
    def from_client(cls, client: RelataClient) -> AsyncStreamingClient:
        return cls(client)

    def _purpose(self, purpose: str | None) -> str:
        eff = purpose or self._client._default_purpose  # noqa: SLF001
        if not eff:
            from relata.exceptions import PurposeError

            raise PurposeError("Streaming queries require a purpose.")
        return eff

    async def query_rows(
        self,
        sql: str,
        *,
        purpose: str | None = None,
    ) -> Any:  # noqa: ANN401
        """Async iterator over NDJSON query rows. Usage::

            stream = AsyncStreamingClient.from_client(client)
            async for row in await stream.query_rows(sql, purpose="x"):
                ...
        """

        eff_purpose = self._purpose(purpose)
        async_transport = self._client._async  # noqa: SLF001

        async def _gen() -> Any:  # noqa: ANN401
            async with async_transport._client.stream(  # noqa: SLF001
                "POST",
                "/v1/query/stream",
                headers={"Accept": "application/x-ndjson"},
                json={"purpose": eff_purpose, "sql": sql},
            ) as resp:
                if not resp.is_success:
                    await _aclassify_capped_error(async_transport, resp)
                decoder = QueryRowDecoder(async_transport._max_response_bytes)  # noqa: SLF001
                try:
                    async for chunk in resp.aiter_bytes():
                        for row in decoder.feed(chunk):
                            yield row
                    for row in decoder.finish():
                        yield row
                except (IncompleteQueryStreamError, ResponseTooLargeError):
                    raise
                except Exception as error:
                    raise decoder.failure("Query stream transport failed before verified completion") from error

        return _gen()

    async def query_arrow_raw(
        self,
        sql: str,
        *,
        purpose: str | None = None,
    ) -> Any:  # noqa: ANN401
        """Async equivalent of :meth:`StreamingClient.query_arrow_raw` — async
        iterator over the raw Arrow IPC byte stream from ``POST /v1/query/arrow``.

        Usage::

            async for chunk in client.streaming.query_arrow_raw(sql, purpose="x"):
                ...
        """
        eff_purpose = self._purpose(purpose)
        async_transport = self._client._async  # noqa: SLF001

        async def _gen() -> Any:  # noqa: ANN401
            async with async_transport._client.stream(  # noqa: SLF001
                "POST",
                "/v1/query/arrow",
                json={"purpose": eff_purpose, "sql": sql},
            ) as resp:
                if not resp.is_success:
                    await _aclassify_capped_error(async_transport, resp)
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        yield chunk

        return _gen()

    async def watch(
        self,
        sql: str,
        *,
        purpose: str,
        reconnect: bool = True,
    ) -> Any:  # noqa: ANN401
        """Async SSE poll consumer for ``GET /v1/watch/stream``.

        The server emits ``result_changed`` hashes, ``gap``, and
        ``evaluation_failed`` notifications; it does not stream changed row
        contents. Reconnects/polls for the same authenticated principal share
        one consumptive queue. Registrations are in-memory and must be
        re-established after a server restart."""
        from urllib.parse import urlencode

        params = urlencode({"sql": sql, "purpose": purpose})
        async_transport = self._client._async  # noqa: SLF001

        async def _gen() -> Any:  # noqa: ANN401
            async with async_transport._client.stream(  # noqa: SLF001
                "GET", f"/v1/watch/stream?{params}"
            ) as resp:
                if not resp.is_success:
                    # The `async with` context closes the response on return;
                    # don't drain the error body (`aread()` was unbounded — a
                    # huge "error" body could OOM the SDK, #3214).
                    return
                event_type: str | None = None
                data_lines: list[str] = []
                async for line in resp.aiter_lines():
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        data_lines.append(line[5:].strip())
                    elif line == "" and data_lines:
                        raw = "\n".join(data_lines)
                        try:
                            payload = json.loads(raw)
                        except json.JSONDecodeError:
                            payload = {"raw": raw}
                        if event_type:
                            payload["event"] = event_type
                        yield payload
                        event_type = None
                        data_lines = []

        return _gen()

    async def watch_stream(
        self,
        type: str,  # noqa: A002
        *,
        since: str | None = None,
    ) -> Any:  # noqa: ANN401
        """Convenience wrapper building a SELECT for ``watch(sql, purpose)``.

        The current server emits query-result notifications, not row-level
        ``RowsAppended`` changes. ``since`` adds a ``system_from`` predicate
        to the generated SELECT; it is not a durable resume cursor and cannot
        resume a watch after a gap or server restart.
        """
        from urllib.parse import urlencode

        sql = _watch_stream_sql(type, since)
        params = urlencode({"sql": sql, "purpose": "watch"})
        async_transport = self._client._async  # noqa: SLF001

        async def _gen() -> Any:  # noqa: ANN401
            async with async_transport._client.stream(  # noqa: SLF001
                "GET", f"/v1/watch/stream?{params}"
            ) as resp:
                if not resp.is_success:
                    # The `async with` context closes the response on return;
                    # don't drain the error body (`aread()` was unbounded — a
                    # huge "error" body could OOM the SDK, #3214).
                    return
                event_type: str | None = None
                data_lines: list[str] = []
                async for line in resp.aiter_lines():
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        data_lines.append(line[5:].strip())
                    elif line == "" and data_lines:
                        raw = "\n".join(data_lines)
                        try:
                            payload = json.loads(raw)
                        except json.JSONDecodeError:
                            payload = {"raw": raw}
                        if event_type:
                            payload["event"] = event_type
                        yield payload
                        event_type = None
                        data_lines = []

        return _gen()

    async def alerts(self, *, reconnect: bool = True) -> Any:  # noqa: ANN401
        """Async SSE consumer for ``GET /v1/alerts/stream``."""
        async_transport = self._client._async  # noqa: SLF001

        async def _gen() -> Any:  # noqa: ANN401
            async with async_transport._client.stream(  # noqa: SLF001
                "GET", "/v1/alerts/stream"
            ) as resp:
                if not resp.is_success:
                    # The `async with` context closes the response on return;
                    # don't drain the error body (`aread()` was unbounded — a
                    # huge "error" body could OOM the SDK, #3214).
                    return
                async for line in resp.aiter_lines():
                    if line.startswith("data:"):
                        try:
                            yield json.loads(line[5:].strip())
                        except json.JSONDecodeError:
                            yield {"raw": line[5:].strip()}

        return _gen()

    async def graph_changes(
        self,
        *,
        type: str | None = None,  # noqa: A002
        tenant_id: str | None = None,
    ) -> Any:  # noqa: ANN401
        """Async SSE consumer for ``GET /v1/graph/changes``."""
        return self._sse_events(_graph_changes_path(type, tenant_id))

    async def a2a_task_stream(self, task_id: str) -> Any:  # noqa: ANN401
        """Async SSE consumer for ``GET /v1/a2a/tasks/:id/stream``."""
        return self._sse_events(f"/v1/a2a/tasks/{path_segment(task_id)}/stream")

    def _sse_events(self, path: str) -> Any:  # noqa: ANN401
        async_transport = self._client._async  # noqa: SLF001

        async def _gen() -> Any:  # noqa: ANN401
            async with async_transport._client.stream("GET", path) as resp:  # noqa: SLF001
                if not resp.is_success:
                    return
                event_type: str | None = None
                data_lines: list[str] = []
                async for line in resp.aiter_lines():
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        data_lines.append(line[5:].strip())
                    elif line == "" and data_lines:
                        raw = "\n".join(data_lines)
                        try:
                            payload = json.loads(raw)
                        except json.JSONDecodeError:
                            payload = {"raw": raw}
                        if event_type:
                            payload["event"] = event_type
                        yield payload
                        event_type = None
                        data_lines = []

        return _gen()
