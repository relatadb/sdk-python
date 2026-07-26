"""Streaming SDK (#75) — SSE consumers + NDJSON row streams + Arrow IPC.

The server ships streaming surfaces at:

- ``POST /query/arrow`` → Arrow IPC stream (columnar).
- ``POST /query`` with ``Accept: text/event-stream`` → NDJSON row stream.
- ``GET /watch/stream`` → SSE — pushes ``RowsAppended`` events on commits.
- ``GET /alerts/stream`` → SSE — pushes alerts to subscribed principals.
- ``GET /a2a/tasks/:id/stream`` → SSE — A2A task lifecycle events.

This module wraps each as a Python iterator (sync) or async iterator (async),
with disconnect/reconnect backoff for SSE.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from relata.client import RelataClient


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

        Uses ``httpx.Client.stream()`` so the response body is consumed
        lazily — the server does not buffer the entire result.
        """
        eff_purpose = self._purpose(purpose)
        # Use the sync transport's underlying httpx client for streaming.
        sync = self._client._sync  # noqa: SLF001
        with sync._client.stream(  # noqa: SLF001
            "POST",
            "/query",
            json={"purpose": eff_purpose, "sql": sql},
        ) as resp:
            if not resp.is_success:
                # Materialise the error body for classification.
                resp.read()
                from relata._http import _classify_error

                try:
                    body = resp.json()
                except Exception:
                    body = {"error": resp.text}
                raise _classify_error(resp.status_code, body)
            for line in resp.iter_lines():
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    # ------------------------------------------------------------------
    # Arrow IPC stream
    # ------------------------------------------------------------------

    def query_arrow_raw(
        self,
        sql: str,
        *,
        purpose: str | None = None,
    ) -> Iterator[bytes]:
        """Stream the raw Arrow IPC byte stream from ``POST /query/arrow``.

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
            "/query/arrow",
            json={"purpose": eff_purpose, "sql": sql},
        ) as resp:
            if not resp.is_success:
                resp.read()
                from relata._http import _classify_error

                try:
                    body = resp.json()
                except Exception:
                    body = {"error": resp.text}
                raise _classify_error(resp.status_code, body)
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
                        resp.read()
                        from relata._http import _classify_error

                        try:
                            body = resp.json()
                        except Exception:
                            body = {"error": resp.text}
                        raise _classify_error(resp.status_code, body)
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
        """SSE consumer for ``GET /watch/stream``. Yields ``RowsAppended``
        events as commit-committing changes match the watch SQL."""
        from urllib.parse import urlencode

        params = urlencode({"sql": sql, "purpose": purpose})
        yield from self._sse_stream(f"/watch/stream?{params}", reconnect=reconnect)

    def watch_stream(
        self,
        type: str,  # noqa: A002
        *,
        since: str | None = None,
        reconnect: bool = True,
    ) -> Iterator[dict[str, Any]]:
        """SSE consumer for ``GET /watch/stream`` filtered to an object type (#1747).

        Yields ``RowsAppended`` events whenever rows of *type* are committed.
        Use *since* (decimal ``system_from`` ns) for incremental consumption.

        Args:
            type: Object type to watch (e.g. ``"HmDispute"``).
            since: Optional system_from cursor; only events after this timestamp.
            reconnect: Reconnect on disconnect with exponential back-off.
        """
        from urllib.parse import urlencode

        sql = f"SELECT * FROM {type}"
        if since:
            sql += f" WHERE system_from > {since}"
        params = urlencode({"sql": sql, "purpose": "watch"})
        yield from self._sse_stream(f"/watch/stream?{params}", reconnect=reconnect)

    def alerts(self, *, reconnect: bool = True) -> Iterator[dict[str, Any]]:
        """SSE consumer for ``GET /alerts/stream``. Yields alert events
        addressed to the caller's principal / tenant."""
        yield from self._sse_stream("/alerts/stream", reconnect=reconnect)


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

            async for row in client.streaming.aquery_rows(sql, purpose="x"):
                ...
        """

        eff_purpose = self._purpose(purpose)
        async_transport = self._client._async  # noqa: SLF001

        async def _gen() -> Any:  # noqa: ANN401
            async with async_transport._client.stream(  # noqa: SLF001
                "POST",
                "/query",
                json={"purpose": eff_purpose, "sql": sql},
            ) as resp:
                if not resp.is_success:
                    await resp.aread()
                    from relata._http import _classify_error

                    try:
                        body = resp.json()
                    except Exception:
                        body = {"error": resp.text}
                    raise _classify_error(resp.status_code, body)
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    yield json.loads(line)

        return _gen()

    async def watch(
        self,
        sql: str,
        *,
        purpose: str,
        reconnect: bool = True,
    ) -> Any:  # noqa: ANN401
        """Async SSE consumer for ``GET /watch/stream``."""
        from urllib.parse import urlencode

        params = urlencode({"sql": sql, "purpose": purpose})
        async_transport = self._client._async  # noqa: SLF001

        async def _gen() -> Any:  # noqa: ANN401
            async with async_transport._client.stream(  # noqa: SLF001
                "GET", f"/watch/stream?{params}"
            ) as resp:
                if not resp.is_success:
                    await resp.aread()
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
        """Async SSE consumer for ``GET /watch/stream`` filtered to an object type (#1747).

        Yields ``RowsAppended`` events for *type*. Use *since* for incremental reads.
        """
        from urllib.parse import urlencode

        sql = f"SELECT * FROM {type}"
        if since:
            sql += f" WHERE system_from > {since}"
        params = urlencode({"sql": sql, "purpose": "watch"})
        async_transport = self._client._async  # noqa: SLF001

        async def _gen() -> Any:  # noqa: ANN401
            async with async_transport._client.stream(  # noqa: SLF001
                "GET", f"/watch/stream?{params}"
            ) as resp:
                if not resp.is_success:
                    await resp.aread()
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
        """Async SSE consumer for ``GET /alerts/stream``."""
        async_transport = self._client._async  # noqa: SLF001

        async def _gen() -> Any:  # noqa: ANN401
            async with async_transport._client.stream(  # noqa: SLF001
                "GET", "/alerts/stream"
            ) as resp:
                if not resp.is_success:
                    await resp.aread()
                    return
                async for line in resp.aiter_lines():
                    if line.startswith("data:"):
                        try:
                            yield json.loads(line[5:].strip())
                        except json.JSONDecodeError:
                            yield {"raw": line[5:].strip()}

        return _gen()
