"""Arrow Flight domain client (#2253, #958, #2492).

The server's Arrow Flight gRPC door (``crates/relata-cli/src/flight_server.rs``
``do_get``) runs a SQL query and streams the governed result set back as Arrow
``FlightData`` frames — no JSON intermediate. This module is Python's canonical
Flight surface, mirroring ``sdks/typescript/src/flight.ts`` and
``sdks/go/relata/flight.go``: the ticket/endpoint builders plus a typed
``FlightClient`` / ``AsyncFlightClient`` pair that drives ``pyarrow.flight``.

``RelataClient`` continues to expose the convenience methods
:meth:`~relata.client.RelataClient.query_flight` /
``aquery_flight``; this module is the dedicated domain home (and the surface
the SDK-parity gate keys on), so the two are intentionally aligned 1:1.

Example::

    from relata import RelataClient
    from relata.flight import FlightClient

    with RelataClient("http://localhost:9090") as relata:
        fc = FlightClient.from_client(relata)
        tbl = fc.query_flight("SELECT * FROM Person LIMIT 1000", purpose="analytics")
        df = tbl.to_pandas()

The Flight door is enabled server-side with ``RELATA_FLIGHT_ENABLE=true``
(default port 8815). Requires ``pyarrow`` (``pip install pyarrow``).
"""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from relata.client import RelataClient

#: The server's default Arrow Flight gRPC port (``RELATA_FLIGHT_PORT``),
#: matching ``crates/relata-cli/src/flight_server.rs``. Mirrors
#: ``DefaultFlightPort`` in the Go SDK.
DEFAULT_FLIGHT_PORT = "8815"


def flight_ticket(sql: str, purpose: str | None) -> str:
    """Assemble an Arrow Flight ``do_get`` ticket from SQL + optional PURPOSE.

    The Flight door reads the ticket as raw UTF-8 SQL and extracts
    ``/* PURPOSE '<id>' */`` from a leading SQL comment, defaulting to
    ``"flight_query"`` when absent (#958). With no *purpose* the SQL is
    returned verbatim. Mirrors Go's ``FlightTicket`` / TS's ``buildFlightTicket``.

    An embedded single quote in *purpose* is doubled so the server's
    ``/* PURPOSE '...' */`` scan is not truncated.
    """
    if not purpose:
        return sql
    escaped = purpose.replace("'", "''")
    return f"/* PURPOSE '{escaped}' */ {sql}"


def resolve_flight_endpoint(base_url: str, flight_endpoint: str | None) -> str:
    """Resolve the Arrow Flight gRPC endpoint.

    Defaults to ``grpc://<base_url host>:<port>`` using
    :data:`DEFAULT_FLIGHT_PORT`. Pass *flight_endpoint* to override (e.g. a
    ``grpcs://host:port`` TLS endpoint). Mirrors Go's ``ResolveFlightEndpoint``
    / TS's ``deriveFlightEndpoint``.
    """
    if flight_endpoint:
        return flight_endpoint
    from urllib.parse import urlparse

    host = urlparse(base_url).hostname or "localhost"
    return f"grpc://{host}:{DEFAULT_FLIGHT_PORT}"


def flight_call_headers(
    client: RelataClient, bearer_token: str | None
) -> list[tuple[bytes, bytes]]:
    """Build the Arrow Flight gRPC metadata for a call (#3213).

    Previously the Flight door only threaded ``authorization`` — the
    tenant/acting-as/delegated-by scope configured on the client was silently
    dropped, so on a multi-tenant cluster the server fell back to
    default-tenant resolution (silent cross-tenant exposure). This mirrors the
    HTTP door's ``_build_headers``: it threads the bearer plus the client's
    ``_extra_headers`` bag (tenant / acting-as / delegated-by / caller-supplied
    headers), lowercased for gRPC metadata, and adds a per-request
    ``x-request-id``. Caller-supplied headers win over the SDK defaults.

    ``bearer_token`` overrides the client's own token when not ``None``
    (matching :meth:`FlightClient.query_flight`'s ``bearer_token`` argument).
    """
    bearer = bearer_token if bearer_token is not None else client._bearer_token  # noqa: SLF001
    headers: dict[str, str] = {}
    if bearer:
        headers["authorization"] = f"Bearer {bearer}"
    extra = client._extra_headers  # noqa: SLF001
    if extra:
        for key, value in extra.items():
            headers[key.lower()] = value
    if "x-request-id" not in headers:
        headers["x-request-id"] = str(uuid.uuid4())
    return [(k.encode("utf-8"), v.encode("utf-8")) for k, v in headers.items()]


class FlightClient:
    """Synchronous Arrow Flight client — ``do_get`` → ``pyarrow.Table``."""

    def __init__(self, client: RelataClient) -> None:
        self._client = client

    @classmethod
    def from_client(cls, client: RelataClient) -> FlightClient:
        return cls(client)

    def query_flight(
        self,
        sql: str,
        *,
        flight_endpoint: str | None = None,
        purpose: str | None = None,
        bearer_token: str | None = None,
    ) -> Any:
        """Execute *sql* over Arrow Flight ``do_get`` and return a zero-copy
        ``pyarrow.Table`` (#2253, #958).

        Opens a real ``pyarrow.flight.FlightClient`` to the server's gRPC
        Flight door and runs ``do_get`` with the SQL (optionally prefixed with
        a ``/* PURPOSE '<id>' */`` comment) as the ticket bytes.

        Args:
            sql: SQL query string used verbatim as the Flight ticket.
            flight_endpoint: Flight gRPC endpoint (e.g.
                ``"grpc://localhost:8815"``). Defaults to
                ``grpc://<base_url host>:<port>``.
            purpose: Optional purpose injected as a ``/* PURPOSE '...' */``
                SQL comment. **Not** resolved from the client default — Flight
                has no purposeless-query gate, so pass it explicitly when wanted.
            bearer_token: Bearer token for Flight metadata. Defaults to the
                client's token.

        Returns:
            A ``pyarrow.Table`` with the governed result set.

        Raises:
            ImportError: if ``pyarrow`` is not installed.
        """
        import pyarrow.flight as flight  # type: ignore[import]

        endpoint = resolve_flight_endpoint(self._client._base_url, flight_endpoint)  # noqa: SLF001
        ticket_sql = flight_ticket(sql, purpose)
        headers = flight_call_headers(self._client, bearer_token)

        client = flight.FlightClient(endpoint)
        options = flight.FlightCallOptions(headers=headers)
        reader = client.do_get(flight.Ticket(ticket_sql.encode("utf-8")), options)
        return reader.read_all()


class AsyncFlightClient:
    """Asynchronous Arrow Flight client.

    ``pyarrow.flight`` is synchronous, so each call is dispatched to a worker
    thread via :func:`asyncio.to_thread`.
    """

    def __init__(self, client: RelataClient) -> None:
        self._client = client

    @classmethod
    def from_client(cls, client: RelataClient) -> AsyncFlightClient:
        return cls(client)

    async def query_flight(
        self,
        sql: str,
        *,
        flight_endpoint: str | None = None,
        purpose: str | None = None,
        bearer_token: str | None = None,
    ) -> Any:
        """Async variant of :meth:`FlightClient.query_flight`."""
        sync = FlightClient(self._client)
        return await asyncio.to_thread(
            sync.query_flight,
            sql,
            flight_endpoint=flight_endpoint,
            purpose=purpose,
            bearer_token=bearer_token,
        )
