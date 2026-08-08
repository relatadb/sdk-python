"""First-party plain gRPC query client with Arrow-IPC ``RowBatch`` opt-in
(#4090 round 3 — Python SDK).

Round 1 (PR #4095, TypeScript's ``grpc-query.ts``) and round 2 (PR #4102,
Go's ``grpc_query.go``) built the same capability for their SDKs: PR #4086
(epic #3879 Wave 6, #3912/#4020 item 2) landed the server-side wire encoding
for a negotiated Arrow-IPC row encoding on the plain gRPC ``RelataQuery``
service (``proto/relata.proto``'s ``QueryRequest.wants_arrow_ipc_rows`` /
``QueryResponse.arrow_ipc_rows`` / ``RowBatch.arrow_ipc_rows``,
``crates/relata-server/src/grpc.rs::encode_rows_arrow_ipc_or_json``),
mirroring ADR-0255's own ``ShardQueryRequest``/``ShardQueryResponse``
negotiated-capability pattern. Both prior rounds investigated Python and
found it has **zero** gRPC scaffolding — no ``grpcio`` dependency, no
generated ``pb2`` stubs, only ``pyarrow.flight`` (:mod:`relata.flight`,
``FlightClient``/``AsyncFlightClient``), which only exposes Arrow Flight's
own ``do_get``/``do_put``/``do_action`` RPCs — there is no generic unary-call
surface on ``pyarrow.flight.FlightClient`` to repurpose for an unrelated gRPC
service like plain ``RelataQuery``. So this module is Python's first gRPC
scaffolding, purpose-built for this capability rather than reusing Flight's.

``grpcio`` is declared as a **real, versioned optional extra**
(``pip install relata-sdk[grpc]``) in ``pyproject.toml``, mirroring how
``langgraph`` is optional (see ``relata_langgraph/__init__.py``) — unlike
``pyarrow`` itself, which stays an *undeclared*, purely lazy-imported
optional dependency across the rest of this SDK (:mod:`relata.flight`,
``RelataClient.query_arrow``). TypeScript's ``@grpc/grpc-js`` is the closest
analogue: also a declared optional peer dependency.

Unlike TypeScript (needs ``grpc.makeGenericClientConstructor`` + a
pass-through serializer pair) and Go (needs ``grpc.ForceCodec`` because
``ClientConn.Invoke`` demands a registered ``encoding.Codec`` even for raw
bytes), Python's ``grpc.Channel.unary_unary(method)`` natively accepts and
returns raw ``bytes`` when constructed with no
``request_serializer``/``response_deserializer`` (grpcio's own contract: the
caller is responsible for wire bytes when neither is supplied) — no codec
workaround needed, the simplest of the three SDK implementations.

``QueryRequest``/``QueryResponse`` are simple enough (every field a scalar or
a ``repeated string``/``bytes`` — no nested messages, no maps, no oneofs)
that the same hand-rolled protobuf wire codec pattern proven twice already
(``grpc-query.ts``, ``grpc_query.go``) is reimplemented here in pure Python —
no ``protoc``/``grpcio-tools`` code generation, no bundled ``.proto`` asset.

Both ``RelataQuery`` RPCs that make sense for a ``wants_arrow_ipc_rows``
caller are wired: the unary ``Execute``
(:meth:`GrpcQueryClient.query_grpc`) and the server-streaming
``ExecuteStream`` (:meth:`GrpcQueryClient.query_grpc_stream`, #4090 round 7 —
mirroring TypeScript's ``GrpcQueryTransport.executeStream``, PR #4228).
``ExecuteStream`` negotiates the Arrow-IPC encoding **per streamed frame**
(``RowBatch.arrow_ipc_rows``), and ``RowBatch``'s field layout differs from
``QueryResponse``'s — see :func:`decode_row_batch`. Both methods normalise to
the same :class:`~relata.models.QueryResult` shape regardless of which wire
encoding the server chose.

Example::

    from relata import RelataClient

    with RelataClient("http://localhost:9090", purpose="analytics") as client:
        result = client.query_grpc("SELECT * FROM Person LIMIT 1000")
        for row in result.rows:
            print(row)

        # Large result set — same result shape, streamed RowBatch frames.
        streamed = client.query_grpc_stream("SELECT * FROM Person")
        print(streamed.row_count)

Requires ``grpcio`` (``pip install relata-sdk[grpc]``, or ``pip install
grpcio`` directly).
"""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from relata.models import QueryResult

if TYPE_CHECKING:
    from relata.client import RelataClient

#: The server's default plain gRPC port (``RELATA_GRPC_PORT``), matching
#: ``crates/relata-cli/src/main.rs``. Distinct from
#: :data:`relata.flight.DEFAULT_FLIGHT_PORT` — the plain ``RelataQuery``
#: service and the Arrow Flight door are two different gRPC endpoints.
DEFAULT_GRPC_QUERY_PORT = "50051"

#: Fully-qualified gRPC method path for the unary ``RelataQuery.Execute`` RPC
#: (``proto/relata.proto``).
_EXECUTE_METHOD = "/relata.RelataQuery/Execute"

#: Fully-qualified gRPC method path for the server-streaming
#: ``RelataQuery.ExecuteStream`` RPC (``proto/relata.proto``), which answers
#: with a stream of ``RowBatch`` frames instead of one ``QueryResponse``.
_EXECUTE_STREAM_METHOD = "/relata.RelataQuery/ExecuteStream"


def resolve_grpc_query_endpoint(base_url: str, grpc_endpoint: str | None) -> str:
    """Resolve the plain gRPC ``RelataQuery`` endpoint.

    Defaults to ``grpc://<base_url host>:50051`` using
    :data:`DEFAULT_GRPC_QUERY_PORT`. Pass ``grpc_endpoint`` to override (e.g.
    a ``grpcs://host:port`` TLS endpoint). Mirrors TypeScript's
    ``deriveGrpcQueryEndpoint`` (always defaults to the plaintext ``grpc://``
    scheme regardless of ``base_url``'s own scheme — same convention here;
    Go's ``ResolveGrpcQueryEndpoint`` additionally infers ``grpcs://`` from an
    ``https://`` base URL, a divergence carried over unchanged from round 2).
    """
    if grpc_endpoint:
        return grpc_endpoint
    host = urlparse(base_url).hostname or "localhost"
    return f"grpc://{host}:{DEFAULT_GRPC_QUERY_PORT}"


def is_secure_grpc_query_endpoint(endpoint: str) -> bool:
    """Report whether *endpoint* requests a TLS-protected gRPC channel.

    ``grpcs://`` / ``tls://`` select TLS; ``grpc://`` (and the legacy bare
    ``http://`` form) stay plaintext. Mirrors ``isSecureGrpcQueryEndpoint``
    (TS) / the scheme check inside ``ResolveGrpcQueryEndpoint`` (Go).
    """
    return endpoint.startswith("grpcs://") or endpoint.startswith("tls://")


def strip_grpc_query_scheme(endpoint: str) -> str:
    """Strip the scheme prefix, leaving the bare ``host:port`` grpcio's
    ``Channel`` constructors want. Mirrors ``stripGrpcQueryScheme`` (TS) /
    ``stripFlightScheme`` reuse (Go).
    """
    for prefix in ("grpcs://", "grpc://", "tls://", "http://"):
        if endpoint.startswith(prefix):
            return endpoint[len(prefix) :]
    return endpoint


# ---------------------------------------------------------------------------
# Minimal protobuf varint + wire helpers (mirrors grpc-query.ts's / grpc_query.go's
# local codec — kept self-contained here rather than shared, same as every
# first-party SDK reimplementing its own minimal framing locally).
# ---------------------------------------------------------------------------


def _encode_varint(n: int) -> bytes:
    """Encode a non-negative integer as a base-128 varint (LEB128)."""
    if n < 0:
        raise ValueError(f"varint must be a non-negative integer, got {n}")
    out = bytearray()
    v = n
    while v > 0x7F:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    out.append(v & 0x7F)
    return bytes(out)


def _read_varint(buf: bytes, offset: int) -> tuple[int, int]:
    """Decode a base-128 varint at *offset*. Returns ``(value, next_offset)``."""
    result = 0
    shift = 0
    i = offset
    n = len(buf)
    while i < n:
        byte = buf[i]
        i += 1
        result |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            return result, i
        shift += 7
        if shift > 63:
            raise ValueError("protobuf varint overflow — value exceeds 64-bit range")
    raise ValueError("truncated protobuf varint — short gRPC query frame")


def _write_string_field(out: bytearray, field: int, s: str) -> None:
    """Append a length-delimited string field, unless *s* is proto3's default
    empty string (proto3 elides default values on the wire)."""
    if not s:
        return
    payload = s.encode("utf-8")
    out.append((field << 3) | 2)
    out.extend(_encode_varint(len(payload)))
    out.extend(payload)


def encode_query_request(
    *,
    purpose: str = "",
    sql: str,
    bearer_token: str = "",
    wants_arrow_ipc_rows: bool = False,
) -> bytes:
    """Encode a ``QueryRequest`` (``proto/relata.proto``) — ``purpose`` (1),
    ``sql`` (2), ``bearer_token`` (3, unused server-side today — auth travels
    as gRPC ``authorization`` metadata, same as :class:`~relata.flight.FlightClient`,
    see ``crates/relata-server/src/grpc.rs::check_bearer_token``),
    ``wants_arrow_ipc_rows`` (4). Proto3 default values (``""``, ``False``)
    are omitted, matching idiomatic proto3 wire output.
    """
    out = bytearray()
    _write_string_field(out, 1, purpose)
    _write_string_field(out, 2, sql)
    _write_string_field(out, 3, bearer_token)
    if wants_arrow_ipc_rows:
        out.append((4 << 3) | 0)
        out.extend(_encode_varint(1))
    return bytes(out)


class DecodedQueryResponse:
    """Decoded ``QueryResponse`` (``proto/relata.proto``).

    ``arrow_ipc_rows`` empty means "read ``rows_json`` instead" per the
    proto's documented fallback contract — never "zero rows."
    """

    __slots__ = ("arrow_ipc_rows", "columns", "row_count", "rows_json")

    def __init__(self) -> None:
        self.columns: list[str] = []
        self.row_count: int = 0
        self.rows_json: list[str] = []
        self.arrow_ipc_rows: bytes = b""


def decode_query_response(buf: bytes) -> DecodedQueryResponse:
    """Decode a ``QueryResponse`` protobuf: ``columns`` (1, repeated string),
    ``row_count`` (2, int64 varint), ``rows_json`` (3, repeated string),
    ``arrow_ipc_rows`` (4, bytes). Unknown wire types are skipped where the
    length is knowable, so the decoder stays forward-compatible with future
    response fields — mirrors ``decodeQueryResponse`` (TS) / Go.
    """
    out = DecodedQueryResponse()
    i = 0
    n = len(buf)
    while i < n:
        tag, i = _read_varint(buf, i)
        field = tag >> 3
        wire_type = tag & 0x7
        if wire_type == 0:  # varint
            value, i = _read_varint(buf, i)
            if field == 2:
                out.row_count = value
        elif wire_type == 2:  # length-delimited
            length, i = _read_varint(buf, i)
            end = i + length
            if end > n:
                raise ValueError(
                    "relata grpc query: truncated protobuf payload in QueryResponse"
                )
            seg = buf[i:end]
            i = end
            if field == 1:
                out.columns.append(seg.decode("utf-8"))
            elif field == 3:
                out.rows_json.append(seg.decode("utf-8"))
            elif field == 4:
                out.arrow_ipc_rows = seg
        elif wire_type == 1:  # fixed64 (unused by QueryResponse today)
            if i + 8 > n:
                raise ValueError("relata grpc query: truncated fixed64 in QueryResponse")
            i += 8
        elif wire_type == 5:  # fixed32 (unused by QueryResponse today)
            if i + 4 > n:
                raise ValueError("relata grpc query: truncated fixed32 in QueryResponse")
            i += 4
        else:
            raise ValueError(
                f"relata grpc query: unsupported protobuf wire type {wire_type} "
                "in QueryResponse"
            )
    return out


class DecodedRowBatch:
    """Decoded ``RowBatch`` (``proto/relata.proto``) — one frame of a streamed
    ``RelataQuery.ExecuteStream`` response.

    The field layout is **not** the same as :class:`DecodedQueryResponse`:
    ``columns`` (1, repeated string), ``rows_json`` (2, repeated string —
    field 2 here, field 3 on ``QueryResponse``), ``done`` (3, bool varint —
    ``QueryResponse`` has no equivalent), ``arrow_ipc_rows`` (4, bytes, same
    field number and fallback contract as ``QueryResponse``). ``RowBatch``
    carries no ``row_count`` — the caller totals the frames it received.
    """

    __slots__ = ("arrow_ipc_rows", "columns", "done", "rows_json")

    def __init__(self) -> None:
        self.columns: list[str] = []
        self.rows_json: list[str] = []
        self.done: bool = False
        self.arrow_ipc_rows: bytes = b""


def decode_row_batch(buf: bytes) -> DecodedRowBatch:
    """Decode a single ``RowBatch`` protobuf frame (see
    :class:`DecodedRowBatch` for the field layout). Unknown wire types are
    skipped where the length is knowable, so the decoder stays
    forward-compatible with future frame fields — mirrors
    :func:`decode_query_response` and ``decodeRowBatch`` (TS).
    """
    out = DecodedRowBatch()
    i = 0
    n = len(buf)
    while i < n:
        tag, i = _read_varint(buf, i)
        field = tag >> 3
        wire_type = tag & 0x7
        if wire_type == 0:  # varint
            value, i = _read_varint(buf, i)
            if field == 3:
                out.done = value != 0
        elif wire_type == 2:  # length-delimited
            length, i = _read_varint(buf, i)
            end = i + length
            if end > n:
                raise ValueError("relata grpc query: truncated protobuf payload in RowBatch")
            seg = buf[i:end]
            i = end
            if field == 1:
                out.columns.append(seg.decode("utf-8"))
            elif field == 2:
                out.rows_json.append(seg.decode("utf-8"))
            elif field == 4:
                out.arrow_ipc_rows = seg
        elif wire_type == 1:  # fixed64 (unused by RowBatch today)
            if i + 8 > n:
                raise ValueError("relata grpc query: truncated fixed64 in RowBatch")
            i += 8
        elif wire_type == 5:  # fixed32 (unused by RowBatch today)
            if i + 4 > n:
                raise ValueError("relata grpc query: truncated fixed32 in RowBatch")
            i += 4
        else:
            raise ValueError(
                f"relata grpc query: unsupported protobuf wire type {wire_type} in RowBatch"
            )
    return out


def _arrow_ipc_rows_to_objects(data: bytes) -> list[dict[str, Any]]:
    """Decode a self-contained Arrow IPC stream (schema + one RecordBatch,
    per ``QueryResponse.arrow_ipc_rows``'s documented shape,
    ``crates/relata-cluster/src/transport.rs::encode_shard_rows_arrow_ipc``)
    into plain row dicts — the same shape ``json.loads(rows_json[i])`` would
    produce, so callers never need to know which wire encoding the server
    chose. Requires ``pyarrow`` (lazy import — already an
    undeclared/optional dependency of the rest of this SDK, see
    :mod:`relata.flight`).

    Raises:
        ImportError: if ``pyarrow`` is not installed.
    """
    import io

    import pyarrow.ipc as ipc  # type: ignore[import]

    reader = ipc.open_stream(io.BytesIO(data))
    table = reader.read_all()
    return table.to_pylist()


def _grpc_query_metadata(
    client: RelataClient, bearer_token: str | None
) -> list[tuple[str, str]]:
    """Build the gRPC call metadata for a ``RelataQuery.Execute`` call
    (#3213-equivalent): threads the bearer plus the client's tenant /
    acting-as / delegated-by / caller-supplied headers, lowercased (gRPC
    metadata keys are always lowercase ASCII), plus a per-request
    ``x-request-id``. Caller-supplied headers win over the SDK defaults.

    Deliberately its own str-valued implementation rather than reusing
    :func:`relata.flight.flight_call_headers` — that helper returns
    ``bytes``-encoded ``(key, value)`` pairs for ``pyarrow.flight``'s
    ``FlightCallOptions(headers=...)`` contract, whereas plain grpcio's
    ``metadata=`` argument requires ``str`` values for any non-``-bin`` key
    (a ``bytes`` value there raises at call time). Same field set, different
    wire-adjacent encoding — same divergence TS/Go each made in their own
    idiom (``grpc.Metadata().set()`` vs. ``metadata.AppendToOutgoingContext``).

    ``bearer_token`` overrides the client's own token when not ``None``
    (matching :meth:`GrpcQueryClient.query_grpc`'s ``bearer_token`` argument).
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
    return list(headers.items())


def _open_grpc_channel(endpoint: str) -> Any:
    """Open a grpcio ``Channel`` to *endpoint*, TLS-protected when the scheme
    asks for it (:func:`is_secure_grpc_query_endpoint`). Shared by the unary
    and streaming call paths.

    Raises:
        ImportError: if ``grpcio`` is not installed.
    """
    import grpc  # type: ignore[import]

    target = strip_grpc_query_scheme(endpoint)
    if is_secure_grpc_query_endpoint(endpoint):
        return grpc.secure_channel(target, grpc.ssl_channel_credentials())
    return grpc.insecure_channel(target)


def _decode_wire_rows(arrow_ipc_rows: bytes, rows_json: list[str]) -> list[dict[str, Any]]:
    """Decode whichever wire encoding the server chose into plain row dicts.

    Non-empty ``arrow_ipc_rows`` wins; empty means "read ``rows_json``
    instead" per the proto's documented fallback contract — never "zero
    rows." Applies identically to a ``QueryResponse`` and to one streamed
    ``RowBatch`` frame (both negotiate the same way, the frame just does it
    per-batch).
    """
    if arrow_ipc_rows:
        return _arrow_ipc_rows_to_objects(arrow_ipc_rows)
    return [json.loads(s) for s in rows_json]


class GrpcQueryClient:
    """Synchronous plain gRPC query client — ``Execute`` / ``ExecuteStream`` →
    :class:`~relata.models.QueryResult`."""

    def __init__(self, client: RelataClient) -> None:
        self._client = client

    @classmethod
    def from_client(cls, client: RelataClient) -> GrpcQueryClient:
        return cls(client)

    def query_grpc(
        self,
        sql: str,
        *,
        grpc_endpoint: str | None = None,
        purpose: str | None = None,
        bearer_token: str | None = None,
        timeout: float = 30.0,
    ) -> QueryResult:
        """Execute *sql* over the plain gRPC ``RelataQuery/Execute`` unary
        RPC, opting into ``QueryRequest.wants_arrow_ipc_rows`` (#4090).

        Opens a real ``grpc.Channel`` to the server's plain gRPC door and
        invokes ``Execute`` with a hand-rolled ``QueryRequest`` encoding.
        Transparently decodes whichever wire shape the server answered with
        (``arrow_ipc_rows`` via ``pyarrow.ipc``, or ``rows_json`` fallback)
        into the same :class:`~relata.models.QueryResult` shape
        :meth:`~relata.client.RelataClient.query` returns.

        Args:
            sql: SQL query string.
            grpc_endpoint: Plain gRPC endpoint (e.g.
                ``"grpc://localhost:50051"``). Defaults to
                ``grpc://<base_url host>:50051``.
            purpose: Optional query purpose (``QueryRequest.purpose``).
                Falls back to the client's default purpose when unset (unlike
                :meth:`~relata.flight.FlightClient.query_flight`, which has no
                default-purpose fallback — the gRPC ``QueryRequest`` has a
                dedicated ``purpose`` field to populate instead of embedding
                it in a SQL comment).
            bearer_token: Bearer token for gRPC metadata. Defaults to the
                client's token.
            timeout: Per-call RPC deadline in seconds (default 30).

        Returns:
            A :class:`~relata.models.QueryResult`.

        Raises:
            ImportError: if ``grpcio`` is not installed (install with
                ``pip install relata-sdk[grpc]``).

        Example::

            result = client.query_grpc(
                "SELECT * FROM Person LIMIT 1000", purpose="analytics",
            )
        """
        endpoint = resolve_grpc_query_endpoint(self._client._base_url, grpc_endpoint)  # noqa: SLF001
        channel = _open_grpc_channel(endpoint)
        try:
            request_bytes = self._encode_request(sql, purpose)
            metadata = _grpc_query_metadata(self._client, bearer_token)
            call = channel.unary_unary(_EXECUTE_METHOD)
            response_bytes = call(request_bytes, timeout=timeout, metadata=metadata)
            decoded = decode_query_response(response_bytes)
        finally:
            channel.close()

        rows = _decode_wire_rows(decoded.arrow_ipc_rows, decoded.rows_json)
        return QueryResult(rows=rows, row_count=decoded.row_count, columns=decoded.columns)

    def query_grpc_stream(
        self,
        sql: str,
        *,
        grpc_endpoint: str | None = None,
        purpose: str | None = None,
        bearer_token: str | None = None,
        timeout: float = 30.0,
    ) -> QueryResult:
        """Execute *sql* over the server-streaming ``RelataQuery/ExecuteStream``
        RPC, opting into ``QueryRequest.wants_arrow_ipc_rows`` (#4090).

        The streaming counterpart to :meth:`query_grpc`: every streamed
        ``RowBatch`` frame independently negotiates the Arrow-IPC encoding
        (``RowBatch.arrow_ipc_rows``, same empty-means-fall-back-to-``rows_json``
        contract as ``QueryResponse``), so a large result set never has to be
        buffered as one response message server-side. Frames are collected
        into the same :class:`~relata.models.QueryResult` shape
        :meth:`query_grpc` returns — callers don't need a different
        consumption model for a large result set versus a small one.

        ``RowBatch`` carries no ``row_count`` field, so the returned
        ``row_count`` is the number of rows actually received (unlike
        :meth:`query_grpc`, which reports the server's own
        ``QueryResponse.row_count``).

        Args:
            sql: SQL query string.
            grpc_endpoint: Plain gRPC endpoint (e.g.
                ``"grpc://localhost:50051"``). Defaults to
                ``grpc://<base_url host>:50051``.
            purpose: Optional query purpose (``QueryRequest.purpose``). Falls
                back to the client's default purpose when unset.
            bearer_token: Bearer token for gRPC metadata. Defaults to the
                client's token.
            timeout: Per-call RPC deadline in seconds (default 30), covering
                the whole stream.

        Returns:
            A :class:`~relata.models.QueryResult` holding every streamed row.

        Raises:
            ImportError: if ``grpcio`` is not installed (install with
                ``pip install relata-sdk[grpc]``).

        Example::

            result = client.query_grpc_stream("SELECT * FROM Person")
        """
        endpoint = resolve_grpc_query_endpoint(self._client._base_url, grpc_endpoint)  # noqa: SLF001
        channel = _open_grpc_channel(endpoint)
        columns: list[str] = []
        rows: list[dict[str, Any]] = []
        try:
            request_bytes = self._encode_request(sql, purpose)
            metadata = _grpc_query_metadata(self._client, bearer_token)
            call = channel.unary_stream(_EXECUTE_STREAM_METHOD)
            for frame_bytes in call(request_bytes, timeout=timeout, metadata=metadata):
                frame = decode_row_batch(frame_bytes)
                if frame.columns:
                    columns = frame.columns
                if frame.arrow_ipc_rows or frame.rows_json:
                    rows.extend(_decode_wire_rows(frame.arrow_ipc_rows, frame.rows_json))
                if frame.done:
                    # Terminal frame per the server's one-batch-lookahead
                    # contract; stop reading rather than waiting on the
                    # iterator's own exhaustion.
                    break
        finally:
            channel.close()

        return QueryResult(rows=rows, row_count=len(rows), columns=columns)

    def _encode_request(self, sql: str, purpose: str | None) -> bytes:
        """Encode the shared ``QueryRequest`` both RPCs send, always opting
        into ``wants_arrow_ipc_rows``. Falls back to the client's default
        purpose when *purpose* is ``None``."""
        effective_purpose = purpose if purpose is not None else self._client._default_purpose  # noqa: SLF001
        return encode_query_request(
            purpose=effective_purpose or "",
            sql=sql,
            wants_arrow_ipc_rows=True,
        )


class AsyncGrpcQueryClient:
    """Asynchronous plain gRPC query client.

    grpcio's synchronous ``Channel`` API is used here (not ``grpc.aio``) so
    both clients share one hand-rolled call path — same rationale as
    :class:`relata.flight.AsyncFlightClient` dispatching to a worker thread
    via :func:`asyncio.to_thread`.
    """

    def __init__(self, client: RelataClient) -> None:
        self._client = client

    @classmethod
    def from_client(cls, client: RelataClient) -> AsyncGrpcQueryClient:
        return cls(client)

    async def query_grpc(
        self,
        sql: str,
        *,
        grpc_endpoint: str | None = None,
        purpose: str | None = None,
        bearer_token: str | None = None,
        timeout: float = 30.0,
    ) -> QueryResult:
        """Async variant of :meth:`GrpcQueryClient.query_grpc`."""
        import asyncio

        sync = GrpcQueryClient(self._client)
        return await asyncio.to_thread(
            sync.query_grpc,
            sql,
            grpc_endpoint=grpc_endpoint,
            purpose=purpose,
            bearer_token=bearer_token,
            timeout=timeout,
        )

    async def query_grpc_stream(
        self,
        sql: str,
        *,
        grpc_endpoint: str | None = None,
        purpose: str | None = None,
        bearer_token: str | None = None,
        timeout: float = 30.0,
    ) -> QueryResult:
        """Async variant of :meth:`GrpcQueryClient.query_grpc_stream`."""
        import asyncio

        sync = GrpcQueryClient(self._client)
        return await asyncio.to_thread(
            sync.query_grpc_stream,
            sql,
            grpc_endpoint=grpc_endpoint,
            purpose=purpose,
            bearer_token=bearer_token,
            timeout=timeout,
        )
