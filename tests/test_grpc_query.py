"""Tests for the plain gRPC query client with Arrow-IPC opt-in (#4090 round 3).

Mirrors ``sdks/typescript/src/grpc-query.test.ts`` (round 1) and
``sdks/go/relata/grpc_query_test.go`` (round 2): pure codec/unit tests need
no ``grpcio`` install at all, plus a real end-to-end round trip against a
local ``grpc.server()`` (Python's grpcio ships a real in-process server with
no bufconn-equivalent module needed — a ``localhost:0`` ephemeral port is
cheap and standard for this). Both the ``grpcio`` and (for the Arrow-IPC
branch) ``pyarrow`` integration tests are skipped via
``pytest.importorskip`` when those optional dependencies aren't installed,
same policy the rest of this SDK's Flight/Arrow tests already follow (see
``test_multimedia_and_flight.py``'s ``test_query_flight_requires_pyarrow``).
"""

from __future__ import annotations

import builtins
import json

import pytest

from relata import RelataClient
from relata.grpc_query import (
    DEFAULT_GRPC_QUERY_PORT,
    _EXECUTE_METHOD,
    _EXECUTE_STREAM_METHOD,
    _encode_varint,
    _grpc_query_metadata,
    _read_varint,
    decode_query_response,
    decode_row_batch,
    encode_query_request,
    is_secure_grpc_query_endpoint,
    resolve_grpc_query_endpoint,
    strip_grpc_query_scheme,
)

BASE = "http://localhost:9090"


# ---------------------------------------------------------------------------
# varint codec
# ---------------------------------------------------------------------------


def test_varint_round_trips_small_and_large_values() -> None:
    for value in (0, 1, 127, 128, 300, 16384, 2**32, 2**53):
        encoded = _encode_varint(value)
        decoded, next_offset = _read_varint(encoded, 0)
        assert decoded == value
        assert next_offset == len(encoded)


def test_varint_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _encode_varint(-1)


def test_read_varint_truncated_raises() -> None:
    with pytest.raises(ValueError, match="truncated"):
        _read_varint(b"\x80\x80", 0)


# ---------------------------------------------------------------------------
# QueryRequest encode
# ---------------------------------------------------------------------------


def test_encode_query_request_encodes_sql_and_purpose() -> None:
    encoded = encode_query_request(purpose="analytics", sql="SELECT 1")
    decoded = _decode_request_for_test(encoded)
    assert decoded["purpose"] == "analytics"
    assert decoded["sql"] == "SELECT 1"
    assert decoded.get("wants_arrow_ipc_rows", False) is False


def test_encode_query_request_omits_empty_purpose_and_bearer() -> None:
    encoded = encode_query_request(sql="SELECT 1")
    decoded = _decode_request_for_test(encoded)
    assert "purpose" not in decoded
    assert "bearer_token" not in decoded
    assert decoded["sql"] == "SELECT 1"


def test_encode_query_request_sets_wants_arrow_ipc_rows_only_when_true() -> None:
    on = encode_query_request(sql="SELECT 1", wants_arrow_ipc_rows=True)
    off = encode_query_request(sql="SELECT 1", wants_arrow_ipc_rows=False)
    assert _decode_request_for_test(on)["wants_arrow_ipc_rows"] is True
    assert "wants_arrow_ipc_rows" not in _decode_request_for_test(off)


def _decode_request_for_test(buf: bytes) -> dict[str, object]:
    """Minimal QueryRequest decoder (purpose=1, sql=2, bearer_token=3,
    wants_arrow_ipc_rows=4) — only used by these tests, mirrors the field
    layout ``decode_query_response`` uses for QueryResponse."""
    out: dict[str, object] = {}
    i = 0
    n = len(buf)
    while i < n:
        tag, i = _read_varint(buf, i)
        field = tag >> 3
        wire_type = tag & 0x7
        if wire_type == 2:
            length, i = _read_varint(buf, i)
            seg = buf[i : i + length]
            i += length
            if field == 1:
                out["purpose"] = seg.decode("utf-8")
            elif field == 2:
                out["sql"] = seg.decode("utf-8")
            elif field == 3:
                out["bearer_token"] = seg.decode("utf-8")
        elif wire_type == 0:
            value, i = _read_varint(buf, i)
            if field == 4:
                out["wants_arrow_ipc_rows"] = bool(value)
    return out


# ---------------------------------------------------------------------------
# QueryResponse decode
# ---------------------------------------------------------------------------


def _encode_response_for_test(
    *,
    columns: list[str] | None = None,
    row_count: int | None = None,
    rows_json: list[str] | None = None,
    arrow_ipc_rows: bytes | None = None,
) -> bytes:
    out = bytearray()
    for c in columns or []:
        payload = c.encode("utf-8")
        out.append((1 << 3) | 2)
        out.extend(_encode_varint(len(payload)))
        out.extend(payload)
    if row_count is not None:
        out.append((2 << 3) | 0)
        out.extend(_encode_varint(row_count))
    for r in rows_json or []:
        payload = r.encode("utf-8")
        out.append((3 << 3) | 2)
        out.extend(_encode_varint(len(payload)))
        out.extend(payload)
    if arrow_ipc_rows is not None:
        out.append((4 << 3) | 2)
        out.extend(_encode_varint(len(arrow_ipc_rows)))
        out.extend(arrow_ipc_rows)
    return bytes(out)


def test_decode_query_response_decodes_columns_row_count_rows_json() -> None:
    buf = _encode_response_for_test(
        columns=["id", "name"],
        row_count=2,
        rows_json=['{"id":1,"name":"a"}', '{"id":2,"name":"b"}'],
    )
    decoded = decode_query_response(buf)
    assert decoded.columns == ["id", "name"]
    assert decoded.row_count == 2
    assert decoded.rows_json == ['{"id":1,"name":"a"}', '{"id":2,"name":"b"}']
    assert decoded.arrow_ipc_rows == b""


def test_decode_query_response_decodes_arrow_ipc_rows() -> None:
    buf = _encode_response_for_test(columns=["id"], row_count=1, arrow_ipc_rows=b"\x01\x02\x03")
    decoded = decode_query_response(buf)
    assert decoded.arrow_ipc_rows == b"\x01\x02\x03"
    assert decoded.rows_json == []


def test_decode_query_response_empty_arrow_ipc_rows_means_read_rows_json() -> None:
    """Never-both, empty-means-fallback contract (per the proto's docs)."""
    buf = _encode_response_for_test(row_count=1, rows_json=["{}"])
    decoded = decode_query_response(buf)
    assert decoded.arrow_ipc_rows == b""
    assert decoded.rows_json == ["{}"]


def test_decode_query_response_tolerates_unknown_fixed_wire_types() -> None:
    """Forward-compat: an unrecognised fixed32/fixed64 field is skipped, not fatal."""
    buf = bytearray()
    # field 99, wire type 1 (fixed64) -- unknown field, should be skipped.
    # The tag itself is a varint: field 99's tag (793) needs two bytes, so it
    # must go through _encode_varint, not bytearray.append (#4090 round 7 —
    # the single-byte form raised ValueError and this test never ran green).
    buf.extend(_encode_varint((99 << 3) | 1))
    buf.extend(b"\x00" * 8)
    buf.extend(_encode_response_for_test(row_count=5))
    decoded = decode_query_response(bytes(buf))
    assert decoded.row_count == 5


# ---------------------------------------------------------------------------
# endpoint resolution
# ---------------------------------------------------------------------------


def test_resolve_grpc_query_endpoint_derives_default() -> None:
    assert resolve_grpc_query_endpoint("http://localhost:9090", None) == (
        f"grpc://localhost:{DEFAULT_GRPC_QUERY_PORT}"
    )
    assert resolve_grpc_query_endpoint("https://relata.example.com", None) == (
        f"grpc://relata.example.com:{DEFAULT_GRPC_QUERY_PORT}"
    )


def test_resolve_grpc_query_endpoint_override_wins() -> None:
    assert resolve_grpc_query_endpoint("http://localhost:9090", "grpc://host:9999") == (
        "grpc://host:9999"
    )


def test_is_secure_grpc_query_endpoint() -> None:
    assert is_secure_grpc_query_endpoint("grpcs://host:1") is True
    assert is_secure_grpc_query_endpoint("tls://host:1") is True
    assert is_secure_grpc_query_endpoint("grpc://host:1") is False
    assert is_secure_grpc_query_endpoint("http://host:1") is False


def test_strip_grpc_query_scheme() -> None:
    assert strip_grpc_query_scheme("grpc://host:50051") == "host:50051"
    assert strip_grpc_query_scheme("grpcs://host:50051") == "host:50051"
    assert strip_grpc_query_scheme("host:50051") == "host:50051"


# ---------------------------------------------------------------------------
# metadata threading (#3213-equivalent)
# ---------------------------------------------------------------------------


def test_grpc_query_metadata_threads_tenant_scope() -> None:
    client = RelataClient(
        BASE,
        bearer_token="tok",
        tenant="acme",
        acting_as="agent-1",
        delegated_by="root-org",
        headers={"X-Verified-Principal": "svc-a"},
    )
    metadata = dict(_grpc_query_metadata(client, None))
    assert metadata["authorization"] == "Bearer tok"
    assert metadata["x-relata-tenant-id"] == "acme"
    assert metadata["x-acting-as"] == "agent-1"
    assert metadata["x-delegated-by"] == "root-org"
    assert metadata["x-verified-principal"] == "svc-a"
    assert "x-request-id" in metadata and metadata["x-request-id"]
    # str values, not bytes -- required by grpcio's non-"-bin"-key contract.
    for value in metadata.values():
        assert isinstance(value, str)


def test_grpc_query_metadata_bearer_override_wins() -> None:
    client = RelataClient(BASE, bearer_token="tok", tenant="acme")
    metadata = dict(_grpc_query_metadata(client, "override-tok"))
    assert metadata["authorization"] == "Bearer override-tok"


def test_grpc_query_metadata_unauthenticated_still_threads_tenant() -> None:
    client = RelataClient(BASE, tenant="acme")
    metadata = dict(_grpc_query_metadata(client, None))
    assert "authorization" not in metadata
    assert metadata["x-relata-tenant-id"] == "acme"


# ---------------------------------------------------------------------------
# grpcio-absent guard
# ---------------------------------------------------------------------------


def test_query_grpc_requires_grpcio(monkeypatch: pytest.MonkeyPatch) -> None:
    """If grpcio is not importable the method raises ImportError, not a crash."""
    real_import = builtins.__import__

    def _block(name: str, *args: object) -> object:
        if name == "grpc" or name.startswith("grpc."):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args)  # type: ignore[misc]

    monkeypatch.setattr(builtins, "__import__", _block)
    client = RelataClient(BASE, purpose="analytics")
    with pytest.raises(ImportError):
        client.query_grpc("SELECT 1")


# ---------------------------------------------------------------------------
# Real end-to-end round trip against a local grpc.server() (#4090). Skipped
# when the optional `grpcio` dependency isn't installed -- run with
# `pip install relata-sdk[grpc]` (or the `dev` extra, which now includes it).
# ---------------------------------------------------------------------------


def test_query_grpc_execute_rows_json_round_trip() -> None:
    grpc = pytest.importorskip("grpc")
    from concurrent import futures

    seen_wants_arrow_ipc_rows: list[bool] = []

    def _handler(request_bytes: bytes, context: object) -> bytes:
        decoded = _decode_request_for_test(request_bytes)
        seen_wants_arrow_ipc_rows.append(bool(decoded.get("wants_arrow_ipc_rows", False)))
        assert decoded["sql"] == "SELECT * FROM Person LIMIT 1"
        return _encode_response_for_test(
            columns=["id", "name"],
            row_count=1,
            rows_json=['{"id": 1, "name": "Ada"}'],
        )

    rpc_handler = grpc.unary_unary_rpc_method_handler(
        _handler, request_deserializer=None, response_serializer=None
    )
    generic_handler = grpc.method_handlers_generic_handler(
        "relata.RelataQuery", {"Execute": rpc_handler}
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    server.add_generic_rpc_handlers((generic_handler,))
    port = server.add_insecure_port("localhost:0")
    server.start()
    try:
        client = RelataClient(BASE, purpose="analytics")
        result = client.query_grpc(
            "SELECT * FROM Person LIMIT 1", grpc_endpoint=f"grpc://localhost:{port}"
        )
        assert result.columns == ["id", "name"]
        assert result.row_count == 1
        assert result.rows == [{"id": 1, "name": "Ada"}]
        assert seen_wants_arrow_ipc_rows == [True]
    finally:
        server.stop(None)


def test_query_grpc_execute_arrow_ipc_rows_round_trip() -> None:
    pytest.importorskip("grpc")
    pyarrow = pytest.importorskip("pyarrow")
    import io

    import pyarrow.ipc as ipc
    from concurrent import futures

    import grpc

    table = pyarrow.table({"id": [1, 2], "name": ["Ada", "Grace"]})
    buf = io.BytesIO()
    with ipc.new_stream(buf, table.schema) as writer:
        writer.write_table(table)
    arrow_bytes = buf.getvalue()

    def _handler(request_bytes: bytes, context: object) -> bytes:
        return _encode_response_for_test(
            columns=["id", "name"], row_count=2, arrow_ipc_rows=arrow_bytes
        )

    rpc_handler = grpc.unary_unary_rpc_method_handler(
        _handler, request_deserializer=None, response_serializer=None
    )
    generic_handler = grpc.method_handlers_generic_handler(
        "relata.RelataQuery", {"Execute": rpc_handler}
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    server.add_generic_rpc_handlers((generic_handler,))
    port = server.add_insecure_port("localhost:0")
    server.start()
    try:
        client = RelataClient(BASE, purpose="analytics")
        result = client.query_grpc(
            "SELECT * FROM Person", grpc_endpoint=f"grpc://localhost:{port}"
        )
        assert result.row_count == 2
        assert result.rows == [
            {"id": 1, "name": "Ada"},
            {"id": 2, "name": "Grace"},
        ]
    finally:
        server.stop(None)


@pytest.mark.asyncio
async def test_aquery_grpc_dispatches_to_thread() -> None:
    grpc = pytest.importorskip("grpc")
    from concurrent import futures

    def _handler(request_bytes: bytes, context: object) -> bytes:
        return _encode_response_for_test(columns=["x"], row_count=1, rows_json=["{\"x\": 1}"])

    rpc_handler = grpc.unary_unary_rpc_method_handler(
        _handler, request_deserializer=None, response_serializer=None
    )
    generic_handler = grpc.method_handlers_generic_handler(
        "relata.RelataQuery", {"Execute": rpc_handler}
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    server.add_generic_rpc_handlers((generic_handler,))
    port = server.add_insecure_port("localhost:0")
    server.start()
    try:
        client = RelataClient(BASE, purpose="analytics")
        result = await client.aquery_grpc(
            "SELECT 1", grpc_endpoint=f"grpc://localhost:{port}"
        )
        assert result.rows == [{"x": 1}]
    finally:
        server.stop(None)


def test_execute_method_path_matches_proto_service_name() -> None:
    assert _EXECUTE_METHOD == "/relata.RelataQuery/Execute"


def test_execute_stream_method_path_matches_proto_service_name() -> None:
    assert _EXECUTE_STREAM_METHOD == "/relata.RelataQuery/ExecuteStream"


# ---------------------------------------------------------------------------
# RowBatch decode (#4090 round 7 — ExecuteStream). RowBatch's field layout is
# NOT QueryResponse's: rows_json is field 2 (not 3), there's a `done` bool at
# field 3, and there is no row_count field at all.
# ---------------------------------------------------------------------------


def _encode_row_batch_for_test(
    *,
    columns: list[str] | None = None,
    rows_json: list[str] | None = None,
    done: bool = False,
    arrow_ipc_rows: bytes | None = None,
) -> bytes:
    out = bytearray()
    for c in columns or []:
        payload = c.encode("utf-8")
        out.append((1 << 3) | 2)
        out.extend(_encode_varint(len(payload)))
        out.extend(payload)
    for r in rows_json or []:
        payload = r.encode("utf-8")
        out.append((2 << 3) | 2)
        out.extend(_encode_varint(len(payload)))
        out.extend(payload)
    if done:
        out.append((3 << 3) | 0)
        out.extend(_encode_varint(1))
    if arrow_ipc_rows is not None:
        out.append((4 << 3) | 2)
        out.extend(_encode_varint(len(arrow_ipc_rows)))
        out.extend(arrow_ipc_rows)
    return bytes(out)


def test_decode_row_batch_reads_field_2_rows_json_not_field_3() -> None:
    """RowBatch.rows_json is field 2 — decoding it as QueryResponse would
    silently read field 3 (`done`) instead and drop every row."""
    buf = _encode_row_batch_for_test(
        columns=["id", "name"], rows_json=['{"id":1,"name":"a"}']
    )
    frame = decode_row_batch(buf)
    assert frame.columns == ["id", "name"]
    assert frame.rows_json == ['{"id":1,"name":"a"}']
    assert frame.done is False
    assert frame.arrow_ipc_rows == b""
    # Cross-check: the same bytes through the QueryResponse decoder do NOT
    # yield rows — proof the two layouts genuinely differ.
    assert decode_query_response(buf).rows_json == []


def test_decode_row_batch_reads_done_flag() -> None:
    frame = decode_row_batch(_encode_row_batch_for_test(done=True))
    assert frame.done is True
    assert frame.rows_json == []


def test_decode_row_batch_reads_arrow_ipc_rows() -> None:
    frame = decode_row_batch(
        _encode_row_batch_for_test(columns=["id"], arrow_ipc_rows=b"\x01\x02\x03")
    )
    assert frame.arrow_ipc_rows == b"\x01\x02\x03"
    assert frame.rows_json == []


def test_decode_row_batch_empty_arrow_ipc_rows_means_read_rows_json() -> None:
    """Never-both, empty-means-fallback contract, applied per frame."""
    frame = decode_row_batch(_encode_row_batch_for_test(rows_json=["{}"]))
    assert frame.arrow_ipc_rows == b""
    assert frame.rows_json == ["{}"]


def test_decode_row_batch_tolerates_unknown_fixed_wire_types() -> None:
    buf = bytearray()
    buf.extend(_encode_varint((99 << 3) | 5))  # unknown field, fixed32
    buf.extend(b"\x00" * 4)
    buf.extend(_encode_row_batch_for_test(rows_json=['{"x":1}'], done=True))
    frame = decode_row_batch(bytes(buf))
    assert frame.rows_json == ['{"x":1}']
    assert frame.done is True


def test_decode_row_batch_truncated_payload_raises() -> None:
    with pytest.raises(ValueError, match="truncated"):
        decode_row_batch(bytes([(1 << 3) | 2, 9, 0x61]))


# ---------------------------------------------------------------------------
# ExecuteStream end-to-end round trips against a local grpc.server().
# ---------------------------------------------------------------------------


def _start_stream_server(grpc: object, frames: list[bytes], seen: list[bytes]) -> tuple:
    """Serve `frames` from a fake `RelataQuery/ExecuteStream`, recording each
    request payload into `seen`. Returns `(server, port)`."""
    from concurrent import futures

    def _handler(request_bytes: bytes, context: object):  # noqa: ANN202
        seen.append(request_bytes)
        yield from frames

    rpc_handler = grpc.unary_stream_rpc_method_handler(  # type: ignore[attr-defined]
        _handler, request_deserializer=None, response_serializer=None
    )
    generic_handler = grpc.method_handlers_generic_handler(  # type: ignore[attr-defined]
        "relata.RelataQuery", {"ExecuteStream": rpc_handler}
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))  # type: ignore[attr-defined]
    server.add_generic_rpc_handlers((generic_handler,))
    port = server.add_insecure_port("localhost:0")
    server.start()
    return server, port


def test_query_grpc_stream_collects_rows_json_frames() -> None:
    grpc = pytest.importorskip("grpc")

    frames = [
        _encode_row_batch_for_test(columns=["id", "name"], rows_json=['{"id": 1, "name": "Ada"}']),
        _encode_row_batch_for_test(
            columns=["id", "name"], rows_json=['{"id": 2, "name": "Grace"}']
        ),
        _encode_row_batch_for_test(done=True),
    ]
    seen: list[bytes] = []
    server, port = _start_stream_server(grpc, frames, seen)
    try:
        client = RelataClient(BASE, purpose="analytics")
        result = client.query_grpc_stream(
            "SELECT * FROM Person", grpc_endpoint=f"grpc://localhost:{port}"
        )
        assert result.columns == ["id", "name"]
        assert result.row_count == 2
        assert result.rows == [{"id": 1, "name": "Ada"}, {"id": 2, "name": "Grace"}]
        # The whole point of #4090: the request opted into Arrow-IPC rows.
        decoded_request = _decode_request_for_test(seen[0])
        assert decoded_request["wants_arrow_ipc_rows"] is True
        assert decoded_request["purpose"] == "analytics"
    finally:
        server.stop(None)


def test_query_grpc_stream_decodes_per_frame_arrow_ipc_rows() -> None:
    grpc = pytest.importorskip("grpc")
    pyarrow = pytest.importorskip("pyarrow")
    import io

    import pyarrow.ipc as ipc

    def _arrow_frame(ids: list[int], names: list[str]) -> bytes:
        table = pyarrow.table({"id": ids, "name": names})
        buf = io.BytesIO()
        with ipc.new_stream(buf, table.schema) as writer:
            writer.write_table(table)
        return _encode_row_batch_for_test(columns=["id", "name"], arrow_ipc_rows=buf.getvalue())

    frames = [
        _arrow_frame([1], ["Ada"]),
        # Mixed stream: the second frame falls back to rows_json, which the
        # proto explicitly allows per frame.
        _encode_row_batch_for_test(columns=["id", "name"], rows_json=['{"id": 2, "name": "Grace"}']),
        _encode_row_batch_for_test(done=True),
    ]
    seen: list[bytes] = []
    server, port = _start_stream_server(grpc, frames, seen)
    try:
        client = RelataClient(BASE, purpose="analytics")
        result = client.query_grpc_stream(
            "SELECT id, name FROM Person", grpc_endpoint=f"grpc://localhost:{port}"
        )
        assert result.row_count == 2
        assert result.rows == [{"id": 1, "name": "Ada"}, {"id": 2, "name": "Grace"}]
    finally:
        server.stop(None)


def test_query_grpc_stream_stops_at_done_frame() -> None:
    """A `done` frame terminates collection — rows after it are not read."""
    grpc = pytest.importorskip("grpc")

    frames = [
        _encode_row_batch_for_test(columns=["x"], rows_json=['{"x": 1}']),
        _encode_row_batch_for_test(done=True),
        _encode_row_batch_for_test(columns=["x"], rows_json=['{"x": 999}']),
    ]
    seen: list[bytes] = []
    server, port = _start_stream_server(grpc, frames, seen)
    try:
        client = RelataClient(BASE)
        result = client.query_grpc_stream("SELECT 1", grpc_endpoint=f"grpc://localhost:{port}")
        assert result.rows == [{"x": 1}]
    finally:
        server.stop(None)


def test_query_grpc_stream_requires_grpcio(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def _block(name: str, *args: object) -> object:
        if name == "grpc" or name.startswith("grpc."):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args)  # type: ignore[misc]

    monkeypatch.setattr(builtins, "__import__", _block)
    client = RelataClient(BASE, purpose="analytics")
    with pytest.raises(ImportError):
        client.query_grpc_stream("SELECT 1")


@pytest.mark.asyncio
async def test_aquery_grpc_stream_dispatches_to_thread() -> None:
    grpc = pytest.importorskip("grpc")

    frames = [
        _encode_row_batch_for_test(columns=["x"], rows_json=['{"x": 7}']),
        _encode_row_batch_for_test(done=True),
    ]
    seen: list[bytes] = []
    server, port = _start_stream_server(grpc, frames, seen)
    try:
        client = RelataClient(BASE, purpose="analytics")
        result = await client.aquery_grpc_stream(
            "SELECT 7", grpc_endpoint=f"grpc://localhost:{port}"
        )
        assert result.rows == [{"x": 7}]
    finally:
        server.stop(None)
