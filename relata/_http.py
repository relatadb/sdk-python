"""
Internal HTTP transport layer.

This module is **not** part of the public API.  All internals are subject to
change without notice.  Callers should use :class:`relata.client.RelataClient`
instead.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any
from urllib.parse import quote

import httpx

from relata._version import __version__
from relata.exceptions import (
    AuthError,
    ConflictError,
    ConnectionError,
    ForbiddenError,
    NotFoundError,
    PurposeError,
    RateLimitedError,
    RelataError,
    ResponseTooLargeError,
    ServerError,
    ValidationError,
)

# Request / response content-type we always send and expect.
_CONTENT_TYPE = "application/json"

#: Default response-body cap (~64 MiB) — a malicious/buggy server can no
#: longer OOM the client by streaming an unbounded body (#3214, parity with
#: Rust #3198). Override via ``RelataClient(max_response_bytes=...)``.
DEFAULT_MAX_RESPONSE_BYTES = 64 * 1024 * 1024


def path_segment(value: str) -> str:
    """Percent-encode a caller-supplied identifier for safe use as ONE URL path
    segment (#3212).

    Every helper that interpolates an id into ``f"/x/{id}"`` must route it
    through this so ``../``, ``?``, ``#``, ``&``, ``/`` can neither traverse the
    path nor smuggle query params. ``safe=""`` encodes even ``/`` — the encoded
    value can never span multiple segments. Mirrors the Rust SDK's
    ``path_segment`` and the Go/TS ``pathSegment`` helpers (parity with #3200).
    """
    return quote(str(value), safe="")

# Server-side error messages that signal a missing/invalid purpose.
_PURPOSE_HINTS = frozenset(
    [
        "purpose",
        "purposeregistry",
        "purpose_registry",
        "missing purpose",
        "invalid purpose",
        "unregistered purpose",
    ]
)

# Default retry config (override via RelataClient constructor).
_DEFAULT_RETRY_ON: frozenset[int] = frozenset({502, 503, 504})
_DEFAULT_MAX_RETRIES = 0  # off by default — caller must opt in
_DEFAULT_RETRY_BACKOFF = 0.5

# HTTP verbs that are safe to retry without risk of double-execution (#2490).
# Parity with the Rust SDK (crates/relata-sdk-rust/src/http.rs:432) and the Go
# SDK (sdks/go/relata/client.go:1259 isIdempotent): POST/DELETE/PATCH/PUT are
# NEVER retried — a gateway timeout after commit would double-execute
# irreversible ops (erase_subject, fuse_identities, session_commit).
_IDEMPOTENT_METHODS: frozenset[str] = frozenset({"GET", "HEAD", "OPTIONS"})


def _is_idempotent(method: str) -> bool:
    """Whether ``method`` is safe to retry (GET/HEAD/OPTIONS only)."""
    return method in _IDEMPOTENT_METHODS


def _opt_float_header(headers: Any, name: str) -> float | None:
    """Parse a numeric response header to float, returning ``None`` when absent/unparseable."""
    raw = headers.get(name)
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _check_content_length(resp: httpx.Response, max_bytes: int) -> None:
    """Reject early when the advertised ``Content-Length`` exceeds the cap
    (#3214) — before buffering a byte of an oversized body."""
    raw = resp.headers.get("content-length")
    if raw is None:
        return
    try:
        length = int(raw)
    except ValueError:
        return
    if length > max_bytes:
        raise ResponseTooLargeError(
            f"Response body of {length} bytes exceeds the max_response_bytes "
            f"cap of {max_bytes} bytes; refusing to buffer it (#3214).",
            max_bytes=max_bytes,
            content_length=length,
        )


# #2321 (ADR-0261): paths mounted only on the loopbound admin control-plane
# listener (RELATA_ADMIN_BIND, default 127.0.0.1:9091) — never the main
# data-plane listener. See crates/relata-cli/src/serve/admin_listener.rs.
_ADMIN_ONLY_PREFIXES = ("/admin/", "/platform/")


def _is_admin_only_path(path: str) -> bool:
    """``True`` for a path served only by the loopbound admin listener."""
    return path.startswith(_ADMIN_ONLY_PREFIXES)


def _response_detail_is_blank(content_type: str, raw_text: str) -> bool:
    """``True`` when a non-2xx response body carries no human-readable error
    text at all -- the shape the server's own RFC 7807 normalisation
    (``normalize_error_body``, ``crates/relata-cli/src/serve.rs``) leaves
    behind for a request that matched no route on the listener that
    answered (``{"type":"about:blank","status":404,"title":"Not Found",
    "detail":""}``), as opposed to a real business error, which always
    carries a non-empty ``detail``/``error``/``message`` by convention
    across this codebase. Also ``True`` for a genuinely empty/whitespace
    non-JSON body.
    """
    if "json" in content_type:
        try:
            parsed = json.loads(raw_text) if raw_text else {}
        except ValueError:
            return not raw_text.strip()
        if not isinstance(parsed, dict):
            return not raw_text.strip()
        return not any(str(parsed.get(k) or "").strip() for k in ("detail", "error", "message"))
    return not raw_text.strip()


def _admin_listener_hint(path: str) -> str:
    """Build the "you're probably pointed at the wrong port" hint for a bare
    404 against an admin/platform-only ``path`` (#2321)."""
    return (
        f"{path} returned a bare 404 with no error detail. This route is "
        "served only by Relata's loopbound admin control-plane listener "
        "(RELATA_ADMIN_BIND, default 127.0.0.1:9091 per ADR-0261) — it is "
        "never mounted on the main data-plane listener, so a bare 404 here "
        "almost always means this client's base_url points at the "
        "data-plane port instead. Pass admin_base_url= to point admin-only "
        "calls at the admin listener, or verify RELATA_ADMIN_BIND. See "
        "docs/src/decisions/0261-zero-trust-authorization-model.md."
    )


def _classify_error(
    status_code: int,
    body: dict[str, Any],
    *,
    request_id: str | None = None,
    retry_after: float | None = None,
    rate_limit_limit: float | None = None,
    rate_limit_remaining: float | None = None,
    rate_limit_reset: float | None = None,
    path: str | None = None,
    content_type: str = "",
    raw_text: str = "",
) -> RelataError:
    """Map an HTTP error response to the most-specific SDK exception.

    Parses both the RFC 7807 ``application/problem+json`` shape (``code``,
    ``type``, ``detail``, ``retryable``, ``request_id``) and the legacy
    ``{"error": "..."}`` shape.
    """
    # RFC 7807 fields — present when the server emits problem+json.
    code: str | None = body.get("code")
    type_url: str | None = body.get("type")
    retryable: bool = bool(body.get("retryable", False))
    rid = request_id or body.get("request_id")
    detail: str = (
        body.get("detail") or body.get("error")
        or body.get("message") or "unknown server error"
    )
    # #2321 (ADR-0261): a bare 404 with no detail text against an
    # admin/platform-only path almost always means this client is pointed at
    # the data-plane listener rather than the loopbound admin listener those
    # routes are exclusively mounted on — surface a hint instead of an
    # uninformative bare 404/"unknown server error".
    if (
        status_code == 404
        and path is not None
        and _is_admin_only_path(path)
        and _response_detail_is_blank(content_type, raw_text)
    ):
        detail = _admin_listener_hint(path)
    lower_detail = detail.lower()

    # Shared kwargs for every subclass.
    base_kwargs: dict[str, Any] = {
        "code": code,
        "type_url": type_url,
        "retryable": retryable,
        "request_id": rid,
    }

    # Purpose-specific detection — works for both legacy and problem+json.
    is_purpose = status_code == 400 and (
        (code and "purpose" in code.lower())
        or any(hint in lower_detail for hint in _PURPOSE_HINTS)
    )
    if is_purpose:
        return PurposeError(
            f"Purpose rejected: {detail}. "
            "Pass purpose= to RelataClient.query() or set a default on the client.",
            status_code=status_code,
        )

    if status_code == 401:
        return AuthError(
            f"Authentication failed: {detail}. ",
            status_code=status_code,
            **base_kwargs,
        )
    if status_code == 403:
        return ForbiddenError(
            f"Forbidden: {detail}",
            status_code=status_code,
            **base_kwargs,
        )
    if status_code == 404:
        return NotFoundError(
            f"Not found: {detail}",
            status_code=status_code,
            **base_kwargs,
        )
    if status_code == 409:
        return ConflictError(
            f"Conflict: {detail}",
            status_code=status_code,
            **base_kwargs,
        )
    if status_code == 422:
        return ValidationError(
            f"Validation error: {detail}",
            status_code=status_code,
            **base_kwargs,
        )
    if status_code == 429:
        return RateLimitedError(
            f"Rate limited: {detail}",
            status_code=status_code,
            retry_after=retry_after,
            rate_limit_limit=rate_limit_limit,
            rate_limit_remaining=rate_limit_remaining,
            rate_limit_reset=rate_limit_reset,
            **base_kwargs,
        )
    if status_code >= 500:
        return ServerError(
            f"Server error: {detail}",
            status_code=status_code,
            **base_kwargs,
        )
    return RelataError(detail, status_code=status_code, **base_kwargs)


def _build_headers(
    bearer_token: str | None,
    extra: dict[str, str] | None = None,
    *,
    compress: bool = False,
) -> dict[str, str]:
    headers: dict[str, str] = {
        "Content-Type": _CONTENT_TYPE,
        "Accept": _CONTENT_TYPE,
        "User-Agent": f"relata-sdk-python/{__version__}",
    }
    if not compress:
        # T9 (#1991): HTTP compression disabled by default — SDK clients are
        # CPU-bound on serialisation, and transparent decompression adds
        # avoidable latency on already-fast localhost / VPC links. Pass
        # compress=True to RelataClient (or set Accept-Encoding in headers=)
        # to re-enable gzip/deflate/br for slow links.
        headers["Accept-Encoding"] = "identity"
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    if extra:
        # Caller-supplied headers win over the SDK defaults so tenants,
        # acting-as, request-id, etc. can be pinned per-request.
        headers.update(extra)
    return headers


# ---------------------------------------------------------------------------
# Sync transport
# ---------------------------------------------------------------------------


class HttpTransport:
    """Synchronous HTTP transport backed by :mod:`httpx`."""

    def __init__(
        self,
        base_url: str,
        bearer_token: str | None,
        timeout: float,
        transport: httpx.BaseTransport | None = None,
        extra_headers: dict[str, str] | None = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        retry_backoff: float = _DEFAULT_RETRY_BACKOFF,
        retry_on: frozenset[int] = _DEFAULT_RETRY_ON,
        compress: bool = False,
        admin_base_url: str | None = None,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    ) -> None:
        headers = _build_headers(bearer_token, extra_headers, compress=compress)
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout,
            transport=transport,
        )
        # #2321 (ADR-0261): /admin/* and /platform/* requests route to a
        # second client bound to admin_base_url when supplied -- those routes
        # are mounted only on the loopbound admin control-plane listener,
        # never the main data-plane listener base_url usually targets.
        # ``None`` (the default) preserves prior behaviour: every request
        # goes to base_url, unchanged.
        self._admin_client: httpx.Client | None = None
        if admin_base_url is not None:
            self._admin_client = httpx.Client(
                base_url=admin_base_url.rstrip("/"),
                headers=headers,
                timeout=timeout,
                transport=transport,
            )
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._retry_on = retry_on
        # #3214: response-body cap enforced on every buffered read.
        self._max_response_bytes = max_response_bytes

    def _request_id(self) -> str:
        return str(uuid.uuid4())

    def _client_for(self, path: str) -> httpx.Client:
        """Resolve the httpx client a request to ``path`` should use:
        the admin client for ``/admin/*``/``/platform/*`` when configured,
        the ordinary client otherwise (unchanged behaviour when unset)."""
        if self._admin_client is not None and _is_admin_only_path(path):
            return self._admin_client
        return self._client

    def _send_with_retry(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
        content: str | bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Send an HTTP request with retry + request_id propagation.

        Retry policy (#2490): only idempotent verbs (GET/HEAD/OPTIONS) are
        retried, and only on transport errors (connect/timeout) or response
        status codes in ``self._retry_on`` (502/503/504 by default). POST and
        other non-idempotent verbs are never retried — parity with the Rust
        and Go SDKs.
        """
        client = self._client_for(path)
        # If the caller already set X-Request-ID in the default headers bag,
        # respect it — only generate a per-attempt UUID when no default exists.
        has_default_rid = "x-request-id" in {k.lower() for k in client.headers}
        idempotent = _is_idempotent(method)
        last_exc: RelataError | None = None
        for attempt in range(self._max_retries + 1):
            req_headers: dict[str, str] = {}
            if not has_default_rid:
                req_headers["X-Request-ID"] = self._request_id()
            if headers:
                req_headers.update(headers)
            try:
                # Stream every request so the response cap can be enforced while
                # reading — a body larger than max_response_bytes is rejected
                # without ever being fully buffered (#3214).
                req = client.build_request(
                    method, path, json=json_payload, content=content, headers=req_headers
                )
                resp = client.send(req, stream=True)
            except httpx.ConnectError as exc:
                last_exc = ConnectionError(
                    f"Cannot connect to Relata server: {exc}. "
                    "Check that the server is running and the base_url is correct."
                )
                if idempotent and attempt < self._max_retries:
                    time.sleep(self._retry_backoff * (2**attempt))
                    continue
                raise last_exc from exc
            except httpx.TimeoutException as exc:
                last_exc = ConnectionError(
                    f"Request timed out: {exc}. "
                    "Increase timeout= on RelataClient or check server health."
                )
                if idempotent and attempt < self._max_retries:
                    time.sleep(self._retry_backoff * (2**attempt))
                    continue
                raise last_exc from exc

            try:
                # Retry on configurable status codes (502/503/504) for idempotent
                # verbs only — non-idempotent verbs fall through to _handle which
                # raises the classified ServerError.
                if (
                    idempotent
                    and resp.status_code in self._retry_on
                    and attempt < self._max_retries
                ):
                    time.sleep(self._retry_backoff * (2**attempt))
                    continue

                # Success or non-retryable error — return / raise via _handle.
                return self._handle(resp)
            finally:
                resp.close()

        # Exhausted retries on a retryable status code (transport errors raise
        # directly from the except block above; only the status-code path can
        # land here, and the final iteration's _handle call always returns or
        # raises — so this is a defensive unreachable fallback).
        assert last_exc is not None
        raise last_exc

    def get(self, path: str) -> dict[str, Any]:
        """Perform a GET request and return the decoded JSON body."""
        return self._send_with_retry("GET", path)

    def post(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Perform a POST request with a JSON body and return the decoded JSON body.

        ``extra_headers`` (e.g. ``x-query-dialect``) are merged on top of the
        default headers when provided (#3265).
        """
        return self._send_with_retry(
            "POST", path, json_payload=payload, headers=extra_headers
        )

    def post_raw(
        self,
        path: str,
        content: bytes | str,
        *,
        content_type: str = "application/x-ndjson",
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """POST a raw (non-JSON) body — used by NDJSON/CSV ingest paths.

        The ``Content-Type`` header is set per ``content_type`` (overriding the
        default ``application/json``); ``extra_headers`` are merged on top
        (e.g. ``X-Detect-Packs``). Response handling + retry + error
        classification + the response-body cap are identical to :meth:`post`.
        """
        headers = {"Content-Type": content_type}
        if extra_headers:
            headers.update(extra_headers)
        return self._send_with_retry("POST", path, content=content, headers=headers)

    def delete(self, path: str) -> dict[str, Any]:
        """Perform a DELETE request and return the decoded JSON body."""
        return self._send_with_retry("DELETE", path)

    def patch(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Perform a PATCH request with an optional JSON body."""
        return self._send_with_retry("PATCH", path, json_payload=payload or {})

    def put(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Perform a PUT request with an optional JSON body."""
        return self._send_with_retry("PUT", path, json_payload=payload or {})

    def close(self) -> None:
        self._client.close()
        if self._admin_client is not None:
            self._admin_client.close()

    def clear_credentials(self) -> None:
        """Drop the ``Authorization`` header from the live client(s) (#3214) —
        called on close so the bearer does not linger in the default header
        bag of a transport that is torn down in two phases (sync + async)."""
        self._client.headers.pop("Authorization", None)
        if self._admin_client is not None:
            self._admin_client.headers.pop("Authorization", None)

    def _read_capped(self, resp: httpx.Response) -> bytes:
        """Read the (streaming) response body, refusing to buffer more than
        ``max_response_bytes`` (#3214). Raises :class:`ResponseTooLargeError`
        the moment the cap is exceeded — the body is never fully buffered."""
        max_bytes = self._max_response_bytes
        _check_content_length(resp, max_bytes)
        chunks: list[bytes] = []
        total = 0
        for chunk in resp.iter_bytes():
            total += len(chunk)
            if total > max_bytes:
                raise ResponseTooLargeError(
                    f"Response body exceeded the max_response_bytes cap of "
                    f"{max_bytes} bytes; aborted the read (#3214).",
                    max_bytes=max_bytes,
                    content_length=None,
                )
            chunks.append(chunk)
        return b"".join(chunks)

    def _handle(self, resp: httpx.Response) -> dict[str, Any]:
        raw = self._read_capped(resp)
        raw_text = raw.decode("utf-8", errors="replace") if raw else ""
        if resp.is_success:
            return json.loads(raw)  # type: ignore[no-any-return]
        try:
            body: dict[str, Any] = json.loads(raw)
            if not isinstance(body, dict):
                raise ValueError("error body is not a JSON object")
        except Exception:
            body = {"error": raw_text or "empty response"}
        request_id = resp.headers.get("x-request-id")
        retry_after_hdr = resp.headers.get("retry-after")
        retry_after = float(retry_after_hdr) if retry_after_hdr else None
        rl_limit = _opt_float_header(resp.headers, "x-ratelimit-limit")
        rl_remaining = _opt_float_header(resp.headers, "x-ratelimit-remaining")
        rl_reset = _opt_float_header(resp.headers, "x-ratelimit-reset")
        raise _classify_error(
            resp.status_code,
            body,
            request_id=request_id,
            retry_after=retry_after,
            rate_limit_limit=rl_limit,
            rate_limit_remaining=rl_remaining,
            rate_limit_reset=rl_reset,
            path=resp.request.url.path,
            content_type=resp.headers.get("content-type", ""),
            raw_text=raw_text,
        )


# ---------------------------------------------------------------------------
# Async transport
# ---------------------------------------------------------------------------


class AsyncHttpTransport:
    """Asynchronous HTTP transport backed by :mod:`httpx`."""

    def __init__(
        self,
        base_url: str,
        bearer_token: str | None,
        timeout: float,
        transport: httpx.AsyncBaseTransport | None = None,
        extra_headers: dict[str, str] | None = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        retry_backoff: float = _DEFAULT_RETRY_BACKOFF,
        retry_on: frozenset[int] = _DEFAULT_RETRY_ON,
        compress: bool = False,
        admin_base_url: str | None = None,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    ) -> None:
        headers = _build_headers(bearer_token, extra_headers, compress=compress)
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout,
            transport=transport,
        )
        # #2321 (ADR-0261): see HttpTransport.__init__ for the rationale --
        # /admin/*|/platform/* route to a second client bound to
        # admin_base_url when supplied; None preserves prior behaviour.
        self._admin_client: httpx.AsyncClient | None = None
        if admin_base_url is not None:
            self._admin_client = httpx.AsyncClient(
                base_url=admin_base_url.rstrip("/"),
                headers=headers,
                timeout=timeout,
                transport=transport,
            )
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._retry_on = retry_on
        # #3214: response-body cap enforced on every buffered read.
        self._max_response_bytes = max_response_bytes

    def _request_id(self) -> str:
        return str(uuid.uuid4())

    def _client_for(self, path: str) -> httpx.AsyncClient:
        """See :meth:`HttpTransport._client_for`."""
        if self._admin_client is not None and _is_admin_only_path(path):
            return self._admin_client
        return self._client

    async def _send_with_retry(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
        content: bytes | str | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Send an async HTTP request with retry + request_id propagation.

        Retry policy mirrors the sync transport (#2490): idempotent verbs
        only (GET/HEAD/OPTIONS), on transport errors or 502/503/504.
        """
        import asyncio

        client = self._client_for(path)
        idempotent = _is_idempotent(method)
        last_exc: RelataError | None = None
        for attempt in range(self._max_retries + 1):
            req_headers: dict[str, str] = {"X-Request-ID": self._request_id()}
            if headers:
                req_headers.update(headers)
            try:
                # Stream every request so the response cap can be enforced while
                # reading — a body larger than max_response_bytes is rejected
                # without ever being fully buffered (#3214).
                if content is not None:
                    req = client.build_request(method, path, content=content, headers=req_headers)
                else:
                    # Parity with the prior async behaviour: POST/PUT/PATCH
                    # always carry a JSON body ({} when no payload is given).
                    body_json = json_payload
                    if body_json is None and method not in ("GET", "HEAD", "DELETE"):
                        body_json = {}
                    req = client.build_request(method, path, json=body_json, headers=req_headers)
                resp = await client.send(req, stream=True)
            except httpx.ConnectError as exc:
                last_exc = ConnectionError(
                    f"Cannot connect to Relata server: {exc}."
                )
                if idempotent and attempt < self._max_retries:
                    await asyncio.sleep(self._retry_backoff * (2**attempt))
                    continue
                raise last_exc from exc
            except httpx.TimeoutException as exc:
                last_exc = ConnectionError(
                    f"Request timed out: {exc}."
                )
                if idempotent and attempt < self._max_retries:
                    await asyncio.sleep(self._retry_backoff * (2**attempt))
                    continue
                raise last_exc from exc
            try:
                if (
                    idempotent
                    and resp.status_code in self._retry_on
                    and attempt < self._max_retries
                ):
                    await asyncio.sleep(self._retry_backoff * (2**attempt))
                    continue
                return await self._handle(resp)
            finally:
                await resp.aclose()
        assert last_exc is not None
        raise last_exc

    async def get(self, path: str) -> dict[str, Any]:
        """Perform an async GET request and return the decoded JSON body."""
        return await self._send_with_retry("GET", path)

    async def post(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Perform an async POST request with a JSON body.

        ``extra_headers`` (e.g. ``x-query-dialect``) are merged on top of the
        default headers when provided (#3265).
        """
        return await self._send_with_retry(
            "POST", path, json_payload=payload, headers=extra_headers
        )

    async def post_raw(
        self,
        path: str,
        content: bytes | str,
        *,
        content_type: str = "application/x-ndjson",
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Async POST a raw (non-JSON) body — NDJSON/CSV ingest path.

        ``extra_headers`` are merged on top of the ``Content-Type`` header
        (e.g. ``X-Detect-Packs``); response handling mirrors :meth:`post`.
        """
        headers = {"Content-Type": content_type}
        if extra_headers:
            headers.update(extra_headers)
        return await self._send_with_retry("POST", path, content=content, headers=headers)

    async def delete(self, path: str) -> dict[str, Any]:
        """Perform an async DELETE request."""
        return await self._send_with_retry("DELETE", path)

    async def patch(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Perform an async PATCH request with an optional JSON body."""
        return await self._send_with_retry("PATCH", path, json_payload=payload or {})

    async def put(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Perform an async PUT request with an optional JSON body."""
        return await self._send_with_retry("PUT", path, json_payload=payload or {})

    async def aclose(self) -> None:
        await self._client.aclose()
        if self._admin_client is not None:
            await self._admin_client.aclose()

    def clear_credentials(self) -> None:
        """Drop the ``Authorization`` header from the live client(s) (#3214).

        See :meth:`HttpTransport.clear_credentials`.
        """
        self._client.headers.pop("Authorization", None)
        if self._admin_client is not None:
            self._admin_client.headers.pop("Authorization", None)

    async def _read_capped(self, resp: httpx.Response) -> bytes:
        """Async twin of :meth:`HttpTransport._read_capped` (#3214)."""
        max_bytes = self._max_response_bytes
        _check_content_length(resp, max_bytes)
        chunks: list[bytes] = []
        total = 0
        async for chunk in resp.aiter_bytes():
            total += len(chunk)
            if total > max_bytes:
                raise ResponseTooLargeError(
                    f"Response body exceeded the max_response_bytes cap of "
                    f"{max_bytes} bytes; aborted the read (#3214).",
                    max_bytes=max_bytes,
                    content_length=None,
                )
            chunks.append(chunk)
        return b"".join(chunks)

    async def _handle(self, resp: httpx.Response) -> dict[str, Any]:
        raw = await self._read_capped(resp)
        raw_text = raw.decode("utf-8", errors="replace") if raw else ""
        if resp.is_success:
            return json.loads(raw)  # type: ignore[no-any-return]
        try:
            body: dict[str, Any] = json.loads(raw)
            if not isinstance(body, dict):
                raise ValueError("error body is not a JSON object")
        except Exception:
            body = {"error": raw_text or "empty response"}
        request_id = resp.headers.get("x-request-id")
        retry_after_hdr = resp.headers.get("retry-after")
        retry_after = float(retry_after_hdr) if retry_after_hdr else None
        rl_limit = _opt_float_header(resp.headers, "x-ratelimit-limit")
        rl_remaining = _opt_float_header(resp.headers, "x-ratelimit-remaining")
        rl_reset = _opt_float_header(resp.headers, "x-ratelimit-reset")
        raise _classify_error(
            resp.status_code,
            body,
            request_id=request_id,
            retry_after=retry_after,
            rate_limit_limit=rl_limit,
            rate_limit_remaining=rl_remaining,
            rate_limit_reset=rl_reset,
            path=resp.request.url.path,
            content_type=resp.headers.get("content-type", ""),
            raw_text=raw_text,
        )
