"""
Internal HTTP transport layer.

This module is **not** part of the public API.  All internals are subject to
change without notice.  Callers should use :class:`relata.client.RelataClient`
instead.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, Literal, overload
from urllib.parse import quote

import httpx

from relata._version import __version__
from relata.exceptions import (
    AuthError,
    ConflictError,
    ConnectionError,
    ForbiddenError,
    NotFoundError,
    PreconditionFailedError,
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


@dataclass(frozen=True)
class _RawHttpResponse:
    """Capped buffered bytes and copied headers; the HTTP body is already closed."""
    status_code: int
    headers: dict[str, str]
    content: bytes


def _parse_retry_after(value: str | None) -> float | None:
    """Parse delta seconds or an HTTP date without replacing the HTTP error."""
    if not value:
        return None
    try:
        seconds = float(value)
        if seconds < 0:
            return None
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
            if when.tzinfo is None:
                when = when.replace(tzinfo=UTC)
            seconds = (when - datetime.now(UTC)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    return (
        max(0.0, seconds) if math.isfinite(seconds) and seconds <= 9_007_199_254_740_991 else None
    )


def path_segment(value: str) -> str:
    """Percent-encode a caller-supplied identifier for safe use as ONE URL path
    segment (#3212).

    Every helper that interpolates an id into ``f"/x/{id}"`` must route it
    through this so ``../``, ``?``, ``#``, ``&``, ``/`` can neither traverse the
    path nor smuggle query params. ``safe=""`` encodes even ``/`` — the encoded
    value can never span multiple segments. Mirrors the Go/TS ``pathSegment``
    helpers (parity with #3200).
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
# Parity with the Go SDK (sdks/go/relata/client.go:1259 isIdempotent):
# POST/DELETE/PATCH/PUT are
# NEVER retried — a gateway timeout after commit would double-execute
# irreversible ops (erase_subject, fuse_identities, session_commit).
_IDEMPOTENT_METHODS: frozenset[str] = frozenset({"GET", "HEAD", "OPTIONS"})


# #6302: a 503 whose body names one of these maintenance pauses and says
# ``retryable: true`` was refused before any handler ran (the router-wide
# admission gate, or a bulk group that wrote nothing), so it is retried
# transparently for every verb, honouring ``Retry-After``, within
# ``pause_retry_budget`` seconds of total waiting. ``poisoned`` and
# ``storage_recovering`` are not maintenance pauses and are never retried here.
_MAINTENANCE_PAUSE_REASONS: frozenset[str] = frozenset(
    {"rollover_begin", "rollover_commit", "backup", "restore"}
)
#: Default total seconds spent waiting out maintenance pauses per request.
DEFAULT_PAUSE_RETRY_BUDGET = 30.0
# Longest single wait between pause retries, whatever ``Retry-After`` says.
_PAUSE_RETRY_MAX_DELAY = 5.0
# Wait used when a pause 503 carries no retry hint.
_PAUSE_RETRY_DEFAULT_DELAY = 1.0
# Shortest wait, so a ``Retry-After: 0`` cannot spin.
_PAUSE_RETRY_MIN_DELAY = 0.1


def _maintenance_pause_delay(status: int, headers: Mapping[str, str], raw: bytes) -> float | None:
    """Seconds to wait before retrying a maintenance-pause 503, or ``None``
    when the response is not one (#6302)."""
    if status != 503:
        return None
    try:
        body = json.loads(raw)
    except ValueError:
        return None
    if (
        not isinstance(body, dict)
        or body.get("pause_reason") not in _MAINTENANCE_PAUSE_REASONS
        or body.get("retryable") is not True
    ):
        return None
    delay = _parse_retry_after(headers.get("retry-after"))
    if delay is None:
        delay = _PAUSE_RETRY_DEFAULT_DELAY
        hint = body.get("retry_after_secs")
        if (
            isinstance(hint, int | float)
            and not isinstance(hint, bool)
            and math.isfinite(hint)
            and hint >= 0
        ):
            delay = float(hint)
    return min(max(delay, _PAUSE_RETRY_MIN_DELAY), _PAUSE_RETRY_MAX_DELAY)


class _MaintenancePause(Exception):
    """Internal signal: the attempt hit a retryable maintenance pause (#6302)."""

    def __init__(self, delay: float, error: RelataError) -> None:
        super().__init__(str(error))
        self.delay = delay
        self.error = error


def _is_idempotent(method: str) -> bool:
    """Whether ``method`` is safe to retry (GET/HEAD/OPTIONS only)."""
    return method in _IDEMPOTENT_METHODS


# #5689 (owner decision D06): every bulk ingest request carries an
# ``Idempotency-Key`` so a retry cannot duplicate keyless rows.
_BULK_INGEST_PATH = "/v1/ingest/bulk"
_IDEMPOTENCY_KEY_HEADER = "Idempotency-Key"


def _is_bulk_ingest(method: str, path: str) -> bool:
    """Whether this request is a ``POST /v1/ingest/bulk``."""
    return method == "POST" and path.split("?", 1)[0].rstrip("/").endswith(_BULK_INGEST_PATH)


def _status_retryable(
    status: int, retry_on: frozenset[int], idempotent: bool, keyed_bulk: bool
) -> bool:
    """Whether a response status is retried. A keyed bulk ingest is retried on
    gateway statuses (502/504) but never on a 503 from Relata itself, which
    carries its own ``retryable`` verdict and ``Retry-After`` (#5689)."""
    if status not in retry_on or not idempotent:
        return False
    return not (keyed_bulk and status == 503)


def _with_ingest_idempotency_key(
    method: str, path: str, headers: dict[str, str] | None
) -> dict[str, str] | None:
    """Add an automatic ``Idempotency-Key`` (a random UUID) to a bulk ingest
    request that does not already carry one (#5689 D06, the Kafka/Stripe
    pattern). Called once per call, before the retry loop, so every attempt of
    the call reuses the same key and the server replays or converges instead of
    writing twice. An explicit key is never replaced; other requests are
    returned unchanged."""
    if not _is_bulk_ingest(method, path):
        return headers
    if headers and any(k.lower() == "idempotency-key" for k in headers):
        return headers
    return {**(headers or {}), _IDEMPOTENCY_KEY_HEADER: str(uuid.uuid4())}


def _opt_float_header(headers: Mapping[str, str], name: str) -> float | None:
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
_ADMIN_ONLY_PREFIXES = ("/admin/",)


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
    404 against an admin-only ``path`` (#2321)."""
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
    *args: Any,
    **kwargs: Any,
) -> RelataError:
    """Classify an HTTP error and attach ingest receipt fields (``task_id``,
    ``accepted``, ``receipt_status``) and two-person confirmation fields
    (``confirmation_id``, ``confirmation_expires_at_ns``) when the body carries
    them."""
    error = _classify_error_inner(*args, **kwargs)
    body = args[1] if len(args) > 1 else kwargs.get("body")
    if isinstance(body, dict):
        if isinstance(body.get("task_id"), str) and body["task_id"]:
            error.task_id = body["task_id"]
        if isinstance(body.get("accepted"), bool):
            error.accepted = body["accepted"]
        if isinstance(body.get("receipt_status"), str):
            error.receipt_status = body["receipt_status"]
        if isinstance(body.get("confirmation_id"), str) and body["confirmation_id"]:
            error.confirmation_id = body["confirmation_id"]
        expires = body.get("expires_at_ns")
        if isinstance(expires, int) and not isinstance(expires, bool):
            error.confirmation_expires_at_ns = expires
    return error


def _classify_error_inner(
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
    # A JSON error body may legally decode to null/list/scalar. Preserve its
    # HTTP status rather than raising AttributeError while classifying it.
    if not isinstance(body, dict):
        body = {"error": raw_text or "non-object HTTP error response"}
    code = body.get("code") if isinstance(body.get("code"), str) else None
    type_url = body.get("type") if isinstance(body.get("type"), str) else None
    query_id = body.get("query_id") if isinstance(body.get("query_id"), str) else None
    retryable = body.get("retryable") is True
    rid = request_id or (
        body.get("request_id") if isinstance(body.get("request_id"), str) else None
    )
    detail = next((body[key] for key in ("detail", "error", "message")
                   if isinstance(body.get(key), str) and body[key]), "unknown server error")
    # #2321 (ADR-0261): a bare 404 with no detail text against an
    # admin-only path almost always means this client is pointed at
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

    # Preserve finite delay guidance on every HTTP error, not only429. A valid
    # header wins (including zero); malformed headers permit a finite body fallback.
    def finite_delay(value: object) -> float | None:
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and 0 <= value <= 9_007_199_254_740_991
            and math.isfinite(value)
        ):
            return float(value)
        return None
    delay = finite_delay(retry_after)
    if delay is None:
        delay = finite_delay(body.get("retry_after", body.get("retry_after_seconds")))

    # Shared kwargs for every subclass.
    base_kwargs: dict[str, Any] = {
        "code": code,
        "type_url": type_url,
        "retryable": retryable,
        "request_id": rid,
        "query_id": query_id,
        "retry_after": delay,
    }

    # Purpose-specific detection — works for both legacy and problem+json.
    # A query-engine ``REL_*`` code is authoritative: a SQL parse error (``REL_PARSE``) echoes
    # the statement, which carries its ``PURPOSE '...'`` clause, and must not read as a purpose
    # rejection (#5666). Free-text hints still apply to every other code and to legacy bodies.
    is_purpose = status_code == 400 and (
        ("purpose" in code.lower())
        if code and code.startswith("REL_")
        else bool(code and "purpose" in code.lower())
        or any(hint in lower_detail for hint in _PURPOSE_HINTS)
    )
    if is_purpose:
        return PurposeError(
            f"Purpose rejected: {detail}. "
            "Pass purpose= to RelataClient.query() or set a default on the client.",
            status_code=status_code,
            **base_kwargs,
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
    if status_code == 412:
        current = body.get("current_version")
        failed = body.get("failed_operation")
        return PreconditionFailedError(
            f"Precondition failed: {detail}",
            current_version=current
            if isinstance(current, int) and not isinstance(current, bool)
            else None,
            failed_operation=failed
            if isinstance(failed, int) and not isinstance(failed, bool)
            else None,
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
        pause_retry_budget: float = DEFAULT_PAUSE_RETRY_BUDGET,
    ) -> None:
        headers = _build_headers(bearer_token, extra_headers, compress=compress)
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout,
            transport=transport,
        )
        # #2321 (ADR-0261): /admin/* requests route to a
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
        # #6302: total wait for maintenance-pause retries; 0 disables them.
        self._pause_retry_budget = pause_retry_budget

    def _request_id(self) -> str:
        return str(uuid.uuid4())

    def _client_for(self, path: str) -> httpx.Client:
        """Resolve the httpx client a request to ``path`` should use:
        the admin client for ``/admin/*`` when configured,
        the ordinary client otherwise (unchanged behaviour when unset)."""
        if self._admin_client is not None and _is_admin_only_path(path):
            return self._admin_client
        return self._client

    @overload
    def _send_with_retry(
        self, method: str, path: str, *, json_payload: dict[str, Any] | None = None,
        content: str | bytes | None = None, headers: dict[str, str] | None = None,
        raw_response: Literal[False] = False,
    ) -> dict[str, Any]: ...

    @overload
    def _send_with_retry(
        self, method: str, path: str, *, json_payload: dict[str, Any] | None = None,
        content: str | bytes | None = None, headers: dict[str, str] | None = None,
        raw_response: Literal[True],
    ) -> _RawHttpResponse: ...

    def _send_with_retry(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
        content: str | bytes | None = None,
        headers: dict[str, str] | None = None,
        raw_response: bool = False,
    ) -> dict[str, Any] | _RawHttpResponse:
        """Send with retry, waiting out maintenance pauses (#6302).

        A 503 maintenance pause (see :func:`_maintenance_pause_delay`) is
        retried for any verb, sleeping its ``Retry-After`` (at most 5 s) until
        ``pause_retry_budget`` seconds have been spent; then its error is
        raised. Pause retries do not count against ``max_retries``.
        """
        waited = 0.0
        while True:
            try:
                return self._send_attempts(
                    method, path, json_payload=json_payload, content=content,
                    headers=headers, raw_response=raw_response,
                )
            except _MaintenancePause as pause:
                delay = min(pause.delay, self._pause_retry_budget - waited)
                if delay <= 0:
                    raise pause.error from None
                time.sleep(delay)
                waited += delay

    def _send_attempts(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
        content: str | bytes | None = None,
        headers: dict[str, str] | None = None,
        raw_response: bool = False,
    ) -> dict[str, Any] | _RawHttpResponse:
        """Send an HTTP request with retry + request_id propagation.

        Retry policy (#2490): only idempotent verbs (GET/HEAD/OPTIONS) are
        retried, and only on transport errors (connect/timeout) or response
        status codes in ``self._retry_on`` (502/503/504 by default). POST and
        other non-idempotent verbs are never retried — parity with the
        TypeScript and Go SDKs. The one exception is ``POST /v1/ingest/bulk``,
        which always carries an ``Idempotency-Key`` (automatic when the caller
        sent none, #5689) reused on every attempt: it is retried on transport
        errors and 502/504, but not on a 503 from Relata (see
        :func:`_status_retryable`).
        """
        client = self._client_for(path)
        # If the caller already set X-Request-ID in the default headers bag,
        # respect it — only generate a per-attempt UUID when no default exists.
        has_default_rid = "x-request-id" in {k.lower() for k in client.headers}
        headers = _with_ingest_idempotency_key(method, path, headers)
        # A keyed bulk ingest is safe to re-send: the server dedupes by key.
        keyed_bulk = _is_bulk_ingest(method, path)
        idempotent = _is_idempotent(method) or keyed_bulk
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
            except httpx.TransportError as exc:
                # Connection reset / protocol error after the request may have
                # been sent: the effect is uncertain, so surface it as a
                # RelataError and never replay a non-idempotent verb.
                last_exc = ConnectionError(
                    f"Connection to Relata server failed mid-request: {exc}. "
                    "The request may have been applied; reconcile before retrying."
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
                    _status_retryable(resp.status_code, self._retry_on, idempotent, keyed_bulk)
                    and attempt < self._max_retries
                ):
                    time.sleep(self._retry_backoff * (2**attempt))
                    continue

                if resp.status_code == 503 and self._pause_retry_budget > 0:
                    raw = self._read_capped(resp)
                    delay = _maintenance_pause_delay(resp.status_code, resp.headers, raw)
                    if delay is not None:
                        try:
                            self._handle(resp, raw)
                        except RelataError as err:
                            raise _MaintenancePause(delay, err) from None
                    return self._handle(resp, raw)

                # Success or non-retryable error — return / raise via _handle.
                if raw_response and resp.is_success:
                    return _RawHttpResponse(
                        resp.status_code, dict(resp.headers), self._read_capped(resp)
                    )
                return self._handle(resp)
            finally:
                resp.close()

        # Exhausted retries on a retryable status code (transport errors raise
        # directly from the except block above; only the status-code path can
        # land here, and the final iteration's _handle call always returns or
        # raises — so this is a defensive unreachable fallback).
        assert last_exc is not None
        raise last_exc

    def request_bytes(
        self, method: str, path: str, *, content: bytes | None = None,
        content_type: str | None = None, accept: str = "application/octet-stream",
        extra_headers: dict[str, str] | None = None,
    ) -> _RawHttpResponse:
        """Use the normal auth/error/retry pipeline with capped byte success bodies."""
        headers = {"Accept": accept}
        if content_type is not None:
            headers["Content-Type"] = content_type
        if extra_headers:
            headers.update(extra_headers)
        return self._send_with_retry(method, path, content=content,
            headers=headers, raw_response=True)

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

    def delete(
        self,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Perform a DELETE request and return the decoded JSON body."""
        return self._send_with_retry("DELETE", path, json_payload=payload, headers=extra_headers)

    def patch(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Perform a PATCH request with an optional JSON body."""
        return self._send_with_retry("PATCH", path, json_payload=payload or {})

    def put(
        self,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Perform a PUT request with an optional JSON body."""
        return self._send_with_retry(
            "PUT", path, json_payload=payload or {}, headers=extra_headers
        )

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

    def _handle(self, resp: httpx.Response, raw: bytes | None = None) -> dict[str, Any]:
        if raw is None:
            raw = self._read_capped(resp)
        raw_text = raw.decode("utf-8", errors="replace") if raw else ""
        if resp.is_success:
            if not raw.strip():  # 204 / empty success body (e.g. DELETE)
                return {}
            return json.loads(raw)  # type: ignore[no-any-return]
        try:
            body: dict[str, Any] = json.loads(raw)
            if not isinstance(body, dict):
                raise ValueError("error body is not a JSON object")
        except Exception:
            body = {"error": raw_text or "empty response"}
        request_id = resp.headers.get("x-request-id")
        retry_after = _parse_retry_after(resp.headers.get("retry-after"))
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
        pause_retry_budget: float = DEFAULT_PAUSE_RETRY_BUDGET,
    ) -> None:
        headers = _build_headers(bearer_token, extra_headers, compress=compress)
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout,
            transport=transport,
        )
        # #2321 (ADR-0261): see HttpTransport.__init__ for the rationale --
        # /admin/* routes to a second client bound to
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
        # #6302: total wait for maintenance-pause retries; 0 disables them.
        self._pause_retry_budget = pause_retry_budget

    def _request_id(self) -> str:
        return str(uuid.uuid4())

    def _client_for(self, path: str) -> httpx.AsyncClient:
        """See :meth:`HttpTransport._client_for`."""
        if self._admin_client is not None and _is_admin_only_path(path):
            return self._admin_client
        return self._client

    @overload
    async def _send_with_retry(
        self, method: str, path: str, *, json_payload: dict[str, Any] | None = None,
        content: str | bytes | None = None, headers: dict[str, str] | None = None,
        raw_response: Literal[False] = False,
    ) -> dict[str, Any]: ...

    @overload
    async def _send_with_retry(
        self, method: str, path: str, *, json_payload: dict[str, Any] | None = None,
        content: str | bytes | None = None, headers: dict[str, str] | None = None,
        raw_response: Literal[True],
    ) -> _RawHttpResponse: ...

    async def _send_with_retry(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
        content: bytes | str | None = None,
        headers: dict[str, str] | None = None,
        raw_response: bool = False,
    ) -> dict[str, Any] | _RawHttpResponse:
        """Async twin of :meth:`HttpTransport._send_with_retry` (#6302)."""
        import asyncio

        waited = 0.0
        while True:
            try:
                return await self._send_attempts(
                    method, path, json_payload=json_payload, content=content,
                    headers=headers, raw_response=raw_response,
                )
            except _MaintenancePause as pause:
                delay = min(pause.delay, self._pause_retry_budget - waited)
                if delay <= 0:
                    raise pause.error from None
                await asyncio.sleep(delay)
                waited += delay

    async def _send_attempts(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
        content: bytes | str | None = None,
        headers: dict[str, str] | None = None,
        raw_response: bool = False,
    ) -> dict[str, Any] | _RawHttpResponse:
        """Send an async HTTP request with retry + request_id propagation.

        Retry policy mirrors the sync transport (#2490): idempotent verbs
        only (GET/HEAD/OPTIONS), on transport errors or 502/503/504.
        """
        import asyncio

        client = self._client_for(path)
        headers = _with_ingest_idempotency_key(method, path, headers)
        # A keyed bulk ingest is safe to re-send: the server dedupes by key.
        keyed_bulk = _is_bulk_ingest(method, path)
        idempotent = _is_idempotent(method) or keyed_bulk
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
            except httpx.TransportError as exc:
                last_exc = ConnectionError(
                    f"Connection to Relata server failed mid-request: {exc}. "
                    "The request may have been applied; reconcile before retrying."
                )
                if idempotent and attempt < self._max_retries:
                    await asyncio.sleep(self._retry_backoff * (2**attempt))
                    continue
                raise last_exc from exc
            try:
                if (
                    _status_retryable(resp.status_code, self._retry_on, idempotent, keyed_bulk)
                    and attempt < self._max_retries
                ):
                    await asyncio.sleep(self._retry_backoff * (2**attempt))
                    continue
                if resp.status_code == 503 and self._pause_retry_budget > 0:
                    raw = await self._read_capped(resp)
                    delay = _maintenance_pause_delay(resp.status_code, resp.headers, raw)
                    if delay is not None:
                        try:
                            await self._handle(resp, raw)
                        except RelataError as err:
                            raise _MaintenancePause(delay, err) from None
                    return await self._handle(resp, raw)
                if raw_response and resp.is_success:
                    return _RawHttpResponse(
                        resp.status_code, dict(resp.headers), await self._read_capped(resp)
                    )
                return await self._handle(resp)
            finally:
                await resp.aclose()
        assert last_exc is not None
        raise last_exc

    async def request_bytes(
        self, method: str, path: str, *, content: bytes | None = None,
        content_type: str | None = None, accept: str = "application/octet-stream",
        extra_headers: dict[str, str] | None = None,
    ) -> _RawHttpResponse:
        """Use the normal auth/error/retry pipeline with capped byte success bodies."""
        headers = {"Accept": accept}
        if content_type is not None:
            headers["Content-Type"] = content_type
        if extra_headers:
            headers.update(extra_headers)
        return await self._send_with_retry(method, path, content=content,
            headers=headers, raw_response=True)

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

    async def delete(
        self,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Perform an async DELETE request."""
        return await self._send_with_retry(
            "DELETE", path, json_payload=payload, headers=extra_headers
        )

    async def patch(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Perform an async PATCH request with an optional JSON body."""
        return await self._send_with_retry("PATCH", path, json_payload=payload or {})

    async def put(
        self,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Perform an async PUT request with an optional JSON body."""
        return await self._send_with_retry(
            "PUT", path, json_payload=payload or {}, headers=extra_headers
        )

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

    async def _handle(self, resp: httpx.Response, raw: bytes | None = None) -> dict[str, Any]:
        if raw is None:
            raw = await self._read_capped(resp)
        raw_text = raw.decode("utf-8", errors="replace") if raw else ""
        if resp.is_success:
            if not raw.strip():  # 204 / empty success body (e.g. DELETE)
                return {}
            return json.loads(raw)  # type: ignore[no-any-return]
        try:
            body: dict[str, Any] = json.loads(raw)
            if not isinstance(body, dict):
                raise ValueError("error body is not a JSON object")
        except Exception:
            body = {"error": raw_text or "empty response"}
        request_id = resp.headers.get("x-request-id")
        retry_after = _parse_retry_after(resp.headers.get("retry-after"))
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
