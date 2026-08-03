"""
Relata SDK exceptions.

All exceptions raised by the SDK are subclasses of :class:`RelataError`.
Callers can catch the base class to handle all SDK errors uniformly, or
catch a specific subclass to handle a particular failure mode.

The v1.1 expansion (#79) adds:

- RFC 7807 ``application/problem+json`` fields (``code``, ``type_url``,
  ``retryable``, ``request_id``) on every server-side error.
- Typed subclasses keyed on HTTP status + ``code``: :class:`ForbiddenError`,
  :class:`NotFoundError`, :class:`ConflictError`, :class:`ValidationError`,
  :class:`RateLimitedError` (with ``retry_after``).
"""

from __future__ import annotations


class RelataError(Exception):
    """Base class for all Relata SDK errors.

    Attributes:
        message: Human-readable error description.
        status_code: HTTP status code from the server, or ``None`` for
            client-side errors (e.g. network failures, validation errors).
        code: The server's RFC 7807 problem+json ``code`` extension field, or
            ``None`` when the response is the legacy ``{"error": "..."}``
            shape. Two vocabularies exist in real responses (#2555): most
            HTTP errors (auth/ACL/ingest/admin/...) use kebab-case, e.g.
            ``"access-denied"``, ``"parse-error"`` (catalogue:
            ``docs/api/error-codes.md``); `/query`'s planner/execution
            errors use the ``QueryError``-derived ``REL_*`` form, e.g.
            ``"REL_PARSE"``, ``"REL_ACL"`` (catalogue:
            ``docs/src/end-users/error-codes.md``). Match on the literal
            string rather than assuming either form — do not rely on
            dotted-form codes like ``"RELATA.QUERY.PURPOSE_REQUIRED"``,
            which were never implemented server-side.
        type_url: RFC 7807 ``type`` URL linking to the error docs.
        retryable: ``True`` when the server says the request can be retried.
        request_id: The ``X-Request-ID`` from the response, when available.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
        type_url: str | None = None,
        retryable: bool = False,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.type_url = type_url
        self.retryable = retryable
        self.request_id = request_id

    def __repr__(self) -> str:
        parts = [f"message={self.message!r}"]
        if self.status_code is not None:
            parts.append(f"status_code={self.status_code}")
        if self.code:
            parts.append(f"code={self.code!r}")
        if self.request_id:
            parts.append(f"request_id={self.request_id!r}")
        return f"{type(self).__name__}({', '.join(parts)})"

    def __str__(self) -> str:
        prefix = f"[HTTP {self.status_code}]" if self.status_code is not None else ""
        rid = f" (request_id={self.request_id})" if self.request_id else ""
        return f"{prefix} {self.message}{rid}".strip()


# ---------------------------------------------------------------------------
# Backward-compatible legacy aliases — the v1.0 constructor signature.
# ---------------------------------------------------------------------------


class PurposeError(RelataError):
    """Raised when a query is rejected due to a missing or invalid purpose."""

    def __init__(
        self,
        message: str,
        status_code: int = 400,
        code: str | None = None,
        type_url: str | None = None,
        retryable: bool = False,
        request_id: str | None = None,
    ) -> None:
        super().__init__(
            message, status_code=status_code, code=code,
            type_url=type_url, retryable=retryable, request_id=request_id,
        )


class QuotaError(RelataError):
    """Raised when the per-principal query-cost quota is exhausted (HTTP 429).

    Alias for :class:`RateLimitedError` kept for backward compat.
    """

    def __init__(
        self,
        message: str,
        status_code: int = 429,
        code: str | None = None,
        type_url: str | None = None,
        retryable: bool = False,
        request_id: str | None = None,
    ) -> None:
        super().__init__(
            message, status_code=status_code, code=code,
            type_url=type_url, retryable=retryable, request_id=request_id,
        )


class AuthError(RelataError):
    """Raised when the server rejects the request as unauthenticated (HTTP 401)."""

    def __init__(
        self,
        message: str,
        status_code: int = 401,
        code: str | None = None,
        type_url: str | None = None,
        retryable: bool = False,
        request_id: str | None = None,
    ) -> None:
        super().__init__(
            message, status_code=status_code, code=code,
            type_url=type_url, retryable=retryable, request_id=request_id,
        )


class ConnectionError(RelataError):
    """Raised when the SDK cannot reach the Relata server."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=None)


class ResponseTooLargeError(RelataError):
    """Raised when a response body exceeds the transport's ``max_response_bytes``
    cap (#3214) — protects the client from OOM via a malicious/buggy server.

    Attributes:
        max_bytes: The configured response cap that was exceeded.
        content_length: The advertised ``Content-Length`` when the server sent
            one (``None`` for chunked/unannounced bodies — the cap is enforced
            on the bytes actually read in that case).
    """

    def __init__(
        self, message: str, *, max_bytes: int, content_length: int | None = None
    ) -> None:
        super().__init__(message, status_code=None)
        self.max_bytes = max_bytes
        self.content_length = content_length


class ServerError(RelataError):
    """Raised when the server returns an unexpected 5xx error."""

    def __init__(
        self,
        message: str,
        status_code: int,
        code: str | None = None,
        type_url: str | None = None,
        retryable: bool = False,
        request_id: str | None = None,
    ) -> None:
        super().__init__(
            message, status_code=status_code, code=code,
            type_url=type_url, retryable=retryable, request_id=request_id,
        )


# ---------------------------------------------------------------------------
# v1.1 typed subclasses (#79)
# ---------------------------------------------------------------------------


class ForbiddenError(RelataError):
    """HTTP 403 — the principal is authenticated but not authorised."""


class NotFoundError(RelataError):
    """HTTP 404 — the requested resource does not exist."""


class ConflictError(RelataError):
    """HTTP 409 — a version or uniqueness conflict."""


class ValidationError(RelataError):
    """HTTP 422 — the request body failed validation."""


class RateLimitedError(RelataError):
    """HTTP 429 — rate limit or quota exceeded.

    Attributes:
        retry_after: Seconds to wait before retrying, from the ``Retry-After``
            header or the problem+json ``retry_after`` extension.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 429,
        retry_after: float | None = None,
        rate_limit_limit: float | None = None,
        rate_limit_remaining: float | None = None,
        rate_limit_reset: float | None = None,
        code: str | None = None,
        type_url: str | None = None,
        retryable: bool = False,
        request_id: str | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=status_code,
            code=code,
            type_url=type_url,
            retryable=retryable,
            request_id=request_id,
        )
        self.retry_after = retry_after
        # #1321: X-RateLimit-* quota headers (None when the server omitted them).
        self.rate_limit_limit = rate_limit_limit
        self.rate_limit_remaining = rate_limit_remaining
        self.rate_limit_reset = rate_limit_reset
