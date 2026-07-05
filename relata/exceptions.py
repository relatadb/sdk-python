"""
Relata SDK exceptions.

All exceptions raised by the SDK are subclasses of :class:`RelataError`.
Callers can catch the base class to handle all SDK errors uniformly, or
catch a specific subclass to handle a particular failure mode.
"""

from __future__ import annotations


class RelataError(Exception):
    """Base class for all Relata SDK errors.

    Attributes:
        message: Human-readable error description.
        status_code: HTTP status code from the server, or ``None`` for
            client-side errors (e.g. network failures, validation errors).
    """

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"message={self.message!r}, "
            f"status_code={self.status_code!r})"
        )

    def __str__(self) -> str:
        if self.status_code is not None:
            return f"[HTTP {self.status_code}] {self.message}"
        return self.message


class PurposeError(RelataError):
    """Raised when a query is rejected due to a missing or invalid purpose.

    Every Relata query must declare a ``purpose`` registered in the tenant's
    ``PurposeRegistry`` (e.g. ``"analytics"``, ``"audit"``, ``"analysis"``).
    Queries that omit the purpose or supply an unregistered purpose string are
    rejected at the wire layer (HTTP 400).

    Fix: pass ``purpose=`` to :meth:`RelataClient.query` or set a default on
    the client with ``RelataClient(..., purpose="analytics")``.
    """

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message, status_code)


class QuotaError(RelataError):
    """Raised when the per-principal query-cost quota is exhausted (HTTP 429).

    Relata enforces hard per-organization / per-principal quotas on query cost units.
    When the quota is exceeded the server returns HTTP 429 and the SDK raises
    this exception rather than silently returning partial results.

    Fix: reduce query scope (add ``LIMIT``, narrow ``WHERE`` clauses, reduce
    ``max_hops`` on graph traversals) or contact your Relata administrator to
    increase the quota for your principal.
    """

    def __init__(self, message: str, status_code: int = 429) -> None:
        super().__init__(message, status_code)


class AuthError(RelataError):
    """Raised when the server rejects the request as unauthenticated (HTTP 401).

    Relata optionally requires a Bearer token set via ``RELATA_BEARER_TOKEN``
    on the server.  Pass ``bearer_token=`` to :class:`RelataClient` to
    authenticate.

    Fix: pass ``bearer_token="<your token>"`` to :class:`RelataClient`.
    """

    def __init__(self, message: str, status_code: int = 401) -> None:
        super().__init__(message, status_code)


class ConnectionError(RelataError):
    """Raised when the SDK cannot reach the Relata server.

    This wraps network-level failures (DNS resolution, TCP connection refused,
    TLS handshake errors, timeouts).  The ``status_code`` is always ``None``
    because no HTTP response was received.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=None)


class ServerError(RelataError):
    """Raised when the server returns an unexpected 5xx error."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message, status_code)
