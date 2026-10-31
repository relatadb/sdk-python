"""Application primitives SDK (#5768, #5783): atomic sequences and a durable job queue.

Wraps ``POST /v1/sequences/{name}/next``, ``GET /v1/sequences/{name}``,
``POST /v1/queues/{name}/jobs``, ``POST /v1/queues/{name}/claim``,
``POST /v1/jobs/{id}/{renew,complete,fail}``, ``GET /v1/queues/{name}/jobs`` and
``GET /v1/queues/{name}/jobs/{id}``.

A claim returns a fencing token; ``renew``/``complete``/``fail`` must present
it, and a stale token raises the transport's 409 error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

from relata._http import HttpTransport, path_segment

if TYPE_CHECKING:
    import httpx

    from relata.client import RelataClient


class AppPrimitivesClient:
    """Synchronous client for sequences and the durable job queue."""

    def __init__(
        self,
        base_url: str,
        *,
        bearer_token: str | None = None,
        timeout: float = 30.0,
        extra_headers: dict[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._t = HttpTransport(
            base_url, bearer_token, timeout, transport=transport, extra_headers=extra_headers
        )

    @classmethod
    def from_client(cls, client: RelataClient) -> AppPrimitivesClient:
        return cls(
            client._base_url,  # noqa: SLF001
            bearer_token=client._bearer_token,  # noqa: SLF001
            timeout=client._timeout,  # noqa: SLF001
            extra_headers=client._extra_headers,  # noqa: SLF001
        )

    # -- sequences ---------------------------------------------------------

    def sequence_next(
        self, name: str, *, count: int | None = None, idempotency_key: str | None = None
    ) -> dict[str, Any]:
        """Allocate ``count`` (default 1) contiguous values; returns
        ``{name, start, count, end, next, duplicate}``."""
        body: dict[str, Any] = {}
        if count is not None:
            body["count"] = count
        if idempotency_key is not None:
            body["idempotencyKey"] = idempotency_key
        return self._t.post(f"/v1/sequences/{path_segment(name)}/next", body)

    def sequence_get(self, name: str) -> dict[str, Any]:
        """Return ``{name, next}`` without allocating."""
        return self._t.get(f"/v1/sequences/{path_segment(name)}")

    # -- queue -------------------------------------------------------------

    def enqueue(
        self,
        queue: str,
        payload: Any = None,
        *,
        priority: int | None = None,
        run_at: str | int | None = None,
        delay_ms: int | None = None,
        max_attempts: int | None = None,
        backoff_ms: int | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Enqueue a job. A repeated ``idempotency_key`` returns the original
        job with ``duplicate: true``."""
        body: dict[str, Any] = {"payload": payload}
        for key, value in (
            ("priority", priority),
            ("runAt", run_at),
            ("delayMs", delay_ms),
            ("maxAttempts", max_attempts),
            ("backoffMs", backoff_ms),
            ("idempotencyKey", idempotency_key),
        ):
            if value is not None:
                body[key] = value
        return self._t.post(f"/v1/queues/{path_segment(queue)}/jobs", body)

    def claim(
        self, queue: str, worker_id: str, *, lease_ms: int | None = None, max: int | None = None
    ) -> list[dict[str, Any]]:
        """Lease up to ``max`` ready jobs; each carries its ``fencingToken``."""
        body: dict[str, Any] = {"workerId": worker_id}
        if lease_ms is not None:
            body["leaseMs"] = lease_ms
        if max is not None:
            body["max"] = max
        resp = self._t.post(f"/v1/queues/{path_segment(queue)}/claim", body)
        return list(resp.get("jobs", []))

    def renew(self, job_id: str, fencing_token: int, *, lease_ms: int | None = None) -> dict[str, Any]:
        """Extend a held lease. A stale token is rejected with 409."""
        body: dict[str, Any] = {"fencingToken": fencing_token}
        if lease_ms is not None:
            body["leaseMs"] = lease_ms
        return self._t.post(f"/v1/jobs/{path_segment(job_id)}/renew", body)

    def complete(self, job_id: str, fencing_token: int, *, result: Any = None) -> dict[str, Any]:
        """Mark a held job done. A stale token is rejected with 409."""
        body: dict[str, Any] = {"fencingToken": fencing_token}
        if result is not None:
            body["result"] = result
        return self._t.post(f"/v1/jobs/{path_segment(job_id)}/complete", body)

    def fail(
        self,
        job_id: str,
        fencing_token: int,
        *,
        error: str | None = None,
        retry_after_ms: int | None = None,
        retryable: bool | None = None,
    ) -> dict[str, Any]:
        """Report failure: retried with backoff, then dead-lettered at ``maxAttempts``."""
        body: dict[str, Any] = {"fencingToken": fencing_token}
        for key, value in (("error", error), ("retryAfterMs", retry_after_ms), ("retryable", retryable)):
            if value is not None:
                body[key] = value
        return self._t.post(f"/v1/jobs/{path_segment(job_id)}/fail", body)

    def list_jobs(
        self, queue: str, *, state: str | None = None, limit: int | None = None
    ) -> dict[str, Any]:
        """Inspect a queue: ``{queue, counts, jobs}`` (``state="dead"`` lists dead letters)."""
        params = {k: v for k, v in (("state", state), ("limit", limit)) if v is not None}
        suffix = f"?{urlencode(params)}" if params else ""
        return self._t.get(f"/v1/queues/{path_segment(queue)}/jobs{suffix}")

    def get_job(self, queue: str, job_id: str) -> dict[str, Any]:
        """Fetch one job."""
        return self._t.get(f"/v1/queues/{path_segment(queue)}/jobs/{path_segment(job_id)}")

    def close(self) -> None:
        self._t.close()

    def __enter__(self) -> AppPrimitivesClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
