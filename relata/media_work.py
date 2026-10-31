"""Media work hand-off for OpenETL (``/v1/media/work``, #5938).

Relata keeps a durable queue of media files that need extraction. A worker
(OpenETL) leases items with :meth:`MediaWorkClient.claim`, keeps the lease alive
with :meth:`MediaWorkClient.heartbeat`, writes the extracted rows through
``IngestClient.bulk(..., source_checkpoint=..., media_work_claim=claim_id)`` and
then acknowledges with :meth:`MediaWorkClient.complete` or
:meth:`MediaWorkClient.fail`. Completion succeeds only when the batch and child
counts equal what Relata durably holds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlencode

from relata._http import AsyncHttpTransport, HttpTransport

if TYPE_CHECKING:
    import httpx

    from relata.client import RelataClient

#: Header every bulk write-back for an ``openetl/media/<work_id>`` source must carry.
CLAIM_HEADER = "X-Relata-Claim"

_STATES = ("queued", "claimed", "writing", "completed", "dead_letter", "cancelled")
_FAILURE_CLASSES = ("retryable", "permanent")


def media_work_claim_headers(claim_id: int) -> dict[str, str]:
    """``X-Relata-Claim`` header for a raw ``POST /v1/ingest/bulk`` write-back."""
    return {CLAIM_HEADER: str(int(claim_id))}


def _item_path(work_id: str, suffix: str = "") -> str:
    return f"/v1/media/work/{quote(work_id, safe='')}{suffix}"


def _list_path(state: str | None, limit: int | None) -> str:
    params: dict[str, str] = {}
    if state is not None:
        if state not in _STATES:
            raise ValueError(f"state must be one of {', '.join(_STATES)}")
        params["state"] = state
    if limit is not None:
        params["limit"] = str(int(limit))
    return "/v1/media/work" + (f"?{urlencode(params)}" if params else "")


def _claim_body(claimant: str, max_items: int | None, lease_ms: int | None) -> dict[str, Any]:
    body: dict[str, Any] = {"claimant": claimant}
    if max_items is not None:
        body["max_items"] = int(max_items)
    if lease_ms is not None:
        body["lease_ms"] = int(lease_ms)
    return body


def _heartbeat_body(claim_id: int, lease_ms: int | None) -> dict[str, Any]:
    body: dict[str, Any] = {"claim_id": int(claim_id)}
    if lease_ms is not None:
        body["lease_ms"] = int(lease_ms)
    return body


def _complete_body(
    claim_id: int, producer_version: str, batches: int, children: int
) -> dict[str, Any]:
    return {
        "claim_id": int(claim_id),
        "producer_version": producer_version,
        "batches": int(batches),
        "children": int(children),
    }


def _fail_body(
    claim_id: int, failure_class: str, code: str | None, detail: str | None
) -> dict[str, Any]:
    if failure_class not in _FAILURE_CLASSES:
        raise ValueError("failure_class must be 'retryable' or 'permanent'")
    body: dict[str, Any] = {"claim_id": int(claim_id), "class": failure_class}
    if code is not None:
        body["code"] = code
    if detail is not None:
        body["detail"] = detail
    return body


class MediaWorkClient:
    """Synchronous media-work client."""

    def __init__(
        self,
        base_url: str,
        *,
        bearer_token: str | None = None,
        timeout: float = 60.0,
        extra_headers: dict[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._t = HttpTransport(
            base_url, bearer_token, timeout, transport=transport, extra_headers=extra_headers
        )

    @classmethod
    def from_client(cls, client: RelataClient) -> MediaWorkClient:
        return cls(
            client._base_url,  # noqa: SLF001
            bearer_token=client._bearer_token,  # noqa: SLF001
            timeout=client._timeout,  # noqa: SLF001
            extra_headers=client._extra_headers,  # noqa: SLF001
        )

    def list(self, *, state: str | None = None, limit: int | None = None) -> dict[str, Any]:
        """``GET /v1/media/work``: the tenant's work items, optionally filtered by state."""
        return dict(self._t.get(_list_path(state, limit)))

    def get(self, work_id: str) -> dict[str, Any]:
        """``GET /v1/media/work/{id}``: durable status; 404 when unknown."""
        return dict(self._t.get(_item_path(work_id)))

    def claim(
        self, claimant: str, *, max_items: int | None = None, lease_ms: int | None = None
    ) -> dict[str, Any]:
        """``POST /v1/media/work/claim``: lease up to ``max_items`` items (``{"items": [...]}``).

        An empty ``items`` list means nothing is deliverable now (no long poll).
        Each item carries the ``claim_id`` the other calls must present.
        """
        return dict(
            self._t.post("/v1/media/work/claim", _claim_body(claimant, max_items, lease_ms))
        )

    def heartbeat(
        self, work_id: str, claim_id: int, *, lease_ms: int | None = None
    ) -> dict[str, Any]:
        """``POST /v1/media/work/{id}/heartbeat``: extend the lease (409 once it lapsed)."""
        return dict(
            self._t.post(_item_path(work_id, "/heartbeat"), _heartbeat_body(claim_id, lease_ms))
        )

    def complete(
        self, work_id: str, claim_id: int, *, producer_version: str, batches: int, children: int
    ) -> dict[str, Any]:
        """``POST /v1/media/work/{id}/complete``: 409 when counts differ from what is stored."""
        return dict(
            self._t.post(
                _item_path(work_id, "/complete"),
                _complete_body(claim_id, producer_version, batches, children),
            )
        )

    def fail(
        self,
        work_id: str,
        claim_id: int,
        *,
        failure_class: str,
        code: str | None = None,
        detail: str | None = None,
    ) -> dict[str, Any]:
        """``POST /v1/media/work/{id}/fail``: ``retryable`` is redelivered, ``permanent``
        dead-letters.
        """
        return dict(
            self._t.post(
                _item_path(work_id, "/fail"), _fail_body(claim_id, failure_class, code, detail)
            )
        )

    def requeue(self, work_id: str) -> dict[str, Any]:
        """``POST /v1/media/work/{id}/requeue``: return an item (for example a
        dead-lettered one) to the queue so it is delivered again."""
        return dict(self._t.post(_item_path(work_id, "/requeue"), {}))

    def close(self) -> None:
        self._t.close()

    def __enter__(self) -> MediaWorkClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class AsyncMediaWorkClient:
    """Asynchronous media-work client -- see :class:`MediaWorkClient`."""

    def __init__(
        self,
        base_url: str,
        *,
        bearer_token: str | None = None,
        timeout: float = 60.0,
        extra_headers: dict[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._t = AsyncHttpTransport(
            base_url, bearer_token, timeout, transport=transport, extra_headers=extra_headers
        )

    @classmethod
    def from_client(cls, client: RelataClient) -> AsyncMediaWorkClient:
        return cls(
            client._base_url,  # noqa: SLF001
            bearer_token=client._bearer_token,  # noqa: SLF001
            timeout=client._timeout,  # noqa: SLF001
            extra_headers=client._extra_headers,  # noqa: SLF001
        )

    async def list(self, *, state: str | None = None, limit: int | None = None) -> dict[str, Any]:
        return dict(await self._t.get(_list_path(state, limit)))

    async def get(self, work_id: str) -> dict[str, Any]:
        return dict(await self._t.get(_item_path(work_id)))

    async def claim(
        self, claimant: str, *, max_items: int | None = None, lease_ms: int | None = None
    ) -> dict[str, Any]:
        return dict(
            await self._t.post("/v1/media/work/claim", _claim_body(claimant, max_items, lease_ms))
        )

    async def heartbeat(
        self, work_id: str, claim_id: int, *, lease_ms: int | None = None
    ) -> dict[str, Any]:
        return dict(
            await self._t.post(
                _item_path(work_id, "/heartbeat"), _heartbeat_body(claim_id, lease_ms)
            )
        )

    async def complete(
        self, work_id: str, claim_id: int, *, producer_version: str, batches: int, children: int
    ) -> dict[str, Any]:
        return dict(
            await self._t.post(
                _item_path(work_id, "/complete"),
                _complete_body(claim_id, producer_version, batches, children),
            )
        )

    async def fail(
        self,
        work_id: str,
        claim_id: int,
        *,
        failure_class: str,
        code: str | None = None,
        detail: str | None = None,
    ) -> dict[str, Any]:
        return dict(
            await self._t.post(
                _item_path(work_id, "/fail"), _fail_body(claim_id, failure_class, code, detail)
            )
        )

    async def requeue(self, work_id: str) -> dict[str, Any]:
        return dict(await self._t.post(_item_path(work_id, "/requeue"), {}))

    async def close(self) -> None:
        await self._t.aclose()

    async def __aenter__(self) -> AsyncMediaWorkClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()
