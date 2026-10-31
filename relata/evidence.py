"""Retrieval evidence (``relata.evidence/v1``, #5707).

Send :func:`evidence_headers` with a ``/v1/query``, ``/v1/search``, ``/v1/graphql`` or
MCP read to record version-pinned citations for its result. Fetch the record
with :meth:`EvidenceClient.get`, and re-read every citation under the
caller's *current* authorization with :meth:`EvidenceClient.replay`. Replay
verifies the recorded citations; it does not recompute ranking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from relata._http import AsyncHttpTransport, HttpTransport

if TYPE_CHECKING:
    import httpx

    from relata.client import RelataClient


def evidence_headers(*, retention_days: int | None = None) -> dict[str, str]:
    """Headers that record evidence for a read (retention 1 to 365 days, default 30)."""
    headers = {"X-Relata-Record-Evidence": "true"}
    if retention_days is not None:
        headers["X-Relata-Evidence-Retention-Days"] = str(int(retention_days))
    return headers


def _replay_body(include_content: bool) -> dict[str, Any]:
    return {"include_content": True} if include_content else {}


def _path(evidence_id: str) -> str:
    return f"/v1/evidence/{quote(evidence_id, safe='')}"


class EvidenceClient:
    """Synchronous evidence client."""

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
    def from_client(cls, client: RelataClient) -> EvidenceClient:
        return cls(
            client._base_url,  # noqa: SLF001
            bearer_token=client._bearer_token,  # noqa: SLF001
            timeout=client._timeout,  # noqa: SLF001
            extra_headers=client._extra_headers,  # noqa: SLF001
        )

    def list(self, *, limit: int | None = None) -> dict[str, Any]:
        """``GET /v1/evidence``: the caller's unexpired records, newest first."""
        path = "/v1/evidence" if limit is None else f"/v1/evidence?limit={int(limit)}"
        return dict(self._t.get(path))

    def get(self, evidence_id: str) -> dict[str, Any]:
        """``GET /v1/evidence/{id}``: 404 when unknown, 410 once expired."""
        return dict(self._t.get(_path(evidence_id)))

    def replay(self, evidence_id: str, *, include_content: bool = False) -> dict[str, Any]:
        """``POST /v1/evidence/{id}/replay``: per-citation
        verified/hash_mismatch/missing/denied/deleted.
        """
        return dict(self._t.post(_path(evidence_id) + "/replay", _replay_body(include_content)))

    def close(self) -> None:
        self._t.close()

    def __enter__(self) -> EvidenceClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class AsyncEvidenceClient:
    """Asynchronous evidence client -- see :class:`EvidenceClient`."""

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
    def from_client(cls, client: RelataClient) -> AsyncEvidenceClient:
        return cls(
            client._base_url,  # noqa: SLF001
            bearer_token=client._bearer_token,  # noqa: SLF001
            timeout=client._timeout,  # noqa: SLF001
            extra_headers=client._extra_headers,  # noqa: SLF001
        )

    async def list(self, *, limit: int | None = None) -> dict[str, Any]:
        path = "/v1/evidence" if limit is None else f"/v1/evidence?limit={int(limit)}"
        return dict(await self._t.get(path))

    async def get(self, evidence_id: str) -> dict[str, Any]:
        return dict(await self._t.get(_path(evidence_id)))

    async def replay(self, evidence_id: str, *, include_content: bool = False) -> dict[str, Any]:
        return dict(
            await self._t.post(_path(evidence_id) + "/replay", _replay_body(include_content))
        )

    async def close(self) -> None:
        await self._t.aclose()

    async def __aenter__(self) -> AsyncEvidenceClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()
