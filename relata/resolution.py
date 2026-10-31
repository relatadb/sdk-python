"""Entity resolution: rule sets, candidate matches, reviewed merge and unmerge (#5712).

Publish a rule set (``publish_ruleset``, under an ``expected_version``
precondition), ``run`` it to generate candidates, inspect a candidate's
evidence (``get_candidate``) and ``approve``/``reject``/``unmerge`` it. A merge
never rewrites the source records; ``lookup`` and ``get_entity`` answer the
canonical id. ``evaluate`` scores a rule set on labeled pairs against its
approved thresholds. A publish that races another gets HTTP 412
``resolution-version-conflict``; approving a candidate whose source changed
gets 409 ``resolution-evidence-stale``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlencode

from relata._http import AsyncHttpTransport, HttpTransport

if TYPE_CHECKING:
    import httpx

    from relata.client import RelataClient

_BASE = "/v1/resolution"


def _ruleset_path(name: str, version: int | None = None, *, suffix: str = "") -> str:
    path = f"{_BASE}/rulesets/{quote(name, safe='')}"
    if suffix:
        path += f"/{suffix}"
    return path + (f"?version={int(version)}" if version is not None else "")


def _candidate_path(candidate_id: str, suffix: str = "") -> str:
    path = f"{_BASE}/candidates/{quote(candidate_id, safe='')}"
    return f"{path}/{suffix}" if suffix else path


class ResolutionClient:
    """Synchronous client for ``/v1/resolution``."""

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
    def from_client(cls, client: RelataClient) -> ResolutionClient:
        return cls(
            client._base_url,  # noqa: SLF001
            bearer_token=client._bearer_token,  # noqa: SLF001
            timeout=client._timeout,  # noqa: SLF001
            extra_headers=client._extra_headers,  # noqa: SLF001
        )

    def list_rulesets(self) -> dict[str, Any]:
        """``GET /v1/resolution/rulesets``."""
        return dict(self._t.get(f"{_BASE}/rulesets"))

    def get_ruleset(self, name: str, *, version: int | None = None) -> dict[str, Any]:
        """``GET /v1/resolution/rulesets/{name}`` (latest, or ``version``)."""
        return dict(self._t.get(_ruleset_path(name, version)))

    def publish_ruleset(
        self, name: str, definition: dict[str, Any], *, expected_version: int = 0
    ) -> dict[str, Any]:
        """``PUT /v1/resolution/rulesets/{name}`` under the ``expected_version`` precondition."""
        body = {"expected_version": int(expected_version), "definition": definition}
        return dict(self._t.put(_ruleset_path(name), body))

    def run(self, name: str, *, proposals: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """``POST .../run``: generate or refresh candidates (repeat while ``complete`` is false)."""
        body: dict[str, Any] = {} if proposals is None else {"proposals": proposals}
        return dict(self._t.post(_ruleset_path(name, suffix="run"), body))

    def evaluate(
        self,
        name: str,
        labeled: list[dict[str, Any]],
        *,
        proposals: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """``POST .../evaluate``: precision and recall on labeled pairs, nothing persisted."""
        body: dict[str, Any] = {"labeled": labeled}
        if proposals is not None:
            body["proposals"] = proposals
        return dict(self._t.post(_ruleset_path(name, suffix="evaluate"), body))

    def list_candidates(
        self, *, ruleset: str | None = None, state: str | None = None, limit: int | None = None
    ) -> dict[str, Any]:
        """``GET /v1/resolution/candidates``."""
        query = {k: str(v) for k, v in (("ruleset", ruleset), ("state", state), ("limit", limit)) if v is not None}
        suffix = f"?{urlencode(query)}" if query else ""
        return dict(self._t.get(f"{_BASE}/candidates{suffix}"))

    def get_candidate(self, candidate_id: str) -> dict[str, Any]:
        """``GET /v1/resolution/candidates/{id}``: evidence, history and live values."""
        return dict(self._t.get(_candidate_path(candidate_id)))

    def decide(self, candidate_id: str, decision: str, reason: str) -> dict[str, Any]:
        """``POST .../decision`` with ``approve``, ``reject`` or ``unmerge``; idempotent."""
        body = {"decision": decision, "reason": reason}
        return dict(self._t.post(_candidate_path(candidate_id, "decision"), body))

    def approve(self, candidate_id: str, reason: str) -> dict[str, Any]:
        """Approve (merge) a candidate."""
        return self.decide(candidate_id, "approve", reason)

    def reject(self, candidate_id: str, reason: str) -> dict[str, Any]:
        """Reject a candidate; the pair is not proposed again."""
        return self.decide(candidate_id, "reject", reason)

    def unmerge(self, candidate_id: str, reason: str) -> dict[str, Any]:
        """Undo an approved merge."""
        return self.decide(candidate_id, "unmerge", reason)

    def get_entity(self, canonical_id: str) -> dict[str, Any]:
        """``GET /v1/resolution/entities/{canonical}``: follows merge aliases."""
        return dict(self._t.get(f"{_BASE}/entities/{quote(canonical_id, safe='')}"))

    def lookup(self, type_name: str, source_id: str) -> dict[str, Any]:
        """``GET /v1/resolution/lookup``: the canonical id of a source record."""
        query = urlencode({"type": type_name, "id": source_id})
        return dict(self._t.get(f"{_BASE}/lookup?{query}"))

    def close(self) -> None:
        self._t.close()

    def __enter__(self) -> ResolutionClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class AsyncResolutionClient:
    """Asynchronous client -- see :class:`ResolutionClient`."""

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
    def from_client(cls, client: RelataClient) -> AsyncResolutionClient:
        return cls(
            client._base_url,  # noqa: SLF001
            bearer_token=client._bearer_token,  # noqa: SLF001
            timeout=client._timeout,  # noqa: SLF001
            extra_headers=client._extra_headers,  # noqa: SLF001
        )

    async def list_rulesets(self) -> dict[str, Any]:
        return dict(await self._t.get(f"{_BASE}/rulesets"))

    async def get_ruleset(self, name: str, *, version: int | None = None) -> dict[str, Any]:
        return dict(await self._t.get(_ruleset_path(name, version)))

    async def publish_ruleset(
        self, name: str, definition: dict[str, Any], *, expected_version: int = 0
    ) -> dict[str, Any]:
        body = {"expected_version": int(expected_version), "definition": definition}
        return dict(await self._t.put(_ruleset_path(name), body))

    async def run(self, name: str, *, proposals: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {} if proposals is None else {"proposals": proposals}
        return dict(await self._t.post(_ruleset_path(name, suffix="run"), body))

    async def evaluate(
        self,
        name: str,
        labeled: list[dict[str, Any]],
        *,
        proposals: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"labeled": labeled}
        if proposals is not None:
            body["proposals"] = proposals
        return dict(await self._t.post(_ruleset_path(name, suffix="evaluate"), body))

    async def list_candidates(
        self, *, ruleset: str | None = None, state: str | None = None, limit: int | None = None
    ) -> dict[str, Any]:
        query = {k: str(v) for k, v in (("ruleset", ruleset), ("state", state), ("limit", limit)) if v is not None}
        suffix = f"?{urlencode(query)}" if query else ""
        return dict(await self._t.get(f"{_BASE}/candidates{suffix}"))

    async def get_candidate(self, candidate_id: str) -> dict[str, Any]:
        return dict(await self._t.get(_candidate_path(candidate_id)))

    async def decide(self, candidate_id: str, decision: str, reason: str) -> dict[str, Any]:
        body = {"decision": decision, "reason": reason}
        return dict(await self._t.post(_candidate_path(candidate_id, "decision"), body))

    async def approve(self, candidate_id: str, reason: str) -> dict[str, Any]:
        return await self.decide(candidate_id, "approve", reason)

    async def reject(self, candidate_id: str, reason: str) -> dict[str, Any]:
        return await self.decide(candidate_id, "reject", reason)

    async def unmerge(self, candidate_id: str, reason: str) -> dict[str, Any]:
        return await self.decide(candidate_id, "unmerge", reason)

    async def get_entity(self, canonical_id: str) -> dict[str, Any]:
        return dict(await self._t.get(f"{_BASE}/entities/{quote(canonical_id, safe='')}"))

    async def lookup(self, type_name: str, source_id: str) -> dict[str, Any]:
        query = urlencode({"type": type_name, "id": source_id})
        return dict(await self._t.get(f"{_BASE}/lookup?{query}"))

    async def close(self) -> None:
        await self._t.aclose()

    async def __aenter__(self) -> AsyncResolutionClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()
