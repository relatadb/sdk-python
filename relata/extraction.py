"""Ontology extraction suggestions and the ETL existence lookup (#5678).

OpenETL extracts entities, relationships and schema proposals; Relata stores
them. ``submit`` a bounded batch (every suggestion starts ``pending``; a
submission cannot carry a review state), work the review queue with ``list`` /
``get``, and ``accept`` / ``reject`` / ``correct`` / ``reopen`` each one. The
original payload is always kept beside the correction and every decision.
Schema suggestions (``object``, ``property``, ``link``) that were accepted fold
into a definition draft with ``apply``, to be published as the next ontology
version. ``entities_exist`` answers, for up to 500 candidates, which already
exist and with which ids. Every ontology read returns an ``ontology_version``
token; pass it back on a suggestion (``ontology_version``) and the queue marks
it ``stale`` once the ontology has moved on.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlencode

from relata._http import AsyncHttpTransport, HttpTransport

if TYPE_CHECKING:
    import httpx

    from relata.client import RelataClient


def _sid(suggestion_id: str, suffix: str = "") -> str:
    path = f"/ontology/suggestions/{quote(suggestion_id, safe='')}"
    return f"{path}/{suffix}" if suffix else path


def _list_path(
    state: str | None,
    kind: str | None,
    definition: str | None,
    low_confidence: bool | None,
    stale: bool | None,
    min_confidence: float | None,
    limit: int | None,
) -> str:
    pairs = (
        ("state", state),
        ("kind", kind),
        ("definition", definition),
        ("low_confidence", None if low_confidence is None else str(bool(low_confidence)).lower()),
        ("stale", None if stale is None else str(bool(stale)).lower()),
        ("min_confidence", min_confidence),
        ("limit", limit),
    )
    query = {k: str(v) for k, v in pairs if v is not None}
    return "/ontology/suggestions" + (f"?{urlencode(query)}" if query else "")


def _decision(
    decision: str,
    reason: str,
    correction: dict[str, Any] | None,
    accept_low_confidence: bool,
) -> dict[str, Any]:
    body: dict[str, Any] = {"decision": decision, "reason": reason}
    if correction is not None:
        body["correction"] = correction
    if accept_low_confidence:
        body["accept_low_confidence"] = True
    return body


def _apply_body(definition: str, suggestion_ids: Sequence[str] | None) -> dict[str, Any]:
    body: dict[str, Any] = {"definition": definition}
    if suggestion_ids is not None:
        body["suggestion_ids"] = list(suggestion_ids)
    return body


def _exist_body(candidates: Sequence[dict[str, Any]], definition: str | None) -> dict[str, Any]:
    body: dict[str, Any] = {"candidates": list(candidates)}
    if definition is not None:
        body["definition"] = definition
    return body


class ExtractionClient:
    """Synchronous client for ``/ontology/suggestions`` and ``/entities/exists``."""

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
    def from_client(cls, client: RelataClient) -> ExtractionClient:
        return cls(
            client._base_url,  # noqa: SLF001
            bearer_token=client._bearer_token,  # noqa: SLF001
            timeout=client._timeout,  # noqa: SLF001
            extra_headers=client._extra_headers,  # noqa: SLF001
        )

    def submit(self, suggestions: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """``POST /ontology/suggestions``: store 1 to 200 suggestions; each result is
        ``created``, ``existing`` (an exact retry) or ``invalid`` with ``issues``."""
        return dict(self._t.post("/ontology/suggestions", {"suggestions": list(suggestions)}))

    def list(
        self,
        *,
        state: str | None = None,
        kind: str | None = None,
        definition: str | None = None,
        low_confidence: bool | None = None,
        stale: bool | None = None,
        min_confidence: float | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """``GET /ontology/suggestions``: the review queue, oldest first."""
        path = _list_path(state, kind, definition, low_confidence, stale, min_confidence, limit)
        return dict(self._t.get(path))

    def get(self, suggestion_id: str) -> dict[str, Any]:
        """``GET /ontology/suggestions/{id}``: with its correction and decision history."""
        return dict(self._t.get(_sid(suggestion_id)))

    def decide(
        self,
        suggestion_id: str,
        decision: str,
        reason: str = "",
        *,
        correction: dict[str, Any] | None = None,
        accept_low_confidence: bool = False,
    ) -> dict[str, Any]:
        """``POST .../decision``: ``accept``, ``reject``, ``correct`` or ``reopen``; idempotent."""
        body = _decision(decision, reason, correction, accept_low_confidence)
        return dict(self._t.post(_sid(suggestion_id, "decision"), body))

    def accept(
        self, suggestion_id: str, reason: str = "", *, accept_low_confidence: bool = False
    ) -> dict[str, Any]:
        """Accept a suggestion as extracted."""
        return self.decide(
            suggestion_id, "accept", reason, accept_low_confidence=accept_low_confidence
        )

    def reject(self, suggestion_id: str, reason: str) -> dict[str, Any]:
        """Reject a suggestion; a reprocessed run of the same claim inherits this."""
        return self.decide(suggestion_id, "reject", reason)

    def correct(
        self, suggestion_id: str, correction: dict[str, Any], reason: str
    ) -> dict[str, Any]:
        """Amend a suggestion's payload; the original is kept."""
        return self.decide(suggestion_id, "correct", reason, correction=correction)

    def reopen(self, suggestion_id: str, reason: str = "") -> dict[str, Any]:
        """Return a decided suggestion to ``pending``."""
        return self.decide(suggestion_id, "reopen", reason)

    def apply(
        self, definition: str, *, suggestion_ids: Sequence[str] | None = None
    ) -> dict[str, Any]:
        """``POST /ontology/suggestions/apply``: fold accepted schema suggestions into
        the definition's draft; publish it with the ontology-definitions client."""
        return dict(
            self._t.post("/ontology/suggestions/apply", _apply_body(definition, suggestion_ids))
        )

    def entities_exist(
        self, candidates: Sequence[dict[str, Any]], *, definition: str | None = None
    ) -> dict[str, Any]:
        """``POST /entities/exists``: for up to 500 ``{"type", "match": {field: value}}``
        candidates, whether a live record matches exactly and with which ids."""
        return dict(self._t.post("/entities/exists", _exist_body(candidates, definition)))

    def close(self) -> None:
        self._t.close()

    def __enter__(self) -> ExtractionClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class AsyncExtractionClient:
    """Asynchronous client -- see :class:`ExtractionClient`."""

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
    def from_client(cls, client: RelataClient) -> AsyncExtractionClient:
        return cls(
            client._base_url,  # noqa: SLF001
            bearer_token=client._bearer_token,  # noqa: SLF001
            timeout=client._timeout,  # noqa: SLF001
            extra_headers=client._extra_headers,  # noqa: SLF001
        )

    async def submit(self, suggestions: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return dict(await self._t.post("/ontology/suggestions", {"suggestions": list(suggestions)}))

    async def list(
        self,
        *,
        state: str | None = None,
        kind: str | None = None,
        definition: str | None = None,
        low_confidence: bool | None = None,
        stale: bool | None = None,
        min_confidence: float | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        path = _list_path(state, kind, definition, low_confidence, stale, min_confidence, limit)
        return dict(await self._t.get(path))

    async def get(self, suggestion_id: str) -> dict[str, Any]:
        return dict(await self._t.get(_sid(suggestion_id)))

    async def decide(
        self,
        suggestion_id: str,
        decision: str,
        reason: str = "",
        *,
        correction: dict[str, Any] | None = None,
        accept_low_confidence: bool = False,
    ) -> dict[str, Any]:
        body = _decision(decision, reason, correction, accept_low_confidence)
        return dict(await self._t.post(_sid(suggestion_id, "decision"), body))

    async def accept(
        self, suggestion_id: str, reason: str = "", *, accept_low_confidence: bool = False
    ) -> dict[str, Any]:
        return await self.decide(
            suggestion_id, "accept", reason, accept_low_confidence=accept_low_confidence
        )

    async def reject(self, suggestion_id: str, reason: str) -> dict[str, Any]:
        return await self.decide(suggestion_id, "reject", reason)

    async def correct(
        self, suggestion_id: str, correction: dict[str, Any], reason: str
    ) -> dict[str, Any]:
        return await self.decide(suggestion_id, "correct", reason, correction=correction)

    async def reopen(self, suggestion_id: str, reason: str = "") -> dict[str, Any]:
        return await self.decide(suggestion_id, "reopen", reason)

    async def apply(
        self, definition: str, *, suggestion_ids: Sequence[str] | None = None
    ) -> dict[str, Any]:
        body = _apply_body(definition, suggestion_ids)
        return dict(await self._t.post("/ontology/suggestions/apply", body))

    async def entities_exist(
        self, candidates: Sequence[dict[str, Any]], *, definition: str | None = None
    ) -> dict[str, Any]:
        return dict(await self._t.post("/entities/exists", _exist_body(candidates, definition)))

    async def close(self) -> None:
        await self._t.aclose()

    async def __aenter__(self) -> AsyncExtractionClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()
