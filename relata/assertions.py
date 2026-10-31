"""Temporal assertions: claims with valid time, recorded time and reviewed corrections (#5713).

``record`` a claim about a subject record (contradictory claims are all kept),
``confirm``/``reject`` it as a reviewer, correct it by recording a new claim
with ``supersedes``, and ``revoke`` it. ``resolve`` answers what is true for a
predicate at a valid time ``valid_at`` as known at ``recorded_as_of`` (both
integer nanoseconds), with the verdict and reason for every assertion
considered. Resolution policies are published under an ``expected_version``
precondition (HTTP 412 ``assertion-version-conflict``). A revoke while a legal
hold or WORM retention covers the subject gets 409 ``assertion-retention-blocked``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlencode

from relata._http import AsyncHttpTransport, HttpTransport

if TYPE_CHECKING:
    import httpx

    from relata.client import RelataClient


def _aid(assertion_id: str, suffix: str = "") -> str:
    path = f"/assertions/{quote(assertion_id, safe='')}"
    return f"{path}/{suffix}" if suffix else path


def _policy(name: str, version: int | None = None) -> str:
    path = f"/assertion-policies/{quote(name, safe='')}"
    return path + (f"?version={int(version)}" if version is not None else "")


def _rid(relation_id: str) -> str:
    return f"/assertion-relations/{quote(relation_id, safe='')}/revoke"


def _graph_path(assertion_id: str, depth, limit, policy, policy_version) -> str:  # noqa: ANN001
    query = {k: str(v) for k, v in (("depth", depth), ("limit", limit)) if v is not None}
    if policy is not None:
        query["policy"] = policy
        if policy_version is not None:
            query["policy_version"] = str(int(policy_version))
    return _aid(assertion_id, "evidence-graph") + (f"?{urlencode(query)}" if query else "")


def _list_path(subject_type, subject_id, predicate, state, limit) -> str:  # noqa: ANN001
    pairs = (
        ("subject_type", subject_type),
        ("subject_id", subject_id),
        ("predicate", predicate),
        ("state", state),
        ("limit", limit),
    )
    query = {k: str(v) for k, v in pairs if v is not None}
    return "/assertions" + (f"?{urlencode(query)}" if query else "")


def _resolve_path(subject_type, subject_id, predicate, valid_at, recorded_as_of, policy, policy_version) -> str:  # noqa: ANN001
    query = {
        "subject_type": subject_type,
        "subject_id": subject_id,
        "predicate": predicate,
        "valid_at": str(int(valid_at)),
    }
    if recorded_as_of is not None:
        query["recorded_as_of"] = str(int(recorded_as_of))
    if policy is not None:
        query["policy"] = policy
        if policy_version is not None:
            query["policy_version"] = str(int(policy_version))
    return f"/assertion-resolve?{urlencode(query)}"


class AssertionsClient:
    """Synchronous client for ``/assertions``."""

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
    def from_client(cls, client: RelataClient) -> AssertionsClient:
        return cls(
            client._base_url,  # noqa: SLF001
            bearer_token=client._bearer_token,  # noqa: SLF001
            timeout=client._timeout,  # noqa: SLF001
            extra_headers=client._extra_headers,  # noqa: SLF001
        )

    def record(self, assertion: dict[str, Any]) -> dict[str, Any]:
        """``POST /assertions``: record a claim; idempotent on claim, source version and derivation."""
        return dict(self._t.post("/assertions", assertion))

    def list(
        self,
        *,
        subject_type: str | None = None,
        subject_id: str | None = None,
        predicate: str | None = None,
        state: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """``GET /assertions``."""
        return dict(self._t.get(_list_path(subject_type, subject_id, predicate, state, limit)))

    def get(self, assertion_id: str) -> dict[str, Any]:
        """``GET /assertions/{id}``."""
        return dict(self._t.get(_aid(assertion_id)))

    def decide(self, assertion_id: str, decision: str, reason: str) -> dict[str, Any]:
        """``POST .../decision`` with ``confirm`` or ``reject``; idempotent."""
        body = {"decision": decision, "reason": reason}
        return dict(self._t.post(_aid(assertion_id, "decision"), body))

    def confirm(self, assertion_id: str, reason: str) -> dict[str, Any]:
        """Confirm an assertion."""
        return self.decide(assertion_id, "confirm", reason)

    def reject(self, assertion_id: str, reason: str) -> dict[str, Any]:
        """Reject an assertion; it never resolves while rejected."""
        return self.decide(assertion_id, "reject", reason)

    def revoke(self, assertion_id: str, reason: str) -> dict[str, Any]:
        """``POST .../revoke``: replace the assertion with a content-free tombstone."""
        return dict(self._t.post(_aid(assertion_id, "revoke"), {"reason": reason}))

    def relate(
        self,
        from_id: str,
        kind: str,
        to_id: str,
        rationale: str,
        *,
        locator: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """``POST /assertion-relations`` (#5721): link two assertions (``supports``,
        ``contradicts`` or ``derived_from``) with a rationale and optional locator."""
        body: dict[str, Any] = {"from": from_id, "to": to_id, "kind": kind, "rationale": rationale}
        if locator is not None:
            body["locator"] = locator
        return dict(self._t.post("/assertion-relations", body))

    def unrelate(self, relation_id: str, reason: str) -> dict[str, Any]:
        """``POST /assertion-relations/{id}/revoke``: withdraw a relation."""
        return dict(self._t.post(_rid(relation_id), {"reason": reason}))

    def assess(self, assertion_id: str, judgment: str, reason: str) -> dict[str, Any]:
        """``POST .../assessment``: an analyst judgment (``corroborated``, ``disputed``,
        ``inconclusive``), kept apart from source reliability and model confidence."""
        body = {"judgment": judgment, "reason": reason}
        return dict(self._t.post(_aid(assertion_id, "assessment"), body))

    def evidence_graph(
        self,
        assertion_id: str,
        *,
        depth: int | None = None,
        limit: int | None = None,
        policy: str | None = None,
        policy_version: int | None = None,
    ) -> dict[str, Any]:
        """``GET .../evidence-graph``: bounded authorized neighbourhood plus the
        corroboration summary (counts and separate components, no truth score)."""
        return dict(self._t.get(_graph_path(assertion_id, depth, limit, policy, policy_version)))

    def publish_policy(
        self, name: str, definition: dict[str, Any], *, expected_version: int = 0
    ) -> dict[str, Any]:
        """``PUT /assertion-policies/{name}`` under the ``expected_version`` precondition."""
        body = {"expected_version": int(expected_version), "definition": definition}
        return dict(self._t.put(_policy(name), body))

    def get_policy(self, name: str, *, version: int | None = None) -> dict[str, Any]:
        """``GET /assertion-policies/{name}`` (latest, or ``version``)."""
        return dict(self._t.get(_policy(name, version)))

    def resolve(
        self,
        subject_type: str,
        subject_id: str,
        predicate: str,
        valid_at: int,
        *,
        recorded_as_of: int | None = None,
        policy: str | None = None,
        policy_version: int | None = None,
    ) -> dict[str, Any]:
        """``GET /assertion-resolve``: the value at ``valid_at`` as known at ``recorded_as_of``, with an explanation."""
        path = _resolve_path(
            subject_type, subject_id, predicate, valid_at, recorded_as_of, policy, policy_version
        )
        return dict(self._t.get(path))

    def close(self) -> None:
        self._t.close()

    def __enter__(self) -> AssertionsClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class AsyncAssertionsClient:
    """Asynchronous client for ``/assertions`` -- see :class:`AssertionsClient`."""

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
    def from_client(cls, client: RelataClient) -> AsyncAssertionsClient:
        return cls(
            client._base_url,  # noqa: SLF001
            bearer_token=client._bearer_token,  # noqa: SLF001
            timeout=client._timeout,  # noqa: SLF001
            extra_headers=client._extra_headers,  # noqa: SLF001
        )

    async def record(self, assertion: dict[str, Any]) -> dict[str, Any]:
        return dict(await self._t.post("/assertions", assertion))

    async def list(
        self,
        *,
        subject_type: str | None = None,
        subject_id: str | None = None,
        predicate: str | None = None,
        state: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        return dict(await self._t.get(_list_path(subject_type, subject_id, predicate, state, limit)))

    async def get(self, assertion_id: str) -> dict[str, Any]:
        return dict(await self._t.get(_aid(assertion_id)))

    async def decide(self, assertion_id: str, decision: str, reason: str) -> dict[str, Any]:
        body = {"decision": decision, "reason": reason}
        return dict(await self._t.post(_aid(assertion_id, "decision"), body))

    async def confirm(self, assertion_id: str, reason: str) -> dict[str, Any]:
        return await self.decide(assertion_id, "confirm", reason)

    async def reject(self, assertion_id: str, reason: str) -> dict[str, Any]:
        return await self.decide(assertion_id, "reject", reason)

    async def revoke(self, assertion_id: str, reason: str) -> dict[str, Any]:
        return dict(await self._t.post(_aid(assertion_id, "revoke"), {"reason": reason}))

    async def relate(
        self,
        from_id: str,
        kind: str,
        to_id: str,
        rationale: str,
        *,
        locator: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"from": from_id, "to": to_id, "kind": kind, "rationale": rationale}
        if locator is not None:
            body["locator"] = locator
        return dict(await self._t.post("/assertion-relations", body))

    async def unrelate(self, relation_id: str, reason: str) -> dict[str, Any]:
        return dict(await self._t.post(_rid(relation_id), {"reason": reason}))

    async def assess(self, assertion_id: str, judgment: str, reason: str) -> dict[str, Any]:
        body = {"judgment": judgment, "reason": reason}
        return dict(await self._t.post(_aid(assertion_id, "assessment"), body))

    async def evidence_graph(
        self,
        assertion_id: str,
        *,
        depth: int | None = None,
        limit: int | None = None,
        policy: str | None = None,
        policy_version: int | None = None,
    ) -> dict[str, Any]:
        path = _graph_path(assertion_id, depth, limit, policy, policy_version)
        return dict(await self._t.get(path))

    async def publish_policy(
        self, name: str, definition: dict[str, Any], *, expected_version: int = 0
    ) -> dict[str, Any]:
        body = {"expected_version": int(expected_version), "definition": definition}
        return dict(await self._t.put(_policy(name), body))

    async def get_policy(self, name: str, *, version: int | None = None) -> dict[str, Any]:
        return dict(await self._t.get(_policy(name, version)))

    async def resolve(
        self,
        subject_type: str,
        subject_id: str,
        predicate: str,
        valid_at: int,
        *,
        recorded_as_of: int | None = None,
        policy: str | None = None,
        policy_version: int | None = None,
    ) -> dict[str, Any]:
        path = _resolve_path(
            subject_type, subject_id, predicate, valid_at, recorded_as_of, policy, policy_version
        )
        return dict(await self._t.get(path))

    async def close(self) -> None:
        await self._t.aclose()

    async def __aenter__(self) -> AsyncAssertionsClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()
