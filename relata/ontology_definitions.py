"""Versioned ontology definitions, publication and migration (#5711).

Create a definition set, edit its draft, ``compare`` it with the published
version (changes classified compatible/migration/breaking/refused, with the
real row impact), ``publish`` under an ``expected_version`` precondition and,
when rows must be rewritten, ``run_migration`` until it completes. A publish
that races another gets HTTP 412 ``ontology-version-conflict``; one that needs
consent gets 409 until ``accept_migration``/``accept_breaking`` is set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlencode

from relata._http import AsyncHttpTransport, HttpTransport

if TYPE_CHECKING:
    import httpx

    from relata.client import RelataClient

_BASE = "/v1/ontology/definitions"


def _p(name: str, *rest: str) -> str:
    return "/".join([f"{_BASE}/{quote(name, safe='')}", *rest])


def _publish_body(
    expected_version: int,
    accept_migration: bool,
    accept_breaking: bool,
    from_version: int | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"expected_version": int(expected_version)}
    if accept_migration:
        body["accept_migration"] = True
    if accept_breaking:
        body["accept_breaking"] = True
    if from_version is not None:
        body["from_version"] = int(from_version)
    return body


def _run_body(max_rows: int | None, retry_failed: bool) -> dict[str, Any]:
    body: dict[str, Any] = {}
    if max_rows is not None:
        body["max_rows"] = int(max_rows)
    if retry_failed:
        body["retry_failed"] = True
    return body


def _compare_path(name: str, from_version: int | str | None, to: int | str | None) -> str:
    query = {k: str(v) for k, v in (("from", from_version), ("to", to)) if v is not None}
    return _p(name, "compare") + (f"?{urlencode(query)}" if query else "")


class OntologyDefinitionsClient:
    """Synchronous client for ``/v1/ontology/definitions``."""

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
    def from_client(cls, client: RelataClient) -> OntologyDefinitionsClient:
        return cls(
            client._base_url,  # noqa: SLF001
            bearer_token=client._bearer_token,  # noqa: SLF001
            timeout=client._timeout,  # noqa: SLF001
            extra_headers=client._extra_headers,  # noqa: SLF001
        )

    def list(self) -> dict[str, Any]:
        """``GET /v1/ontology/definitions``."""
        return dict(self._t.get(_BASE))

    def create(self, name: str, definition: dict[str, Any]) -> dict[str, Any]:
        """``POST /v1/ontology/definitions``: a new set with an initial draft."""
        return dict(self._t.post(_BASE, {"name": name, "definition": definition}))

    def get(self, name: str) -> dict[str, Any]:
        """``GET /v1/ontology/definitions/{name}``: head and current draft."""
        return dict(self._t.get(_p(name)))

    def save_draft(self, name: str, definition: dict[str, Any]) -> dict[str, Any]:
        """``PUT /v1/ontology/definitions/{name}/draft``."""
        return dict(self._t.put(_p(name, "draft"), {"definition": definition}))

    def validate(self, name: str, definition: dict[str, Any] | None = None) -> dict[str, Any]:
        """``POST .../validate``: the draft, or ``definition`` when given."""
        body = {} if definition is None else {"definition": definition}
        return dict(self._t.post(_p(name, "validate"), body))

    def compare(
        self, name: str, *, from_version: int | str | None = None, to: int | str | None = None
    ) -> dict[str, Any]:
        """``GET .../compare``: classified changes plus row impact."""
        return dict(self._t.get(_compare_path(name, from_version, to)))

    def publish(
        self,
        name: str,
        *,
        expected_version: int,
        accept_migration: bool = False,
        accept_breaking: bool = False,
        from_version: int | None = None,
    ) -> dict[str, Any]:
        """``POST .../publish`` under the ``expected_version`` precondition."""
        body = _publish_body(expected_version, accept_migration, accept_breaking, from_version)
        return dict(self._t.post(_p(name, "publish"), body))

    def version(self, name: str, version: int) -> dict[str, Any]:
        """``GET .../versions/{version}``: an immutable published version."""
        return dict(self._t.get(_p(name, "versions", str(int(version)))))

    def status(self, name: str) -> dict[str, Any]:
        """``GET .../status``: row counts by definition version, stale rows."""
        return dict(self._t.get(_p(name, "status")))

    def migration(self, name: str, version: int) -> dict[str, Any]:
        """``GET .../migrations/{version}``: progress, failures, cursor."""
        return dict(self._t.get(_p(name, "migrations", str(int(version)))))

    def run_migration(
        self,
        name: str,
        version: int,
        *,
        max_rows: int | None = None,
        retry_failed: bool = False,
    ) -> dict[str, Any]:
        """``POST .../migrations/{version}/run``: rewrite up to ``max_rows`` and resume later."""
        path = _p(name, "migrations", str(int(version)), "run")
        return dict(self._t.post(path, _run_body(max_rows, retry_failed)))

    def close(self) -> None:
        self._t.close()

    def __enter__(self) -> OntologyDefinitionsClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class AsyncOntologyDefinitionsClient:
    """Asynchronous client -- see :class:`OntologyDefinitionsClient`."""

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
    def from_client(cls, client: RelataClient) -> AsyncOntologyDefinitionsClient:
        return cls(
            client._base_url,  # noqa: SLF001
            bearer_token=client._bearer_token,  # noqa: SLF001
            timeout=client._timeout,  # noqa: SLF001
            extra_headers=client._extra_headers,  # noqa: SLF001
        )

    async def list(self) -> dict[str, Any]:
        return dict(await self._t.get(_BASE))

    async def create(self, name: str, definition: dict[str, Any]) -> dict[str, Any]:
        return dict(await self._t.post(_BASE, {"name": name, "definition": definition}))

    async def get(self, name: str) -> dict[str, Any]:
        return dict(await self._t.get(_p(name)))

    async def save_draft(self, name: str, definition: dict[str, Any]) -> dict[str, Any]:
        return dict(await self._t.put(_p(name, "draft"), {"definition": definition}))

    async def validate(self, name: str, definition: dict[str, Any] | None = None) -> dict[str, Any]:
        body = {} if definition is None else {"definition": definition}
        return dict(await self._t.post(_p(name, "validate"), body))

    async def compare(
        self, name: str, *, from_version: int | str | None = None, to: int | str | None = None
    ) -> dict[str, Any]:
        return dict(await self._t.get(_compare_path(name, from_version, to)))

    async def publish(
        self,
        name: str,
        *,
        expected_version: int,
        accept_migration: bool = False,
        accept_breaking: bool = False,
        from_version: int | None = None,
    ) -> dict[str, Any]:
        body = _publish_body(expected_version, accept_migration, accept_breaking, from_version)
        return dict(await self._t.post(_p(name, "publish"), body))

    async def version(self, name: str, version: int) -> dict[str, Any]:
        return dict(await self._t.get(_p(name, "versions", str(int(version)))))

    async def status(self, name: str) -> dict[str, Any]:
        return dict(await self._t.get(_p(name, "status")))

    async def migration(self, name: str, version: int) -> dict[str, Any]:
        return dict(await self._t.get(_p(name, "migrations", str(int(version)))))

    async def run_migration(
        self,
        name: str,
        version: int,
        *,
        max_rows: int | None = None,
        retry_failed: bool = False,
    ) -> dict[str, Any]:
        path = _p(name, "migrations", str(int(version)), "run")
        return dict(await self._t.post(path, _run_body(max_rows, retry_failed)))

    async def close(self) -> None:
        await self._t.aclose()

    async def __aenter__(self) -> AsyncOntologyDefinitionsClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()
