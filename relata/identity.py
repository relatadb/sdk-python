"""Identity + lookup + GDPR SDK (#79).

Wraps ``/v1/identity/label``, ``/v1/identity/uncertainty``, ``/v1/lookup``,
``/v1/lookup/register``, plus the ``ERASE SUBJECT`` SQL command. The
``ERASE SUBJECT`` path goes through the parent ``RelataClient.query()`` because
the server exposes it as a SQL statement, not a dedicated HTTP route.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

from relata._http import AsyncHttpTransport, HttpTransport, path_segment

if TYPE_CHECKING:
    import httpx

    from relata.client import RelataClient


_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _register_lookup_statement(
    name: str, server_path: str, key_column: str, value_columns: list[str] | None
) -> str:
    cols = [key_column, *(value_columns or [])]
    if not _IDENT.fullmatch(name) or not all(_IDENT.fullmatch(c) for c in cols):
        raise ValueError(
            "register_lookup: name and column names must be identifiers ([A-Za-z_][A-Za-z0-9_]*)"
        )
    if not server_path or any(ch in server_path for ch in "'\r\n"):
        raise ValueError(
            "register_lookup: server_path must be a non-empty path without quotes or newlines"
        )
    fields = f" FIELDS ({', '.join(value_columns)})" if value_columns else ""
    return f"REGISTER LOOKUP {name} FROM '{server_path}' KEY ({key_column}){fields}"


def _lookup_names(data: dict[str, Any]) -> list[dict[str, Any]]:
    names = data.get("lookup_tables")
    return [{"name": n} for n in names] if isinstance(names, list) else []


def _uncertainty_query(
    identity: str,
    reason: str,
    suggested_alternatives: list[str] | None,
    candidates: list[str] | None,
    k: int | None,
) -> str:
    params: dict[str, str] = {"identity": identity, "reason": reason}
    if suggested_alternatives:
        params["suggested_alternatives"] = ",".join(suggested_alternatives)
    if candidates:
        params["candidates"] = ",".join(candidates)
    if k is not None:
        params["k"] = str(k)
    return urlencode(params)


class IdentityClient:
    """Synchronous identity + lookup client."""

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
    def from_client(cls, client: RelataClient) -> IdentityClient:
        return cls(
            client._base_url,
            bearer_token=client._bearer_token,
            timeout=client._timeout,
            extra_headers=client._extra_headers,
        )

    # ------------------------------------------------------------------
    # Identity (#79)
    # ------------------------------------------------------------------

    def label(self, left: str, right: str, *, is_match: bool = True) -> dict[str, Any]:
        """Record a human label on a candidate identity pair.

        ``is_match`` says whether ``left`` and ``right`` refer to the same entity.
        The server takes one learning step on the caller's tenant resolver weights
        and persists the labelled history. Wraps ``POST /v1/identity/label`` with
        ``{left, right, is_match}``.
        """
        return self._t.post(
            "/v1/identity/label", {"left": left, "right": right, "is_match": is_match}
        )

    def record_uncertainty(
        self,
        identity: str,
        reason: str,
        *,
        suggested_alternatives: list[str] | None = None,
        candidates: list[str] | None = None,
        k: int | None = None,
    ) -> dict[str, Any]:
        """Record uncertainty about an identity match. The server ranks
        ``candidates`` (``a|b`` pairs) and returns the top ``k`` most uncertain;
        ``identity``/``reason`` are sent for audit only."""
        return self._t.get(
            "/v1/identity/uncertainty?"
            + _uncertainty_query(identity, reason, suggested_alternatives, candidates, k)
        )

    # ------------------------------------------------------------------
    # Lookup tables (#79) — Splunk `lookup` parity
    # ------------------------------------------------------------------

    def register_lookup(
        self,
        name: str,
        server_path: str,
        *,
        key_column: str,
        value_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Register a lookup table from a CSV file that already exists ON THE SERVER.

        The file must live under ``<data dir>/lookups`` (files elsewhere, or over
        16 MiB, are refused). Wraps ``POST /v1/lookup/register`` with a
        ``text/plain`` ``REGISTER LOOKUP <name> FROM '<path>' KEY (<col>) FIELDS
        (<cols>)`` statement -- the only body the server accepts; it cannot receive
        CSV text. Returns ``{registered, rows}``.
        """
        statement = _register_lookup_statement(name, server_path, key_column, value_columns)
        return self._t.post_raw("/v1/lookup/register", statement, content_type="text/plain")

    def list_lookups(self) -> list[dict[str, Any]]:
        """List registered lookup tables as ``[{"name": ...}]`` (the route returns
        ``{"lookup_tables": [names]}``)."""
        return _lookup_names(self._t.get("/v1/lookup"))

    def invoke_lookup(self, name: str, key: str) -> dict[str, Any]:
        """Fetch the enrichment row for ``key`` in lookup table ``name``.

        Wraps ``GET /v1/lookup/{name}/{key}``; returns ``{name, key, row}`` and
        raises a not-found error when the table or key is absent.
        """
        return self._t.get(f"/v1/lookup/{path_segment(name)}/{path_segment(key)}")

    # ------------------------------------------------------------------
    # ERASE SUBJECT (#79, #61) — SQL path
    # ------------------------------------------------------------------

    def erase_subject(
        self,
        subject_identity: str,
        reason: str,
        *,
        certify: bool | None = None,
        purpose: str,
    ) -> dict[str, Any]:
        """Issue ``ERASE SUBJECT '<id>' REASON '<r>' [CERTIFY]`` via the parent
        client's query path. Returns the server's Art. 17 receipt. ``subject_identity``
        and ``reason`` are bound as ``$1``/``$2`` (#3211).

        ``certify`` is destructive **opt-in**: omitted or ``False`` omits the
        CERTIFY keyword, and the server refuses the erasure with
        "CERTIFY keyword required to confirm irreversible erasure" — matching
        the engine's own safety rail. Only an explicit ``certify=True``
        opts into the request, not provider key-destruction certification. (Historically the omitted
        default was ``True``, which silently confirmed destruction for
        callers that never considered the flag.)

        Pairs with #61 — today the server does governed-tombstone only (the
        DEK survives). Once #61 ships the per-subject DEK destroy, this same
        SDK call must be inspected for actual local acknowledgments; provider/all-copy
        completion remains unverified.
        """
        # This helper is provided on IdentityClient for ergonomics but needs
        # the parent query path; the caller must supply a purpose. We do not
        # hold a back-reference to RelataClient to avoid a circular dep, so
        # the actual call is documented for the caller to make via
        # ``client.query(...)`` using the SQL built here.
        certify_kw = " CERTIFY" if certify else ""
        # subject/reason are bound as $1/$2 via the server-side parameterized
        # path — never interpolated into the SQL text (#3211).
        sql = f"ERASE SUBJECT $1 REASON $2{certify_kw}"
        # We send via /v1/query on this client's transport — the caller's purpose
        # is required and is forwarded as the body purpose.
        return self._t.post(
            "/v1/query", {"purpose": purpose, "sql": sql, "params": [subject_identity, reason]}
        )

    def close(self) -> None:
        self._t.close()

    def __enter__(self) -> IdentityClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class AsyncIdentityClient:
    """Asynchronous identity + lookup client — see :class:`IdentityClient`."""

    def __init__(
        self,
        base_url: str,
        *,
        bearer_token: str | None = None,
        timeout: float = 30.0,
        extra_headers: dict[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._t = AsyncHttpTransport(
            base_url, bearer_token, timeout, transport=transport, extra_headers=extra_headers
        )

    @classmethod
    def from_client(cls, client: RelataClient) -> AsyncIdentityClient:
        return cls(
            client._base_url,
            bearer_token=client._bearer_token,
            timeout=client._timeout,
            extra_headers=client._extra_headers,
        )

    async def label(self, left: str, right: str, *, is_match: bool = True) -> dict[str, Any]:
        return await self._t.post(
            "/v1/identity/label", {"left": left, "right": right, "is_match": is_match}
        )

    async def record_uncertainty(
        self,
        identity: str,
        reason: str,
        *,
        suggested_alternatives: list[str] | None = None,
        candidates: list[str] | None = None,
        k: int | None = None,
    ) -> dict[str, Any]:
        return await self._t.get(
            "/v1/identity/uncertainty?"
            + _uncertainty_query(identity, reason, suggested_alternatives, candidates, k)
        )

    async def register_lookup(
        self,
        name: str,
        server_path: str,
        *,
        key_column: str,
        value_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        statement = _register_lookup_statement(name, server_path, key_column, value_columns)
        return await self._t.post_raw("/v1/lookup/register", statement, content_type="text/plain")

    async def list_lookups(self) -> list[dict[str, Any]]:
        return _lookup_names(await self._t.get("/v1/lookup"))

    async def invoke_lookup(self, name: str, key: str) -> dict[str, Any]:
        return await self._t.get(f"/v1/lookup/{path_segment(name)}/{path_segment(key)}")

    async def erase_subject(
        self,
        subject_identity: str,
        reason: str,
        *,
        certify: bool | None = None,
        purpose: str,
    ) -> dict[str, Any]:
        """Async variant — ``certify`` is destructive opt-in (see the sync
        :meth:`IdentityClient.erase_subject` docstring): omitted or ``False``
        omits the CERTIFY keyword and the server refuses the erasure."""
        certify_kw = " CERTIFY" if certify else ""
        sql = f"ERASE SUBJECT $1 REASON $2{certify_kw}"
        return await self._t.post(
            "/v1/query", {"purpose": purpose, "sql": sql, "params": [subject_identity, reason]}
        )

    async def close(self) -> None:
        await self._t.aclose()

    async def __aenter__(self) -> AsyncIdentityClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()
