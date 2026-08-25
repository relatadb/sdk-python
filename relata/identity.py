"""Identity + lookup + GDPR SDK (#79).

Wraps ``/identity/label``, ``/identity/uncertainty``, ``/lookup``,
``/lookup/register``, plus the ``ERASE SUBJECT`` SQL command. The
``ERASE SUBJECT`` path goes through the parent ``RelataClient.query()`` because
the server exposes it as a SQL statement, not a dedicated HTTP route.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from relata._http import AsyncHttpTransport, HttpTransport

if TYPE_CHECKING:
    import httpx

    from relata.client import RelataClient


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

    def label(
        self,
        identity: str,
        label: str,
        *,
        confidence: float = 1.0,
        source_id: str | None = None,
    ) -> dict[str, Any]:
        """Assign a canonical label to an identity."""
        payload: dict[str, Any] = {
            "identity": identity,
            "label": label,
            "confidence": confidence,
        }
        if source_id is not None:
            payload["source_id"] = source_id
        return self._t.post("/identity/label", payload)

    def record_uncertainty(
        self,
        identity: str,
        reason: str,
        *,
        suggested_alternatives: list[str] | None = None,
    ) -> dict[str, Any]:
        """Record uncertainty about an identity match. The server surfaces
        these for human review."""
        payload: dict[str, Any] = {"identity": identity, "reason": reason}
        if suggested_alternatives is not None:
            payload["suggested_alternatives"] = suggested_alternatives
        # Server exposes uncertainty as GET with query params — adapt.
        from urllib.parse import urlencode

        return self._t.get("/identity/uncertainty?" + urlencode(payload))

    # ------------------------------------------------------------------
    # Lookup tables (#79) — Splunk `lookup` parity
    # ------------------------------------------------------------------

    def register_lookup(
        self,
        name: str,
        csv_data: str,
        *,
        key_column: str,
        value_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Register a lookup table (CSV). Matches Splunk's ``lookup`` semantics."""
        payload: dict[str, Any] = {
            "name": name,
            "csv_data": csv_data,
            "key_column": key_column,
        }
        if value_columns is not None:
            payload["value_columns"] = value_columns
        return self._t.post("/lookup/register", payload)

    def list_lookups(self) -> list[dict[str, Any]]:
        """List registered lookup tables."""
        data = self._t.get("/lookup")
        lookups = data.get("lookups") if isinstance(data, dict) else data
        return lookups if isinstance(lookups, list) else []

    def invoke_lookup(self, name: str, key: str) -> dict[str, Any]:
        """Invoke a registered lookup. Returns the row(s) matching ``key``."""
        from urllib.parse import urlencode

        return self._t.get("/lookup?" + urlencode({"name": name, "key": key}))

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
        confirms the irreversible crypto-shred. (Historically the omitted
        default was ``True``, which silently confirmed destruction for
        callers that never considered the flag.)

        Pairs with #61 — today the server does governed-tombstone only (the
        DEK survives). Once #61 ships the per-subject DEK destroy, this same
        SDK call will produce crypto-shred semantics.
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
        # We send via /query on this client's transport — the caller's purpose
        # is required and is forwarded as the body purpose.
        return self._t.post(
            "/query", {"purpose": purpose, "sql": sql, "params": [subject_identity, reason]}
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

    async def label(
        self,
        identity: str,
        label: str,
        *,
        confidence: float = 1.0,
        source_id: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "identity": identity,
            "label": label,
            "confidence": confidence,
        }
        if source_id is not None:
            payload["source_id"] = source_id
        return await self._t.post("/identity/label", payload)

    async def record_uncertainty(
        self,
        identity: str,
        reason: str,
        *,
        suggested_alternatives: list[str] | None = None,
    ) -> dict[str, Any]:
        from urllib.parse import urlencode

        payload: dict[str, Any] = {"identity": identity, "reason": reason}
        if suggested_alternatives is not None:
            payload["suggested_alternatives"] = suggested_alternatives
        return await self._t.get("/identity/uncertainty?" + urlencode(payload))

    async def register_lookup(
        self,
        name: str,
        csv_data: str,
        *,
        key_column: str,
        value_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": name,
            "csv_data": csv_data,
            "key_column": key_column,
        }
        if value_columns is not None:
            payload["value_columns"] = value_columns
        return await self._t.post("/lookup/register", payload)

    async def list_lookups(self) -> list[dict[str, Any]]:
        data = await self._t.get("/lookup")
        lookups = data.get("lookups") if isinstance(data, dict) else data
        return lookups if isinstance(lookups, list) else []

    async def invoke_lookup(self, name: str, key: str) -> dict[str, Any]:
        from urllib.parse import urlencode

        return await self._t.get("/lookup?" + urlencode({"name": name, "key": key}))

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
            "/query", {"purpose": purpose, "sql": sql, "params": [subject_identity, reason]}
        )

    async def close(self) -> None:
        await self._t.aclose()

    async def __aenter__(self) -> AsyncIdentityClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()
