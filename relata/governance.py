"""Governance SDK — rules, retention, breakglass, alerts, DSAR (#74).

A thin governed wrapper over the server's HTTP surface so a Python operator
can:

- Define and manage detection rules (including Sigma rule ingestion).
- Place and lift legal holds; configure WORM retention per object type.
- Request and approve emergency HUMINT breakglass access (4h, two-person).
- List and ack alerts; tail the alert SSE stream.
- File a GDPR Data Subject Access Request.

Every call carries the caller's ``purpose`` and ``tenant`` (set on the parent
``RelataClient``); governance is on by default.

Synchronous and asynchronous variants are provided for every method. Use as a
context manager to guarantee the underlying HTTP connection pool is closed::

    from relata.governance import GovernanceClient

    with GovernanceClient.from_client(client) as g:
        rule_id = g.create_rule({...})
        g.import_sigma(sigma_yaml)
        hold_id = g.place_legal_hold("case-42", "Person")

For async apps::

    async with AsyncGovernanceClient.from_client(client) as g:
        rule_id = await g.create_rule({...})
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from relata._http import AsyncHttpTransport, HttpTransport

if TYPE_CHECKING:
    import httpx

    from relata.client import RelataClient


def _optional_str_or_none(v: object) -> str | None:
    """Return ``str(v)`` if truthy else ``None`` — keeps query-string builds tidy."""
    if v is None:
        return None
    s = str(v)
    return s or None


class _BaseGovernance:
    """Shared state for sync and async governance clients."""

    def __init__(
        self,
        base_url: str,
        *,
        bearer_token: str | None = None,
        purpose: str | None = None,
        tenant: str | None = None,
        timeout: float = 30.0,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self._purpose = purpose
        self._tenant = tenant
        # Pre-build the headers bag so callers using ``from_client`` inherit
        # the parent client's tenant/acting-as/delegated-by plus their own
        # caller-supplied headers.
        extra: dict[str, str] = {}
        if tenant is not None:
            extra["X-Organization-Id"] = tenant
        if extra_headers:
            extra.update(extra_headers)
        self._extra_headers = extra or None
        self._base_url = base_url
        self._bearer_token = bearer_token
        self._timeout = timeout

    @classmethod
    def from_client(cls, client: RelataClient) -> _BaseGovernance:
        """Construct a governance client that inherits the parent's auth/tenant.

        The returned client opens its own HTTP transport so closing the parent
        does not close the governance child. Use as a context manager.
        """
        return cls(
            client._base_url,
            bearer_token=client._bearer_token,
            purpose=client._default_purpose,
            tenant=client._tenant,
            timeout=client._timeout,
            extra_headers=client._extra_headers,
        )


class GovernanceClient(_BaseGovernance):
    """Synchronous governance surface — rules / retention / breakglass / alerts / DSAR."""

    def __init__(
        self,
        base_url: str,
        *,
        bearer_token: str | None = None,
        purpose: str | None = None,
        tenant: str | None = None,
        timeout: float = 30.0,
        extra_headers: dict[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(
            base_url,
            bearer_token=bearer_token,
            purpose=purpose,
            tenant=tenant,
            timeout=timeout,
            extra_headers=extra_headers,
        )
        self._t = HttpTransport(
            base_url,
            bearer_token,
            timeout,
            transport=transport,
            extra_headers=self._extra_headers,
        )

    # ------------------------------------------------------------------
    # Rules (#74)
    # ------------------------------------------------------------------

    def list_rules(self, *, object_type: str | None = None) -> list[dict[str, Any]]:
        """List detection rules. Optional ``object_type`` filter."""
        path = "/rules"
        if object_type:
            path += f"?object_type={object_type}"
        data = self._t.get(path)
        rules = data.get("rules") if isinstance(data, dict) else data
        return rules if isinstance(rules, list) else []

    def create_rule(self, rule: dict[str, Any]) -> dict[str, Any]:
        """Create a detection rule. The dict shape matches the server's
        ``RuleSpec`` (``name``, ``object_type``, ``condition``, ``action``, ...).
        Returns the created rule record including its server-assigned ``id``.
        """
        return self._t.post("/rules", rule)

    def disable_rule(self, rule_id: str) -> dict[str, Any]:
        """Disable (logically delete) a rule by id."""
        return self._t.delete(f"/rules/{rule_id}")

    def import_sigma(self, sigma_yaml: str) -> dict[str, Any]:
        """Import a Sigma rule (YAML string). Returns the import summary
        (``rules_imported``, ``rules_skipped``, ``errors``)."""
        return self._t.post("/rules/sigma", {"sigma": sigma_yaml})

    # ------------------------------------------------------------------
    # Retention (#74)
    # ------------------------------------------------------------------

    def list_retention_policies(self) -> list[dict[str, Any]]:
        """List configured retention policies."""
        data = self._t.get("/retention/policies")
        policies = data.get("policies") if isinstance(data, dict) else data
        return policies if isinstance(policies, list) else []

    def list_legal_holds(self) -> list[dict[str, Any]]:
        """List active legal holds."""
        data = self._t.get("/retention/holds")
        holds = data.get("holds") if isinstance(data, dict) else data
        return holds if isinstance(holds, list) else []

    def place_legal_hold(
        self,
        case_id: str,
        object_type: str,
        *,
        object_id: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Place a legal hold on ``object_type`` (optionally on a specific row).

        Args:
            case_id: Caller-supplied case identifier (becomes the hold id).
            object_type: Type to hold.
            object_id: Optional specific row id; omit for type-wide hold.
            reason: Optional human-friendly reason.
        """
        payload: dict[str, Any] = {
            "case_id": case_id,
            "object_type": object_type,
        }
        if object_id is not None:
            payload["object_id"] = object_id
        if reason is not None:
            payload["reason"] = reason
        return self._t.post("/retention/holds", payload)

    def lift_legal_hold(self, case_id: str) -> dict[str, Any]:
        """Lift (remove) a legal hold by case id."""
        return self._t.delete(f"/retention/holds/{case_id}")

    def list_worm_policies(self) -> list[dict[str, Any]]:
        """List WORM (write-once-read-many) retention policies."""
        data = self._t.get("/retention/worm")
        policies = data.get("policies") if isinstance(data, dict) else data
        return policies if isinstance(policies, list) else []

    def set_worm_policy(
        self,
        object_type: str,
        *,
        retention_secs: int,
    ) -> dict[str, Any]:
        """Set WORM retention for ``object_type``. Rows cannot be mutated or
        purged until ``retention_secs`` elapses from their ``system_from``."""
        return self._t.post(
            f"/retention/worm/{object_type}",
            {"retention_secs": retention_secs},
        )

    # ------------------------------------------------------------------
    # Breakglass (#74)
    # ------------------------------------------------------------------

    def request_breakglass(
        self,
        reason: str,
        *,
        scope: str | None = None,
        duration_secs: int = 4 * 60 * 60,
    ) -> dict[str, Any]:
        """Request emergency HUMINT breakglass access (default 4 h).

        Returns the request record including ``request_id`` and ``status``.
        Approval requires two distinct officers (``approve_breakglass``).
        """
        payload: dict[str, Any] = {
            "reason": reason,
            "duration_secs": duration_secs,
        }
        if scope is not None:
            payload["scope"] = scope
        return self._t.post("/humint/breakglass/request", payload)

    def approve_breakglass(
        self,
        request_id: str,
        *,
        approver_note: str | None = None,
    ) -> dict[str, Any]:
        """Approve a breakglass request (second-officer sign-off).

        The server enforces: requester cannot approve their own request, the
        approver must belong to the same tenant, and two distinct approvals
        are required before access is granted.
        """
        payload: dict[str, Any] = {"request_id": request_id}
        if approver_note is not None:
            payload["note"] = approver_note
        return self._t.post("/humint/breakglass/approve", payload)

    def breakglass_status(self, request_id: str) -> dict[str, Any]:
        """Look up the status of a breakglass request."""
        return self._t.get(f"/humint/breakglass/status/{request_id}")

    # ------------------------------------------------------------------
    # Alerts (#74)
    # ------------------------------------------------------------------

    def list_alerts(
        self,
        *,
        severity: str | None = None,
        since_ns: int | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List alerts, optionally filtered by severity / since-cursor."""
        params: dict[str, str] = {"limit": str(limit)}
        if severity:
            params["severity"] = severity
        if since_ns is not None:
            params["since_ns"] = str(since_ns)
        from urllib.parse import urlencode

        data = self._t.get("/alerts/list?" + urlencode(params))
        alerts = data.get("alerts") if isinstance(data, dict) else data
        return alerts if isinstance(alerts, list) else []

    def update_alert(
        self,
        alert_id: str,
        *,
        status: str | None = None,
        assignee: str | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        """Update an alert (ack, assign, close, add a note)."""
        payload: dict[str, Any] = {}
        if status is not None:
            payload["status"] = status
        if assignee is not None:
            payload["assignee"] = assignee
        if note is not None:
            payload["note"] = note
        return self._t.patch(f"/alerts/update/{alert_id}", payload)

    # ------------------------------------------------------------------
    # DSAR (#74, #79)
    # ------------------------------------------------------------------

    def submit_dsar(
        self,
        subject_identity: str,
        *,
        reason: str,
        scope: str | None = None,
    ) -> dict[str, Any]:
        """File a GDPR Data Subject Access Request.

        Args:
            subject_identity: The canonical identifier of the subject
                (email, phone, national id, ...).
            reason: Grounds for the request (``"gdpr-art-15"``, ``"court-order"``, ...).
            scope: Optional scope filter (``"email"``, ``"financial"``, ...).
        """
        from urllib.parse import urlencode

        params: dict[str, str] = {
            "subject_identity": subject_identity,
            "reason": reason,
        }
        if scope is not None:
            params["scope"] = scope
        return self._t.get("/gdpr/dsar?" + urlencode(params))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._t.close()

    def __enter__(self) -> GovernanceClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class AsyncGovernanceClient(_BaseGovernance):
    """Asynchronous governance surface — see :class:`GovernanceClient`."""

    def __init__(
        self,
        base_url: str,
        *,
        bearer_token: str | None = None,
        purpose: str | None = None,
        tenant: str | None = None,
        timeout: float = 30.0,
        extra_headers: dict[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(
            base_url,
            bearer_token=bearer_token,
            purpose=purpose,
            tenant=tenant,
            timeout=timeout,
            extra_headers=extra_headers,
        )
        self._t = AsyncHttpTransport(
            base_url,
            bearer_token,
            timeout,
            transport=transport,
            extra_headers=self._extra_headers,
        )

    async def list_rules(self, *, object_type: str | None = None) -> list[dict[str, Any]]:
        path = "/rules"
        if object_type:
            path += f"?object_type={object_type}"
        data = await self._t.get(path)
        rules = data.get("rules") if isinstance(data, dict) else data
        return rules if isinstance(rules, list) else []

    async def create_rule(self, rule: dict[str, Any]) -> dict[str, Any]:
        return await self._t.post("/rules", rule)

    async def disable_rule(self, rule_id: str) -> dict[str, Any]:
        return await self._t.delete(f"/rules/{rule_id}")

    async def import_sigma(self, sigma_yaml: str) -> dict[str, Any]:
        return await self._t.post("/rules/sigma", {"sigma": sigma_yaml})

    async def list_legal_holds(self) -> list[dict[str, Any]]:
        data = await self._t.get("/retention/holds")
        holds = data.get("holds") if isinstance(data, dict) else data
        return holds if isinstance(holds, list) else []

    async def place_legal_hold(
        self,
        case_id: str,
        object_type: str,
        *,
        object_id: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"case_id": case_id, "object_type": object_type}
        if object_id is not None:
            payload["object_id"] = object_id
        if reason is not None:
            payload["reason"] = reason
        return await self._t.post("/retention/holds", payload)

    async def lift_legal_hold(self, case_id: str) -> dict[str, Any]:
        return await self._t.delete(f"/retention/holds/{case_id}")

    async def list_worm_policies(self) -> list[dict[str, Any]]:
        data = await self._t.get("/retention/worm")
        policies = data.get("policies") if isinstance(data, dict) else data
        return policies if isinstance(policies, list) else []

    async def set_worm_policy(self, object_type: str, *, retention_secs: int) -> dict[str, Any]:
        return await self._t.post(
            f"/retention/worm/{object_type}",
            {"retention_secs": retention_secs},
        )

    async def request_breakglass(
        self,
        reason: str,
        *,
        scope: str | None = None,
        duration_secs: int = 4 * 60 * 60,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"reason": reason, "duration_secs": duration_secs}
        if scope is not None:
            payload["scope"] = scope
        return await self._t.post("/humint/breakglass/request", payload)

    async def approve_breakglass(
        self,
        request_id: str,
        *,
        approver_note: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"request_id": request_id}
        if approver_note is not None:
            payload["note"] = approver_note
        return await self._t.post("/humint/breakglass/approve", payload)

    async def breakglass_status(self, request_id: str) -> dict[str, Any]:
        return await self._t.get(f"/humint/breakglass/status/{request_id}")

    async def list_alerts(
        self,
        *,
        severity: str | None = None,
        since_ns: int | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        from urllib.parse import urlencode

        params: dict[str, str] = {"limit": str(limit)}
        if severity:
            params["severity"] = severity
        if since_ns is not None:
            params["since_ns"] = str(since_ns)
        data = await self._t.get("/alerts/list?" + urlencode(params))
        alerts = data.get("alerts") if isinstance(data, dict) else data
        return alerts if isinstance(alerts, list) else []

    async def submit_dsar(
        self,
        subject_identity: str,
        *,
        reason: str,
        scope: str | None = None,
    ) -> dict[str, Any]:
        from urllib.parse import urlencode

        params: dict[str, str] = {
            "subject_identity": subject_identity,
            "reason": reason,
        }
        if scope is not None:
            params["scope"] = scope
        return await self._t.get("/gdpr/dsar?" + urlencode(params))

    async def close(self) -> None:
        await self._t.aclose()

    async def __aenter__(self) -> AsyncGovernanceClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()
