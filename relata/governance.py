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

from relata._http import AsyncHttpTransport, HttpTransport, path_segment

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
            extra["X-Relata-Tenant-Id"] = tenant
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

    def _effective_purpose(self, purpose: str | None) -> str:
        """Resolve ``purpose`` (per-call override, else the client default),
        raising :class:`~relata.exceptions.PurposeError` if neither is set.
        """
        eff = purpose or self._purpose
        if not eff:
            from relata.exceptions import PurposeError

            raise PurposeError(
                "This governance call requires a purpose. Pass purpose= to "
                "the call or set a default on the RelataClient."
            )
        return eff


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
        from urllib.parse import urlencode

        path = "/rules"
        if object_type:
            path += "?" + urlencode({"object_type": object_type})
        data = self._t.get(path)
        rules = data.get("rules") if isinstance(data, dict) else data
        return rules if isinstance(rules, list) else []

    def create_rule(
        self, rule: dict[str, Any], *, purpose: str | None = None
    ) -> dict[str, Any]:
        """Create a detection rule. The dict shape matches the server's
        ``RuleSpec`` (``name``, ``object_type``, ``condition``, ``action``, ...).
        Returns the created rule record including its server-assigned ``id``.

        ``purpose`` is mandatory server-side (``POST /rules?purpose=<p>``) —
        pass it here or set a default on the parent ``RelataClient``.
        """
        eff = self._effective_purpose(purpose)
        return self._t.post(f"/rules?purpose={path_segment(eff)}", rule)

    def disable_rule(self, rule_id: str) -> dict[str, Any]:
        """Disable (logically delete) a rule by id."""
        return self._t.delete(f"/rules/{path_segment(rule_id)}")

    def import_sigma(
        self, sigma_yaml: str, *, purpose: str | None = None
    ) -> dict[str, Any]:
        """Import a Sigma rule (YAML string). Returns the import summary
        (``rules_imported``, ``rules_skipped``, ``errors``).

        The server (``POST /rules/sigma?purpose=<p>``) requires ``purpose``
        as a query param and the raw YAML text as the request body — not a
        JSON envelope.
        """
        eff = self._effective_purpose(purpose)
        return self._t.post_raw(
            f"/rules/sigma?purpose={path_segment(eff)}",
            sigma_yaml,
            content_type="application/x-yaml",
        )

    def snooze_rule(self, rule_id: str, duration_secs: int) -> dict[str, Any]:
        """Temporarily disable a rule for ``duration_secs`` (#967)."""
        return self._t.post(
            f"/rules/{path_segment(rule_id)}/snooze", {"duration_secs": duration_secs}
        )

    def suppress_rule(
        self, rule_id: str, entity_id: str, *, condition: str | None = None
    ) -> dict[str, Any]:
        """Permanently suppress future matches of the rule for ``entity_id``
        (#967). ``condition`` is an optional informational note."""
        payload: dict[str, Any] = {"entity_id": entity_id}
        if condition is not None:
            payload["condition"] = condition
        return self._t.post(f"/rules/{path_segment(rule_id)}/suppress", payload)

    def add_rule_exception(self, rule_id: str, exception: dict[str, Any]) -> dict[str, Any]:
        """Add an exception entry so specific matches are ignored (#967)."""
        return self._t.post(f"/rules/{path_segment(rule_id)}/exceptions", exception)

    def get_rule_tuning(self, rule_id: str) -> dict[str, Any]:
        """Retrieve the tuning state (snoozes, suppressions, exceptions) (#967)."""
        return self._t.get(f"/rules/{path_segment(rule_id)}/tuning")

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
        field: str | None = None,
        value: str | None = None,
    ) -> dict[str, Any]:
        """Place a legal hold on ``object_type`` (optionally scoped to rows
        where ``field`` equals ``value``).

        Args:
            case_id: Caller-supplied case identifier (becomes the hold id).
            object_type: Type to hold.
            field: Optional column name to scope the hold to; pass both
                ``field`` and ``value`` together, or omit both for a
                type-wide hold. The server's ``POST /retention/holds`` only
                reads ``case_id``/``object_type``/``field``/``value`` — any
                other key is silently ignored (#2465), so ``field``/``value``
                is the only supported row-scoping mechanism.
            value: Column value to match when ``field`` is set.
        """
        payload: dict[str, Any] = {
            "case_id": case_id,
            "object_type": object_type,
        }
        if field is not None:
            payload["field"] = field
        if value is not None:
            payload["value"] = value
        return self._t.post("/retention/holds", payload)

    def lift_legal_hold(self, case_id: str) -> dict[str, Any]:
        """Lift (remove) a legal hold by case id."""
        return self._t.delete(f"/retention/holds/{path_segment(case_id)}")

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
            f"/retention/worm/{path_segment(object_type)}",
            {"retention_secs": retention_secs},
        )

    # ------------------------------------------------------------------
    # Breakglass (#74)
    # ------------------------------------------------------------------

    def request_breakglass(
        self,
        source_id: str,
        *,
        purpose: str | None = None,
        justification: str | None = None,
    ) -> dict[str, Any]:
        """Request emergency HUMINT breakglass access (fixed 4 h window).

        Args:
            source_id: The masked source ID being unmasked (e.g. ``"SRC-0042"``).
            purpose: Declared query purpose. Server defaults to
                ``"humint_unmask"`` when omitted.
            justification: Free-text justification for the emergency access,
                captured for audit.

        Returns the request record including ``request_id`` and ``status``.
        Approval requires two distinct officers (``approve_breakglass``).
        """
        payload: dict[str, Any] = {"source_id": source_id}
        if purpose is not None:
            payload["purpose"] = purpose
        if justification is not None:
            payload["justification"] = justification
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
        return self._t.get(f"/humint/breakglass/status/{path_segment(request_id)}")

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
        return self._t.patch(f"/alerts/update/{path_segment(alert_id)}", payload)

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
        """Close the underlying HTTP connection pool and drop the bearer
        credential (#3214)."""
        self._t.close()
        self._bearer_token = None

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
        from urllib.parse import urlencode

        path = "/rules"
        if object_type:
            path += "?" + urlencode({"object_type": object_type})
        data = await self._t.get(path)
        rules = data.get("rules") if isinstance(data, dict) else data
        return rules if isinstance(rules, list) else []

    async def create_rule(
        self, rule: dict[str, Any], *, purpose: str | None = None
    ) -> dict[str, Any]:
        eff = self._effective_purpose(purpose)
        return await self._t.post(f"/rules?purpose={path_segment(eff)}", rule)

    async def disable_rule(self, rule_id: str) -> dict[str, Any]:
        return await self._t.delete(f"/rules/{path_segment(rule_id)}")

    async def import_sigma(
        self, sigma_yaml: str, *, purpose: str | None = None
    ) -> dict[str, Any]:
        eff = self._effective_purpose(purpose)
        return await self._t.post_raw(
            f"/rules/sigma?purpose={path_segment(eff)}",
            sigma_yaml,
            content_type="application/x-yaml",
        )

    async def snooze_rule(self, rule_id: str, duration_secs: int) -> dict[str, Any]:
        return await self._t.post(
            f"/rules/{path_segment(rule_id)}/snooze", {"duration_secs": duration_secs}
        )

    async def suppress_rule(
        self, rule_id: str, entity_id: str, *, condition: str | None = None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"entity_id": entity_id}
        if condition is not None:
            payload["condition"] = condition
        return await self._t.post(f"/rules/{path_segment(rule_id)}/suppress", payload)

    async def add_rule_exception(self, rule_id: str, exception: dict[str, Any]) -> dict[str, Any]:
        return await self._t.post(f"/rules/{path_segment(rule_id)}/exceptions", exception)

    async def get_rule_tuning(self, rule_id: str) -> dict[str, Any]:
        return await self._t.get(f"/rules/{path_segment(rule_id)}/tuning")

    async def list_retention_policies(self) -> list[dict[str, Any]]:
        data = await self._t.get("/retention/policies")
        policies = data.get("policies") if isinstance(data, dict) else data
        return policies if isinstance(policies, list) else []

    async def list_legal_holds(self) -> list[dict[str, Any]]:
        data = await self._t.get("/retention/holds")
        holds = data.get("holds") if isinstance(data, dict) else data
        return holds if isinstance(holds, list) else []

    async def place_legal_hold(
        self,
        case_id: str,
        object_type: str,
        *,
        field: str | None = None,
        value: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"case_id": case_id, "object_type": object_type}
        if field is not None:
            payload["field"] = field
        if value is not None:
            payload["value"] = value
        return await self._t.post("/retention/holds", payload)

    async def lift_legal_hold(self, case_id: str) -> dict[str, Any]:
        return await self._t.delete(f"/retention/holds/{path_segment(case_id)}")

    async def list_worm_policies(self) -> list[dict[str, Any]]:
        data = await self._t.get("/retention/worm")
        policies = data.get("policies") if isinstance(data, dict) else data
        return policies if isinstance(policies, list) else []

    async def set_worm_policy(self, object_type: str, *, retention_secs: int) -> dict[str, Any]:
        return await self._t.post(
            f"/retention/worm/{path_segment(object_type)}",
            {"retention_secs": retention_secs},
        )

    async def request_breakglass(
        self,
        source_id: str,
        *,
        purpose: str | None = None,
        justification: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"source_id": source_id}
        if purpose is not None:
            payload["purpose"] = purpose
        if justification is not None:
            payload["justification"] = justification
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
        return await self._t.get(f"/humint/breakglass/status/{path_segment(request_id)}")

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
        self._bearer_token = None

    async def __aenter__(self) -> AsyncGovernanceClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()
