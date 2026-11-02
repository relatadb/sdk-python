"""Governance SDK — rules, retention, alerts, DSAR (#74).

A thin governed wrapper over the server's HTTP surface so a Python operator
can:

- Define and manage prepared detection rules.
- Place and lift legal holds; configure WORM retention per object type.
- List and ack alerts; tail the alert SSE stream.
- File a GDPR Data Subject Access Request.

Every call carries the caller's ``purpose`` and ``tenant`` (set on the parent
``RelataClient``); governance is on by default.

Synchronous and asynchronous variants are provided for every method. Use as a
context manager to guarantee the underlying HTTP connection pool is closed::

    from relata.governance import GovernanceClient

    with GovernanceClient.from_client(client) as g:
        rule = g.create_rule({
            "name": "high-transfer",
            "condition_sql": "amount > 1000000",
            "target_type": "Transaction",
            "severity": "high",
        })
        hold_id = g.place_legal_hold("case-42", "Person")

For async apps::

    async with AsyncGovernanceClient.from_client(client) as g:
        rule = await g.create_rule({
            "name": "high-transfer",
            "condition_sql": "amount > 1000000",
            "target_type": "Transaction",
            "severity": "high",
        })
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



def _worm_policies(data: Any) -> list[dict[str, Any]]:
    """Normalise ``GET /v1/retention/worm`` (``{"worm_types": [...], "total": n}``).

    Each WORM type becomes ``{"object_type": <name>, "worm": True}``, the same
    shape :meth:`GovernanceClient.set_worm_policy` returns. A legacy
    ``policies`` list is passed through.
    """
    if isinstance(data, dict):
        types = data.get("worm_types")
        if isinstance(types, list):
            return [{"object_type": t, "worm": True} for t in types]
        data = data.get("policies")
    return data if isinstance(data, list) else []


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
    """Synchronous governance surface — rules / retention / alerts / DSAR."""

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

        path = "/v1/rules"
        if object_type:
            path += "?" + urlencode({"object_type": object_type})
        data = self._t.get(path)
        rules = data.get("rules") if isinstance(data, dict) else data
        return rules if isinstance(rules, list) else []

    def create_rule(
        self, rule: dict[str, Any], *, purpose: str | None = None
    ) -> dict[str, Any]:
        """Create a prepared detection rule. The server requires ``name``,
        ``condition_sql``, and ``target_type``; it accepts optional
        ``severity``, ``mitre_technique``, ``trigger``, and ``action`` fields.
        Returns the created rule record including its server-assigned ``id``.

        ``purpose`` is mandatory server-side (``POST /v1/rules?purpose=<p>``) —
        pass it here or set a default on the parent ``RelataClient``.
        """
        eff = self._effective_purpose(purpose)
        return self._t.post(f"/v1/rules?purpose={path_segment(eff)}", rule)

    def disable_rule(self, rule_id: str) -> dict[str, Any]:
        """Disable (logically delete) a rule by id."""
        return self._t.delete(f"/v1/rules/{path_segment(rule_id)}")

    def snooze_rule(self, rule_id: str, duration_secs: int) -> dict[str, Any]:
        """Temporarily disable a rule for ``duration_secs`` (#967)."""
        return self._t.post(
            f"/v1/rules/{path_segment(rule_id)}/snooze", {"duration_secs": duration_secs}
        )

    def suppress_rule(
        self, rule_id: str, entity_id: str, *, condition: str | None = None
    ) -> dict[str, Any]:
        """Permanently suppress future matches of the rule for ``entity_id``
        (#967). ``condition`` is an optional informational note."""
        payload: dict[str, Any] = {"entity_id": entity_id}
        if condition is not None:
            payload["condition"] = condition
        return self._t.post(f"/v1/rules/{path_segment(rule_id)}/suppress", payload)

    def add_rule_exception(self, rule_id: str, exception: dict[str, Any]) -> dict[str, Any]:
        """Add an exception entry so specific matches are ignored (#967)."""
        return self._t.post(f"/v1/rules/{path_segment(rule_id)}/exceptions", exception)

    def get_rule_tuning(self, rule_id: str) -> dict[str, Any]:
        """Retrieve the tuning state (snoozes, suppressions, exceptions) (#967)."""
        return self._t.get(f"/v1/rules/{path_segment(rule_id)}/tuning")

    def get_rule(self, rule_id: str) -> dict[str, Any]:
        """Fetch one detection rule by id (``GET /v1/rules/:id``)."""
        return self._t.get(f"/v1/rules/{path_segment(rule_id)}")

    def remove_rule_exception(self, rule_id: str, exception_id: str) -> dict[str, Any]:
        """Remove a tuning exception (``DELETE /v1/rules/:id/exceptions/:eid``)."""
        return self._t.delete(
            f"/v1/rules/{path_segment(rule_id)}/exceptions/{path_segment(exception_id)}"
        )

    # ------------------------------------------------------------------
    # Retention (#74)
    # ------------------------------------------------------------------

    def set_retention_policy(
        self, object_type: str, max_age_days: int, action: str = "delete"
    ) -> dict[str, Any]:
        """Set a retention policy for a type (``POST /v1/retention/policies``).

        ``action`` is ``"delete"`` or ``"tier:cold"``.
        """
        return self._t.post(
            "/v1/retention/policies",
            {"object_type": object_type, "max_age_days": max_age_days, "action": action},
        )

    def list_retention_policies(self) -> list[dict[str, Any]]:
        """List configured retention policies."""
        data = self._t.get("/v1/retention/policies")
        policies = data.get("policies") if isinstance(data, dict) else data
        return policies if isinstance(policies, list) else []

    def list_legal_holds(self) -> list[dict[str, Any]]:
        """List active legal holds."""
        data = self._t.get("/v1/retention/holds")
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
                type-wide hold. The server's ``POST /v1/retention/holds`` only
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
        return self._t.post("/v1/retention/holds", payload)

    def lift_legal_hold(self, case_id: str) -> dict[str, Any]:
        """Lift (remove) a legal hold by case id."""
        return self._t.delete(f"/v1/retention/holds/{path_segment(case_id)}")

    def list_worm_policies(self) -> list[dict[str, Any]]:
        """List WORM (write-once-read-many) retention policies."""
        return _worm_policies(self._t.get("/v1/retention/worm"))

    def set_worm_policy(
        self,
        object_type: str,
        *,
        retention_secs: int,
    ) -> dict[str, Any]:
        """Set WORM retention for ``object_type``. Rows cannot be mutated or
        purged until ``retention_secs`` elapses from their ``system_from``."""
        return self._t.post(
            f"/v1/retention/worm/{path_segment(object_type)}",
            {"retention_secs": retention_secs},
        )

    # ------------------------------------------------------------------
    # Named HUMINT workflow retired; generic admin break-glass remains.
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

        data = self._t.get("/v1/alerts?" + urlencode(params))
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
        return self._t.patch(f"/v1/alerts/{path_segment(alert_id)}", payload)

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
        return self._t.get("/v1/gdpr/dsar?" + urlencode(params))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # GDPR erasure (POST /v1/gdpr/erase, deletions, verify-receipt)
    # ------------------------------------------------------------------

    def erase(
        self,
        subject_id: str,
        *,
        reason: str | None = None,
        tenant_id: str | None = None,
        physical: bool = False,
    ) -> dict[str, Any]:
        """File a GDPR erasure for ``subject_id`` (``POST /v1/gdpr/erase``).

        ``reason`` is recorded on the erasure receipt; ``tenant_id`` only applies
        to unscoped admin callers; ``physical`` requests physical erasure in
        addition to the logical tombstone. Returns the receipt / operation
        response -- inspect its ``receipt_scope`` and per-copy states.
        """
        payload: dict[str, Any] = {"subject_id": subject_id}
        if reason:
            payload["reason"] = reason
        if tenant_id:
            payload["tenant_id"] = tenant_id
        if physical:
            payload["physical"] = True
        return self._t.post("/v1/gdpr/erase", payload)

    def deletion_status(self, operation_id: str) -> dict[str, Any]:
        """Progress of a deletion operation (``GET /v1/gdpr/deletions/{operation_id}``)."""
        return self._t.get(f"/v1/gdpr/deletions/{path_segment(operation_id)}")

    def verify_receipt(self, receipt: dict[str, Any]) -> dict[str, Any]:
        """Verify a signed erasure receipt (``POST /v1/gdpr/verify-receipt``).

        ``receipt`` carries ``receipt_id``, ``subject``, ``erased_at``,
        ``rows_shredded`` and ``receipt_signature`` exactly as returned by
        :meth:`erase`.
        """
        return self._t.post("/v1/gdpr/verify-receipt", receipt)

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

        path = "/v1/rules"
        if object_type:
            path += "?" + urlencode({"object_type": object_type})
        data = await self._t.get(path)
        rules = data.get("rules") if isinstance(data, dict) else data
        return rules if isinstance(rules, list) else []

    async def create_rule(
        self, rule: dict[str, Any], *, purpose: str | None = None
    ) -> dict[str, Any]:
        eff = self._effective_purpose(purpose)
        return await self._t.post(f"/v1/rules?purpose={path_segment(eff)}", rule)

    async def disable_rule(self, rule_id: str) -> dict[str, Any]:
        return await self._t.delete(f"/v1/rules/{path_segment(rule_id)}")

    async def snooze_rule(self, rule_id: str, duration_secs: int) -> dict[str, Any]:
        return await self._t.post(
            f"/v1/rules/{path_segment(rule_id)}/snooze", {"duration_secs": duration_secs}
        )

    async def suppress_rule(
        self, rule_id: str, entity_id: str, *, condition: str | None = None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"entity_id": entity_id}
        if condition is not None:
            payload["condition"] = condition
        return await self._t.post(f"/v1/rules/{path_segment(rule_id)}/suppress", payload)

    async def add_rule_exception(self, rule_id: str, exception: dict[str, Any]) -> dict[str, Any]:
        return await self._t.post(f"/v1/rules/{path_segment(rule_id)}/exceptions", exception)

    async def get_rule_tuning(self, rule_id: str) -> dict[str, Any]:
        return await self._t.get(f"/v1/rules/{path_segment(rule_id)}/tuning")

    async def get_rule(self, rule_id: str) -> dict[str, Any]:
        return await self._t.get(f"/v1/rules/{path_segment(rule_id)}")

    async def remove_rule_exception(self, rule_id: str, exception_id: str) -> dict[str, Any]:
        return await self._t.delete(
            f"/v1/rules/{path_segment(rule_id)}/exceptions/{path_segment(exception_id)}"
        )

    async def set_retention_policy(
        self, object_type: str, max_age_days: int, action: str = "delete"
    ) -> dict[str, Any]:
        return await self._t.post(
            "/v1/retention/policies",
            {"object_type": object_type, "max_age_days": max_age_days, "action": action},
        )

    async def list_retention_policies(self) -> list[dict[str, Any]]:
        data = await self._t.get("/v1/retention/policies")
        policies = data.get("policies") if isinstance(data, dict) else data
        return policies if isinstance(policies, list) else []

    async def list_legal_holds(self) -> list[dict[str, Any]]:
        data = await self._t.get("/v1/retention/holds")
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
        return await self._t.post("/v1/retention/holds", payload)

    async def lift_legal_hold(self, case_id: str) -> dict[str, Any]:
        return await self._t.delete(f"/v1/retention/holds/{path_segment(case_id)}")

    async def list_worm_policies(self) -> list[dict[str, Any]]:
        return _worm_policies(await self._t.get("/v1/retention/worm"))

    async def set_worm_policy(self, object_type: str, *, retention_secs: int) -> dict[str, Any]:
        return await self._t.post(
            f"/v1/retention/worm/{path_segment(object_type)}",
            {"retention_secs": retention_secs},
        )

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
        data = await self._t.get("/v1/alerts?" + urlencode(params))
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
        return await self._t.get("/v1/gdpr/dsar?" + urlencode(params))

    async def erase(
        self,
        subject_id: str,
        *,
        reason: str | None = None,
        tenant_id: str | None = None,
        physical: bool = False,
    ) -> dict[str, Any]:
        """Async variant of :meth:`GovernanceClient.erase`."""
        payload: dict[str, Any] = {"subject_id": subject_id}
        if reason:
            payload["reason"] = reason
        if tenant_id:
            payload["tenant_id"] = tenant_id
        if physical:
            payload["physical"] = True
        return await self._t.post("/v1/gdpr/erase", payload)

    async def deletion_status(self, operation_id: str) -> dict[str, Any]:
        """Async variant of :meth:`GovernanceClient.deletion_status`."""
        return await self._t.get(f"/v1/gdpr/deletions/{path_segment(operation_id)}")

    async def verify_receipt(self, receipt: dict[str, Any]) -> dict[str, Any]:
        """Async variant of :meth:`GovernanceClient.verify_receipt`."""
        return await self._t.post("/v1/gdpr/verify-receipt", receipt)

    async def close(self) -> None:
        await self._t.aclose()
        self._bearer_token = None

    async def __aenter__(self) -> AsyncGovernanceClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()
