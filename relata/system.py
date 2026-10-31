"""System / runtime SDK for generic jobs and operator metadata.

Model-provider configuration and completion callbacks belong to applications.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlencode

from relata._http import AsyncHttpTransport, HttpTransport

if TYPE_CHECKING:
    import httpx

    from relata.client import RelataClient


def _query(**params: str | int | None) -> str:
    """Render only the supplied query parameters as a ``?k=v`` suffix."""
    kept = {k: v for k, v in params.items() if v is not None}
    return f"?{urlencode(kept)}" if kept else ""


class SystemClient:
    """Synchronous system/runtime client for generic operator surfaces."""

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
    def from_client(cls, client: RelataClient) -> SystemClient:
        return cls(
            client._base_url,  # noqa: SLF001
            bearer_token=client._bearer_token,  # noqa: SLF001
            timeout=client._timeout,  # noqa: SLF001
            extra_headers=client._extra_headers,  # noqa: SLF001
        )

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    def jobs_status(self) -> dict[str, Any]:
        """Return the status of every background job (continuous-pattern
        detectors, MV refresh, embedder worker, orphan-blob sweep, ...).
        Useful for agent dashboards that need to know whether the detection
        pipeline is healthy before relying on its outputs.
        """
        return self._t.get("/v1/jobs")

    def job_status(self, name: str) -> dict[str, Any]:
        """Status of a single background job by name. Wraps ``GET /v1/jobs/:name``."""
        return self._t.get(f"/v1/jobs/{quote(name, safe='')}")

    # ── notifications ─────────────────────────────────────────────────────────
    # Webhooks are a delivery channel of notification rules: set
    # ``webhook_url`` (and optionally ``webhook_headers``) on the rule body.

    def notification_rules(self) -> list[dict[str, Any]]:
        """GET /v1/notification-rules — list notification rules."""
        data = self._t.get("/v1/notification-rules")
        return data if isinstance(data, list) else data.get("rules", [])

    def create_notification_rule(self, rule: dict[str, Any]) -> dict[str, Any]:
        """POST /v1/notification-rules — create a notification rule.

        Webhook delivery is a channel of the rule itself: include
        ``webhook_url`` (and optional ``webhook_headers``) in ``rule``.
        """
        return self._t.post("/v1/notification-rules", rule)

    def delete_notification_rule(self, rule_id: str) -> dict[str, Any]:
        """DELETE /v1/notification-rules/:id — delete a notification rule."""
        return self._t.delete(f"/v1/notification-rules/{quote(rule_id, safe='')}")

    def update_notification_rule(self, rule_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        """PATCH /v1/notification-rules/:id — update fields of a notification rule."""
        return self._t.patch(f"/v1/notification-rules/{quote(rule_id, safe='')}", patch)

    # ── runtime configuration, specs, MCP sessions ──────────

    def system_config(self) -> dict[str, Any]:
        """Runtime feature configuration. Wraps ``GET /v1/system/config``."""
        return self._t.get("/v1/system/config")

    def system_specs(self) -> dict[str, Any]:
        """Capability specification map. Wraps ``GET /v1/system/specs``."""
        return self._t.get("/v1/system/specs")

    def mcp_sessions(
        self, *, session_id: str | None = None, limit: int | None = None
    ) -> dict[str, Any]:
        """List MCP agent sessions with their tool-call history.

        Wraps ``GET /v1/mcp/sessions``; ``session_id`` filters to one session,
        ``limit`` caps the result (server default 20, max 100).
        """
        return self._t.get("/v1/mcp/sessions" + _query(session_id=session_id, limit=limit))

    def close(self) -> None:
        self._t.close()

    def __enter__(self) -> SystemClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class AsyncSystemClient:
    """Asynchronous system/runtime client — see :class:`SystemClient`."""

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
    def from_client(cls, client: RelataClient) -> AsyncSystemClient:
        return cls(
            client._base_url,  # noqa: SLF001
            bearer_token=client._bearer_token,  # noqa: SLF001
            timeout=client._timeout,  # noqa: SLF001
            extra_headers=client._extra_headers,  # noqa: SLF001
        )

    async def jobs_status(self) -> dict[str, Any]:
        return await self._t.get("/v1/jobs")

    async def job_status(self, name: str) -> dict[str, Any]:
        return await self._t.get(f"/v1/jobs/{quote(name, safe='')}")

    async def notification_rules(self) -> list[dict[str, Any]]:
        data = await self._t.get("/v1/notification-rules")
        return data if isinstance(data, list) else data.get("rules", [])

    async def create_notification_rule(self, rule: dict[str, Any]) -> dict[str, Any]:
        return await self._t.post("/v1/notification-rules", rule)

    async def delete_notification_rule(self, rule_id: str) -> dict[str, Any]:
        return await self._t.delete(f"/v1/notification-rules/{quote(rule_id, safe='')}")

    async def update_notification_rule(self, rule_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        return await self._t.patch(f"/v1/notification-rules/{quote(rule_id, safe='')}", patch)

    async def system_config(self) -> dict[str, Any]:
        return await self._t.get("/v1/system/config")

    async def system_specs(self) -> dict[str, Any]:
        return await self._t.get("/v1/system/specs")

    async def mcp_sessions(
        self, *, session_id: str | None = None, limit: int | None = None
    ) -> dict[str, Any]:
        return await self._t.get("/v1/mcp/sessions" + _query(session_id=session_id, limit=limit))

    async def close(self) -> None:
        await self._t.aclose()

    async def __aenter__(self) -> AsyncSystemClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()
