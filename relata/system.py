"""System / runtime SDK for generic jobs and operator metadata.

Model-provider configuration and completion callbacks belong to applications.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from relata._http import AsyncHttpTransport, HttpTransport

if TYPE_CHECKING:
    import httpx

    from relata.client import RelataClient


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
        return self._t.get("/jobs")

    def job_status(self, name: str) -> dict[str, Any]:
        """Status of a single background job by name. Wraps ``GET /jobs/:name``."""
        return self._t.get(f"/jobs/{quote(name, safe='')}")

    # ── feed ─────────────────────────────────────────────────────────────────

    def feed_health(self) -> dict[str, Any]:
        """GET /feed/health — RIFN feed broker health."""
        return self._t.get("/feed/health")

    def feed_channels(self) -> dict[str, Any]:
        """GET /feed/channels — list available feed channels."""
        return self._t.get("/feed/channels")

    def feed_publish(self, channel: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST /feed/publish — publish a record to a channel."""
        return self._t.post("/feed/publish", {"channel": channel, "payload": payload})

    # ── notifications ─────────────────────────────────────────────────────────

    def notification_rules(self) -> list[dict[str, Any]]:
        """GET /notifications/rules — list notification rules."""
        data = self._t.get("/notifications/rules")
        return data if isinstance(data, list) else data.get("rules", [])

    def create_notification_rule(self, rule: dict[str, Any]) -> dict[str, Any]:
        """POST /notifications/rules — create a notification rule."""
        return self._t.post("/notifications/rules", rule)

    def delete_notification_rule(self, rule_id: str) -> dict[str, Any]:
        """DELETE /notifications/rules/:id — delete a notification rule."""
        return self._t.delete(f"/notifications/rules/{quote(rule_id, safe='')}")

    # ── pipelines ─────────────────────────────────────────────────────────────

    def list_pipelines(self) -> list[dict[str, Any]]:
        """GET /pipelines — list registered ingest pipelines."""
        data = self._t.get("/pipelines")
        return data if isinstance(data, list) else data.get("pipelines", [])

    def define_pipeline(self, definition: dict[str, Any]) -> dict[str, Any]:
        """POST /pipelines — define a new ingest pipeline."""
        return self._t.post("/pipelines", definition)

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
        return await self._t.get("/jobs")

    async def job_status(self, name: str) -> dict[str, Any]:
        return await self._t.get(f"/jobs/{quote(name, safe='')}")

    async def feed_health(self) -> dict[str, Any]:
        return await self._t.get("/feed/health")

    async def feed_channels(self) -> dict[str, Any]:
        return await self._t.get("/feed/channels")

    async def feed_publish(self, channel: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._t.post("/feed/publish", {"channel": channel, "payload": payload})

    async def notification_rules(self) -> list[dict[str, Any]]:
        data = await self._t.get("/notifications/rules")
        return data if isinstance(data, list) else data.get("rules", [])

    async def create_notification_rule(self, rule: dict[str, Any]) -> dict[str, Any]:
        return await self._t.post("/notifications/rules", rule)

    async def delete_notification_rule(self, rule_id: str) -> dict[str, Any]:
        return await self._t.delete(f"/notifications/rules/{quote(rule_id, safe='')}")

    async def list_pipelines(self) -> list[dict[str, Any]]:
        data = await self._t.get("/pipelines")
        return data if isinstance(data, list) else data.get("pipelines", [])

    async def define_pipeline(self, definition: dict[str, Any]) -> dict[str, Any]:
        return await self._t.post("/pipelines", definition)

    async def close(self) -> None:
        await self._t.aclose()

    async def __aenter__(self) -> AsyncSystemClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()
