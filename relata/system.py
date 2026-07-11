"""System / runtime SDK — LLM configuration, jobs status, agent card.

Pairs the agent surfaces with the operator-side runtime config so a Python
caller can drive the full agent lifecycle: configure the LLM, submit an A2A
task, watch the jobs queue, inspect the agent card.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from relata._http import AsyncHttpTransport, HttpTransport

if TYPE_CHECKING:
    import httpx

    from relata.client import RelataClient


class SystemClient:
    """Synchronous system/runtime client — LLM config + jobs."""

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
    # LLM configuration (operator-side; agent-side is in McpClient)
    # ------------------------------------------------------------------

    def llm_config(self) -> dict[str, Any]:
        """Return the configured LLM endpoint + model roster."""
        return self._t.get("/config/llm")

    def test_llm(self, *, prompt: str, model: str | None = None) -> dict[str, Any]:
        """Send a test prompt to the configured LLM endpoint (or a specific
        ``model=``) and return the round-trip result + latency. Used by
        operators to verify connectivity before pointing agents at the server.
        """
        payload: dict[str, Any] = {"prompt": prompt}
        if model is not None:
            payload["model"] = model
        return self._t.post("/config/llm/test", payload)

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

    # ------------------------------------------------------------------
    # Workflows (#615/#619; hardened by #706/#709/#711)
    # ------------------------------------------------------------------

    def list_workflows(self) -> dict[str, Any]:
        """List registered workflow definition names. Wraps ``GET /workflows``."""
        return self._t.get("/workflows")

    def get_workflow(self, name: str) -> dict[str, Any]:
        """Fetch one workflow definition. Wraps ``GET /workflows/:name``."""
        return self._t.get(f"/workflows/{quote(name, safe='')}")

    def register_workflow(self, name: str, steps: list[dict[str, Any]]) -> dict[str, Any]:
        """Register a workflow definition. Wraps ``POST /workflows``.

        ``steps`` is a list of step dicts, e.g.
        ``{"name": "..", "kind": "query", "sql": "..", "depends_on": [0]}``.
        """
        return self._t.post("/workflows", {"name": name, "steps": steps})

    def run_workflow(self, name: str) -> dict[str, Any]:
        """Start a workflow execution; returns ``run_id`` + runnable steps.

        Wraps ``POST /workflows/:name/run``.
        """
        return self._t.post(f"/workflows/{quote(name, safe='')}/run", {})

    def workflow_status(self, name: str) -> dict[str, Any]:
        """Latest run status for a workflow (by name). Wraps
        ``GET /workflows/:name/status``."""
        return self._t.get(f"/workflows/{quote(name, safe='')}/status")

    def workflow_run(self, run_id: str) -> dict[str, Any]:
        """Step-level detail of a run by ``run_id`` — surfaces ``attempt_count``,
        ``last_error``, per-step status (the fields #711/#713 made accurate).
        Wraps ``GET /workflows/runs/:run_id``.
        """
        return self._t.get(f"/workflows/runs/{quote(run_id, safe='')}")

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

    async def llm_config(self) -> dict[str, Any]:
        return await self._t.get("/config/llm")

    async def test_llm(self, *, prompt: str, model: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"prompt": prompt}
        if model is not None:
            payload["model"] = model
        return await self._t.post("/config/llm/test", payload)

    async def jobs_status(self) -> dict[str, Any]:
        return await self._t.get("/jobs")

    async def job_status(self, name: str) -> dict[str, Any]:
        return await self._t.get(f"/jobs/{quote(name, safe='')}")

    async def list_workflows(self) -> dict[str, Any]:
        return await self._t.get("/workflows")

    async def get_workflow(self, name: str) -> dict[str, Any]:
        return await self._t.get(f"/workflows/{quote(name, safe='')}")

    async def register_workflow(self, name: str, steps: list[dict[str, Any]]) -> dict[str, Any]:
        return await self._t.post("/workflows", {"name": name, "steps": steps})

    async def run_workflow(self, name: str) -> dict[str, Any]:
        return await self._t.post(f"/workflows/{quote(name, safe='')}/run", {})

    async def workflow_status(self, name: str) -> dict[str, Any]:
        return await self._t.get(f"/workflows/{quote(name, safe='')}/status")

    async def workflow_run(self, run_id: str) -> dict[str, Any]:
        return await self._t.get(f"/workflows/runs/{quote(run_id, safe='')}")

    async def close(self) -> None:
        await self._t.aclose()

    async def __aenter__(self) -> AsyncSystemClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()
