"""System / runtime SDK — LLM configuration, jobs status, agent card.

Pairs the agent surfaces with the operator-side runtime config so a Python
caller can drive the full agent lifecycle: configure the LLM, submit an A2A
task, watch the jobs queue, inspect the agent card.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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

    async def close(self) -> None:
        await self._t.aclose()

    async def __aenter__(self) -> AsyncSystemClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()
