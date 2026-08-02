"""Backup / restore / PITR SDK (#88 -- server routes now shipped).

Wraps POST /admin/backup, GET /admin/backups,
POST /admin/restore, GET /admin/restore/:id.
All routes require admin auth.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from relata._http import AsyncHttpTransport, HttpTransport

if TYPE_CHECKING:
    import httpx

    from relata.client import RelataClient


class BackupRef:
    """A backup snapshot reference."""

    __slots__ = (
        "snapshot_id", "kind", "written_at_ns",
        "size_bytes", "sha256", "row_count", "path",
    )

    def __init__(self, raw: dict[str, Any]) -> None:
        self.snapshot_id = str(raw.get("snapshot_id", ""))
        self.kind = str(raw.get("kind", "full"))
        self.written_at_ns = int(raw.get("written_at_ns", 0))
        self.size_bytes = int(raw.get("size_bytes", 0))
        self.sha256 = str(raw.get("sha256", ""))
        self.row_count = int(raw.get("row_count", 0))
        self.path = str(raw.get("path", ""))

    def __repr__(self) -> str:
        return (
            f"BackupRef(snapshot_id={self.snapshot_id!r}, "
            f"kind={self.kind!r}, size_bytes={self.size_bytes})"
        )


class RestoreStatus:
    """Restore job status."""

    __slots__ = ("restore_id", "snapshot_id", "status", "rto_actual_secs", "error")

    def __init__(self, raw: dict[str, Any]) -> None:
        self.restore_id = str(raw.get("restore_id", ""))
        self.snapshot_id = str(raw.get("snapshot_id", ""))
        self.status = str(raw.get("status", "unknown"))
        self.rto_actual_secs = float(raw.get("rto_actual_secs", 0.0))
        self.error = raw.get("error")  # str | None

    @property
    def is_completed(self) -> bool:
        return self.status == "completed"

    @property
    def is_failed(self) -> bool:
        return self.status == "failed"

    def __repr__(self) -> str:
        return (
            f"RestoreStatus(restore_id={self.restore_id!r}, "
            f"status={self.status!r}, "
            f"rto_actual_secs={self.rto_actual_secs})"
        )


class BackupClient:
    """Synchronous backup / restore client (admin-auth required)."""

    def __init__(
        self,
        base_url: str,
        *,
        bearer_token: str | None = None,
        timeout: float = 120.0,
        extra_headers: dict[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
        admin_base_url: str | None = None,
    ) -> None:
        # #2321 (ADR-0261): every route this client wraps is /admin/*, which
        # is mounted only on the loopbound admin control-plane listener
        # (RELATA_ADMIN_BIND, default 127.0.0.1:9091) on a hardened
        # server/cluster deployment -- pass admin_base_url to reach it there.
        # Unset (the default) preserves prior behaviour: every request goes
        # to base_url, correct when the admin listener isn't split (e.g.
        # local/free-profile dev).
        self._t = HttpTransport(
            base_url,
            bearer_token,
            timeout,
            transport=transport,
            extra_headers=extra_headers,
            admin_base_url=admin_base_url,
        )

    @classmethod
    def from_client(cls, client: RelataClient) -> BackupClient:
        return cls(
            client._base_url,  # noqa: SLF001
            bearer_token=client._bearer_token,  # noqa: SLF001
            timeout=client._timeout,  # noqa: SLF001
            extra_headers=client._extra_headers,  # noqa: SLF001
            admin_base_url=client._admin_base_url,  # noqa: SLF001
        )

    def create(
        self,
        *,
        kind: str = "full",
        tenant: str | None = None,
    ) -> BackupRef:
        """Trigger a backup. ``kind`` is ``"full"`` (default) or
        ``"incremental"``. Returns a :class:`BackupRef`."""
        payload: dict[str, Any] = {"kind": kind}
        if tenant is not None:
            payload["tenant"] = tenant
        resp = self._t.post("/admin/backup", payload)
        return BackupRef(resp)

    def list(self) -> list[BackupRef]:
        """List all backup snapshots."""
        resp = self._t.get("/admin/backups")
        raw_list = resp.get("backups", []) if isinstance(resp, dict) else []
        return [BackupRef(r) for r in raw_list]

    def compact(self) -> dict[str, Any]:
        """Trigger background compaction of the row store (#967)."""
        return self._t.post("/admin/compact", {})

    def restore(
        self,
        snapshot_id: str,
        *,
        tenant: str | None = None,
        point_in_time_ns: int | None = None,
    ) -> str:
        """Trigger a restore from ``snapshot_id``. Returns the ``restore_id``
        (job runs asynchronously; poll via :meth:`restore_status`)."""
        payload: dict[str, Any] = {"snapshot_id": snapshot_id}
        if tenant is not None:
            payload["tenant"] = tenant
        if point_in_time_ns is not None:
            payload["point_in_time_ns"] = point_in_time_ns
        resp = self._t.post("/admin/restore", payload)
        return str(resp.get("restore_id", ""))

    def restore_status(self, restore_id: str) -> RestoreStatus:
        """Poll restore job status."""
        resp = self._t.get(f"/admin/restore/{restore_id}")
        return RestoreStatus(resp)

    def wait_for_restore(
        self,
        restore_id: str,
        *,
        timeout_secs: float = 300.0,
        poll_interval_secs: float = 2.0,
    ) -> RestoreStatus:
        """Block until the restore completes or fails. Raises
        :class:`TimeoutError` if ``timeout_secs`` elapses."""
        import time

        deadline = time.monotonic() + timeout_secs
        while time.monotonic() < deadline:
            status = self.restore_status(restore_id)
            if status.is_completed or status.is_failed:
                return status
            time.sleep(poll_interval_secs)
        raise TimeoutError(
            f"Restore {restore_id} did not complete within {timeout_secs}s"
        )

    def close(self) -> None:
        self._t.close()

    def __enter__(self) -> BackupClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class AsyncBackupClient:
    """Asynchronous backup / restore client -- see BackupClient."""

    def __init__(
        self,
        base_url: str,
        *,
        bearer_token: str | None = None,
        timeout: float = 120.0,
        extra_headers: dict[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        admin_base_url: str | None = None,
    ) -> None:
        # See BackupClient.__init__ (#2321, ADR-0261).
        self._t = AsyncHttpTransport(
            base_url,
            bearer_token,
            timeout,
            transport=transport,
            extra_headers=extra_headers,
            admin_base_url=admin_base_url,
        )

    @classmethod
    def from_client(cls, client: RelataClient) -> AsyncBackupClient:
        return cls(
            client._base_url,  # noqa: SLF001
            bearer_token=client._bearer_token,  # noqa: SLF001
            timeout=client._timeout,  # noqa: SLF001
            extra_headers=client._extra_headers,  # noqa: SLF001
            admin_base_url=client._admin_base_url,  # noqa: SLF001
        )

    async def create(
        self,
        *,
        kind: str = "full",
        tenant: str | None = None,
    ) -> BackupRef:
        payload: dict[str, Any] = {"kind": kind}
        if tenant is not None:
            payload["tenant"] = tenant
        resp = await self._t.post("/admin/backup", payload)
        return BackupRef(resp)

    async def list(self) -> list[BackupRef]:
        resp = await self._t.get("/admin/backups")
        raw_list = resp.get("backups", []) if isinstance(resp, dict) else []
        return [BackupRef(r) for r in raw_list]

    async def compact(self) -> dict[str, Any]:
        """Async equivalent of :meth:`BackupClient.compact`."""
        return await self._t.post("/admin/compact", {})

    async def restore(
        self,
        snapshot_id: str,
        *,
        tenant: str | None = None,
        point_in_time_ns: int | None = None,
    ) -> str:
        payload: dict[str, Any] = {"snapshot_id": snapshot_id}
        if tenant is not None:
            payload["tenant"] = tenant
        if point_in_time_ns is not None:
            payload["point_in_time_ns"] = point_in_time_ns
        resp = await self._t.post("/admin/restore", payload)
        return str(resp.get("restore_id", ""))

    async def restore_status(self, restore_id: str) -> RestoreStatus:
        resp = await self._t.get(f"/admin/restore/{restore_id}")
        return RestoreStatus(resp)

    async def wait_for_restore(
        self,
        restore_id: str,
        *,
        timeout_secs: float = 300.0,
        poll_interval_secs: float = 2.0,
    ) -> RestoreStatus:
        import asyncio

        deadline = asyncio.get_event_loop().time() + timeout_secs
        while asyncio.get_event_loop().time() < deadline:
            status = await self.restore_status(restore_id)
            if status.is_completed or status.is_failed:
                return status
            await asyncio.sleep(poll_interval_secs)
        raise TimeoutError(
            f"Restore {restore_id} did not complete within {timeout_secs}s"
        )

    async def close(self) -> None:
        await self._t.aclose()

    async def __aenter__(self) -> AsyncBackupClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()
