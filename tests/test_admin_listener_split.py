"""Tests for #2321 (ADR-0261): the admin control-plane listener split.

/admin/* and /platform/* are mounted ONLY on the loopbound admin listener
(RELATA_ADMIN_BIND, default 127.0.0.1:9091) on a hardened server/cluster
deployment -- never the main data-plane listener. These tests pin two
things:

1. ``admin_base_url``, when configured (on ``RelataClient`` or directly on
   ``BackupClient``/``TenantAdminClient``), routes admin/platform-only
   requests to the admin listener instead of ``base_url`` -- spot-checked
   through the real typed clients (not just the raw transport).
2. A bare, detail-free 404 against an admin/platform-only path -- the shape
   the server's own RFC 7807 normalisation leaves behind for a request that
   matched no route on the listener that answered -- surfaces a "you're
   probably pointed at the wrong port" hint instead of an uninformative bare
   404.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from relata._http import _admin_listener_hint, _is_admin_only_path, _response_detail_is_blank
from relata.backup import AsyncBackupClient, BackupClient
from relata.client import RelataClient
from relata.exceptions import NotFoundError
from relata.tenants import TenantAdminClient

BASE = "http://data-plane.example"
ADMIN = "http://admin-plane.example"

Handler = Callable[[httpx.Request], httpx.Response]


def _wrap(handler: Handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


# ---------------------------------------------------------------------------
# Pure helper unit tests
# ---------------------------------------------------------------------------


def test_is_admin_only_path() -> None:
    assert _is_admin_only_path("/admin/backup")
    assert _is_admin_only_path("/platform/tenants")
    assert not _is_admin_only_path("/query")
    assert not _is_admin_only_path("/tenants/me")
    assert not _is_admin_only_path("/administer")  # no trailing slash after "admin"


def test_response_detail_is_blank_true_for_bare_normalized_404() -> None:
    body = '{"type":"about:blank","status":404,"title":"Not Found","detail":""}'
    assert _response_detail_is_blank("application/problem+json", body)


def test_response_detail_is_blank_true_for_empty_body() -> None:
    assert _response_detail_is_blank("", "")
    assert _response_detail_is_blank("text/plain", "   ")


def test_response_detail_is_blank_false_for_real_business_404() -> None:
    body = (
        '{"type":"about:blank","status":404,"title":"Not Found",'
        '"detail":"tenant \'acme\' not found"}'
    )
    assert not _response_detail_is_blank("application/problem+json", body)
    assert not _response_detail_is_blank("text/plain", "restore job not found")


def test_admin_listener_hint_names_the_path_and_env_var() -> None:
    hint = _admin_listener_hint("/admin/backup")
    assert "/admin/backup" in hint
    assert "RELATA_ADMIN_BIND" in hint
    assert "admin_base_url" in hint


# ---------------------------------------------------------------------------
# admin_base_url routing (through the real typed clients)
# ---------------------------------------------------------------------------


def test_backup_client_falls_back_to_base_url_when_admin_unset() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.host == "data-plane.example"
        return httpx.Response(200, json={"backups": []})

    c = BackupClient(BASE, transport=_wrap(handler))
    assert c.list() == []


def test_backup_client_routes_to_admin_base_url_when_configured() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.host == "admin-plane.example"
        assert req.url.path == "/admin/backups"
        return httpx.Response(200, json={"backups": []})

    c = BackupClient(BASE, admin_base_url=ADMIN, transport=_wrap(handler))
    assert c.list() == []


def test_tenant_admin_client_splits_tenants_and_platform_across_planes() -> None:
    seen_hosts: dict[str, str] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen_hosts[req.url.path] = req.url.host
        return httpx.Response(200, json={"usage": {}})

    c = TenantAdminClient(BASE, admin_base_url=ADMIN, transport=_wrap(handler))
    c.usage()  # /tenants/usage -- ordinary data-plane route
    c.platform_usage()  # /platform/usage -- admin-listener-only route

    assert seen_hosts["/tenants/usage"] == "data-plane.example"
    assert seen_hosts["/platform/usage"] == "admin-plane.example"


def test_backup_client_from_client_inherits_admin_base_url() -> None:
    client = RelataClient(BASE, admin_base_url=ADMIN, purpose="analytics")
    backup = BackupClient.from_client(client)
    assert backup._t._admin_client is not None  # noqa: SLF001
    assert str(backup._t._admin_client.base_url).rstrip("/") == ADMIN  # noqa: SLF001


@pytest.mark.asyncio
async def test_async_backup_client_routes_to_admin_base_url_when_configured() -> None:
    async def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.host == "admin-plane.example"
        return httpx.Response(200, json={"backups": []})

    c = AsyncBackupClient(BASE, admin_base_url=ADMIN, transport=httpx.MockTransport(handler))
    assert await c.list() == []
    await c.close()


# ---------------------------------------------------------------------------
# "You're probably pointed at the wrong port" hint
# ---------------------------------------------------------------------------


def test_bare_404_against_admin_only_path_surfaces_hint() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        # The shape normalize_error_body (crates/relata-cli/src/serve.rs)
        # leaves behind for a request matching no route on this listener.
        return httpx.Response(
            404,
            headers={"content-type": "application/problem+json"},
            content=json.dumps(
                {"type": "about:blank", "status": 404, "title": "Not Found", "detail": ""}
            ),
        )

    c = BackupClient(BASE, transport=_wrap(handler))  # admin_base_url deliberately unset
    with pytest.raises(NotFoundError) as exc_info:
        c.list()
    msg = str(exc_info.value)
    assert "RELATA_ADMIN_BIND" in msg
    assert "/admin/backups" in msg


def test_real_business_404_on_platform_route_does_not_get_hint() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            headers={"content-type": "application/problem+json"},
            content=json.dumps(
                {
                    "type": "about:blank",
                    "status": 404,
                    "title": "Not Found",
                    "detail": "tenant 'acme' not found",
                }
            ),
        )

    c = TenantAdminClient(BASE, transport=_wrap(handler))
    with pytest.raises(NotFoundError) as exc_info:
        c.get("acme")
    msg = str(exc_info.value)
    assert "RELATA_ADMIN_BIND" not in msg
    assert "tenant 'acme' not found" in msg


def test_bare_404_on_data_plane_path_does_not_get_hint() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            headers={"content-type": "application/problem+json"},
            content=json.dumps(
                {"type": "about:blank", "status": 404, "title": "Not Found", "detail": ""}
            ),
        )

    c = TenantAdminClient(BASE, transport=_wrap(handler))
    with pytest.raises(NotFoundError) as exc_info:
        c.tenant_usage("acme")  # /tenants/:id/usage -- not admin/platform-only
    assert "RELATA_ADMIN_BIND" not in str(exc_info.value)
