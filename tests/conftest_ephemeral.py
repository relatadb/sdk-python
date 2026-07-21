"""Pytest fixture that spawns an ephemeral Relata server for integration tests.

Usage in any test file::

    from tests.conftest_ephemeral import relata_server

    def test_health(relata_server):
        import httpx
        r = httpx.get(f"http://127.0.0.1:{relata_server['port']}/health",
                      headers={"Authorization": f"Bearer {relata_server['token']}"})
        assert r.status_code == 200
"""
import os
import socket
import subprocess
import time

import pytest


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def relata_server():
    """Start a relata serve process on a random port; yield config dict; kill on teardown."""
    port = _free_port()
    token = os.environ.get("RELATA_TEST_TOKEN", "relata-test")
    bin_path = os.environ.get("RELATA_BIN", "relata")

    proc = subprocess.Popen(
        [bin_path, "serve"],
        env={**os.environ, "RELATA_PORT": str(port), "RELATA_BEARER_TOKEN": token, "RELATA_LOG_LEVEL": "warn"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.1)
    else:
        proc.kill()
        raise RuntimeError(f"relata on port {port} did not start within 10 s")

    yield {"port": port, "token": token, "base_url": f"http://127.0.0.1:{port}"}

    proc.kill()
    proc.wait()
