"""
Ephemeral server — spawn a temporary `relata serve`, run a health check, exit.

Useful for tests, demos, and CI: pick a free port, start the server, wait for
it to be ready, run a query, then tear down.

Requires the `relata` binary on `$PATH` (or `RELATA_BIN`).

Run:
    python -m examples.ephemeral_server
"""

from __future__ import annotations

import os
import socket
import subprocess
import time

from relata import RelataClient


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_port(port: int, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"server on port {port} did not start within {timeout}s")


def main() -> None:
    binary = os.getenv("RELATA_BIN", "relata")
    port = free_port()
    token = "relata-test"

    print(f"Spawning {binary} serve on port {port} ...")
    child = subprocess.Popen(
        [binary, "serve"],
        env={
            **os.environ,
            "RELATA_PORT": str(port),
            "RELATA_BEARER_TOKEN": token,
            "RELATA_LOG_LEVEL": "warn",
        },
    )
    try:
        wait_for_port(port)
        url = f"http://127.0.0.1:{port}"
        print(f"Server ready at {url}")

        relata = RelataClient(url, bearer_token=token, purpose="analytics")
        health = relata.health()
        print(f"Health: status={health.status}  profile={health.profile}  node={health.node_id}")
    finally:
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()


if __name__ == "__main__":
    main()
