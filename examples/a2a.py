"""
A2A — Agent-to-Agent task lifecycle + checkpoints (#967).

The A2A door (`.well-known/agent.json` + `/a2a/tasks/*` + `/a2a/checkpoints/*`)
lets one agent ask another agent to do work and track its progress. It is the
production-grade replacement for ad-hoc LLM-to-LLM HTTP plumbing.

Flow demonstrated:
    1. Discover the agent card (GET /.well-known/agent.json)
    2. Submit a task (POST /a2a/tasks) with a JSON-RPC-style payload
    3. Poll task status (GET /a2a/tasks/:id)
    4. List LangGraph-style checkpoints for a thread (GET /a2a/checkpoints/:thread_id)

Run:
    RELATA_TOKEN=secret python -m examples.a2a
"""

from __future__ import annotations

import os
import time

from relata import RelataClient
from relata.a2a import A2AClient


def main() -> None:
    url = os.getenv("RELATA_URL", "http://localhost:9090")
    token = os.getenv("RELATA_TOKEN")

    with RelataClient(url, bearer_token=token, purpose="agent") as relata:
        a2a = A2AClient.from_client(relata)

        # ── 1. Agent card (public discovery) ───────────────────────────────
        print("=== 1. GET /.well-known/agent.json ===")
        card = a2a.agent_card()
        print(
            f"  name={card.get('name')}  version={card.get('version')}  "
            f"description={card.get('description')}"
        )
        skills = card.get("skills", []) if isinstance(card, dict) else []
        print(f"  {len(skills)} skill(s):")
        for s in skills[:5]:
            print(f"    - {s.get('id')} : {s.get('name')}")

        # ── 2. Submit a task ───────────────────────────────────────────────
        print("\n=== 2. POST /a2a/tasks ===")
        task_id = f"task-{int(time.time() * 1000)}"
        payload = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {
                "id": task_id,
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": "Summarise the latest 5 audit entries."}],
                },
            },
        }
        created = a2a.submit_task(payload)
        task_id = (
            (created.get("result") or {}).get("id")
            or created.get("id")
            or task_id
        )
        print(f"  task_id={task_id}")
        print(f"  response: {created}")

        # ── 3. Poll status ─────────────────────────────────────────────────
        print(f"\n=== 3. GET /a2a/tasks/{task_id} ===")
        status = a2a.get_task(task_id)
        state = (
            ((status.get("result") or {}).get("status") or {}).get("state")
            or (status.get("status") or {}).get("state")
            or status.get("state")
            or "unknown"
        )
        print(f"  state={state}")

        # ── 4. Checkpoints (LangGraph thread state) ────────────────────────
        thread_id = os.getenv("THREAD_ID", task_id)
        print(f"\n=== 4. GET /a2a/checkpoints/{thread_id} ===")
        checkpoints = a2a.list_checkpoints(thread_id)
        print(f"  {len(checkpoints)} checkpoint(s) for thread {thread_id}")


if __name__ == "__main__":
    main()
