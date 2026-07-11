"""Example: jobs & workflows client methods (#723).

Shows the 8 canonical system methods across the synchronous and async clients.
Requires a running Relata server (``cargo run -p relata-cli -- serve``).
"""

import asyncio
from relata.system import AsyncSystemClient, SystemClient

BASE = "http://localhost:8080"
TOKEN = "demo"


def sync_example() -> None:
    with SystemClient(BASE, bearer_token=TOKEN) as sys:
        print("jobs:", sys.jobs_status())
        print("job/c2:", sys.job_status("c2_beacon_detect"))
        print("workflows:", sys.list_workflows())

        sys.register_workflow("demo_flow", [{"name": "s1", "kind": "query", "sql": "SELECT 1"}])
        print("get_workflow:", sys.get_workflow("demo_flow"))

        run = sys.run_workflow("demo_flow")
        run_id = run.get("run_id", "")
        print("run:", run)
        print("workflow_status:", sys.workflow_status("demo_flow"))
        if run_id:
            print("workflow_run:", sys.workflow_run(run_id))


async def async_example() -> None:
    async with AsyncSystemClient(BASE, bearer_token=TOKEN) as sys:
        print("async jobs:", await sys.jobs_status())
        print("async job/c2:", await sys.job_status("c2_beacon_detect"))
        print("async workflows:", await sys.list_workflows())
        run = await sys.run_workflow("demo_flow")
        run_id = run.get("run_id", "")
        if run_id:
            print("async workflow_run:", await sys.workflow_run(run_id))


if __name__ == "__main__":
    sync_example()
    asyncio.run(async_example())
