"""Compose a governed Relata query with a local OpenAI-compatible model.

Install the Python SDK, start a local OpenAI-compatible inference server with a
model already available, then run this example with RELATA_URL,
LOCAL_MODEL_URL, and LOCAL_MODEL_ID set. This example makes no remote provider
calls and downloads no model.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from urllib.parse import urlparse

import httpx

from relata import RelataClient, query_in_natural_language


def local_completion(
    endpoint: str, model: str, timeout: float = 30.0
) -> Callable[[str, int], dict[str, str]]:
    """Return a callback for a loopback OpenAI-compatible chat endpoint."""
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        raise ValueError("local model endpoint must use a loopback host")
    if not model.strip():
        raise ValueError("LOCAL_MODEL_ID is required")

    def complete(prompt: str, max_tokens: int) -> dict[str, str]:
        response = httpx.post(
            endpoint,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0,
            },
            timeout=timeout,
            trust_env=False,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("local model response content must be text")
        return {"text": content, "model_id": model}

    return complete


def main() -> int:
    base_url = os.environ.get("RELATA_URL", "http://127.0.0.1:9090")
    endpoint = os.environ.get(
        "LOCAL_MODEL_URL", "http://127.0.0.1:11434/v1/chat/completions"
    )
    model = os.environ.get("LOCAL_MODEL_ID", "")
    token = os.environ.get("RELATA_BEARER_TOKEN")
    question = " ".join(sys.argv[1:]).strip() or "List the most recent cases"
    if not model:
        raise SystemExit("Set LOCAL_MODEL_ID to a model already available locally.")

    with RelataClient(base_url, purpose="analytics", bearer_token=token) as client:
        result = query_in_natural_language(
            client,
            question,
            purpose="analytics",
            complete=local_completion(endpoint, model),
        )
    print("Proposed SQL:", result["sql"])
    print("Governed result:", result["result"].rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
