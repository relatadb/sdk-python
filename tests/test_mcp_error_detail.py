"""#6779: an MCP tool error keeps the server's ``code`` and reason instead of
collapsing to ``unknown server error``. Bodies are recorded from
``POST /v1/mcp/tools/call``; ``httpx.MockTransport``, no live server.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from relata import McpClient, RelataError

BASE = "http://localhost:9090"

RULE_DETAIL = "invalid condition_sql: parse error: expected '=', found 'SELECT'"

# Current server: `detail`, `code` and the HTTP problem in `structuredContent`.
CREATE_RULE_BODY: dict[str, Any] = {
    "content": [{"type": "text", "text": RULE_DETAIL}],
    "isError": True,
    "code": "rule-condition-invalid",
    "detail": RULE_DETAIL,
    "structuredContent": {
        "type": "https://relatadb.dev/errors/rule-condition-invalid",
        "title": "Bad Request",
        "status": 400,
        "detail": RULE_DETAIL,
        "code": "rule-condition-invalid",
    },
}

# v3.0.0-rc.4 server: the reason only in `content[0].text`.
RECOGNIZE_TEXT = "field 'id' has wrong type (expected string, got object)"
RC4_RECOGNIZE_BODY: dict[str, Any] = {
    "content": [{"type": "text", "text": RECOGNIZE_TEXT}],
    "isError": True,
    "code": "invalid_params",
}


def _client(body: dict[str, Any]) -> McpClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json=body)

    return McpClient(BASE, transport=httpx.MockTransport(handler))


def test_create_rule_error_keeps_code_and_detail() -> None:
    with pytest.raises(RelataError) as exc:
        _client(CREATE_RULE_BODY).create_rule(
            "r1", "SELECT * FROM Person WHERE name = 'x'", "Person"
        )
    assert exc.value.status_code == 400
    assert exc.value.code == "rule-condition-invalid"
    assert RULE_DETAIL in str(exc.value)
    assert "unknown server error" not in str(exc.value)


def test_mcp_envelope_text_is_the_fallback_detail() -> None:
    with pytest.raises(RelataError) as exc:
        _client(RC4_RECOGNIZE_BODY).recognize({"id": "m1"})  # type: ignore[arg-type]
    assert exc.value.code == "invalid_params"
    assert RECOGNIZE_TEXT in str(exc.value)
    assert "unknown server error" not in str(exc.value)
