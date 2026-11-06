"""``GET /v1/retention/worm`` returns each WORM type with its retention (#6045);
``POST /v1/retention/worm/{type}`` takes an optional ``retention_secs``."""

import json

import httpx

from relata.governance import GovernanceClient, _worm_policies

LIVE = {
    "policies": [
        {"object_type": "A", "worm": True, "retention": "indefinite", "retention_secs": None},
        {"object_type": "B", "worm": True, "retention": "indefinite", "retention_secs": None},
    ],
    "worm_types": ["A", "B"],
    "total": 2,
}


def test_policies_shape_is_returned_as_is() -> None:
    assert _worm_policies(LIVE) == LIVE["policies"]


def test_worm_types_only_and_empty_shapes() -> None:
    assert _worm_policies({"worm_types": ["A"], "total": 1}) == [{"object_type": "A", "worm": True}]
    assert _worm_policies({"policies": [], "worm_types": [], "total": 0}) == []
    assert _worm_policies(None) == []


def test_set_worm_policy_body() -> None:
    seen: list[object] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        return httpx.Response(201, json={"object_type": "A", "worm": True})

    client = GovernanceClient("http://x", transport=httpx.MockTransport(handler))
    client.set_worm_policy("A")
    client.set_worm_policy("A", retention_secs=60)
    assert seen == [{}, {"retention_secs": 60}]
