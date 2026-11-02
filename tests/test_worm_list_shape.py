"""``GET /v1/retention/worm`` answers ``{"worm_types": [...], "total": n}``; the SDK must not
silently return ``[]`` for it."""

from relata.governance import _worm_policies


def test_worm_types_shape_is_surfaced() -> None:
    out = _worm_policies({"worm_types": ["A", "B"], "total": 2})
    assert out == [{"object_type": "A", "worm": True}, {"object_type": "B", "worm": True}]


def test_legacy_and_empty_shapes() -> None:
    assert _worm_policies({"policies": [{"object_type": "A"}]}) == [{"object_type": "A"}]
    assert _worm_policies({"worm_types": [], "total": 0}) == []
    assert _worm_policies(None) == []
