"""``GET /v1/memory/episodes`` answers ``{"episodes": [...]}``; the SDK must not return ``[]`` for it."""

from relata.memory import _episode_rows


def test_episodes_key_is_read() -> None:
    wire = {"session_id": "s", "episodes": [{"id": "e1", "item_count": 2}], "count": 1}
    assert _episode_rows(wire) == [{"id": "e1", "item_count": 2}]


def test_legacy_rows_and_empty() -> None:
    assert _episode_rows({"rows": [{"id": "e"}]}) == [{"id": "e"}]
    assert _episode_rows({"episodes": []}) == []
    assert _episode_rows({}) == []
