"""Tests for ObjectClient.get() / delete() (#747) and QueryBuilder.where_param() (#748)."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

BASE = "http://localhost:8080"
Handler = Callable[[httpx.Request], httpx.Response]


def _mock(handler: Handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


# ---------------------------------------------------------------------------
# ObjectClient.get()
# ---------------------------------------------------------------------------


def test_get_returns_row_when_found() -> None:
    from relata.objects import ObjectClient

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/query"
        body = json.loads(req.content)
        assert "SELECT * FROM HmSkill WHERE id = 'biosec.ioc'" in body["sql"]
        assert body["purpose"] == "hm_ledger"
        return httpx.Response(200, json={"data": [{"id": "biosec.ioc", "content_b64": "abc"}], "rows": 1})

    c = ObjectClient(BASE, purpose="hm_ledger", transport=_mock(handler))
    row = c.get("HmSkill", "biosec.ioc")
    assert row is not None
    assert row["id"] == "biosec.ioc"
    assert row["content_b64"] == "abc"


def test_get_returns_none_when_empty() -> None:
    from relata.objects import ObjectClient

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [], "rows": 0})

    c = ObjectClient(BASE, purpose="hm_ledger", transport=_mock(handler))
    assert c.get("HmSkill", "nonexistent") is None


def test_get_returns_none_on_404() -> None:
    from relata.objects import ObjectClient

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "not found"})

    c = ObjectClient(BASE, purpose="hm_ledger", transport=_mock(handler))
    assert c.get("HmSkill", "gone") is None


def test_get_escapes_single_quote_in_id() -> None:
    from relata.objects import ObjectClient

    captured: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["sql"] = json.loads(req.content)["sql"]
        return httpx.Response(200, json={"data": [], "rows": 0})

    c = ObjectClient(BASE, purpose="p", transport=_mock(handler))
    c.get("T", "it's-tricky")
    assert "it''s-tricky" in captured["sql"]


def test_get_rejects_invalid_object_type() -> None:
    from relata.objects import ObjectClient

    c = ObjectClient(BASE, purpose="p", transport=_mock(lambda r: httpx.Response(200, json={})))
    with pytest.raises(ValueError, match="Invalid object_type"):
        c.get("DROP TABLE--", "id")


# ---------------------------------------------------------------------------
# ObjectClient.delete()
# ---------------------------------------------------------------------------


def test_delete_sends_deleted_flag() -> None:
    from relata.objects import ObjectClient

    def handler(req: httpx.Request) -> httpx.Response:
        assert "object_type=HmSkill" in str(req.url)
        rows = [json.loads(line) for line in req.content.decode().splitlines()]
        assert rows[0]["id"] == "biosec.ioc"
        assert rows[0]["_deleted"] is True
        return httpx.Response(200, json={"accepted": 1})

    c = ObjectClient(BASE, purpose="hm_ledger", transport=_mock(handler))
    c.delete("HmSkill", "biosec.ioc")  # must not raise


def test_delete_rejects_invalid_object_type() -> None:
    from relata.objects import ObjectClient

    c = ObjectClient(BASE, purpose="p", transport=_mock(lambda r: httpx.Response(200, json={})))
    with pytest.raises(ValueError, match="Invalid object_type"):
        c.delete("Bad Type!", "id")


# ---------------------------------------------------------------------------
# QueryBuilder.where_param()
# ---------------------------------------------------------------------------


def test_where_param_str_single_quote_escaped() -> None:
    from relata.query import select

    sql = select("HmSkill").where_param("id = ?", "it's here").sql()
    assert "id = 'it''s here'" in sql


def test_where_param_int() -> None:
    from relata.query import select

    sql = select("T").where_param("score > ?", 42).sql()
    assert "score > 42" in sql


def test_where_param_float() -> None:
    from relata.query import select

    sql = select("T").where_param("conf > ?", 0.75).sql()
    assert "conf > 0.75" in sql


def test_where_param_multi_placeholder() -> None:
    from relata.query import select

    sql = select("T").where_param("a = ? AND b = ?", "x", "y").sql()
    assert "a = 'x' AND b = 'y'" in sql


def test_where_param_mismatch_raises() -> None:
    from relata.query import select

    with pytest.raises(ValueError, match="placeholder"):
        select("T").where_param("a = ? AND b = ?", "only-one")


def test_where_param_bool_raises() -> None:
    from relata.query import select

    with pytest.raises(TypeError, match="bool"):
        select("T").where_param("flag = ?", True)  # type: ignore[arg-type]


def test_where_param_bad_type_raises() -> None:
    from relata.query import select

    with pytest.raises(TypeError, match="list"):
        select("T").where_param("id = ?", ["a", "b"])  # type: ignore[arg-type]


def test_where_param_chains_with_where() -> None:
    from relata.query import select

    sql = (
        select("HmSkill")
        .where("updated_ms > 0")
        .where_param("id = ?", "biosec.ioc_search")
        .limit(1)
        .sql()
    )
    assert "updated_ms > 0" in sql
    assert "id = 'biosec.ioc_search'" in sql
