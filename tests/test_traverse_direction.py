"""graph_traverse maps out/in/both onto the server's outbound/inbound/any vocabulary."""

import httpx

from relata.client import RelataClient


def test_aliases_are_mapped(monkeypatch) -> None:
    sent: list = []

    def fake_get(self, path, **kw):  # HttpTransport.get
        sent.append(httpx.URL("http://x" + path).params.get("direction"))
        return {"nodes": [], "edges": [], "depth_reached": 0}

    from relata._http import HttpTransport

    monkeypatch.setattr(HttpTransport, "get", fake_get)
    c = RelataClient("http://x", bearer_token="t")
    for d in ("out", "in", "both", "inbound"):
        c.graph_traverse("a", direction=d)
    assert sent == ["outbound", "inbound", "any", "inbound"]
