"""A 204/empty success body (``DELETE /v1/auth/tokens/self/{id}``) must decode to ``{}``,
not raise ``JSONDecodeError`` after the server already acted."""

import httpx

from relata import TokenClient


def test_revoke_self_accepts_204() -> None:
    t = TokenClient(
        "http://x",
        bearer_token="t",
        transport=httpx.MockTransport(lambda req: httpx.Response(204)),
    )
    assert t.revoke_self("abc") == {}
