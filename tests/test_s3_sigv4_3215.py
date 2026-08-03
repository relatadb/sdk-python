"""S3 SigV4 door-scoped secret regression tests (#3215).

The audit found ``S3Client.boto3()`` passing the single platform bearer as
BOTH the SigV4 access key id AND the secret key. boto3's SigV4 signer embeds
HMAC material derived from the secret into every signed request's
``Authorization`` header — replayable from any S3 access log, so one leaked
S3 request log leaked the bearer for every other door.

The fix derives a per-door secret via ``HMAC-SHA256(bearer, "relata-s3-door")``
so the SigV4 secret is door-scoped and revocable independently. The access
key id stays the bearer (the door matches the SigV4 ``Credential=`` field
against it).
"""

from __future__ import annotations

import hashlib
import hmac
import sys
import types
from unittest.mock import MagicMock

import pytest

from relata.s3 import AsyncS3Client, S3Client, derive_s3_door_secret

BASE = "http://localhost:9191"
BEARER = "platform-bearer-s3cr3t"


def _expected_secret(bearer: str) -> str:
    return hmac.new(bearer.encode(), b"relata-s3-door", hashlib.sha256).hexdigest()


def _install_fake_boto3(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Install a stub boto3/botocore so boto3() runs without the real deps."""
    fake_client = MagicMock(name="boto3.client")

    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = fake_client  # type: ignore[attr-defined]

    class _Config:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    fake_config = types.ModuleType("botocore.config")
    fake_config.Config = _Config  # type: ignore[attr-defined]
    fake_botocore = types.ModuleType("botocore")
    fake_botocore.config = fake_config  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    monkeypatch.setitem(sys.modules, "botocore", fake_botocore)
    monkeypatch.setitem(sys.modules, "botocore.config", fake_config)
    return fake_client


def test_sigv4_secret_is_door_scoped_not_the_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Acceptance: boto3's aws_secret_access_key != the bearer token."""
    boto3_client = _install_fake_boto3(monkeypatch)
    client = S3Client(BASE, bearer_token=BEARER)
    client.boto3()

    kwargs = boto3_client.call_args.kwargs
    # The literal acceptance criterion from the ticket.
    assert kwargs["aws_secret_access_key"] != client._bearer_token
    # The access key stays the bearer (the door checks Credential= against it).
    assert kwargs["aws_access_key_id"] == BEARER
    # The secret is exactly the door-scoped derivation.
    assert kwargs["aws_secret_access_key"] == _expected_secret(BEARER)


def test_sigv4_secret_differs_across_bearers() -> None:
    assert derive_s3_door_secret("bearer-a") != derive_s3_door_secret("bearer-b")
    # Deterministic — the server derives the same value from the same bearer.
    assert derive_s3_door_secret(BEARER) == _expected_secret(BEARER)


def test_sigv4_secret_never_equals_any_bearer() -> None:
    for bearer in ("tok", "relata-anonymous", BEARER, "x" * 64):
        assert derive_s3_door_secret(bearer) != bearer


def test_anonymous_fallback_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    boto3_client = _install_fake_boto3(monkeypatch)
    client = S3Client(BASE)  # no bearer
    client.boto3()
    kwargs = boto3_client.call_args.kwargs
    assert kwargs["aws_access_key_id"] == "relata-anonymous"
    assert kwargs["aws_secret_access_key"] == "relata-anonymous"


async def test_async_aio_secret_is_door_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    """The aiobotocore path gets the same door-scoped secret."""
    create_client = MagicMock(name="session.create_client")

    fake_session_mod = types.ModuleType("aiobotocore.session")
    fake_session_mod.get_session = lambda: types.SimpleNamespace(  # type: ignore[attr-defined]
        create_client=create_client
    )
    fake_aiobotocore = types.ModuleType("aiobotocore")
    fake_aiobotocore.session = fake_session_mod  # type: ignore[attr-defined]

    class _Config:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    fake_config = types.ModuleType("botocore.config")
    fake_config.Config = _Config  # type: ignore[attr-defined]
    fake_botocore = types.ModuleType("botocore")
    fake_botocore.config = fake_config  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "aiobotocore", fake_aiobotocore)
    monkeypatch.setitem(sys.modules, "aiobotocore.session", fake_session_mod)
    monkeypatch.setitem(sys.modules, "botocore", fake_botocore)
    monkeypatch.setitem(sys.modules, "botocore.config", fake_config)

    client = AsyncS3Client(BASE, bearer_token=BEARER)
    client.aio()

    kwargs = create_client.call_args.kwargs
    assert kwargs["aws_secret_access_key"] != client._bearer_token
    assert kwargs["aws_access_key_id"] == BEARER
    assert kwargs["aws_secret_access_key"] == _expected_secret(BEARER)


def test_docstring_warns_about_sigv4_credential_in_logs() -> None:
    """Acceptance: the S3Client docstring warns about SigV4-derived material."""
    doc = S3Client.__doc__ or ""
    assert "log" in doc.lower()
    assert "SigV4" in doc
