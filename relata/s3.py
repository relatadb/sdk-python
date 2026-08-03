"""S3 protocol-door client (#81) — thin wrapper that configures ``boto3`` (or
``aiobotocore`` for async) against RelataDB's built-in S3-compatible server.

The server ships a working S3 door at ``crates/relata-cli/src/s3_server.rs``
with ``ListBuckets``, ``CreateBucket``, ``PutObject``, ``GetObject``, and
``DeleteObject``. This module removes the configuration friction so a Python
caller does not have to hand-build the ``endpoint_url=`` / credential map.

For the typed content-addressed blob API (``client.blobs.put/get``), see #81
— those routes are server-side additions; this module covers the S3-door half
that ships today.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from relata.client import RelataClient

#: HMAC info string used to derive the door-scoped SigV4 secret (#3215).
_S3_DOOR_SIGV4_INFO = b"relata-s3-door"


def derive_s3_door_secret(bearer_token: str) -> str:
    """Derive the S3-door-scoped SigV4 secret from the platform bearer (#3215).

    ``HMAC-SHA256(bearer, "relata-s3-door")`` (hex). boto3's SigV4 signer
    embeds HMAC material derived from the secret key into every signed
    request's ``Authorization`` header — that material is replayable from any
    S3 access log. Deriving the secret per-door means a leaked S3 request log
    exposes only the S3 door's credential, not the platform bearer that opens
    every other door. The derivation is deterministic, so the server is
    configured with the same value via ``RELATA_S3_SECRET_KEY``.
    """
    return hmac.new(
        bearer_token.encode("utf-8"), _S3_DOOR_SIGV4_INFO, hashlib.sha256
    ).hexdigest()


class S3Client:
    """Configure a ``boto3`` S3 client pointed at RelataDB's S3 door.

    Usage::

        from relata.s3 import S3Client

        with S3Client("http://localhost:9090",
                      bearer_token="...",
                      tenant="acme").boto3() as s3:
            s3.create_bucket(Bucket="acme-intel")
            s3.put_object(Bucket="acme-intel", Key="report.pdf", Body=b"...")
            obj = s3.get_object(Bucket="acme-intel", Key="report.pdf")
            data = obj["Body"].read()

    The bearer token becomes the ``aws_access_key_id`` (the S3 door checks the
    SigV4 ``Credential=`` access-key field against it), while the
    ``aws_secret_access_key`` is the door-scoped derivation
    ``HMAC-SHA256(bearer, "relata-s3-door")`` (#3215) — never the bearer
    itself, so a leaked S3 request log does not leak the platform credential.

    .. warning::
        SigV4 signs every request with HMAC material derived from the secret
        key, and that material travels in the ``Authorization`` header of
        every S3 request — it ends up in any S3 access/proxy log. The
        door-scoped derivation bounds the blast radius to the S3 door (rotate
        the bearer, or the door secret, independently), but treat S3 request
        logs as credential material. The server must be configured with the
        same derived secret (``RELATA_S3_SECRET_KEY=<derive_s3_door_secret(bearer)>``)
        or SigV4 signature verification will reject these requests.

    For an air-gapped / no-boto3 deployment, use :meth:`httpx` which returns a
    plain ``httpx.Client`` pre-configured against the door.

    Bucket/key mapping: RelataDB stores each object as a typed ``S3Object``
    row plus an optional content-addressed blob (when the body is at/above
    ``RELATA_S3_BLOB_THRESHOLD_MB``, default 4 MiB). ETag is SHA-256 of the
    body. Multipart parts are in-memory only today (lost on restart — open
    #37).
    """

    def __init__(
        self,
        endpoint_url: str,
        *,
        bearer_token: str | None = None,
        tenant: str | None = None,
        region: str = "us-east-1",
    ) -> None:
        self.endpoint_url = endpoint_url.rstrip("/")
        # #3214: the credential is private — no public accessor, and
        # :meth:`close` clears it.
        self._bearer_token = bearer_token
        self.tenant = tenant
        self.region = region

    @classmethod
    def from_client(cls, client: RelataClient) -> S3Client:
        return cls(
            client._base_url,  # noqa: SLF001
            bearer_token=client._bearer_token,  # noqa: SLF001
            tenant=client._tenant,  # noqa: SLF001
        )

    def boto3(self) -> Any:  # noqa: ANN401 — boto3's type is dynamic; depends on botocore
        """Return a configured ``boto3.client('s3')`` resource.

        Caller is responsible for closing the underlying HTTP session when
        done; ``boto3.client`` does not need explicit close in normal use.
        Raises ``ImportError`` if ``boto3`` is not installed.
        """
        try:
            import boto3  # type: ignore[import-not-found]
            from botocore.config import Config  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover — environment-dependent
            raise ImportError(
                "S3Client.boto3() requires `boto3`. "
                "Install with `pip install boto3` or use S3Client.httpx() for "
                "a zero-dependency httpx-based client."
            ) from exc

        # The bearer token is the access key (the door matches the SigV4
        # ``Credential=`` field against it), but the SigV4 SECRET is the
        # door-scoped derivation HMAC-SHA256(bearer, "relata-s3-door") (#3215):
        # boto3 embeds HMAC material derived from the secret into every signed
        # request, and that material is replayable from S3 access logs — it
        # must never be the platform bearer itself.
        if self._bearer_token:
            access_key = self._bearer_token
            secret_key = derive_s3_door_secret(self._bearer_token)
        else:
            access_key = "relata-anonymous"
            secret_key = "relata-anonymous"
        return boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=self.region,
            config=Config(
                # RelataDB's S3 door uses path-style addressing.
                s3={"addressing_style": "path"},
                # Bypass boto's attempt to verify AWS-style redirects.
                signature_version="s3v4",
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        )

    def httpx(self, *, timeout: float = 30.0) -> Any:  # noqa: ANN401 — httpx.Client
        """Return a plain ``httpx.Client`` pre-configured against the S3 door.

        Useful for callers that don't want a boto3 dependency. The caller
        uses standard S3 REST verbs against ``/<bucket>/<key>``.
        """
        import httpx

        headers: dict[str, str] = {}
        if self._bearer_token:
            headers["Authorization"] = f"Bearer {self._bearer_token}"
        if self.tenant:
            headers["X-Relata-Tenant-Id"] = self.tenant
        return httpx.Client(
            base_url=self.endpoint_url,
            headers=headers,
            timeout=timeout,
        )

    def close(self) -> None:
        """Clear the bearer credential (#3214).

        boto3/httpx resources handed out earlier keep their own copies of the
        credential until they are closed; this drops the SDK's copy so it does
        not linger on the client object.
        """
        self._bearer_token = None


class AsyncS3Client:
    """Configure an ``aiobotocore`` S3 client pointed at RelataDB's S3 door.

    Usage::

        from relata.s3 import AsyncS3Client

        async with AsyncS3Client(...).aio() as s3:
            await s3.create_bucket(Bucket="acme-intel")
            await s3.put_object(Bucket="acme-intel", Key="report.pdf", Body=b"...")
    """

    def __init__(
        self,
        endpoint_url: str,
        *,
        bearer_token: str | None = None,
        tenant: str | None = None,
        region: str = "us-east-1",
    ) -> None:
        self.endpoint_url = endpoint_url.rstrip("/")
        # #3214: the credential is private — no public accessor, and
        # :meth:`aclose` clears it.
        self._bearer_token = bearer_token
        self.tenant = tenant
        self.region = region

    @classmethod
    def from_client(cls, client: RelataClient) -> AsyncS3Client:
        return cls(
            client._base_url,  # noqa: SLF001
            bearer_token=client._bearer_token,  # noqa: SLF001
            tenant=client._tenant,  # noqa: SLF001
        )

    def aio(self) -> Any:  # noqa: ANN401 — aiobotocore client type is dynamic
        """Return an ``aiobotocore`` AioBaseClient.

        Raises ``ImportError`` if ``aiobotocore`` is not installed.
        """
        try:
            from aiobotocore.session import get_session  # type: ignore[import-not-found]
            from botocore.config import Config
        except ImportError as exc:  # pragma: no cover — environment-dependent
            raise ImportError(
                "AsyncS3Client.aio() requires `aiobotocore`. "
                "Install with `pip install aiobotocore`."
            ) from exc

        # Door-scoped SigV4 secret — see S3Client.boto3() (#3215).
        if self._bearer_token:
            access_key = self._bearer_token
            secret_key = derive_s3_door_secret(self._bearer_token)
        else:
            access_key = "relata-anonymous"
            secret_key = "relata-anonymous"
        session = get_session()
        return session.create_client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=self.region,
            config=Config(
                s3={"addressing_style": "path"},
                signature_version="s3v4",
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        )

    async def aclose(self) -> None:
        """Clear the bearer credential (#3214). See :meth:`S3Client.close`."""
        self._bearer_token = None
