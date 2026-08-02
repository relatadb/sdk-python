"""Typed vector / hybrid-search SDK (#88).

The server does not expose dedicated ``/similar`` / ``/hybrid_search`` HTTP
routes today — vector search is reachable via the ``HYBRID_SEARCH`` and
``SIMILAR TO`` SQL operators. This module wraps those operators as a typed
client surface so a Python caller does not have to hand-build SQL.

When (or if) dedicated HTTP routes ship, this module is the natural place to
migrate to them; the public API stays stable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from relata.client import RelataClient


class VectorClient:
    """Synchronous typed vector client — backs onto ``RelataClient.query``.

    Construct with :meth:`from_client` so the SQL executes under the parent
    client's auth, tenant, and purpose context.
    """

    def __init__(self, client: RelataClient) -> None:
        self._client = client

    @classmethod
    def from_client(cls, client: RelataClient) -> VectorClient:
        return cls(client)

    def _purpose(self, purpose: str | None) -> str:
        eff = purpose or self._client._default_purpose  # noqa: SLF001
        if not eff:
            from relata.exceptions import PurposeError

            raise PurposeError(
                "Vector operations require a purpose. Pass purpose= to the call "
                "or set a default on the RelataClient."
            )
        return eff

    def knn_search(
        self,
        object_type: str,
        embedding_slot: str,
        query_embedding: list[float],
        *,
        k: int = 10,
        ef_search: int | None = None,
        purpose: str | None = None,
    ) -> list[dict[str, Any]]:
        """Pure KNN search over a named embedding slot.

        Emits ``SELECT * FROM <Type> ORDER BY <slot> <=> '[...]' LIMIT k`` —
        the pgvector cosine form the server understands natively. ``<=>`` is
        cosine, ``<->`` is L2, ``<#>`` is negative inner product; this helper
        uses cosine because the HNSW index is cosine-trained.
        """
        import json

        emb_str = json.dumps(query_embedding)
        sql = (
            f"SELECT * FROM {object_type} "
            f"ORDER BY {embedding_slot} <=> '{emb_str}' LIMIT {k}"
        )
        result = self._client.query(sql, purpose=self._purpose(purpose))
        return result.rows

    def hybrid_search(
        self,
        object_type: str,
        *,
        query_text: str | None = None,
        query_embedding: list[float] | None = None,
        embedding_slot: str | None = None,
        k: int = 10,
        purpose: str | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid BM25 + vector search via the ``HYBRID_SEARCH`` operator.

        Caller must supply at least one of ``query_text`` (BM25 leg) or
        ``query_embedding`` (vector leg, requires ``embedding_slot``). When
        both are supplied the server fuses via reciprocal rank fusion
        (ADR-175).
        """
        if query_text is None and query_embedding is None:
            raise ValueError("hybrid_search requires query_text or query_embedding")

        import json

        # Build the HYBRID_SEARCH TVF call. The server's TVF form accepts
        # textual + vector queries as named parameters.
        args: list[str] = [f"from => '{object_type}'", f"limit => {k}"]
        if query_text is not None:
            escaped = query_text.replace("'", "''")
            args.append(f"query_text => '{escaped}'")
        if query_embedding is not None and embedding_slot is not None:
            emb_str = json.dumps(query_embedding).replace("'", "''")
            args.append(f"query_embedding => '{emb_str}'")
            args.append(f"embedding_slot => '{embedding_slot}'")
        sql = f"SELECT * FROM HYBRID_SEARCH({', '.join(args)})"
        result = self._client.query(sql, purpose=self._purpose(purpose))
        return result.rows

    def similar_to(
        self,
        object_type: str,
        reference_id: str,
        *,
        k: int = 10,
        purpose: str | None = None,
    ) -> list[dict[str, Any]]:
        """Multi-vector similarity (``SIMILAR TO``) — ranks by max-pool cosine
        over every ``_emb_*`` slot on the reference row (#1013).
        """
        escaped = reference_id.replace("'", "''")
        sql = (
            f"SELECT * FROM SIMILAR TO {object_type} "
            f"WHERE id = '{escaped}' LIMIT {k}"
        )
        result = self._client.query(sql, purpose=self._purpose(purpose))
        return result.rows

    def embed(self, text: str, *, model: str | None = None) -> dict[str, Any]:
        """Embed a single text string via ``POST /embed`` (#1172).

        Uses the server's built-in CPU lexical embedder when no sidecar is
        configured, or the GPU sidecar (``RELATA_ACCEL_ENDPOINT``) when set.
        Returns ``{"embedding": [...], "model": ..., "dim": ...}``.
        """
        payload: dict[str, Any] = {"text": text}
        if model is not None:
            payload["model"] = model
        return self._client._sync.post("/embed", payload)  # noqa: SLF001

    def embed_batch(
        self, texts: list[str], *, model: str | None = None
    ) -> dict[str, Any]:
        """Embed multiple texts in one call via ``POST /embed/batch`` (#1172).

        Returns ``{"embeddings": [[...], ...], "model": ..., "dim": ..., "count": ...}``.
        """
        payload: dict[str, Any] = {"texts": texts}
        if model is not None:
            payload["model"] = model
        return self._client._sync.post("/embed/batch", payload)  # noqa: SLF001

    def _embed_media(
        self, modality: str, bytes_b64: str, *, model: str | None = None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"bytes_b64": bytes_b64}
        if model is not None:
            payload["model"] = model
        return self._client._sync.post(f"/embed/{modality}", payload)  # noqa: SLF001

    def embed_image(
        self, bytes_b64: str, *, model: str | None = None
    ) -> dict[str, Any]:
        """Embed a single image via ``POST /embed/image`` (#2444, ADR-0276).

        ``bytes_b64`` is the base64-encoded image. Uses the server's
        configured embedder's sidecar (``RELATA_ACCEL_ENDPOINT``, ADR-177);
        raises :class:`~relata.exceptions.RelataError` (503) when the active
        embedder does not support media. Returns
        ``{"embedding": [...], "model": ..., "dim": ...}``.
        """
        return self._embed_media("image", bytes_b64, model=model)

    def embed_face(
        self, bytes_b64: str, *, model: str | None = None
    ) -> dict[str, Any]:
        """Embed a single face crop via ``POST /embed/face`` (#2444, ADR-0276).

        Returns ``{"embedding": [...], "model": ..., "dim": ...}``.
        """
        return self._embed_media("face", bytes_b64, model=model)

    def embed_audio(
        self, bytes_b64: str, *, model: str | None = None
    ) -> dict[str, Any]:
        """Embed a single audio clip via ``POST /embed/audio`` (#2444, ADR-0276).

        Returns ``{"embedding": [...], "model": ..., "dim": ...}``.
        """
        return self._embed_media("audio", bytes_b64, model=model)

    def embed_video(
        self, bytes_b64: str, *, model: str | None = None
    ) -> dict[str, Any]:
        """Embed a single video clip/keyframe via ``POST /embed/video``
        (#2444, ADR-0276).

        Returns ``{"embedding": [...], "model": ..., "dim": ...}``.
        """
        return self._embed_media("video", bytes_b64, model=model)


class AsyncVectorClient:
    """Asynchronous typed vector client — see :class:`VectorClient`."""

    def __init__(self, client: RelataClient) -> None:
        self._client = client

    @classmethod
    def from_client(cls, client: RelataClient) -> AsyncVectorClient:
        return cls(client)

    def _purpose(self, purpose: str | None) -> str:
        eff = purpose or self._client._default_purpose  # noqa: SLF001
        if not eff:
            from relata.exceptions import PurposeError

            raise PurposeError(
                "Vector operations require a purpose. Pass purpose= to the call "
                "or set a default on the RelataClient."
            )
        return eff

    async def knn_search(
        self,
        object_type: str,
        embedding_slot: str,
        query_embedding: list[float],
        *,
        k: int = 10,
        purpose: str | None = None,
    ) -> list[dict[str, Any]]:
        import json

        emb_str = json.dumps(query_embedding)
        sql = (
            f"SELECT * FROM {object_type} "
            f"ORDER BY {embedding_slot} <=> '{emb_str}' LIMIT {k}"
        )
        result = await self._client.aquery(sql, purpose=self._purpose(purpose))
        return result.rows

    async def hybrid_search(
        self,
        object_type: str,
        *,
        query_text: str | None = None,
        query_embedding: list[float] | None = None,
        embedding_slot: str | None = None,
        k: int = 10,
        purpose: str | None = None,
    ) -> list[dict[str, Any]]:
        if query_text is None and query_embedding is None:
            raise ValueError("hybrid_search requires query_text or query_embedding")

        import json

        args: list[str] = [f"from => '{object_type}'", f"limit => {k}"]
        if query_text is not None:
            escaped = query_text.replace("'", "''")
            args.append(f"query_text => '{escaped}'")
        if query_embedding is not None and embedding_slot is not None:
            emb_str = json.dumps(query_embedding).replace("'", "''")
            args.append(f"query_embedding => '{emb_str}'")
            args.append(f"embedding_slot => '{embedding_slot}'")
        sql = f"SELECT * FROM HYBRID_SEARCH({', '.join(args)})"
        result = await self._client.aquery(sql, purpose=self._purpose(purpose))
        return result.rows

    async def similar_to(
        self,
        object_type: str,
        reference_id: str,
        *,
        k: int = 10,
        purpose: str | None = None,
    ) -> list[dict[str, Any]]:
        escaped = reference_id.replace("'", "''")
        sql = (
            f"SELECT * FROM SIMILAR TO {object_type} "
            f"WHERE id = '{escaped}' LIMIT {k}"
        )
        result = await self._client.aquery(sql, purpose=self._purpose(purpose))
        return result.rows

    async def embed(self, text: str, *, model: str | None = None) -> dict[str, Any]:
        """Embed a single text string via ``POST /embed`` (#1172).

        Returns ``{"embedding": [...], "model": ..., "dim": ...}``.
        """
        payload: dict[str, Any] = {"text": text}
        if model is not None:
            payload["model"] = model
        return await self._client._async.post("/embed", payload)  # noqa: SLF001

    async def embed_batch(
        self, texts: list[str], *, model: str | None = None
    ) -> dict[str, Any]:
        """Embed multiple texts in one call via ``POST /embed/batch`` (#1172).

        Returns ``{"embeddings": [[...], ...], "model": ..., "dim": ..., "count": ...}``.
        """
        payload: dict[str, Any] = {"texts": texts}
        if model is not None:
            payload["model"] = model
        return await self._client._async.post("/embed/batch", payload)  # noqa: SLF001

    async def _embed_media(
        self, modality: str, bytes_b64: str, *, model: str | None = None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"bytes_b64": bytes_b64}
        if model is not None:
            payload["model"] = model
        return await self._client._async.post(f"/embed/{modality}", payload)  # noqa: SLF001

    async def embed_image(
        self, bytes_b64: str, *, model: str | None = None
    ) -> dict[str, Any]:
        """Embed a single image via ``POST /embed/image`` (#2444, ADR-0276).

        Returns ``{"embedding": [...], "model": ..., "dim": ...}``.
        """
        return await self._embed_media("image", bytes_b64, model=model)

    async def embed_face(
        self, bytes_b64: str, *, model: str | None = None
    ) -> dict[str, Any]:
        """Embed a single face crop via ``POST /embed/face`` (#2444, ADR-0276).

        Returns ``{"embedding": [...], "model": ..., "dim": ...}``.
        """
        return await self._embed_media("face", bytes_b64, model=model)

    async def embed_audio(
        self, bytes_b64: str, *, model: str | None = None
    ) -> dict[str, Any]:
        """Embed a single audio clip via ``POST /embed/audio`` (#2444, ADR-0276).

        Returns ``{"embedding": [...], "model": ..., "dim": ...}``.
        """
        return await self._embed_media("audio", bytes_b64, model=model)

    async def embed_video(
        self, bytes_b64: str, *, model: str | None = None
    ) -> dict[str, Any]:
        """Embed a single video clip/keyframe via ``POST /embed/video``
        (#2444, ADR-0276).

        Returns ``{"embedding": [...], "model": ..., "dim": ...}``.
        """
        return await self._embed_media("video", bytes_b64, model=model)
