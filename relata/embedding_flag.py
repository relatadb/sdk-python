"""Query-embedding flag on search / RAG / recall results (#5726).

Under the server default ``RELATA_QUERY_EMBED_POLICY=lenient`` a
``HYBRID_SEARCH``, ``SIMILAR``, ``RAG_RETRIEVE`` or agent-memory recall can fall
back to lexical search. The server marks each such result with an ``embedding``
object; the SDK parses it and logs a warning when ``semantic`` is false.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict

__all__ = ["EmbeddingFlag", "warn_if_lexical"]

_log = logging.getLogger("relata")


class EmbeddingFlag(BaseModel):
    """How the query side of a search / RAG / recall result was embedded (#5726).

    Attributes:
        mode: ``"semantic"``, ``"lexical-fallback"`` or ``"caller-vector"``.
        semantic: ``False`` when the result is a lexical fallback.
        model: Model tag of the query vector, when one exists.
        reason: ``"no endpoint"``, ``"endpoint error"``, ``"model mismatch"`` or ``None``.
    """

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    mode: str = "semantic"
    semantic: bool = True
    model: str | None = None
    reason: str | None = None


def warn_if_lexical(flag: EmbeddingFlag | dict[str, Any] | None) -> None:
    """Log a warning when a result's ``embedding`` flag says it is not semantic (#5726)."""
    if isinstance(flag, dict):
        flag = EmbeddingFlag.model_validate(flag)
    if flag is not None and not flag.semantic:
        _log.warning(
            "relata: result is a lexical fallback, not semantic search (mode=%s, reason=%s); "
            "set RELATA_EMBED_URL, or RELATA_QUERY_EMBED_POLICY=strict to refuse instead",
            flag.mode,
            flag.reason,
        )
