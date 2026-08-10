"""Session-scoped coreference (anaphora) resolution — RAG epic (#4530).

Multi-turn conversational callers of ``/rag/query`` (#4514) lose their
subject across turns: "What is India's capital?" then "Where is it located?"
has no resolvable antecedent for "it" once the second query reaches
retrieval in isolation, because ``/rag/query`` is deliberately stateless and
LLM-free per ADR-0299 — that contract does not change here.

This module is the SDK-side fix: a **bounded state object**, not a replayed
transcript. After each turn the caller tells :class:`CorefResolver` the
resolved subject (typically the winning hit's first ``entity_ids`` entry —
see :func:`subject_from_hit`); before the next turn the caller runs the
query through :meth:`CorefResolver.resolve`, which only pays a network cost
when a cheap regex-level anaphora check trips.

Storage is a governed ``MemoryItem`` via ``POST /memory/remember`` /
``POST /memory/consolidate`` (:mod:`relata.memory`'s :class:`~relata.memory.Memory`)
keyed by session id — **not an SDK-local cache of the subject value itself**,
so a second process picking up the same session id still resolves correctly.
RelataDB's bi-temporal supersession means each new subject naturally
supersedes the last via ``consolidate`` rather than accumulating: the state
is structurally one row per session, never a growing transcript.

Example::

    from relata import CorefResolver, Memory, RelataClient

    with RelataClient(base_url, purpose="research") as client:
        memory = Memory(base_url, purpose="research")
        coref = CorefResolver(memory)

        hits = client.rag.query("What is India's capital?", "DocumentChunk",
                                 purpose="research").hits
        if hits:
            subject = subject_from_hit(hits[0])
            if subject:
                coref.remember_subject("session-1", subject)

        # Next turn — "it" has no local antecedent, gets rewritten.
        resolved = coref.resolve("Where is it located?", "session-1")
        # resolved == "Where is India located?"
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from relata.memory import AsyncMemory, Memory
    from relata.models import RagHit

#: Pronouns with no fixed antecedent that a stateless single-shot query
#: cannot resolve on its own. Deliberately small and cheap — this is a
#: regex-level guard, not an NLP coreference model (same cost class as
#: #4524's numeric-intent-word guard).
_PRONOUNS = frozenset({"it", "there", "that", "this", "they", "them", "these", "those"})

_PRONOUN_RE = re.compile(
    r"\b(" + "|".join(sorted(_PRONOUNS, key=len, reverse=True)) + r")\b", re.IGNORECASE
)

#: Words that look like a local antecedent (capitalised, mid-sentence) but
#: aren't proper nouns — excluded so a stray "I" or a sentence-initial
#: capital doesn't suppress a real trip.
_NOT_ANTECEDENTS = frozenset({"i"})

#: memory_class used for coref-state rows — "episodic" matches the existing
#: turn-scoped-fact convention `Memory.add`'s default docstring describes.
_COREF_MEMORY_CLASS = "episodic"


def _coref_session_key(session_id: str) -> str:
    """Derive the storage session id coref state is filed under.

    Deliberately distinct from the caller's own ``session_id`` so this
    feature's one-row-per-session state never collides with (or is scanned
    alongside) unrelated ``MemoryItem`` rows the caller stores under the
    same conversational session.
    """
    return f"{session_id}::coref-subject"


def _has_unresolved_pronoun(query: str) -> bool:
    """Cheap regex-level anaphora check.

    Trips when ``query`` contains one of :data:`_PRONOUNS` **and** no local
    antecedent (a capitalised, non-sentence-initial word) is already present
    in the same query — e.g. "Where is it located?" trips, but "What did
    Marie Curie discover and how did it change physics?" does not, because
    "Marie Curie" is a local antecedent for "it".
    """
    if not _PRONOUN_RE.search(query):
        return False
    words = query.split()
    for word in words[1:]:  # skip the sentence-initial word
        stripped = word.strip(".,!?;:'\"")
        if (
            stripped
            and stripped[0].isupper()
            and len(stripped) > 1
            and stripped.lower() not in _PRONOUNS
            and stripped.lower() not in _NOT_ANTECEDENTS
        ):
            return False
    return True


def _rewrite_with_subject(query: str, subject: str) -> str:
    """Pure string substitution: replace the first unresolved pronoun with
    ``subject``. This is the "simple case" from the mechanism — an LLM call
    given only ``{last_subject, new_query}`` is a documented alternative for
    ambiguous phrasing but is not implemented here (out of scope for the
    minimal fix; nothing about this contract prevents adding it later)."""
    return _PRONOUN_RE.sub(subject, query, count=1)


def subject_from_hit(hit: RagHit) -> str | None:
    """Extract the resolved-subject candidate from a winning ``/rag/query`` hit.

    Per the mechanism (#4530 step 1): the hit's first ``entity_ids`` entry
    (already in #4514's frozen response contract — no new field needed)
    becomes the anchor subject for anaphora resolution on the next turn.
    Returns ``None`` when the hit carries no ``entity_ids`` — nothing to
    remember, and the caller should skip :meth:`CorefResolver.remember_subject`.
    """
    return hit.entity_ids[0] if hit.entity_ids else None


class CorefResolver:
    """Session-scoped anaphora resolution over a :class:`~relata.memory.Memory` client.

    Two calls drive the whole lifecycle:

    - :meth:`remember_subject` — call after a turn resolves a subject.
    - :meth:`resolve` — call before the next turn's ``/rag/query`` call;
      rewrites an unresolved pronoun using the last remembered subject, or
      returns the query unchanged when nothing trips.

    Single-turn callers pay zero extra cost: no ``session_id`` or a
    query with no unresolved pronoun both short-circuit before any network
    call — the regex check in :func:`_has_unresolved_pronoun` is the only
    always-on cost.
    """

    def __init__(self, memory: Memory) -> None:
        self._memory = memory

    def remember_subject(self, session_id: str, subject: str) -> None:
        """Persist ``subject`` as the resolved anchor for ``session_id``.

        Structurally one row: looks up any existing coref-state row for this
        session and supersedes it via ``consolidate`` (:meth:`Memory.update`)
        instead of adding a second row, so state never grows with turn count.
        """
        if not session_id or not subject:
            return
        storage_session = _coref_session_key(session_id)
        existing = self._memory.search(subject, session_id=storage_session, top_k=1)
        if existing:
            self._memory.update(str(existing[0].get("id", "")), subject)
        else:
            self._memory.add(
                subject, session_id=storage_session, memory_class=_COREF_MEMORY_CLASS
            )

    def resolve(self, query: str, session_id: str) -> str:
        """Rewrite ``query`` using the last subject remembered for ``session_id``.

        Returns ``query`` unchanged (no network call) when ``session_id`` is
        empty or the anaphora check does not trip. When it trips, recalls
        the session's coref state (a single bounded ``top_k=1`` lookup); if
        no subject was ever stored, ``query`` is returned unchanged rather
        than guessing.
        """
        if not session_id or not _has_unresolved_pronoun(query):
            return query
        storage_session = _coref_session_key(session_id)
        rows = self._memory.search(query, session_id=storage_session, top_k=1)
        if not rows:
            return query
        subject = str(rows[0].get("content") or "")
        if not subject:
            return query
        return _rewrite_with_subject(query, subject)


class AsyncCorefResolver:
    """Async counterpart of :class:`CorefResolver` — see its docstring."""

    def __init__(self, memory: AsyncMemory) -> None:
        self._memory = memory

    async def remember_subject(self, session_id: str, subject: str) -> None:
        """Async equivalent of :meth:`CorefResolver.remember_subject`."""
        if not session_id or not subject:
            return
        storage_session = _coref_session_key(session_id)
        existing = await self._memory.search(subject, session_id=storage_session, top_k=1)
        if existing:
            await self._memory.update(str(existing[0].get("id", "")), subject)
        else:
            await self._memory.add(
                subject, session_id=storage_session, memory_class=_COREF_MEMORY_CLASS
            )

    async def resolve(self, query: str, session_id: str) -> str:
        """Async equivalent of :meth:`CorefResolver.resolve`."""
        if not session_id or not _has_unresolved_pronoun(query):
            return query
        storage_session = _coref_session_key(session_id)
        rows = await self._memory.search(query, session_id=storage_session, top_k=1)
        if not rows:
            return query
        subject = str(rows[0].get("content") or "")
        if not subject:
            return query
        return _rewrite_with_subject(query, subject)
