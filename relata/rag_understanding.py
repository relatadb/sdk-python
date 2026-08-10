"""Query-shape dispatch + HyDE + decomposition — RAG epic SDK-side query
understanding (#4524), extended by #4536 with a content-safety pre-filter
gate and structured-attribute-filter query routing, and by #4535 with
aggregation/negation/boolean/ranking SQL routing.

Nine cooperating pieces that run **before** (or instead of) the first
``/rag/query`` call, sharing this one dispatch layer rather than a parallel
one (#4536's explicit instruction, reaffirmed by #4535 — every new shape
extends :func:`classify_query_shape`, none of them fork it):

0. **Content-safety pre-filter** (:func:`check_content_safety`) — a
   lightweight, no-LLM, pattern-based gate that runs ahead of everything
   else. On a match it short-circuits with a :class:`~relata.models.Refusal`
   (analogous to the ``clarification`` shape, but a distinct ``refused``
   reason) *before* any ``/rag/query`` or ``/query`` call is constructed.
   Off by default — a deployment opts in by passing a category->pattern
   mapping (see :data:`DANGEROUS_CONTENT_PATTERNS` for a reviewed starter
   set); this is a deployment-specific policy decision, not something
   RelataDB should impose universally. This is a coarse pre-filter, not a
   substitute for governance already enforced in-scan by RelataDB's
   ACL/cell-masking (ADR-0299 §11).
1. **Query-shape dispatch** (:func:`classify_query_shape`) — regex
   classification of the query text. Conjunction shapes (``"who ... and
   ..."``, ``"with X and Y"``) route toward ``expand_window=True`` so the
   answer's supporting evidence can span more of a single document.
   Enumeration shapes (``"which"``/``"list"``/``"how many"``) route toward a
   wider ``top_k`` so document-level aggregation over the hits has enough
   candidates to work with. **Attribute-filter** shapes (#4536, e.g. "list of
   persons above 6ft tall with moustache, fair complexion") name physical/
   descriptive properties rather than semantic content — these route toward
   SQL, not retrieval; see :func:`route_attribute_filter_query`.
2. **HyDE** (:func:`expand_query_hyde`) — generate a hypothetical answer via
   one caller-supplied LLM call, then send that hypothetical answer as the
   ``/rag/query`` search text instead of the bare query — **except** when
   numeric intent is detected (:func:`is_numeric_intent`: revenue,
   percentage, count, dosage, ...), where HyDE is skipped entirely since a
   hallucinated hypothetical answer would invent a plausible-but-wrong
   number.
3. **Decomposition** (:func:`decompose_query`, :func:`rrf_merge`) — split a
   multi-part question into sub-queries, issue one ``/rag/query`` call per
   sub-query (in parallel), and Reciprocal-Rank-Fuse the results back
   together using an auto-scaling dampening constant,
   ``k = max(10, 60 / n)`` (:func:`rrf_k_for_fanout`) — **not** the fixed
   ``k=60`` RelataDB uses server-side to fuse BM25⊕vector *within* one
   ``/rag/query`` call (``crates/relata-query/src/hybrid.rs::RRF_K``). That
   is a different merge at a different layer; reusing its fixed constant
   here would under-weight top ranks as the fan-out grows.
4. **Structured-attribute-filter routing** (:func:`route_attribute_filter_query`
   / :func:`aroute_attribute_filter_query`) — when the target ``Type``'s
   schema has canonical fields matching the query's named attributes, route
   to governed SQL ``WHERE`` predicates instead of a semantic-similarity
   guess; when no matching field exists, fall back to retrieval with an
   explicit ``low_confidence`` signal rather than silently guessing.
5. **Aggregation/negation/boolean/ranking routing** (#4535,
   :func:`route_aggregation_query`, :func:`route_negation_query`,
   :func:`route_boolean_query`, :func:`route_ranking_query`, and their
   ``aroute_*`` async twins) — a pure-retrieval top-k search structurally
   cannot answer "how many", "which are NOT", "X AND Y", or "top N" shaped
   questions (there is no notion of "all"/"not"/"count"/"order by" in a
   similarity search). Each shape routes to governed SQL instead:
   aggregation to ``COUNT(*)``, negation to a ``NOT ILIKE``/``!=``
   predicate, boolean to multiple predicates joined by the query's own
   ``AND``/``OR``, ranking to ``ORDER BY ... LIMIT N``. Aggregation always
   attempts SQL routing (a bare ``COUNT(*)`` needs no domain vocabulary);
   negation/boolean/ranking require a caller-supplied ``structured_field_map``
   (keyword -> canonical field — deployment-specific vocabulary, unlike
   :data:`ATTRIBUTE_FIELD_KEYWORDS`'s general-purpose physical-attribute
   defaults) and fall back to retrieval with ``low_confidence`` set when
   they can't build a predicate, mirroring #4536's attribute-filter
   fallback contract exactly.

:func:`smart_rag_query` / :func:`asmart_rag_query` are the one entry point
that composes all nine ahead of calling
:class:`~relata.rag.RagClient`/:class:`~relata.rag.AsyncRagClient`.

Per ADR-0298, RelataDB has no server-side agent loop and orchestration lives
in the SDK — this module is deliberately Python-only (the canonical SDK for
this epic's orchestration layer, per #4523's ticket body); ``RagClient``
itself stays a thin pass-through on the TypeScript/Go/Rust surfaces with no
equivalent of this module.

This module never calls an LLM itself: :data:`HypothesisGenerator` is a
caller-supplied callable (RelataDB runs no LLM, ADR-013).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import re
from collections.abc import Collection, Mapping
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from relata.client import RelataClient
    from relata.rag import AsyncRagClient, RagClient

from relata.models import QueryResult, RagHit, RagQueryResponse, Refusal
from relata.rag import DEFAULT_TOP_K

# ---------------------------------------------------------------------------
# 0. Content-safety pre-filter (#4536) — runs before everything else, before
#    any /rag/query or /query call is constructed.
# ---------------------------------------------------------------------------

#: A reviewed starter set of dangerous-content categories -> compiled regex,
#: for a counter-terrorism-intelligence-style deployment. **Not applied
#: unless a caller explicitly passes it** (or its own mapping) — the gate
#: itself is off by default (see :func:`check_content_safety`). Patterns are
#: precision-reviewed to require construction/how-to intent, not mere
#: mention, so a news article or countermeasures/deactivation discussion
#: about the same subject matter is not falsely refused.
DANGEROUS_CONTENT_PATTERNS: dict[str, re.Pattern[str]] = {
    "weapons_explosives_construction": re.compile(
        r"\bhow\s+to\s+(?:build|make|construct|assemble|manufacture)\s+"
        r"(?:an?\s+)?(?:ied|bomb|explosive\w*|pipe\s*bomb|detonator)\b"
        r"|\b(?:ied|bomb|explosive)\s+construction\s+(?:guide|instructions?|manual)\b",
        re.IGNORECASE,
    ),
}


def check_content_safety(
    query: str,
    *,
    patterns: Mapping[str, str | re.Pattern[str]] | None = None,
) -> Refusal | None:
    """Coarse, no-LLM content-safety pre-filter (#4536), run ahead of every
    other technique in this module.

    Off by default: ``patterns=None`` (the default) never refuses anything.
    A deployment opts in explicitly by passing a category-label -> pattern
    mapping — :data:`DANGEROUS_CONTENT_PATTERNS` is a reviewed starter set,
    but this is a deployment-specific policy decision (a counter-terrorism
    intelligence platform and a general enterprise deployment have very
    different thresholds for what's in-scope), so nothing is applied by
    default.

    Returns a :class:`~relata.models.Refusal` on the first matching category
    (mapping iteration order), or ``None`` when nothing matches (or the gate
    is disabled). This is a coarse pre-filter, not a substitute for
    governance already enforced in-scan by RelataDB's ACL/cell-masking
    (ADR-0299 §11) — it exists to avoid ever constructing a retrieval or SQL
    call for clearly out-of-scope content, not to replace access control.
    """
    if not patterns:
        return None
    for category, pattern in patterns.items():
        compiled = (
            pattern if isinstance(pattern, re.Pattern) else re.compile(pattern, re.IGNORECASE)
        )
        if compiled.search(query):
            return Refusal(
                category=category,
                message=(
                    f"This query was refused by the content-safety pre-filter "
                    f"(category: {category!r}) before any retrieval or SQL call was made."
                ),
            )
    return None


# ---------------------------------------------------------------------------
# 1. Query-shape dispatch
# ---------------------------------------------------------------------------


class QueryShape(str, Enum):
    """Detected shape of a query text, per :func:`classify_query_shape`."""

    ATTRIBUTE_FILTER = "attribute_filter"
    AGGREGATION = "aggregation"
    NEGATION = "negation"
    BOOLEAN = "boolean"
    RANKING = "ranking"
    CONJUNCTION = "conjunction"
    ENUMERATION = "enumeration"
    SIMPLE = "simple"


#: Shapes routed to governed SQL instead of retrieval (#4536's
#: ATTRIBUTE_FILTER, extended by #4535's four). A pure-retrieval top-k
#: search has no notion of "all"/"not"/"count"/"order by" — see the module
#: docstring's item 5.
SQL_ROUTABLE_SHAPES = frozenset(
    {
        QueryShape.ATTRIBUTE_FILTER,
        QueryShape.AGGREGATION,
        QueryShape.NEGATION,
        QueryShape.BOOLEAN,
        QueryShape.RANKING,
    }
)


#: Physical/descriptive-attribute keyword -> canonical field name this
#: keyword's presence implies (#4536's structured-attribute-filter shape).
#: Verified against the sampled production query "list of persons above 6ft
#: tall with moustache, fair complexion". Deliberately example-driven and
#: conservative — callers extend/override via the ``field_map`` argument to
#: :func:`extract_attribute_filters` / :func:`route_attribute_filter_query`
#: (e.g. a deployment's own ontology field names).
ATTRIBUTE_FIELD_KEYWORDS: dict[str, str] = {
    "tall": "height",
    "height": "height",
    "moustache": "facial_hair",
    "mustache": "facial_hair",
    "beard": "facial_hair",
    "clean-shaven": "facial_hair",
    "complexion": "complexion",
    "fair": "complexion",
    "dark-complexioned": "complexion",
    "wheatish": "complexion",
    "olive-skinned": "complexion",
    "build": "build",
    "athletic": "build",
    "heavyset": "build",
    "slim": "build",
    "lean": "build",
    "scar": "distinguishing_marks",
    "tattoo": "distinguishing_marks",
    "birthmark": "distinguishing_marks",
    "mole": "distinguishing_marks",
}

_ATTRIBUTE_FILTER_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(ATTRIBUTE_FIELD_KEYWORDS, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)


def is_attribute_filter_intent(query: str) -> bool:
    """``True`` when ``query`` names at least one physical/descriptive
    attribute keyword from :data:`ATTRIBUTE_FIELD_KEYWORDS` — #4536's
    structured-attribute-filter shape (e.g. "list of persons above 6ft tall
    with moustache, fair complexion"), structurally distinct from
    semantic-content retrieval and from #4535's aggregation/negation/
    boolean/ranking shapes."""
    return bool(_ATTRIBUTE_FILTER_RE.search(query))


#: Conjunction patterns: "who ... and ...", "with X and Y", "both ... and ...".
_CONJUNCTION_RE = re.compile(
    r"\bwho\b.*\band\b|\bwith\b\s+\S+\s+\band\b\s+\S+|\bboth\b.*\band\b",
    re.IGNORECASE,
)

#: Enumeration patterns: a query opening with "which"/"list"/"how many"/etc.
_ENUMERATION_RE = re.compile(
    r"^\s*(which|what\s+are\s+all|list|enumerate|name\s+all|how\s+many)\b",
    re.IGNORECASE,
)

#: Enumeration queries aggregate across more of the corpus than a single-fact
#: lookup, so they widen top_k past #4514's server default
#: (``RAG_RETRIEVE_DEFAULT_TOP_K`` = 8) to give document-level aggregation
#: enough candidates to work with.
ENUMERATION_TOP_K = 25

# ---------------------------------------------------------------------------
# 1b. Aggregation/negation/boolean/ranking detection (#4535) — a pure-
#     retrieval top-k search structurally cannot answer these; each routes
#     to governed SQL (see routers in section 5 below) instead of retrieval.
# ---------------------------------------------------------------------------

#: Lexical cues for a countable/aggregate intent ("how many", "count of",
#: "total number of", "average", ...). Deliberately does not include bare
#: "percentage"/"revenue"/etc (those are :data:`NUMERIC_INTENT_WORDS`'s HyDE
#: guard, a different concern — a numeric-intent query that isn't
#: aggregation-shaped, e.g. "What was the total revenue for Q1?", still goes
#: through retrieval with HyDE skipped, not SQL routing).
_AGGREGATION_RE = re.compile(
    r"\bhow\s+many\b|\bhow\s+much\b|\bcount\s+of\b|\btotal\s+number\s+of\b|"
    r"\btotal\s+count\b|\baverage\b|\bavg\b|\bsum\s+of\b",
    re.IGNORECASE,
)

#: Negation markers ("not", "except", "excluding", "without", ...) adjacent
#: to an entity/attribute reference (e.g. "Which SIMI members are NOT in
#: custody?").
_NEGATION_RE = re.compile(
    r"\bnot\b|\bexcept\b|\bexcluding\b|\bwithout\b|\bisn't\b|\baren't\b|"
    r"\bdoesn't\b|\bdon't\b",
    re.IGNORECASE,
)

#: Boolean/set-operation shape: "members of X and Y", "belongs to X or Y",
#: "affiliated with X and Y" — multiple entity mentions joined by and/or
#: where the intent is a set operation (e.g. "Members of SIMI AND LeT"),
#: structurally distinct from #4524's CONJUNCTION shape (which is about a
#: single answer spanning more of one document's content, not a set
#: operation over named entities).
_BOOLEAN_RE = re.compile(
    r"\b(?:members?\s+of|belongs?\s+to|affiliated\s+with|part\s+of)\s+"
    r"[\w&\-]+(?:\s+[\w&\-]+){0,3}\s+(?:and|or)\s+[\w&\-]+",
    re.IGNORECASE,
)

#: Ranking/top-N shape: "top N", "first N", "most X", "highest"/"lowest".
_RANKING_RE = re.compile(
    r"\btop\s+\d+\b|\bfirst\s+\d+\b|\bmost\s+\w+\b|\bhighest\b|\blowest\b|\bleast\s+\w+\b",
    re.IGNORECASE,
)


def is_aggregation_intent(query: str) -> bool:
    """``True`` when ``query`` carries a countable/aggregate lexical cue
    (#4535, e.g. "how many", "total number of")."""
    return bool(_AGGREGATION_RE.search(query))


def is_negation_intent(query: str) -> bool:
    """``True`` when ``query`` carries a negation marker (#4535, e.g. "not",
    "except", "excluding", "without")."""
    return bool(_NEGATION_RE.search(query))


def is_boolean_intent(query: str) -> bool:
    """``True`` when ``query`` names multiple entities joined by "and"/"or"
    in a set-operation shape (#4535, e.g. "members of SIMI and LeT")."""
    return bool(_BOOLEAN_RE.search(query))


def is_ranking_intent(query: str) -> bool:
    """``True`` when ``query`` carries a ranking/top-N lexical cue (#4535,
    e.g. "top 5", "most active", "highest")."""
    return bool(_RANKING_RE.search(query))


def classify_query_shape(query: str) -> QueryShape:
    """Classify ``query`` as a :class:`QueryShape` via regex — no LLM call,
    cheap enough to run on every query.

    Checked in this order: ATTRIBUTE_FILTER, AGGREGATION, NEGATION,
    BOOLEAN, RANKING, CONJUNCTION, ENUMERATION, SIMPLE.

    ATTRIBUTE_FILTER is checked first: it is a structurally different query
    type (routes toward SQL, not retrieval — #4536), and its trigger words
    would otherwise sometimes overlap with ENUMERATION's opening-word cues
    (e.g. "list of persons above 6ft tall ..." opens with "list").

    AGGREGATION/NEGATION/BOOLEAN/RANKING (#4535) are checked next, ahead of
    CONJUNCTION/ENUMERATION, for the same reason: "how many incidents were
    filed last year?" opens with an ENUMERATION-style intent word ("how
    many" used to fall through to ENUMERATION's `how\\s+many` alternation)
    but must classify as AGGREGATION so it routes to a real ``COUNT(*)``
    instead of a top-k retrieval-and-hope-the-LLM-counts-them guess — the
    exact failure mode #4535 exists to close.
    """
    if is_attribute_filter_intent(query):
        return QueryShape.ATTRIBUTE_FILTER
    if is_aggregation_intent(query):
        return QueryShape.AGGREGATION
    if is_negation_intent(query):
        return QueryShape.NEGATION
    if is_boolean_intent(query):
        return QueryShape.BOOLEAN
    if is_ranking_intent(query):
        return QueryShape.RANKING
    if _CONJUNCTION_RE.search(query):
        return QueryShape.CONJUNCTION
    if _ENUMERATION_RE.search(query):
        return QueryShape.ENUMERATION
    return QueryShape.SIMPLE


def _apply_query_shape(shape: QueryShape, rag_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``rag_kwargs`` adjusted for the detected ``shape``.

    Conjunction queries route toward ``expand_window=True`` (only if the
    caller did not already set it explicitly). Enumeration queries widen
    ``top_k`` to at least :data:`ENUMERATION_TOP_K`, taking the max with
    whatever the caller already requested so an explicit larger value is
    never shrunk.
    """
    out = dict(rag_kwargs)
    if shape is QueryShape.CONJUNCTION:
        out.setdefault("expand_window", True)
    elif shape is QueryShape.ENUMERATION:
        out["top_k"] = max(out.get("top_k", DEFAULT_TOP_K), ENUMERATION_TOP_K)
    return out


# ---------------------------------------------------------------------------
# 2. HyDE, with the numeric-intent guard
# ---------------------------------------------------------------------------

#: Signature for a caller-supplied hypothetical-answer generator. RelataDB
#: runs no LLM (ADR-013) — this is always provided by the application, never
#: called by RelataDB itself.
HypothesisGenerator = Callable[[str], str]

#: Numeric-intent trigger words (verified list from the reference toolkit).
#: A query containing one of these usually wants an exact number; HyDE's
#: hallucinated hypothetical answer would invent a plausible-but-wrong one,
#: so HyDE is skipped entirely for these rather than risk it.
NUMERIC_INTENT_WORDS = frozenset(
    {
        "revenue",
        "percentage",
        "percent",
        "count",
        "how many",
        "total",
        "sum",
        "average",
        "mean",
        "median",
        "dosage",
        "dose",
        "amount",
        "quantity",
        "number of",
        "price",
        "cost",
        "rate",
        "ratio",
        "margin",
        "growth rate",
    }
)

_NUMERIC_INTENT_ALTERNATION = "|".join(
    re.escape(w) for w in sorted(NUMERIC_INTENT_WORDS, key=len, reverse=True)
)
_NUMERIC_INTENT_RE = re.compile(r"\b(" + _NUMERIC_INTENT_ALTERNATION + r")\b", re.IGNORECASE)


def is_numeric_intent(query: str) -> bool:
    """``True`` when ``query`` contains a :data:`NUMERIC_INTENT_WORDS` trigger
    word — the guard :func:`expand_query_hyde` uses to skip HyDE entirely."""
    return bool(_NUMERIC_INTENT_RE.search(query))


def expand_query_hyde(query: str, *, hypothesis_fn: HypothesisGenerator) -> str:
    """Return the HyDE-expanded search text for ``query``, or ``query``
    unchanged when :func:`is_numeric_intent` trips.

    ``hypothesis_fn`` is called with ``query`` and must return a hypothetical
    answer (plain text); that hypothetical answer becomes the ``query`` field
    sent to ``POST /rag/query`` (#4514), which embeds and searches it
    server-side — this function does not embed anything itself.
    """
    if is_numeric_intent(query):
        return query
    hypothetical = hypothesis_fn(query)
    return hypothetical or query


# ---------------------------------------------------------------------------
# 3. Decomposition + RRF merge with auto-scaling k
# ---------------------------------------------------------------------------

#: Splits a multi-part question into sub-queries on (a) a `?` immediately
#: followed by a new capitalised sentence, or (b) an "and" conjoining two
#: question clauses ("... and when ...", "... and how ...").
_DECOMPOSITION_SPLIT_RE = re.compile(
    r"\?\s+(?=[A-Z])|\s+and\s+(?=(?:what|when|where|who|which|how|why)\b)",
    re.IGNORECASE,
)


def decompose_query(query: str) -> list[str]:
    """Split ``query`` into independently-retrievable sub-queries.

    Returns ``[query]`` unchanged (a single-element list) when no split
    point is detected — callers should treat ``len(...) == 1`` as "no
    decomposition happened," not as an error.
    """
    parts = [p.strip() for p in _DECOMPOSITION_SPLIT_RE.split(query) if p.strip()]
    return parts if parts else [query]


def rrf_k_for_fanout(n_subqueries: int) -> float:
    """Auto-scaling RRF dampening constant for merging ``n_subqueries``
    parallel result lists: ``max(10, 60 / n)`` (verified in the reference
    toolkit). Deliberately *not* RelataDB's fixed internal ``RRF_K=60``
    (``crates/relata-query/src/hybrid.rs``), which fuses BM25⊕vector within
    one ``/rag/query`` call — a different merge at a different layer.
    """
    if n_subqueries <= 0:
        raise ValueError("n_subqueries must be positive")
    return max(10.0, 60.0 / n_subqueries)


def _rrf_scores(responses: list[RagQueryResponse], k: float) -> dict[str, float]:
    """Per-``chunk_id`` summed RRF contribution ``1 / (k + rank)`` across
    ``responses``. Split out from :func:`rrf_merge` so the exact arithmetic
    (and, in particular, which ``k`` it was actually called with) is directly
    assertable in tests without depending on final hit ordering."""
    scores: dict[str, float] = {}
    for response in responses:
        for rank, hit in enumerate(response.hits, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


def rrf_merge(responses: list[RagQueryResponse], *, k: float) -> RagQueryResponse:
    """Reciprocal-Rank-Fuse ``responses`` (one per sub-query) into a single
    :class:`~relata.models.RagQueryResponse`, deduped by ``chunk_id`` and
    re-ordered by the summed RRF contribution ``1 / (k + rank)`` (see
    :func:`_rrf_scores`).

    Each hit's own ``bm25_score``/``vector_score`` (#4514's per-channel
    contract) is passed through untouched — RRF only decides final
    ordering/dedup here, it never fuses or overwrites those fields.
    """
    scores = _rrf_scores(responses, k)
    first_seen: dict[str, RagHit] = {}
    for response in responses:
        for hit in response.hits:
            first_seen.setdefault(hit.chunk_id, hit)
    ordered_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    hits = [first_seen[cid] for cid in ordered_ids]
    return RagQueryResponse(hits=hits, total=len(hits))


# ---------------------------------------------------------------------------
# 4. Structured-attribute-filter routing (#4536)
# ---------------------------------------------------------------------------

#: "above/over/taller than N (ft|feet|cm|in|inches)" — the >= half of the
#: height-comparison shape. Height is handled via these comparison patterns
#: only, never as a bare-keyword ILIKE match (see :func:`extract_attribute_filters`).
_ATTRIBUTE_ABOVE_RE = re.compile(
    r"\b(?:above|over|taller\s+than|greater\s+than|more\s+than)\s+"
    r"(\d+(?:\.\d+)?)\s*(ft|feet|cm|in|inches)?\b",
    re.IGNORECASE,
)
#: "below/under/shorter than N (ft|feet|cm|in|inches)" — the <= half.
_ATTRIBUTE_BELOW_RE = re.compile(
    r"\b(?:below|under|shorter\s+than|less\s+than)\s+"
    r"(\d+(?:\.\d+)?)\s*(ft|feet|cm|in|inches)?\b",
    re.IGNORECASE,
)

_FEET_TO_CM = 30.48
_INCH_TO_CM = 2.54


def _height_value_cm(raw: str, unit: str | None) -> float:
    """Normalise a matched height value + optional unit to centimeters
    (``ft``/``feet`` is the assumed default unit when none is given, since
    that is how the sampled production query phrased it: "above 6ft tall")."""
    value = float(raw)
    unit = (unit or "ft").lower()
    if unit in ("ft", "feet"):
        return round(value * _FEET_TO_CM, 1)
    if unit in ("in", "inches"):
        return round(value * _INCH_TO_CM, 1)
    return value  # already centimeters


def extract_attribute_filters(
    query: str,
    *,
    field_map: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Extract ``[{field, op, value}]`` predicates for an attribute-filter-
    shaped query (#4536) — the same filter-list shape
    :func:`~relata.rag.filters_for_entity_ids` already produces for
    ``/rag/query``, reused here as the input to SQL ``WHERE`` construction
    (:func:`route_attribute_filter_query`) rather than a ``filters=`` entry.

    ``field_map`` overrides/extends :data:`ATTRIBUTE_FIELD_KEYWORDS`
    (keyword -> canonical field name); pass a deployment's own ontology
    field names when they differ from the built-in defaults.

    Height comparisons (e.g. "above 6ft tall") become a ``>=``/``<=``
    predicate on the mapped field, converted to centimeters. Every other
    matched keyword (e.g. "moustache", "fair complexion", "scar") becomes an
    ``ILIKE`` predicate. Returns ``[]`` when nothing in the query matches
    :data:`ATTRIBUTE_FIELD_KEYWORDS` — callers must treat that as "could not
    extract a filter," not an error.
    """
    keywords = dict(ATTRIBUTE_FIELD_KEYWORDS)
    if field_map:
        keywords.update(field_map)
    height_field = keywords.get("height", "height")

    filters: list[dict[str, Any]] = []

    above = _ATTRIBUTE_ABOVE_RE.search(query)
    if above:
        above_cm = _height_value_cm(above.group(1), above.group(2))
        filters.append({"field": height_field, "op": ">=", "value": above_cm})
    below = _ATTRIBUTE_BELOW_RE.search(query)
    if below:
        below_cm = _height_value_cm(below.group(1), below.group(2))
        filters.append({"field": height_field, "op": "<=", "value": below_cm})

    seen_fields = {f["field"] for f in filters}
    for keyword in sorted(keywords, key=len, reverse=True):
        field = keywords[keyword]
        if field == height_field or field in seen_fields or keyword == field:
            # Height is comparison-only (above/below), never ILIKE.
            # `keyword == field` (e.g. "complexion" -> "complexion",
            # "build" -> "build") is a bare field-name mention used only to
            # detect intent (:func:`is_attribute_filter_intent`) — it names
            # no actual value, so it must not become a self-referential
            # `complexion ILIKE '%complexion%'` predicate. Every other field
            # takes its first matching keyword only, so "fair complexion,
            # dark eyes" doesn't produce two conflicting predicates on one
            # field.
            continue
        if re.search(rf"\b{re.escape(keyword)}\b", query, re.IGNORECASE):
            filters.append({"field": field, "op": "ILIKE", "value": f"%{keyword}%"})
            seen_fields.add(field)

    return filters


def _sql_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


def _filter_to_sql_predicate(f: dict[str, Any]) -> str:
    field, op, value = f["field"], f["op"], f["value"]
    if op == "IN":
        values = ", ".join(_sql_literal(v) for v in value)
        return f"{field} IN ({values})"
    return f"{field} {op} {_sql_literal(value)}"


def _attribute_filter_sql(
    query: str,
    type: str,  # noqa: A002
    *,
    field_map: Mapping[str, str] | None,
    known_fields: Collection[str] | None,
) -> str | None:
    """Shared by :func:`route_attribute_filter_query` and
    :func:`aroute_attribute_filter_query`: build the SQL statement, or
    return ``None`` when no (schema-confirmed) filter could be built."""
    filters = extract_attribute_filters(query, field_map=field_map)
    if known_fields is not None:
        filters = [f for f in filters if f["field"] in known_fields]
    if not filters:
        return None
    where_clause = " AND ".join(_filter_to_sql_predicate(f) for f in filters)
    return f"SELECT * FROM {type} WHERE {where_clause}"


def route_attribute_filter_query(
    relata_client: RelataClient,
    query: str,
    type: str,  # noqa: A002
    *,
    purpose: str | None = None,
    field_map: Mapping[str, str] | None = None,
    known_fields: Collection[str] | None = None,
) -> QueryResult | None:
    """Route an attribute-filter-shaped query (#4536) to a governed SQL
    ``/query`` call instead of a semantic-similarity retrieval guess.

    Returns ``None`` (never raises) when :func:`extract_attribute_filters`
    found nothing, or when ``known_fields`` is given and none of the
    extracted filters' fields are present in it — the caller falls back to
    retrieval in that case rather than silently guessing against fields that
    don't exist on ``type``'s schema. Pass ``known_fields`` (e.g. from a
    prior ``relata_client.get_type(type)`` schema lookup) whenever it is
    available; without it, every extracted filter is sent as-is.
    """
    sql = _attribute_filter_sql(query, type, field_map=field_map, known_fields=known_fields)
    if sql is None:
        return None
    return relata_client.query(sql, purpose=purpose)


async def aroute_attribute_filter_query(
    relata_client: RelataClient,
    query: str,
    type: str,  # noqa: A002
    *,
    purpose: str | None = None,
    field_map: Mapping[str, str] | None = None,
    known_fields: Collection[str] | None = None,
) -> QueryResult | None:
    """Async variant of :func:`route_attribute_filter_query`."""
    sql = _attribute_filter_sql(query, type, field_map=field_map, known_fields=known_fields)
    if sql is None:
        return None
    return await relata_client.aquery(sql, purpose=purpose)


# ---------------------------------------------------------------------------
# 5. Aggregation/negation/boolean/ranking SQL routing (#4535)
# ---------------------------------------------------------------------------
#
# Unlike :data:`ATTRIBUTE_FIELD_KEYWORDS` (a general-purpose default for
# physical/descriptive attributes), there is no sensible universal keyword
# vocabulary for these four shapes — "which organizations", "which status
# values" are entirely deployment-specific. Callers supply their own
# ``structured_field_map`` (keyword -> canonical field name, ideally sourced
# from schema introspection per the design doc); without one, NEGATION/
# BOOLEAN/RANKING can't build a predicate and fall back to retrieval.
# AGGREGATION is the one exception: a bare ``COUNT(*)`` needs no domain
# vocabulary at all, so it always attempts SQL routing.
#
# ponytail: this section covers #4535's "small-N" routing only (a bare
# ``COUNT(*)`` or a bounded ``SELECT ... WHERE`` — one row or a small,
# synthesizable result). #4535's second, separately-scoped half — the
# large-result-set policy (governed `S3Object` export + a new `relata-jobs`
# job type + `bucket/key`-shaped response for an enumeration query that
# legitimately returns thousands-to-millions of rows) is genuinely
# substantial new cross-crate infrastructure (relata-jobs, s3_server,
# job-status polling) and is deliberately deferred out of this pass rather
# than forced through half-built. `route_aggregation_query` already routes
# a "how many" question to a real one-row `COUNT(*)` per the ticket's
# acceptance criteria; a "give me all X" enumeration question with a large
# result still falls back to retrieval today rather than to the S3Object/
# job export path the design doc describes — see the ticket for tracking.


def extract_keyword_filters(
    query: str,
    field_map: Mapping[str, str],
    *,
    op: str = "ILIKE",
    dedupe_fields: bool = True,
) -> list[dict[str, Any]]:
    """Generic keyword -> field predicate extraction, shared by
    :func:`route_aggregation_query`/:func:`route_negation_query`/
    :func:`route_boolean_query` (#4535) — the same mechanism as
    :func:`extract_attribute_filters`'s ``ILIKE`` branch, but against a
    caller-required ``field_map`` rather than a built-in default.

    Each keyword present in ``query`` (as a whole word, case-insensitive)
    becomes one ``{field, op, value}`` predicate using ``field_map``'s
    mapped field name. ``value`` is the keyword itself, wrapped in
    ``%...%`` for the ``ILIKE``/``NOT ILIKE`` operator family.

    ``dedupe_fields=True`` (the default) keeps only the first matching
    keyword per field, matching :func:`extract_attribute_filters`'s
    behaviour. :func:`route_boolean_query` passes ``dedupe_fields=False``
    because a boolean shape's whole point is multiple predicates over the
    *same* field (e.g. ``organization = 'SIMI' AND organization = 'LeT'``).
    """
    filters: list[dict[str, Any]] = []
    seen_fields: set[str] = set()
    for keyword in sorted(field_map, key=len, reverse=True):
        field = field_map[keyword]
        if dedupe_fields and field in seen_fields:
            continue
        if re.search(rf"\b{re.escape(keyword)}\b", query, re.IGNORECASE):
            value = f"%{keyword}%" if op in ("ILIKE", "NOT ILIKE") else keyword
            filters.append({"field": field, "op": op, "value": value})
            seen_fields.add(field)
    return filters


def _first_matching_field(query: str, field_map: Mapping[str, str]) -> str | None:
    """Return the canonical field for the first ``field_map`` keyword found
    in ``query`` (longest keyword first), or ``None`` when nothing matches —
    used by :func:`route_ranking_query` to pick the ``ORDER BY`` column."""
    for keyword in sorted(field_map, key=len, reverse=True):
        if re.search(rf"\b{re.escape(keyword)}\b", query, re.IGNORECASE):
            return field_map[keyword]
    return None


def route_aggregation_query(
    relata_client: RelataClient,
    query: str,
    type: str,  # noqa: A002
    *,
    purpose: str | None = None,
    field_map: Mapping[str, str] | None = None,
    known_fields: Collection[str] | None = None,
) -> QueryResult | None:
    """Route an aggregation-shaped query (#4535, e.g. "how many incidents
    happened in 2023?") to a governed ``SELECT COUNT(*)`` instead of a
    retrieval top-k the LLM would have to guess-count from.

    Always attempts SQL routing (returns a real ``QueryResult``, never
    ``None``) when no ``field_map`` is given, or when ``field_map`` is given
    but names nothing found in ``query`` — a bare ``COUNT(*)`` over ``type``
    needs no domain vocabulary. Returns ``None`` (fall back to retrieval)
    only when ``field_map`` *did* extract a keyword filter but
    ``known_fields`` rules every extracted field out — i.e. the query named
    a specific criterion that can't be mapped onto the schema, so an
    unfiltered count would silently answer the wrong question.
    """
    filters = extract_keyword_filters(query, field_map) if field_map else []
    if known_fields is not None and filters:
        resolved = [f for f in filters if f["field"] in known_fields]
        if not resolved:
            return None
        filters = resolved
    sql = f"SELECT COUNT(*) AS count FROM {type}"
    if filters:
        where_clause = " AND ".join(_filter_to_sql_predicate(f) for f in filters)
        sql += f" WHERE {where_clause}"
    return relata_client.query(sql, purpose=purpose)


async def aroute_aggregation_query(
    relata_client: RelataClient,
    query: str,
    type: str,  # noqa: A002
    *,
    purpose: str | None = None,
    field_map: Mapping[str, str] | None = None,
    known_fields: Collection[str] | None = None,
) -> QueryResult | None:
    """Async variant of :func:`route_aggregation_query`."""
    filters = extract_keyword_filters(query, field_map) if field_map else []
    if known_fields is not None and filters:
        resolved = [f for f in filters if f["field"] in known_fields]
        if not resolved:
            return None
        filters = resolved
    sql = f"SELECT COUNT(*) AS count FROM {type}"
    if filters:
        where_clause = " AND ".join(_filter_to_sql_predicate(f) for f in filters)
        sql += f" WHERE {where_clause}"
    return await relata_client.aquery(sql, purpose=purpose)


def route_negation_query(
    relata_client: RelataClient,
    query: str,
    type: str,  # noqa: A002
    *,
    purpose: str | None = None,
    field_map: Mapping[str, str] | None = None,
    known_fields: Collection[str] | None = None,
) -> QueryResult | None:
    """Route a negation-shaped query (#4535, e.g. "Which SIMI members are
    NOT in custody?") to a governed ``NOT ILIKE`` predicate instead of
    retrieval — retrieval has no notion of "not".

    Requires ``field_map`` (no built-in default; negation vocabulary is
    deployment-specific). Returns ``None`` when ``field_map`` is absent, or
    extracts no keyword, or (with ``known_fields`` given) resolves to no
    schema-known field — every case falls back to retrieval rather than
    guessing.
    """
    if not field_map:
        return None
    filters = extract_keyword_filters(query, field_map, op="NOT ILIKE")
    if known_fields is not None:
        filters = [f for f in filters if f["field"] in known_fields]
    if not filters:
        return None
    where_clause = " AND ".join(_filter_to_sql_predicate(f) for f in filters)
    return relata_client.query(f"SELECT * FROM {type} WHERE {where_clause}", purpose=purpose)


async def aroute_negation_query(
    relata_client: RelataClient,
    query: str,
    type: str,  # noqa: A002
    *,
    purpose: str | None = None,
    field_map: Mapping[str, str] | None = None,
    known_fields: Collection[str] | None = None,
) -> QueryResult | None:
    """Async variant of :func:`route_negation_query`."""
    if not field_map:
        return None
    filters = extract_keyword_filters(query, field_map, op="NOT ILIKE")
    if known_fields is not None:
        filters = [f for f in filters if f["field"] in known_fields]
    if not filters:
        return None
    where_clause = " AND ".join(_filter_to_sql_predicate(f) for f in filters)
    return await relata_client.aquery(f"SELECT * FROM {type} WHERE {where_clause}", purpose=purpose)


#: Matches the conjunction word actually used in a BOOLEAN-shaped query, so
#: :func:`route_boolean_query` joins its predicates with the same operator
#: the question asked for ("or" -> SQL ``OR``, everything else -> ``AND``).
_BOOLEAN_CONJUNCTION_WORD_RE = re.compile(r"\b(and|or)\b", re.IGNORECASE)


def route_boolean_query(
    relata_client: RelataClient,
    query: str,
    type: str,  # noqa: A002
    *,
    purpose: str | None = None,
    field_map: Mapping[str, str] | None = None,
    known_fields: Collection[str] | None = None,
) -> QueryResult | None:
    """Route a boolean/set-operation-shaped query (#4535, e.g. "Members of
    SIMI AND LeT") to governed SQL predicates joined by the query's own
    ``AND``/``OR`` instead of a semantic-similarity guess.

    Requires ``field_map`` and at least two resolved keyword matches (a
    boolean shape is meaningless with only one) — anything less returns
    ``None`` and the caller falls back to retrieval.
    """
    if not field_map:
        return None
    filters = extract_keyword_filters(query, field_map, op="=", dedupe_fields=False)
    if known_fields is not None:
        filters = [f for f in filters if f["field"] in known_fields]
    if len(filters) < 2:
        return None
    conj = _BOOLEAN_CONJUNCTION_WORD_RE.search(query)
    joiner = " OR " if (conj and conj.group(1).lower() == "or") else " AND "
    where_clause = joiner.join(_filter_to_sql_predicate(f) for f in filters)
    return relata_client.query(f"SELECT * FROM {type} WHERE {where_clause}", purpose=purpose)


async def aroute_boolean_query(
    relata_client: RelataClient,
    query: str,
    type: str,  # noqa: A002
    *,
    purpose: str | None = None,
    field_map: Mapping[str, str] | None = None,
    known_fields: Collection[str] | None = None,
) -> QueryResult | None:
    """Async variant of :func:`route_boolean_query`."""
    if not field_map:
        return None
    filters = extract_keyword_filters(query, field_map, op="=", dedupe_fields=False)
    if known_fields is not None:
        filters = [f for f in filters if f["field"] in known_fields]
    if len(filters) < 2:
        return None
    conj = _BOOLEAN_CONJUNCTION_WORD_RE.search(query)
    joiner = " OR " if (conj and conj.group(1).lower() == "or") else " AND "
    where_clause = joiner.join(_filter_to_sql_predicate(f) for f in filters)
    return await relata_client.aquery(f"SELECT * FROM {type} WHERE {where_clause}", purpose=purpose)


#: Explicit "top N" / "first N" count, when given.
_RANKING_N_RE = re.compile(r"\btop\s+(\d+)\b|\bfirst\s+(\d+)\b", re.IGNORECASE)
#: "highest"/"most" default to descending; no explicit N given.
_RANKING_ASCENDING_RE = re.compile(r"\blowest\b|\bleast\b", re.IGNORECASE)
#: Default LIMIT for a ranking query with no explicit "top N"/"first N".
DEFAULT_RANKING_LIMIT = 10


def _ranking_limit(query: str) -> int:
    m = _RANKING_N_RE.search(query)
    if not m:
        return DEFAULT_RANKING_LIMIT
    return int(m.group(1) or m.group(2))


def route_ranking_query(
    relata_client: RelataClient,
    query: str,
    type: str,  # noqa: A002
    *,
    purpose: str | None = None,
    field_map: Mapping[str, str] | None = None,
    known_fields: Collection[str] | None = None,
) -> QueryResult | None:
    """Route a ranking/top-N-shaped query (#4535, e.g. "Top 5 most active
    members") to a governed ``ORDER BY ... LIMIT N`` instead of retrieval
    (which has no notion of "order by").

    Requires ``field_map`` naming the sortable attribute (e.g.
    ``{"active": "activity_count"}``) — returns ``None`` (fall back to
    retrieval) when ``field_map`` is absent, resolves to no field, or (with
    ``known_fields`` given) the resolved field isn't schema-known.
    """
    if not field_map:
        return None
    order_field = _first_matching_field(query, field_map)
    if order_field is None:
        return None
    if known_fields is not None and order_field not in known_fields:
        return None
    limit = _ranking_limit(query)
    direction = "ASC" if _RANKING_ASCENDING_RE.search(query) else "DESC"
    sql = f"SELECT * FROM {type} ORDER BY {order_field} {direction} LIMIT {limit}"
    return relata_client.query(sql, purpose=purpose)


async def aroute_ranking_query(
    relata_client: RelataClient,
    query: str,
    type: str,  # noqa: A002
    *,
    purpose: str | None = None,
    field_map: Mapping[str, str] | None = None,
    known_fields: Collection[str] | None = None,
) -> QueryResult | None:
    """Async variant of :func:`route_ranking_query`."""
    if not field_map:
        return None
    order_field = _first_matching_field(query, field_map)
    if order_field is None:
        return None
    if known_fields is not None and order_field not in known_fields:
        return None
    limit = _ranking_limit(query)
    direction = "ASC" if _RANKING_ASCENDING_RE.search(query) else "DESC"
    sql = f"SELECT * FROM {type} ORDER BY {order_field} {direction} LIMIT {limit}"
    return await relata_client.aquery(sql, purpose=purpose)


# ---------------------------------------------------------------------------
# Entry point — composes all nine ahead of the first /rag/query call
# ---------------------------------------------------------------------------


def smart_rag_query(
    rag_client: RagClient,
    query: str,
    type: str,  # noqa: A002 — matches RagClient.query's wire-facing param name
    *,
    purpose: str | None = None,
    hypothesis_fn: HypothesisGenerator | None = None,
    max_workers: int = 4,
    content_safety_patterns: Mapping[str, str | re.Pattern[str]] | None = None,
    attribute_field_map: Mapping[str, str] | None = None,
    attribute_known_fields: Collection[str] | None = None,
    structured_field_map: Mapping[str, str] | None = None,
    structured_known_fields: Collection[str] | None = None,
    **rag_kwargs: Any,
) -> RagQueryResponse:
    """Run ``query`` through the content-safety gate, structured-attribute
    routing, decomposition, per-sub-query shape dispatch, and optional HyDE,
    then issue the (possibly several, parallel) ``/rag/query`` calls via
    ``rag_client`` and RRF-merge the results.

    Args:
        rag_client: A :class:`~relata.rag.RagClient` (sync).
        query: The raw user question, possibly multi-part.
        type: Object type to search (e.g. ``"DocumentChunk"``).
        purpose: PURPOSE token, passed through to every underlying call.
        hypothesis_fn: When given, each sub-query's search text is HyDE-
            expanded via :func:`expand_query_hyde` before the call (skipped
            per sub-query when :func:`is_numeric_intent` trips).
        max_workers: Thread-pool size for the parallel sub-query fan-out
            when decomposition produces more than one sub-query.
        content_safety_patterns: Category-label -> pattern mapping for
            :func:`check_content_safety` (#4536). ``None`` (the default)
            disables the gate entirely — pass
            :data:`DANGEROUS_CONTENT_PATTERNS` (or a deployment's own
            mapping) to opt in.
        attribute_field_map: Overrides/extends
            :data:`ATTRIBUTE_FIELD_KEYWORDS` for #4536's structured-
            attribute-filter routing.
        attribute_known_fields: When given, restricts attribute-filter
            routing to fields actually present on ``type``'s schema (#4536)
            — an extracted filter naming any other field is dropped, and if
            that empties the filter set, the query falls back to retrieval
            with :attr:`~relata.models.RagQueryResponse.low_confidence` set.
        structured_field_map: Keyword -> canonical field name for #4535's
            aggregation/negation/boolean/ranking routing (deployment-
            specific vocabulary — e.g. ``{"simi": "organization", "custody":
            "status", "active": "activity_count"}``). AGGREGATION works
            without it (a bare ``COUNT(*)`` needs no vocabulary);
            NEGATION/BOOLEAN/RANKING need it to build any predicate at all.
        structured_known_fields: When given, restricts #4535's routing to
            fields actually present on ``type``'s schema, the same
            fallback-not-guess contract as ``attribute_known_fields``.
        **rag_kwargs: Forwarded to :meth:`RagClient.query` for every
            sub-query call (``top_k``, ``rerank``, ``search_mode``,
            ``embedding_slot``, ``filters``, ``as_of``, ``expand_window``,
            ``graph_hops``), after :func:`classify_query_shape`'s per-shape
            adjustments are applied on top.

    Returns:
        A single :class:`~relata.models.RagQueryResponse`. ``refused`` is
        set (empty ``hits``) when the content-safety gate matched, before
        any ``/rag/query``/``/query`` call was made. ``sql_result`` is set
        (empty ``hits``) when an attribute-filter/aggregation/negation/
        boolean/ranking-shaped query was routed to SQL instead of
        retrieval. Otherwise this is the (possibly RRF-merged) retrieval
        result, exactly as before #4536/#4535 — with ``low_confidence`` set
        when a SQL-routable shape could not be routed and fell back to
        retrieval.
    """
    refusal = check_content_safety(query, patterns=content_safety_patterns)
    if refusal is not None:
        return RagQueryResponse(hits=[], refused=refusal)

    low_confidence = False
    shape = classify_query_shape(query)
    if shape in SQL_ROUTABLE_SHAPES:
        client = rag_client._client  # noqa: SLF001
        if shape is QueryShape.ATTRIBUTE_FILTER:
            sql_result = route_attribute_filter_query(
                client, query, type,
                purpose=purpose, field_map=attribute_field_map,
                known_fields=attribute_known_fields,
            )
        elif shape is QueryShape.AGGREGATION:
            sql_result = route_aggregation_query(
                client, query, type,
                purpose=purpose, field_map=structured_field_map,
                known_fields=structured_known_fields,
            )
        elif shape is QueryShape.NEGATION:
            sql_result = route_negation_query(
                client, query, type,
                purpose=purpose, field_map=structured_field_map,
                known_fields=structured_known_fields,
            )
        elif shape is QueryShape.BOOLEAN:
            sql_result = route_boolean_query(
                client, query, type,
                purpose=purpose, field_map=structured_field_map,
                known_fields=structured_known_fields,
            )
        else:  # QueryShape.RANKING
            sql_result = route_ranking_query(
                client, query, type,
                purpose=purpose, field_map=structured_field_map,
                known_fields=structured_known_fields,
            )
        if sql_result is not None:
            return RagQueryResponse(hits=[], sql_result=sql_result)
        low_confidence = True

    sub_queries = decompose_query(query)
    n = len(sub_queries)

    def _run(sub_query: str) -> RagQueryResponse:
        sub_shape = classify_query_shape(sub_query)
        call_kwargs = _apply_query_shape(sub_shape, rag_kwargs)
        search_text = sub_query
        if hypothesis_fn is not None:
            search_text = expand_query_hyde(sub_query, hypothesis_fn=hypothesis_fn)
        return rag_client.query(search_text, type, purpose=purpose, **call_kwargs)

    if n == 1:
        response = _run(sub_queries[0])
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, n)) as pool:
            responses = list(pool.map(_run, sub_queries))
        response = rrf_merge(responses, k=rrf_k_for_fanout(n))

    if low_confidence:
        response.low_confidence = True
        response.low_confidence_reason = (
            f"{shape.value}-shaped query could not be routed to SQL "
            f"(no matching canonical field on {type!r}); fell back to retrieval "
            f"(#4536/#4535)"
        )
    return response


async def asmart_rag_query(
    rag_client: AsyncRagClient,
    query: str,
    type: str,  # noqa: A002
    *,
    purpose: str | None = None,
    hypothesis_fn: HypothesisGenerator | None = None,
    content_safety_patterns: Mapping[str, str | re.Pattern[str]] | None = None,
    attribute_field_map: Mapping[str, str] | None = None,
    attribute_known_fields: Collection[str] | None = None,
    structured_field_map: Mapping[str, str] | None = None,
    structured_known_fields: Collection[str] | None = None,
    **rag_kwargs: Any,
) -> RagQueryResponse:
    """Async variant of :func:`smart_rag_query` — issues the sub-query fan-out
    concurrently via ``asyncio.gather`` instead of a thread pool."""
    refusal = check_content_safety(query, patterns=content_safety_patterns)
    if refusal is not None:
        return RagQueryResponse(hits=[], refused=refusal)

    low_confidence = False
    shape = classify_query_shape(query)
    if shape in SQL_ROUTABLE_SHAPES:
        client = rag_client._inner._client  # noqa: SLF001
        if shape is QueryShape.ATTRIBUTE_FILTER:
            sql_result = await aroute_attribute_filter_query(
                client, query, type,
                purpose=purpose, field_map=attribute_field_map,
                known_fields=attribute_known_fields,
            )
        elif shape is QueryShape.AGGREGATION:
            sql_result = await aroute_aggregation_query(
                client, query, type,
                purpose=purpose, field_map=structured_field_map,
                known_fields=structured_known_fields,
            )
        elif shape is QueryShape.NEGATION:
            sql_result = await aroute_negation_query(
                client, query, type,
                purpose=purpose, field_map=structured_field_map,
                known_fields=structured_known_fields,
            )
        elif shape is QueryShape.BOOLEAN:
            sql_result = await aroute_boolean_query(
                client, query, type,
                purpose=purpose, field_map=structured_field_map,
                known_fields=structured_known_fields,
            )
        else:  # QueryShape.RANKING
            sql_result = await aroute_ranking_query(
                client, query, type,
                purpose=purpose, field_map=structured_field_map,
                known_fields=structured_known_fields,
            )
        if sql_result is not None:
            return RagQueryResponse(hits=[], sql_result=sql_result)
        low_confidence = True

    sub_queries = decompose_query(query)
    n = len(sub_queries)

    async def _run(sub_query: str) -> RagQueryResponse:
        sub_shape = classify_query_shape(sub_query)
        call_kwargs = _apply_query_shape(sub_shape, rag_kwargs)
        search_text = sub_query
        if hypothesis_fn is not None:
            search_text = expand_query_hyde(sub_query, hypothesis_fn=hypothesis_fn)
        return await rag_client.query(search_text, type, purpose=purpose, **call_kwargs)

    if n == 1:
        response = await _run(sub_queries[0])
    else:
        responses = list(await asyncio.gather(*(_run(sq) for sq in sub_queries)))
        response = rrf_merge(responses, k=rrf_k_for_fanout(n))

    if low_confidence:
        response.low_confidence = True
        response.low_confidence_reason = (
            f"{shape.value}-shaped query could not be routed to SQL "
            f"(no matching canonical field on {type!r}); fell back to retrieval "
            f"(#4536/#4535)"
        )
    return response
