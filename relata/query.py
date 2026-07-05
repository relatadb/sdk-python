"""
Fluent SQL query builder for the Relata engine.

The ``QueryBuilder`` constructs extended-SQL strings that the Relata engine
understands, including bi-temporal ``AS OF``, ``WITH PROVENANCE``, graph
operators, and face/identity lookups.

Typical usage::

    result = (
        relata_client.select("Person")
        .purpose("analytics")
        .where("name LIKE 'Ahmed%'")
        .where("nationality = 'IN'")
        .as_of("2025-01-01")
        .with_provenance()
        .limit(20)
        .execute()
    )

All methods return ``self`` so calls can be chained.  Call :meth:`execute` or
:meth:`sql` to materialise the query.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from relata.client import RelataClient
    from relata.models import QueryResult


class QueryBuilder:
    """Fluent builder that assembles a Relata SQL query.

    Do not instantiate directly — use :meth:`relata.client.RelataClient.select`
    or the module-level :func:`select` helper instead.

    The builder tracks individual clauses and assembles them into a single SQL
    string when :meth:`sql` or :meth:`execute` is called.  Because the builder
    produces standard SQL extended with Relata-specific keywords, you can
    inspect the generated SQL before executing it.

    Supported Relata SQL extensions:
    - ``AS OF 'timestamp'`` — point-in-time bi-temporal query
    - ``WITH PROVENANCE`` — include PROV-O provenance columns
    - ``PATHS_BETWEEN(src, dst, max_hops => N)`` — graph traversal
    - ``MATCH_FACE(face_col, candidate_col, threshold)`` — face similarity
    - ``LOOKUP_IDENTITY('value')`` — universal identity resolution
    - ``HYBRID_SCORE(query_text)`` — BM25 + vector hybrid search
    """

    def __init__(self, *, client: "RelataClient") -> None:
        self._client = client
        self._select_cols: list[str] = []
        self._from_table: str = ""
        self._where_clauses: list[str] = []
        self._as_of_ts: str | None = None
        self._with_provenance: bool = False
        self._limit_val: int | None = None
        self._offset_val: int | None = None
        self._order_by_cols: list[str] = []
        self._purpose_override: str | None = None
        # Graph / advanced operator state
        self._paths_between: tuple[str, str, int] | None = None
        self._match_face: tuple[str, str, float] | None = None
        self._lookup_identity: str | None = None
        self._hybrid_score: str | None = None
        self._raw_sql: str | None = None

    # ------------------------------------------------------------------
    # Core clauses
    # ------------------------------------------------------------------

    def select(self, *columns_or_table: str) -> "QueryBuilder":
        """Set the SELECT target or shorthand for ``SELECT * FROM <table>``.

        When called with a single argument that looks like a table name (no
        spaces, no commas, no ``*``), it is treated as ``SELECT * FROM <table>``.
        Pass explicit column names to project only specific fields::

            # SELECT * FROM Person
            builder.select("Person")

            # SELECT name, dob, nationality FROM Person
            builder.select("name", "dob", "nationality").from_("Person")

        Args:
            *columns_or_table: Table name (shorthand) or column list.

        Returns:
            ``self``
        """
        if len(columns_or_table) == 1 and _looks_like_table(columns_or_table[0]):
            # Shorthand: select("Person") → SELECT * FROM Person
            self._select_cols = ["*"]
            self._from_table = columns_or_table[0]
        else:
            self._select_cols = list(columns_or_table) if columns_or_table else ["*"]
        return self

    def from_(self, table: str) -> "QueryBuilder":
        """Set the FROM table.

        Args:
            table: Table / object-type name.

        Returns:
            ``self``
        """
        self._from_table = table
        return self

    def where(self, condition: str) -> "QueryBuilder":
        """Add a WHERE condition.

        Multiple calls are joined with ``AND``::

            builder.where("age > 18").where("nationality = 'IN'")
            # → WHERE age > 18 AND nationality = 'IN'

        Args:
            condition: SQL predicate expression.

        Returns:
            ``self``
        """
        self._where_clauses.append(condition)
        return self

    def limit(self, n: int) -> "QueryBuilder":
        """Set a LIMIT on result rows.

        Args:
            n: Maximum number of rows to return.  Must be a positive integer.

        Returns:
            ``self``
        """
        if n < 1:
            raise ValueError(f"limit must be a positive integer, got {n}")
        self._limit_val = n
        return self

    def offset(self, n: int) -> "QueryBuilder":
        """Set an OFFSET for result pagination.

        Args:
            n: Number of rows to skip.  Must be non-negative.

        Returns:
            ``self``
        """
        if n < 0:
            raise ValueError(f"offset must be non-negative, got {n}")
        self._offset_val = n
        return self

    def order_by(self, *columns: str) -> "QueryBuilder":
        """Add ORDER BY columns.

        Args:
            *columns: Column expressions, e.g. ``"valid_from DESC"``,
                ``"name ASC"``.

        Returns:
            ``self``
        """
        self._order_by_cols.extend(columns)
        return self

    # ------------------------------------------------------------------
    # Relata extensions
    # ------------------------------------------------------------------

    def as_of(self, timestamp: str) -> "QueryBuilder":
        """Apply a bi-temporal ``AS OF`` clause.

        Restricts the query to rows whose bi-temporal validity interval
        contains the given point in time.  The timestamp is parsed by the
        engine as RFC 3339 (e.g. ``"2025-01-01T00:00:00Z"``).  A bare date
        (``"2025-01-01"``) is accepted as ``2025-01-01T00:00:00Z``.

        Args:
            timestamp: RFC 3339 timestamp string or bare date.

        Returns:
            ``self``

        Examples::

            # All Person records that were valid on 1 Jan 2025
            builder.select("Person").as_of("2025-01-01")
        """
        self._as_of_ts = timestamp
        return self

    def with_provenance(self) -> "QueryBuilder":
        """Append ``WITH PROVENANCE`` to include PROV-O columns.

        When set, each result row includes additional columns:
        ``source_id``, ``collector``, ``method``, ``confidence``,
        ``observed_at``, ``recorded_at``.

        Returns:
            ``self``
        """
        self._with_provenance = True
        return self

    def purpose(self, purpose: str) -> "QueryBuilder":
        """Override the query purpose for this specific query.

        Overrides the client-level default set in
        :class:`~relata.client.RelataClient`.

        Args:
            purpose: Purpose string registered in the tenant's
                ``PurposeRegistry``, e.g. ``"analytics"``, ``"audit"``.

        Returns:
            ``self``
        """
        self._purpose_override = purpose
        return self

    # ------------------------------------------------------------------
    # Graph operators
    # ------------------------------------------------------------------

    def paths_between(
        self, src: str, dst: str, max_hops: int = 4
    ) -> "QueryBuilder":
        """Use the ``PATHS_BETWEEN`` graph operator.

        Replaces the standard ``FROM`` clause with a graph traversal that
        finds all paths between two entity IDs up to ``max_hops`` edges.

        Args:
            src: Source entity ID string.
            dst: Destination entity ID string.
            max_hops: Maximum number of graph hops (default 4).  Higher
                values increase cost significantly; keep at ≤ 6 for
                interactive queries.

        Returns:
            ``self``

        Examples::

            result = (
                client.select("*")
                .paths_between("person-123", "org-456", max_hops=3)
                .purpose("analytics")
                .execute()
            )
        """
        self._paths_between = (src, dst, max_hops)
        return self

    def match_face(
        self, face_column: str, candidate_column: str, threshold: float = 0.92
    ) -> "QueryBuilder":
        """Add a ``MATCH_FACE`` filter for face-embedding similarity search.

        Args:
            face_column: Column containing the probe face embedding.
            candidate_column: Column containing candidate face embeddings.
            threshold: Cosine similarity threshold, 0.0–1.0 (default 0.92).

        Returns:
            ``self``

        Examples::

            result = (
                client.select("FaceRecord")
                .match_face("probe_embedding", "stored_embedding", threshold=0.92)
                .purpose("analytics")
                .limit(20)
                .execute()
            )
        """
        self._match_face = (face_column, candidate_column, threshold)
        return self

    def lookup_identity(self, value: str) -> "QueryBuilder":
        """Resolve any identifier via the ``LOOKUP_IDENTITY`` operator.

        Queries the ``IdentityIndex`` using a universal lookup across all
        canonical identifier types (phone, email, IP, AADHAAR, passport, etc.)

        Args:
            value: Any identifier string.  SmartIngest will detect the type
                automatically.

        Returns:
            ``self``

        Examples::

            result = (
                client.select("*")
                .lookup_identity("+919876543210")
                .purpose("analytics")
                .execute()
            )
        """
        self._lookup_identity = value
        return self

    def hybrid_score(self, query_text: str) -> "QueryBuilder":
        """Add a ``HYBRID_SCORE`` (BM25 + vector) ranking clause.

        Args:
            query_text: Free-text query for hybrid retrieval.

        Returns:
            ``self``
        """
        self._hybrid_score = query_text
        return self

    # ------------------------------------------------------------------
    # Raw SQL escape hatch
    # ------------------------------------------------------------------

    def raw(self, sql: str) -> "QueryBuilder":
        """Set a completely raw SQL string, bypassing the builder.

        Use this escape hatch when the builder does not yet support a
        particular Relata SQL extension.  Calling :meth:`execute` on a builder
        with a raw SQL string sends that string verbatim to the server.

        Args:
            sql: Complete SQL statement.

        Returns:
            ``self``
        """
        self._raw_sql = sql
        return self

    # ------------------------------------------------------------------
    # SQL assembly
    # ------------------------------------------------------------------

    def sql(self) -> str:
        """Assemble and return the SQL string without executing it.

        Returns:
            The generated SQL string.

        Raises:
            ValueError: If the builder state is incomplete (no table set).
        """
        if self._raw_sql is not None:
            return self._raw_sql

        parts: list[str] = []

        # SELECT
        cols = ", ".join(self._select_cols) if self._select_cols else "*"
        parts.append(f"SELECT {cols}")

        # HYBRID_SCORE wraps the column list
        if self._hybrid_score:
            parts.append(f", HYBRID_SCORE({_quote(self._hybrid_score)}) AS score")

        # FROM — graph operators override the standard FROM
        if self._paths_between:
            src, dst, hops = self._paths_between
            parts.append(
                f"FROM PATHS_BETWEEN({_quote(src)}, {_quote(dst)}, max_hops => {hops})"
            )
        elif self._lookup_identity:
            parts.append(f"FROM LOOKUP_IDENTITY({_quote(self._lookup_identity)})")
        else:
            if not self._from_table:
                raise ValueError(
                    "No table specified.  Call .from_('TableName') or use the "
                    ".select('TableName') shorthand."
                )
            parts.append(f"FROM {self._from_table}")

        # AS OF
        if self._as_of_ts:
            parts.append(f"AS OF '{self._as_of_ts}'")

        # MATCH_FACE as a WHERE predicate
        where_clauses = list(self._where_clauses)
        if self._match_face:
            fc, cc, th = self._match_face
            where_clauses.append(f"MATCH_FACE({fc}, {cc}, {th})")

        # WHERE
        if where_clauses:
            parts.append("WHERE " + " AND ".join(f"({c})" for c in where_clauses))

        # ORDER BY
        if self._order_by_cols:
            parts.append("ORDER BY " + ", ".join(self._order_by_cols))

        # LIMIT / OFFSET
        if self._limit_val is not None:
            parts.append(f"LIMIT {self._limit_val}")
        if self._offset_val is not None:
            parts.append(f"OFFSET {self._offset_val}")

        # WITH PROVENANCE must appear at the end of the statement
        if self._with_provenance:
            parts.append("WITH PROVENANCE")

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self) -> "QueryResult":
        """Execute the built query (synchronous).

        Returns:
            :class:`~relata.models.QueryResult`

        Raises:
            :class:`~relata.exceptions.PurposeError`: No purpose set.
            :class:`~relata.exceptions.QuotaError`: Quota exhausted.
            :class:`~relata.exceptions.AuthError`: Invalid/missing token.
            :class:`~relata.exceptions.ConnectionError`: Server unreachable.
            :class:`~relata.exceptions.RelataError`: Any other server error.
        """
        return self._client.query(self.sql(), purpose=self._purpose_override)

    async def aexecute(self) -> "QueryResult":
        """Execute the built query (asynchronous).

        Returns:
            :class:`~relata.models.QueryResult`
        """
        return await self._client.aquery(self.sql(), purpose=self._purpose_override)

    # ------------------------------------------------------------------
    # Repr / debug
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        try:
            sql_preview = self.sql()
        except ValueError:
            sql_preview = "<incomplete>"
        return f"QueryBuilder(sql={sql_preview!r})"


# ---------------------------------------------------------------------------
# Module-level convenience constructor
# ---------------------------------------------------------------------------


def select(*columns_or_table: str) -> QueryBuilder:
    """Create a ``QueryBuilder`` without a bound client.

    Useful for constructing queries before you have a client instance, or for
    inspecting the generated SQL in tests::

        from relata.query import select

        sql = (
            select("Person")
            .where("name LIKE 'Ahmed%'")
            .as_of("2025-01-01")
            .with_provenance()
            .limit(20)
            .sql()
        )
        print(sql)

    To execute the query, bind a client::

        builder = select("Person").where("status = 'wanted'").limit(50)
        with RelataClient("http://localhost:8080", purpose="analytics") as client:
            result = builder.purpose("analytics").execute()

    .. note::
        Calling :meth:`~QueryBuilder.execute` on an unbound builder raises
        ``AttributeError`` because there is no client to send the request.
        Use :meth:`~RelataClient.select` on a client instance for a fully bound
        builder.

    Args:
        *columns_or_table: Table name or column list (see
            :meth:`QueryBuilder.select`).

    Returns:
        An unbound :class:`QueryBuilder`.
    """
    # We pass a sentinel client=None here; execute() will fail informatively
    # if called on an unbound builder.
    builder: QueryBuilder = object.__new__(QueryBuilder)
    QueryBuilder.__init__(builder, client=None)  # type: ignore[arg-type]
    return builder.select(*columns_or_table)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _looks_like_table(s: str) -> bool:
    """Return True when a string looks like a table/type name rather than a column list."""
    return " " not in s and "," not in s and "*" not in s and "(" not in s


def _quote(s: str) -> str:
    """Wrap a string in single quotes, escaping internal single quotes."""
    return "'" + s.replace("'", "''") + "'"
