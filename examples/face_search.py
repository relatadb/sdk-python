"""
Face search example — MATCH_FACE operator.

Demonstrates:
- Using the MATCH_FACE operator for face-embedding similarity search
- Adjusting similarity thresholds for precision vs. recall trade-offs
- Combining face search with identity resolution
- Bi-temporal face search (who was captured in CCTV on a given date?)

The MATCH_FACE operator computes cosine similarity between a probe embedding
and stored embeddings, returning rows where the similarity meets or exceeds
the threshold.  Embeddings are stored as FLOAT32[] (Lance vector column) in
the FaceRecord table.

Operator signature::

    MATCH_FACE(probe_column, candidate_column, threshold FLOAT)

or in a WHERE predicate::

    WHERE MATCH_FACE(face_embedding, stored_embedding, 0.92)
"""

import os

from relata import RelataClient
from relata.exceptions import RelataError


def search_by_face_id(
    relata: RelataClient,
    face_record_id: str,
    threshold: float = 0.92,
    limit: int = 20,
) -> None:
    """Search for persons matching a stored face record.

    Looks up the probe embedding from the FaceRecord table, then runs a
    similarity search against all stored person-face pairs.

    Args:
        relata: Bound ``RelataClient`` instance.
        face_record_id: ID of the FaceRecord to use as the probe.
        threshold: Cosine similarity threshold (0.0–1.0).  Higher values
            are more precise; lower values increase recall.
        limit: Maximum number of matches to return.
    """
    print(f"\n=== Face search: probe={face_record_id!r}, threshold={threshold} ===")

    result = (
        relata.select(
            "p.person_id",
            "p.name",
            "p.dob",
            "p.nationality",
            "fr.face_record_id",
            "fr.captured_at",
            "fr.capture_source",
            "fr.match_score",
        )
        .from_("FaceRecord fr")
        .where(
            f"MATCH_FACE("
            f"  (SELECT embedding FROM FaceRecord WHERE face_record_id = '{face_record_id}'),"
            f"  fr.embedding,"
            f"  {threshold}"
            f")"
        )
        .where(f"fr.face_record_id != '{face_record_id}'")
        .order_by("fr.match_score DESC")
        .limit(limit)
        .execute()
    )

    print(f"Matches found: {result.row_count}  ({result.elapsed_ms} ms)")
    print()

    for rank, row in enumerate(result, start=1):
        score = row.get("match_score", "—")
        name = row.get("name", "UNKNOWN")
        dob = row.get("dob", "—")
        nat = row.get("nationality", "—")
        captured_at = row.get("captured_at", "—")
        source = row.get("capture_source", "—")
        person_id = row.get("person_id", "UNLINKED")

        print(f"  #{rank:02d}  score={score:.4f}  [{person_id}]")
        print(f"        Name       : {name}")
        print(f"        DOB        : {dob}  Nationality: {nat}")
        print(f"        Captured at: {captured_at}  Source: {source}")
        print()


def search_by_face_embedding_cctv(
    relata: RelataClient,
    cctv_face_record_id: str,
    location: str,
    start_date: str,
    end_date: str,
    threshold: float = 0.88,
) -> None:
    """Search CCTV face captures for a specific person within a date range.

    This query combines:
    - MATCH_FACE for similarity filtering
    - AS OF for bi-temporal point-in-time data
    - WHERE predicates for location and time range
    - WITH PROVENANCE for chain-of-custody

    Args:
        relata: Bound ``RelataClient`` instance.
        cctv_face_record_id: Face record ID from the CCTV feed to use as probe.
        location: Camera location identifier (e.g. ``"GATE-7-IGAI"``).
        start_date: ISO 8601 start timestamp for capture range.
        end_date: ISO 8601 end timestamp for capture range.
        threshold: Similarity threshold (default 0.88 for CCTV-quality images).
    """
    print(f"\n=== CCTV face search: location={location!r} [{start_date} → {end_date}] ===")

    result = relata.query(
        f"""
        SELECT
            cf.frame_id,
            cf.camera_id,
            cf.captured_at,
            cf.match_score,
            cf.face_record_id,
            p.person_id,
            p.name,
            p.nationality,
            p.risk_score
        FROM CctvFrame cf
        LEFT JOIN Person p ON cf.linked_person_id = p.person_id
        AS OF '{end_date}'
        WHERE MATCH_FACE(
            (SELECT embedding FROM FaceRecord WHERE face_record_id = '{cctv_face_record_id}'),
            cf.face_embedding,
            {threshold}
        )
        AND cf.camera_id LIKE '{location}%'
        AND cf.captured_at BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY cf.match_score DESC, cf.captured_at DESC
        LIMIT 50
        WITH PROVENANCE
        """
    )

    print(f"CCTV matches: {result.row_count}  ({result.elapsed_ms} ms)")
    print()

    for row in result:
        score = row.get("match_score", "—")
        cam = row.get("camera_id", "—")
        ts = row.get("captured_at", "—")
        name = row.get("name") or "UNIDENTIFIED"
        risk = row.get("risk_score", "—")
        print(f"  [{ts}]  camera={cam}  score={score:.4f}  {name}  risk={risk}")
        # Provenance columns (added by WITH PROVENANCE)
        if row.get("source_id"):
            print(
                f"    source={row.get('source_id')}  "
                f"collector={row.get('collector')}  "
                f"confidence={row.get('confidence')}"
            )


def multi_threshold_search(
    relata: RelataClient,
    probe_face_id: str,
) -> None:
    """Run the same face search at three thresholds to illustrate precision/recall.

    Args:
        relata: Bound ``RelataClient`` instance.
        probe_face_id: Face record ID to use as the search probe.
    """
    print(f"\n=== Multi-threshold comparison: probe={probe_face_id!r} ===")
    print(f"{'Threshold':>12}  {'Matches':>8}  {'Elapsed':>8}")
    print("-" * 35)

    for threshold in (0.95, 0.92, 0.88, 0.82):
        result = (
            relata.select("face_record_id", "match_score")
            .match_face("probe_embedding", "stored_embedding", threshold)
            .from_("FaceRecord")
            .where(f"face_record_id != '{probe_face_id}'")
            .limit(500)
            .execute()
        )
        print(f"{threshold:>12.2f}  {result.row_count:>8}  {result.elapsed_ms:>6} ms")


if __name__ == "__main__":
    RELATA_URL = os.getenv("RELATA_URL", "http://localhost:9090")
    RELATA_TOKEN = os.getenv("RELATA_TOKEN")

    PROBE_FACE_ID = os.getenv("PROBE_FACE_ID", "face-001")
    CCTV_FACE_ID = os.getenv("CCTV_FACE_ID", "cctv-face-007")

    with RelataClient(
        RELATA_URL,
        bearer_token=RELATA_TOKEN,
        purpose="analytics",
        timeout=60.0,
    ) as relata:
        try:
            # 1. Basic face-to-person search
            search_by_face_id(relata, face_record_id=PROBE_FACE_ID, threshold=0.92)

            # 2. CCTV search at a specific location and time window
            search_by_face_embedding_cctv(
                relata,
                cctv_face_record_id=CCTV_FACE_ID,
                location="GATE-7-IGAI",
                start_date="2025-03-01T00:00:00Z",
                end_date="2025-03-07T23:59:59Z",
                threshold=0.88,
            )

            # 3. Compare thresholds
            multi_threshold_search(relata, probe_face_id=PROBE_FACE_ID)

        except RelataError as exc:
            print(f"Error: {exc}")
            raise SystemExit(1) from exc
