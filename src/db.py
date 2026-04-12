"""
SQLite database layer for Discharge Summary POC.
Simulates Azure Cosmos DB audit logging using a local SQLite database.

Tables:
    - audit_log: Tracks every pipeline run end-to-end
    - failed_documents_tracking: Records documents that failed processing
"""

import sqlite3
from datetime import datetime

from src.config import DATABASE_DIR

# ---------------------------------------------------------------------------
# Database file path
# ---------------------------------------------------------------------------
DB_PATH = DATABASE_DIR / "audit.db"


def _utc_now() -> str:
    """Return current UTC timestamp as ISO 8601 string."""
    return datetime.utcnow().isoformat() + "Z"


def _get_connection() -> sqlite3.Connection:
    """Create a new SQLite connection with row factory set to sqlite3.Row."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


# ============================================================================
# TABLE CREATION
# ============================================================================

_CREATE_AUDIT_LOG = """
CREATE TABLE IF NOT EXISTS audit_log (
    -- Identifiers
    audit_id                    TEXT PRIMARY KEY,
    pipeline_run_id             TEXT,
    patient_id                  TEXT,
    unique_doc_num              TEXT,
    service_date                TEXT,
    original_file_name          TEXT,
    blob_uri                    TEXT,
    file_size_kb                REAL,
    upload_timestamp            TEXT,
    event_id                    TEXT,

    -- Ingestion
    ingestion_status            TEXT,
    event_triggered_at          TEXT,
    processing_mode             TEXT,

    -- Pre-Validation
    file_name_validation_status TEXT,
    file_size_validation_status TEXT,
    validation_failure_reason   TEXT,
    is_eligible_for_processing  INTEGER,

    -- Metadata Extraction
    metadata_extraction_status  TEXT,
    metadata_json_stored        INTEGER,
    metadata_extraction_time_ms INTEGER,
    metadata_error_reason       TEXT,

    -- Raw Text Extraction
    vision_model_used           TEXT,
    raw_text_extraction_status  TEXT,
    raw_text_blob_uri           TEXT,
    raw_text_token_count        INTEGER,
    raw_text_character_count    INTEGER,
    raw_text_quality_score      REAL,
    vision_prompt_version       TEXT,
    raw_extraction_time_ms      INTEGER,

    -- Ontology Extraction
    ontology_yaml_version       TEXT,
    reasoning_model_used        TEXT,
    structured_fields_total     INTEGER,
    structured_fields_filled    INTEGER,
    reasoning_prompt_version    TEXT,
    ontology_extraction_time_ms INTEGER,

    -- Validation Layer 1
    structure_validation_status TEXT,
    filled_to_total_ratio       REAL,
    acceptance_threshold        REAL,
    missing_critical_sections   TEXT,

    -- Validation Layer 2
    hallucination_check_status  TEXT,
    non_traceable_fields        TEXT,
    fallback_triggered          INTEGER,
    reprocessing_attempt_count  INTEGER,
    hallucination_check_time_ms INTEGER,

    -- Final Output
    final_json_stored           INTEGER,
    final_summary_stored        INTEGER,
    output_version              TEXT,
    final_cosmos_container      TEXT,
    finalized_at                TEXT,

    -- Replication
    replication_api_status      TEXT,
    replication_attempt_count   INTEGER,
    replication_error_reason    TEXT,
    replicated_at               TEXT,

    -- Overall Status
    overall_pipeline_status     TEXT,
    failure_stage               TEXT,
    is_document_unrefined       INTEGER,
    retention_expiry_date       TEXT,
    failed_blob_uri             TEXT,
    reuploaded_from             TEXT,
    created_at                  TEXT,
    last_updated_at             TEXT
);
"""

_CREATE_FAILED_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS failed_documents_tracking (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    audit_id            TEXT,
    patient_id          TEXT,
    unique_doc_num      TEXT,
    original_file_name  TEXT,
    failure_stage       TEXT,
    failure_reason      TEXT,
    failed_blob_uri     TEXT,
    failed_at           TEXT
);
"""


# ============================================================================
# PUBLIC FUNCTIONS
# ============================================================================

def init_db() -> None:
    """
    Create both tables if they don't exist. Called on application startup.
    Also ensures the database directory exists.
    """
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    with _get_connection() as conn:
        conn.execute(_CREATE_AUDIT_LOG)
        conn.execute(_CREATE_FAILED_DOCUMENTS)
        conn.commit()


def create_audit_record(
    audit_id: str,
    pipeline_run_id: str,
    patient_id: str,
    unique_doc_num: str,
    service_date: str,
    original_file_name: str,
    blob_uri: str,
    file_size_kb: float,
    upload_timestamp: str,
    event_id: str,
) -> None:
    """
    Insert a new audit_log row with initial identifiers.
    Sets overall_pipeline_status to 'IN_PROGRESS' and timestamps to now UTC.
    """
    now = _utc_now()
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO audit_log (
                audit_id, pipeline_run_id, patient_id, unique_doc_num,
                service_date, original_file_name, blob_uri, file_size_kb,
                upload_timestamp, event_id,
                overall_pipeline_status, created_at, last_updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                audit_id, pipeline_run_id, patient_id, unique_doc_num,
                service_date, original_file_name, blob_uri, file_size_kb,
                upload_timestamp, event_id,
                "IN_PROGRESS", now, now,
            ),
        )
        conn.commit()


def update_audit_record(audit_id: str, **kwargs) -> None:
    """
    Update any set of columns on an existing audit_log row.
    Always sets last_updated_at to the current UTC timestamp.

    Usage:
        update_audit_record("AUD-001", ingestion_status="COMPLETED",
                            vision_model_used="gemini-1.5-pro")
    """
    kwargs["last_updated_at"] = _utc_now()

    columns = ", ".join(f"{col} = ?" for col in kwargs)
    values = list(kwargs.values())
    values.append(audit_id)

    with _get_connection() as conn:
        conn.execute(
            f"UPDATE audit_log SET {columns} WHERE audit_id = ?",
            values,
        )
        conn.commit()


def log_failed_document(
    audit_id: str,
    patient_id: str,
    unique_doc_num: str,
    original_file_name: str,
    failure_stage: str,
    failure_reason: str,
    failed_blob_uri: str,
) -> None:
    """Insert a record into the failed_documents_tracking table."""
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO failed_documents_tracking (
                audit_id, patient_id, unique_doc_num, original_file_name,
                failure_stage, failure_reason, failed_blob_uri, failed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                audit_id, patient_id, unique_doc_num, original_file_name,
                failure_stage, failure_reason, failed_blob_uri, _utc_now(),
            ),
        )
        conn.commit()


def get_audit_record(audit_id: str) -> dict | None:
    """
    Retrieve a single audit_log row by audit_id.
    Returns the full row as a dict, or None if not found.
    """
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM audit_log WHERE audit_id = ?", (audit_id,)
        ).fetchone()
    return dict(row) if row else None


def get_all_failed_documents() -> list[dict]:
    """Return all rows from failed_documents_tracking as a list of dicts."""
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM failed_documents_tracking ORDER BY failed_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def get_pipeline_summary() -> dict:
    """
    Return a summary dict with counts of pipeline runs by status.

    Returns:
        {
            "total": int,
            "completed": int,
            "failed": int,
            "in_progress": int
        }
    """
    with _get_connection() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM audit_log"
        ).fetchone()[0]

        completed = conn.execute(
            "SELECT COUNT(*) FROM audit_log WHERE overall_pipeline_status = 'COMPLETED'"
        ).fetchone()[0]

        failed = conn.execute(
            "SELECT COUNT(*) FROM audit_log WHERE overall_pipeline_status = 'FAILED'"
        ).fetchone()[0]

        in_progress = conn.execute(
            "SELECT COUNT(*) FROM audit_log WHERE overall_pipeline_status = 'IN_PROGRESS'"
        ).fetchone()[0]

    return {
        "total": total,
        "completed": completed,
        "failed": failed,
        "in_progress": in_progress,
    }
