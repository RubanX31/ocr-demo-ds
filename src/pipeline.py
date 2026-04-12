"""
Main pipeline orchestrator for Discharge Summary POC.

Wires together all processing modules in the correct sequence:
    Step 0: Setup (generate IDs, create audit record)
    Step 1: Pre-validation (filename + file size)
    Step 2: Vision LLM text extraction (PDF → raw text)
    Step 3: Ontology extraction (raw text → structured JSON)
    Step 4: Structure validation (field-fill ratio check)
    Step 5: Hallucination check (traceability verification)
    Step 6: Final storage (persist JSON + summary)
    Step 7: API replication (call mock client API)
    Final : Move PDF to processed folder

On any failure at Steps 1–5 the PDF is moved to ``failed/`` and the
pipeline returns immediately with a failure result.
"""

import json
import logging
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path

import requests

from src.config import (
    FAILED_DIR,
    PROCESSED_DIR,
    TEXT_DUMPS_DIR,
    MOCK_API_PORT,
)
from src import db
from src.validator import run_pre_validation
from src.vision_extractor import extract_raw_text
from src.ontology_extractor import extract_ontology_fields
from src.validation_layer1 import run_structure_validation, format_validation_report
from src.validation_layer2 import run_hallucination_check

logger = logging.getLogger(__name__)


# ============================================================================
# 1. ID generation
# ============================================================================

def generate_ids() -> tuple:
    """
    Generate unique identifiers for a pipeline run.

    Returns:
        Tuple of (audit_id, pipeline_run_id), both UUID4 strings.
    """
    audit_id = str(uuid.uuid4())
    pipeline_run_id = str(uuid.uuid4())
    return audit_id, pipeline_run_id


# ============================================================================
# 2. File movement helpers
# ============================================================================

def move_to_failed(
    pdf_path: Path,
    audit_id: str,
    patient_id: str = "",
    unique_doc_num: str = "",
    failure_stage: str = "",
    failure_reason: str = "",
) -> Path:
    """
    Move a PDF from its current location to the ``failed/`` directory.

    Also updates the audit log with the new path and records the failure
    in the ``failed_documents_tracking`` table.

    Args:
        pdf_path:       Current absolute path to the PDF.
        audit_id:       Audit record ID.
        patient_id:     Patient ID (may be empty if pre-validation failed).
        unique_doc_num: Document number (may be empty).
        failure_stage:  Pipeline stage where the failure occurred.
        failure_reason: Human-readable failure reason.

    Returns:
        New path of the PDF in the ``failed/`` directory.
    """
    FAILED_DIR.mkdir(parents=True, exist_ok=True)
    dest = FAILED_DIR / pdf_path.name

    # Avoid overwriting if a file with the same name already exists
    if dest.exists():
        stem = pdf_path.stem
        suffix = pdf_path.suffix
        dest = FAILED_DIR / f"{stem}_{audit_id[:8]}{suffix}"

    shutil.move(str(pdf_path), str(dest))
    logger.info("Moved PDF to failed: %s → %s", pdf_path.name, dest)

    # Update audit log
    db.update_audit_record(
        audit_id,
        failed_blob_uri=str(dest),
    )

    # Log to failed documents tracking table
    db.log_failed_document(
        audit_id=audit_id,
        patient_id=patient_id,
        unique_doc_num=unique_doc_num,
        original_file_name=pdf_path.name,
        failure_stage=failure_stage,
        failure_reason=failure_reason,
        failed_blob_uri=str(dest),
    )

    return dest


def move_to_processed(pdf_path: Path) -> Path:
    """
    Move a PDF from its current location to the ``processed/`` directory.

    Args:
        pdf_path: Current absolute path to the PDF.

    Returns:
        New path of the PDF in the ``processed/`` directory.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dest = PROCESSED_DIR / pdf_path.name

    # Avoid overwriting
    if dest.exists():
        stem = pdf_path.stem
        suffix = pdf_path.suffix
        dest = PROCESSED_DIR / f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"

    shutil.move(str(pdf_path), str(dest))
    logger.info("Moved PDF to processed: %s → %s", pdf_path.name, dest)
    return dest


# ============================================================================
# 3. Mock API call
# ============================================================================

def call_mock_api(
    patient_id: str,
    unique_doc_num: str,
    service_date: str,
    summary_text: str,
    json_object: dict,
    audit_id: str,
) -> dict:
    """
    Send extracted data to the mock client replication API.

    Makes an HTTP POST to ``http://localhost:{MOCK_API_PORT}/replicate``
    with the patient data payload.

    Args:
        patient_id:     Patient ID.
        unique_doc_num: Unique document number.
        service_date:   Service date string.
        summary_text:   Human-readable summary of the discharge.
        json_object:    Full extracted JSON object.
        audit_id:       Audit record ID.

    Returns:
        dict with ``success`` (bool) and optionally ``error`` (str).
    """
    url = f"http://localhost:{MOCK_API_PORT}/replicate"
    payload = {
        "patientId": patient_id,
        "uniqueDocNum": unique_doc_num,
        "serviceDate": service_date,
        "summaryText": summary_text,
        "jsonObject": json_object,
    }

    try:
        logger.info("Calling mock API at %s — audit_id=%s", url, audit_id)
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()

        now = datetime.utcnow().isoformat() + "Z"
        db.update_audit_record(
            audit_id,
            replication_api_status="SUCCESS",
            replication_attempt_count=1,
            replicated_at=now,
        )
        logger.info("Mock API call succeeded — audit_id=%s", audit_id)
        return {"success": True}

    except Exception as exc:
        error_msg = str(exc)
        logger.warning(
            "Mock API call failed — audit_id=%s, error=%s", audit_id, error_msg
        )
        db.update_audit_record(
            audit_id,
            replication_api_status="FAILED",
            replication_attempt_count=1,
            replication_error_reason=error_msg,
        )
        return {"success": False, "error": error_msg}


# ============================================================================
# 4. Short summary generator (pure Python — no LLM)
# ============================================================================

def generate_short_summary(extracted_json: dict) -> str:
    """
    Build a human-readable 3–5 sentence summary from the extracted JSON.

    Includes patient name, age, admission/discharge dates, primary diagnosis,
    medication count, and follow-up date. Handles missing fields gracefully.

    Args:
        extracted_json: The structured JSON produced by ontology extraction.

    Returns:
        A short summary string.
    """
    def _get(section: str, field: str, default: str = "Not recorded") -> str:
        val = extracted_json.get(section, {}).get(field)
        if val is None or (isinstance(val, str) and val.strip() == ""):
            return default
        return str(val)

    patient_name = _get("patient_demographics", "patient_name")
    age = _get("patient_demographics", "age")
    gender = _get("patient_demographics", "gender")

    admission_date = _get("admission_details", "admission_date")
    discharge_date = _get("admission_details", "discharge_date")
    length_of_stay = _get("admission_details", "length_of_stay_days")

    primary_dx = _get("diagnoses", "primary_diagnosis")

    # Count medications
    meds = extracted_json.get("medications", {}).get("medications_on_discharge", [])
    med_count = len(meds) if isinstance(meds, list) else 0

    follow_up_date = _get("follow_up", "follow_up_date")
    discharge_condition = _get("discharge_details", "discharge_condition")

    sentences = [
        f"Patient {patient_name}, {age} year(s) old ({gender}), "
        f"was admitted on {admission_date} and discharged on {discharge_date} "
        f"(length of stay: {length_of_stay} day(s)).",

        f"Primary diagnosis: {primary_dx}.",

        f"The patient was discharged in {discharge_condition} condition "
        f"with {med_count} medication(s) prescribed.",

        f"Follow-up scheduled for {follow_up_date}.",
    ]

    return " ".join(sentences)


# ============================================================================
# 5. Main pipeline orchestrator
# ============================================================================

def run_pipeline(pdf_path: Path) -> dict:
    """
    Execute the full discharge summary processing pipeline.

    Steps:
        0. Setup — generate IDs, create audit record
        1. Pre-validation — filename + file size checks
        2. Vision LLM extraction — PDF → raw text
        3. Ontology extraction — raw text → structured JSON
        4. Validation Layer 1 — field-fill ratio check
        5. Validation Layer 2 — hallucination traceability check
        6. Final storage — persist JSON + summary
        7. API replication — call mock client API
        Final — move PDF to processed folder

    Args:
        pdf_path: Absolute path to the PDF file to process.

    Returns:
        dict with keys:
            - audit_id (str)
            - overall_status (str): "COMPLETED" or "FAILED"
            - failure_stage (str | None): None if successful
            - processing_time_ms (int): total end-to-end time
    """
    pipeline_start = time.perf_counter()

    # Tracking variables (populated as pipeline progresses)
    patient_id = ""
    unique_doc_num = ""
    service_date = ""
    file_size_kb = 0.0

    # ==================================================================
    # STEP 0: Setup
    # ==================================================================
    try:
        audit_id, pipeline_run_id = generate_ids()
        upload_timestamp = datetime.utcnow().isoformat() + "Z"

        logger.info(
            "══════════════════════════════════════════════════════"
        )
        logger.info(
            "PIPELINE START — audit_id=%s, file=%s", audit_id, pdf_path.name
        )
        logger.info(
            "══════════════════════════════════════════════════════"
        )

        # Initialize DB tables (idempotent)
        db.init_db()

        # Create initial audit record
        db.create_audit_record(
            audit_id=audit_id,
            pipeline_run_id=pipeline_run_id,
            patient_id="",
            unique_doc_num="",
            service_date="",
            original_file_name=pdf_path.name,
            blob_uri=str(pdf_path),
            file_size_kb=0.0,
            upload_timestamp=upload_timestamp,
            event_id=str(uuid.uuid4()),
        )

        db.update_audit_record(
            audit_id,
            ingestion_status="RECEIVED",
            event_triggered_at=upload_timestamp,
            processing_mode="AUTOMATIC",
        )

    except Exception as exc:
        elapsed = int((time.perf_counter() - pipeline_start) * 1000)
        logger.exception("STEP 0 (Setup) failed: %s", exc)
        return {
            "audit_id": "UNKNOWN",
            "overall_status": "FAILED",
            "failure_stage": "Setup",
            "processing_time_ms": elapsed,
        }

    # ==================================================================
    # STEP 1: Pre-Validation
    # ==================================================================
    try:
        logger.info("[Step 1/7] Pre-validation — %s", pdf_path.name)
        pre_val = run_pre_validation(filepath=pdf_path, audit_id=audit_id)

        if not pre_val["passed"]:
            failure_reason = "Pre-validation failed"
            logger.warning("Step 1 FAILED: %s", failure_reason)
            db.update_audit_record(
                audit_id,
                overall_pipeline_status="FAILED",
                failure_stage="PreValidation",
            )
            move_to_failed(
                pdf_path, audit_id,
                patient_id=pre_val.get("patient_id", ""),
                unique_doc_num=pre_val.get("unique_doc_num", ""),
                failure_stage="PreValidation",
                failure_reason=failure_reason,
            )
            elapsed = int((time.perf_counter() - pipeline_start) * 1000)
            return {
                "audit_id": audit_id,
                "overall_status": "FAILED",
                "failure_stage": "PreValidation",
                "processing_time_ms": elapsed,
            }

        # Capture parsed identifiers
        patient_id = pre_val["patient_id"]
        unique_doc_num = pre_val["unique_doc_num"]
        service_date = pre_val["service_date"]
        file_size_kb = pre_val["file_size_kb"]

        # Back-fill audit record with parsed identifiers
        db.update_audit_record(
            audit_id,
            patient_id=patient_id,
            unique_doc_num=unique_doc_num,
            service_date=service_date,
            file_size_kb=file_size_kb,
        )

        logger.info(
            "Step 1 PASSED — patient=%s, doc=%s, date=%s, size=%.1fKB",
            patient_id, unique_doc_num, service_date, file_size_kb,
        )

    except Exception as exc:
        elapsed = int((time.perf_counter() - pipeline_start) * 1000)
        logger.exception("STEP 1 (Pre-validation) exception: %s", exc)
        db.update_audit_record(
            audit_id,
            overall_pipeline_status="FAILED",
            failure_stage="PreValidation",
        )
        move_to_failed(
            pdf_path, audit_id,
            failure_stage="PreValidation",
            failure_reason=str(exc),
        )
        return {
            "audit_id": audit_id,
            "overall_status": "FAILED",
            "failure_stage": "PreValidation",
            "processing_time_ms": elapsed,
        }

    # ==================================================================
    # STEP 2: Vision LLM Extraction
    # ==================================================================
    try:
        logger.info("[Step 2/7] Vision LLM text extraction")
        vision_result = extract_raw_text(
            pdf_path=pdf_path,
            audit_id=audit_id,
            file_size_kb=file_size_kb,
        )

        if not vision_result["success"]:
            logger.warning(
                "Step 2 FAILED: %s", vision_result.get("failure_reason")
            )
            move_to_failed(
                pdf_path, audit_id,
                patient_id=patient_id,
                unique_doc_num=unique_doc_num,
                failure_stage="RawTextExtraction",
                failure_reason=vision_result.get("failure_reason", "Unknown"),
            )
            elapsed = int((time.perf_counter() - pipeline_start) * 1000)
            return {
                "audit_id": audit_id,
                "overall_status": "FAILED",
                "failure_stage": "RawTextExtraction",
                "processing_time_ms": elapsed,
            }

        raw_text = vision_result["raw_text"]
        logger.info(
            "Step 2 PASSED — quality=%.4f, chars=%d",
            vision_result["quality_score"], len(raw_text),
        )

    except Exception as exc:
        elapsed = int((time.perf_counter() - pipeline_start) * 1000)
        logger.exception("STEP 2 (Vision extraction) exception: %s", exc)
        db.update_audit_record(
            audit_id,
            overall_pipeline_status="FAILED",
            failure_stage="RawTextExtraction",
        )
        move_to_failed(
            pdf_path, audit_id,
            patient_id=patient_id,
            unique_doc_num=unique_doc_num,
            failure_stage="RawTextExtraction",
            failure_reason=str(exc),
        )
        return {
            "audit_id": audit_id,
            "overall_status": "FAILED",
            "failure_stage": "RawTextExtraction",
            "processing_time_ms": elapsed,
        }

    # ==================================================================
    # STEP 3: Ontology Extraction
    # ==================================================================
    try:
        logger.info("[Step 3/7] Reasoning LLM ontology extraction")
        onto_result = extract_ontology_fields(
            raw_text=raw_text,
            audit_id=audit_id,
        )

        if not onto_result["success"]:
            logger.warning(
                "Step 3 FAILED: %s", onto_result.get("failure_reason")
            )
            move_to_failed(
                pdf_path, audit_id,
                patient_id=patient_id,
                unique_doc_num=unique_doc_num,
                failure_stage="OntologyExtraction",
                failure_reason=onto_result.get("failure_reason", "Unknown"),
            )
            elapsed = int((time.perf_counter() - pipeline_start) * 1000)
            return {
                "audit_id": audit_id,
                "overall_status": "FAILED",
                "failure_stage": "OntologyExtraction",
                "processing_time_ms": elapsed,
            }

        extracted_json = onto_result["extracted_json"]
        logger.info(
            "Step 3 PASSED — fields filled=%d/%d",
            onto_result["fields_filled"], onto_result["fields_total"],
        )

    except Exception as exc:
        elapsed = int((time.perf_counter() - pipeline_start) * 1000)
        logger.exception("STEP 3 (Ontology extraction) exception: %s", exc)
        db.update_audit_record(
            audit_id,
            overall_pipeline_status="FAILED",
            failure_stage="OntologyExtraction",
        )
        move_to_failed(
            pdf_path, audit_id,
            patient_id=patient_id,
            unique_doc_num=unique_doc_num,
            failure_stage="OntologyExtraction",
            failure_reason=str(exc),
        )
        return {
            "audit_id": audit_id,
            "overall_status": "FAILED",
            "failure_stage": "OntologyExtraction",
            "processing_time_ms": elapsed,
        }

    # ==================================================================
    # STEP 4: Validation Layer 1 — Structure Validation
    # ==================================================================
    try:
        logger.info("[Step 4/7] Structure validation (Layer 1)")
        val1_result = run_structure_validation(
            extracted_json=extracted_json,
            fields_total=onto_result["fields_total"],
            fields_filled=onto_result["fields_filled"],
            empty_fields=onto_result["empty_fields"],
            audit_id=audit_id,
        )

        report = format_validation_report(val1_result)
        logger.info("Validation Layer 1 report:\n%s", report)

        if not val1_result["passed"]:
            logger.warning(
                "Step 4 FAILED: %s", val1_result.get("failure_reason")
            )
            move_to_failed(
                pdf_path, audit_id,
                patient_id=patient_id,
                unique_doc_num=unique_doc_num,
                failure_stage="ValidationLayer1",
                failure_reason=val1_result.get("failure_reason", "Unknown"),
            )
            elapsed = int((time.perf_counter() - pipeline_start) * 1000)
            return {
                "audit_id": audit_id,
                "overall_status": "FAILED",
                "failure_stage": "ValidationLayer1",
                "processing_time_ms": elapsed,
            }

        logger.info(
            "Step 4 PASSED — ratio=%.2f%%, threshold=%.0f%%",
            val1_result["ratio"] * 100, val1_result["threshold"] * 100,
        )

    except Exception as exc:
        elapsed = int((time.perf_counter() - pipeline_start) * 1000)
        logger.exception("STEP 4 (Structure validation) exception: %s", exc)
        db.update_audit_record(
            audit_id,
            overall_pipeline_status="FAILED",
            failure_stage="ValidationLayer1",
        )
        move_to_failed(
            pdf_path, audit_id,
            patient_id=patient_id,
            unique_doc_num=unique_doc_num,
            failure_stage="ValidationLayer1",
            failure_reason=str(exc),
        )
        return {
            "audit_id": audit_id,
            "overall_status": "FAILED",
            "failure_stage": "ValidationLayer1",
            "processing_time_ms": elapsed,
        }

    # ==================================================================
    # STEP 5: Validation Layer 2 — Hallucination Check
    # ==================================================================
    try:
        logger.info("[Step 5/7] Hallucination check (Layer 2)")
        val2_result = run_hallucination_check(
            raw_text=raw_text,
            extracted_json=extracted_json,
            audit_id=audit_id,
        )

        if not val2_result["passed"]:
            logger.warning(
                "Step 5 FAILED: %s", val2_result.get("failure_reason")
            )
            move_to_failed(
                pdf_path, audit_id,
                patient_id=patient_id,
                unique_doc_num=unique_doc_num,
                failure_stage="ValidationLayer2",
                failure_reason=val2_result.get("failure_reason", "Unknown"),
            )
            elapsed = int((time.perf_counter() - pipeline_start) * 1000)
            return {
                "audit_id": audit_id,
                "overall_status": "FAILED",
                "failure_stage": "ValidationLayer2",
                "processing_time_ms": elapsed,
            }

        logger.info(
            "Step 5 PASSED — summary: %s", val2_result.get("summary")
        )

    except Exception as exc:
        elapsed = int((time.perf_counter() - pipeline_start) * 1000)
        logger.exception("STEP 5 (Hallucination check) exception: %s", exc)
        db.update_audit_record(
            audit_id,
            overall_pipeline_status="FAILED",
            failure_stage="ValidationLayer2",
        )
        move_to_failed(
            pdf_path, audit_id,
            patient_id=patient_id,
            unique_doc_num=unique_doc_num,
            failure_stage="ValidationLayer2",
            failure_reason=str(exc),
        )
        return {
            "audit_id": audit_id,
            "overall_status": "FAILED",
            "failure_stage": "ValidationLayer2",
            "processing_time_ms": elapsed,
        }

    # ==================================================================
    # STEP 6: Final Storage
    # ==================================================================
    try:
        logger.info("[Step 6/7] Final storage")

        summary_text = generate_short_summary(extracted_json)

        # Save final output JSON (includes both extraction and summary)
        TEXT_DUMPS_DIR.mkdir(parents=True, exist_ok=True)
        final_output = {
            "audit_id": audit_id,
            "patient_id": patient_id,
            "unique_doc_num": unique_doc_num,
            "service_date": service_date,
            "summary_text": summary_text,
            "extracted_json": extracted_json,
        }
        final_path = TEXT_DUMPS_DIR / f"{audit_id}_final_output.json"
        final_path.write_text(
            json.dumps(final_output, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Saved final output to %s", final_path)

        # Update audit log
        finalized_at = datetime.utcnow().isoformat() + "Z"
        db.update_audit_record(
            audit_id,
            final_json_stored=1,
            final_summary_stored=1,
            output_version="1.0",
            final_cosmos_container="Active",
            finalized_at=finalized_at,
            overall_pipeline_status="COMPLETED",
        )

        logger.info("Step 6 PASSED — pipeline marked COMPLETED")

    except Exception as exc:
        elapsed = int((time.perf_counter() - pipeline_start) * 1000)
        logger.exception("STEP 6 (Final storage) exception: %s", exc)
        db.update_audit_record(
            audit_id,
            overall_pipeline_status="FAILED",
            failure_stage="FinalStorage",
        )
        move_to_failed(
            pdf_path, audit_id,
            patient_id=patient_id,
            unique_doc_num=unique_doc_num,
            failure_stage="FinalStorage",
            failure_reason=str(exc),
        )
        return {
            "audit_id": audit_id,
            "overall_status": "FAILED",
            "failure_stage": "FinalStorage",
            "processing_time_ms": elapsed,
        }

    # ==================================================================
    # STEP 7: API Replication
    # (Failure here does NOT fail the pipeline)
    # ==================================================================
    try:
        logger.info("[Step 7/7] API replication")
        api_result = call_mock_api(
            patient_id=patient_id,
            unique_doc_num=unique_doc_num,
            service_date=service_date,
            summary_text=summary_text,
            json_object=extracted_json,
            audit_id=audit_id,
        )
        if api_result["success"]:
            logger.info("Step 7 PASSED — API replication successful")
        else:
            logger.warning(
                "Step 7 WARNING — API replication failed (non-blocking): %s",
                api_result.get("error"),
            )
    except Exception as exc:
        logger.warning(
            "Step 7 WARNING — API replication exception (non-blocking): %s",
            exc,
        )

    # ==================================================================
    # FINAL: Move to processed
    # ==================================================================
    try:
        move_to_processed(pdf_path)
    except Exception as exc:
        logger.warning(
            "Could not move PDF to processed (non-blocking): %s", exc
        )

    elapsed = int((time.perf_counter() - pipeline_start) * 1000)

    logger.info(
        "══════════════════════════════════════════════════════"
    )
    logger.info(
        "PIPELINE COMPLETE — audit_id=%s, status=COMPLETED, time=%dms",
        audit_id, elapsed,
    )
    logger.info(
        "══════════════════════════════════════════════════════"
    )

    return {
        "audit_id": audit_id,
        "overall_status": "COMPLETED",
        "failure_stage": None,
        "processing_time_ms": elapsed,
    }
