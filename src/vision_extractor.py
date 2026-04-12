"""
Step 5 — Vision LLM text extraction.

Sends discharge summary PDFs to Google Gemini (vision-capable model) to
extract raw text. Includes a quality gate (via :func:`validator.check_text_quality`)
and an automatic retry with a more aggressive prompt if the first extraction
produces low-quality output.

No text is summarised or interpreted at this stage — the goal is a faithful,
verbatim transcription of the PDF content.
"""

import logging
from pathlib import Path

from src.config import TEXT_DUMPS_DIR, VISION_PROMPT_VERSION
from src.gemini_client import gemini
from src.validator import check_text_quality
from src import db

logger = logging.getLogger(__name__)


# ============================================================================
# Prompt V1 — Primary extraction prompt
# ============================================================================

VISION_PROMPT_V1 = """\
You are a medical document text extraction engine. You are given a hospital \
discharge summary as a PDF. Your task is to extract ALL text from the document \
exactly as it appears. Follow these rules strictly:

1. EXTRACT EVERY WORD, NUMBER, DATE, AND SYMBOL exactly as printed in the \
document. Do NOT summarise, paraphrase, interpret, or rephrase anything.

2. PRESERVE ALL SECTION HEADERS, LABELS, AND FIELD NAMES exactly as they are \
written (e.g., "Chief Complaint:", "Discharge Medications:", "ICD Code:"). \
Do not rename or reorganise them.

3. EXTRACT ALL TABLE CONTENT ROW BY ROW. Maintain the column alignment as \
closely as possible using spaces or tabs. Include every header row.

4. EXTRACT ALL MEDICATION DETAILS with full precision — drug names, dosages \
(e.g., "75 mg"), frequencies (e.g., "twice daily"), routes (e.g., "oral"), \
and durations. Do not truncate or round any values.

5. EXTRACT ALL DATES in whatever format they appear (DD/MM/YYYY, YYYY-MM-DD, \
"12th March 2026", etc.). Do not convert or reformat them.

6. EXTRACT ALL NUMERIC VALUES exactly — lab results, vitals, scores, counts. \
Include units and reference ranges if present.

7. If a section is PARTIALLY READABLE (e.g., faded, cut off, or obscured), \
extract whatever portion is visible. Do NOT skip the section entirely.

8. PRESERVE LINE BREAKS between sections and paragraphs for readability. \
Do not merge everything into a single block of text.

9. Do NOT add any commentary, preamble, explanation, or notes of your own. \
Output ONLY the extracted text from the document. No "Here is the text..." \
or "The document contains..." — just the raw text.

10. If the document has multiple pages, extract text from EVERY page in order.

Begin extraction now.
"""


# ============================================================================
# Prompt V1 Retry — More aggressive fallback for poor-quality first pass
# ============================================================================

VISION_PROMPT_V1_RETRY = """\
IMPORTANT: This is a RETRY attempt. A previous extraction of this document \
produced poor-quality or incomplete output. You must try harder this time.

You are a medical document text extraction engine. You are given a hospital \
discharge summary as a PDF that may contain low-contrast text, skewed or \
rotated content, handwritten annotations, stamps, or partially obscured areas.

Follow these instructions with maximum effort:

1. GO PAGE BY PAGE through the entire document. For each page, extract every \
single piece of visible text — printed, typed, stamped, or handwritten.

2. EXTRACT EVERY WORD, NUMBER, DATE, AND SYMBOL exactly as it appears. Do \
NOT summarise, interpret, or rephrase anything.

3. HANDLE DIFFICULT CONTENT:
   - Low contrast or faded text: extract your best reading of it.
   - Skewed, rotated, or misaligned text: read and extract it in correct order.
   - Partially obscured text: extract whatever characters or words are visible. \
     If only a partial word is readable, output it as-is (e.g., "Amlo..." or \
     "...dipine 5 mg").
   - Stamps, watermarks, headers, footers: include them.

4. PRESERVE ALL SECTION HEADERS, LABELS, FIELD NAMES, AND TABLE STRUCTURES \
exactly as written. Extract table content row by row.

5. EXTRACT ALL MEDICATION DETAILS, LAB VALUES, DATES, AND NUMERIC VALUES \
with full precision. Do not truncate, round, or skip any values.

6. PRESERVE LINE BREAKS between sections and paragraphs for readability.

7. Do NOT add any commentary, preamble, explanation, or notes of your own. \
Output ONLY the extracted text. No "Here is the text..." — just the raw text.

8. If the document has multiple pages, process EVERY page and output text \
in page order.

This extraction must be as complete as possible. Every piece of visible text \
matters for patient care.

Begin extraction now.
"""


# ============================================================================
# Main extraction function
# ============================================================================

def extract_raw_text(
    pdf_path: Path,
    audit_id: str,
    file_size_kb: float,
) -> dict:
    """
    Extract raw text from a discharge summary PDF using the Vision LLM.

    Workflow:
        1. Send the PDF + ``VISION_PROMPT_V1`` to Gemini.
        2. If the API call fails → record failure, return immediately.
        3. Run :func:`validator.check_text_quality` on the extracted text.
        4. If quality passes → save text, update audit log, return success.
        5. If quality fails → retry with ``VISION_PROMPT_V1_RETRY``.
        6. If retry passes → save text, update audit log, return success.
        7. If retry also fails → update audit log with FAILED status, return failure.

    Args:
        pdf_path:     Absolute path to the PDF file.
        audit_id:     The audit record ID to update in the database.
        file_size_kb: The PDF file size in kilobytes (used for quality scoring).

    Returns:
        dict with keys:
            - success (bool)
            - text_path (Path | None): path to saved text file, if successful
            - raw_text (str | None): extracted text, if successful
            - quality_score (float | None): quality score, if successful
            - failure_reason (str | None): reason for failure, if unsuccessful
    """
    logger.info(
        "Starting Vision LLM extraction — audit_id=%s, file=%s, size=%.1fKB",
        audit_id, pdf_path.name, file_size_kb,
    )

    # ------------------------------------------------------------------
    # Attempt 1: Primary prompt
    # ------------------------------------------------------------------
    llm_result = gemini.send_prompt_with_pdf(
        prompt=VISION_PROMPT_V1,
        pdf_path=pdf_path,
        prompt_version=VISION_PROMPT_VERSION,
    )

    if not llm_result["success"]:
        logger.error(
            "Vision LLM API call failed — audit_id=%s, error=%s",
            audit_id, llm_result["error"],
        )
        db.update_audit_record(
            audit_id,
            vision_model_used=llm_result["model_used"],
            raw_text_extraction_status="FAILED",
            vision_prompt_version=VISION_PROMPT_VERSION,
            raw_extraction_time_ms=llm_result["time_ms"],
            overall_pipeline_status="FAILED",
            failure_stage="RawTextExtraction",
        )
        return {
            "success": False,
            "text_path": None,
            "raw_text": None,
            "quality_score": None,
            "failure_reason": f"Gemini API error: {llm_result['error']}",
        }

    # ------------------------------------------------------------------
    # Quality check on first attempt
    # ------------------------------------------------------------------
    raw_text = llm_result["response_text"]
    quality = check_text_quality(raw_text, file_size_kb)

    if quality["passed"]:
        logger.info(
            "Quality check PASSED (attempt 1) — audit_id=%s, score=%.4f",
            audit_id, quality["quality_score"],
        )
        return _save_and_record(
            audit_id=audit_id,
            raw_text=raw_text,
            quality=quality,
            llm_result=llm_result,
            prompt_version=VISION_PROMPT_VERSION,
        )

    # ------------------------------------------------------------------
    # Attempt 2: Retry with aggressive prompt
    # ------------------------------------------------------------------
    logger.warning(
        "Quality check FAILED (attempt 1) — audit_id=%s, score=%.4f, "
        "reason=%s. Retrying with aggressive prompt...",
        audit_id, quality["quality_score"], quality["failure_reason"],
    )

    retry_prompt_version = f"{VISION_PROMPT_VERSION}-retry"

    retry_result = gemini.send_prompt_with_pdf(
        prompt=VISION_PROMPT_V1_RETRY,
        pdf_path=pdf_path,
        prompt_version=retry_prompt_version,
    )

    if not retry_result["success"]:
        logger.error(
            "Vision LLM retry API call also failed — audit_id=%s, error=%s",
            audit_id, retry_result["error"],
        )
        db.update_audit_record(
            audit_id,
            vision_model_used=retry_result["model_used"],
            raw_text_extraction_status="FAILED",
            raw_text_quality_score=quality["quality_score"],
            vision_prompt_version=retry_prompt_version,
            raw_extraction_time_ms=(
                llm_result["time_ms"] + retry_result["time_ms"]
            ),
            overall_pipeline_status="FAILED",
            failure_stage="RawTextExtraction",
        )
        return {
            "success": False,
            "text_path": None,
            "raw_text": None,
            "quality_score": quality["quality_score"],
            "failure_reason": (
                f"Retry API error: {retry_result['error']}; "
                f"original quality failure: {quality['failure_reason']}"
            ),
        }

    # Quality check on retry
    retry_text = retry_result["response_text"]
    retry_quality = check_text_quality(retry_text, file_size_kb)

    if retry_quality["passed"]:
        logger.info(
            "Quality check PASSED (retry) — audit_id=%s, score=%.4f",
            audit_id, retry_quality["quality_score"],
        )
        # Use combined time from both attempts
        retry_result["time_ms"] = llm_result["time_ms"] + retry_result["time_ms"]
        return _save_and_record(
            audit_id=audit_id,
            raw_text=retry_text,
            quality=retry_quality,
            llm_result=retry_result,
            prompt_version=retry_prompt_version,
        )

    # ------------------------------------------------------------------
    # Both attempts failed quality check
    # ------------------------------------------------------------------
    logger.error(
        "Quality check FAILED on both attempts — audit_id=%s, "
        "attempt1_score=%.4f, retry_score=%.4f, reason=%s",
        audit_id,
        quality["quality_score"],
        retry_quality["quality_score"],
        retry_quality["failure_reason"],
    )

    # Use the better of the two scores for the audit record
    best_quality = (
        retry_quality if retry_quality["quality_score"] >= quality["quality_score"]
        else quality
    )

    db.update_audit_record(
        audit_id,
        vision_model_used=retry_result["model_used"],
        raw_text_extraction_status="FAILED",
        raw_text_token_count=retry_result["token_count"],
        raw_text_character_count=best_quality["char_count"],
        raw_text_quality_score=best_quality["quality_score"],
        vision_prompt_version=retry_prompt_version,
        raw_extraction_time_ms=(
            llm_result["time_ms"] + retry_result["time_ms"]
        ),
        overall_pipeline_status="FAILED",
        failure_stage="RawTextExtraction",
    )

    return {
        "success": False,
        "text_path": None,
        "raw_text": None,
        "quality_score": best_quality["quality_score"],
        "failure_reason": (
            f"Text quality too low after retry: {retry_quality['failure_reason']}"
        ),
    }


# ============================================================================
# Private helpers
# ============================================================================

def _save_and_record(
    audit_id: str,
    raw_text: str,
    quality: dict,
    llm_result: dict,
    prompt_version: str,
) -> dict:
    """
    Save extracted text to disk and update the audit log with success data.

    Args:
        audit_id:       Audit record ID.
        raw_text:       The extracted text string.
        quality:        Result dict from :func:`check_text_quality`.
        llm_result:     Result dict from :meth:`GeminiClient.send_prompt_with_pdf`.
        prompt_version: Prompt version tag to record in the audit log.

    Returns:
        Success dict for the caller.
    """
    # Ensure text_dumps directory exists
    TEXT_DUMPS_DIR.mkdir(parents=True, exist_ok=True)

    # Save raw text to file
    text_path = TEXT_DUMPS_DIR / f"{audit_id}_raw.txt"
    text_path.write_text(raw_text, encoding="utf-8")

    logger.info("Saved raw text to %s (%d chars)", text_path, len(raw_text))

    # Update audit log
    db.update_audit_record(
        audit_id,
        vision_model_used=llm_result["model_used"],
        raw_text_extraction_status="SUCCESS",
        raw_text_blob_uri=str(text_path),
        raw_text_token_count=llm_result["token_count"],
        raw_text_character_count=quality["char_count"],
        raw_text_quality_score=quality["quality_score"],
        vision_prompt_version=prompt_version,
        raw_extraction_time_ms=llm_result["time_ms"],
    )

    return {
        "success": True,
        "text_path": text_path,
        "raw_text": raw_text,
        "quality_score": quality["quality_score"],
        "failure_reason": None,
    }
