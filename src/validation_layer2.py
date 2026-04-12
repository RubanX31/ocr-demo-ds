"""
Step 8 — Hallucination check (Validation Layer 2).

Sends the raw source text and extracted JSON back to Gemini to verify that
every non-null field in the extraction is **traceable** to the source text.
Fields that cannot be found in the source are flagged as hallucinations.

If *any* field is non-traceable, the verdict is FAIL and the pipeline is
marked as failed at this stage.
"""

import json
import logging

from src.config import VALIDATION_PROMPT_VERSION
from src.gemini_client import gemini
from src import db

logger = logging.getLogger(__name__)


# ============================================================================
# Hallucination check prompt template
# ============================================================================

HALLUCINATION_CHECK_PROMPT_TEMPLATE = """\
You are a medical data verification engine. Your task is to check whether \
every value in the EXTRACTED DATA is supported by the SOURCE TEXT.

=== INSTRUCTIONS ===

1. Go through EACH non-null field in the EXTRACTED DATA.

2. For each field value, search the SOURCE TEXT to determine whether that \
value can be found or reasonably inferred from what is written there.
   - "Reasonably inferred" means the value is a direct, obvious consequence \
of what the text states (e.g., length_of_stay_days = 8 when admission is \
2026-03-20 and discharge is 2026-03-28). It does NOT mean guessing or \
interpreting ambiguous text.

3. Classify each field:
   - TRACEABLE: the value appears in or is directly calculable from the \
SOURCE TEXT.
   - NON_TRACEABLE: the value does NOT appear anywhere in the SOURCE TEXT \
and cannot be directly calculated from it. This means it was hallucinated.

4. Be STRICT. If you are uncertain whether a value is supported by the \
source text, mark it as NON_TRACEABLE. It is better to flag a false positive \
than to miss a hallucination.

5. Return your result as a single valid JSON object with this exact structure:
{{
  "verdict": "PASS" or "FAIL",
  "non_traceable_fields": ["section.field_name", ...],
  "summary": "Brief one-sentence summary of your findings"
}}

Rules for the verdict:
- "PASS" if ALL non-null fields are TRACEABLE (non_traceable_fields is empty).
- "FAIL" if ANY non-null field is NON_TRACEABLE.

6. Do NOT add any commentary, preamble, or explanation outside the JSON. \
Return the JSON object only.

=== SOURCE TEXT ===

{raw_text}

=== EXTRACTED DATA ===

{extracted_json}

=== YOUR VERIFICATION RESULT (JSON only) ===
"""


# ============================================================================
# Main hallucination check function
# ============================================================================

def run_hallucination_check(
    raw_text: str,
    extracted_json: dict,
    audit_id: str,
) -> dict:
    """
    Verify that all extracted fields are traceable to the raw source text.

    Sends both the raw text and extracted JSON to Gemini, which checks each
    non-null field for traceability. Any field that cannot be found in the
    source text is flagged as a hallucination.

    Args:
        raw_text:       Verbatim text extracted from the PDF by the Vision LLM.
        extracted_json: The structured JSON produced by the Reasoning LLM.
        audit_id:       The audit record ID to update in the database.

    Returns:
        dict with keys:
            - passed (bool)
            - non_traceable_fields (list[str])
            - summary (str)
            - failure_reason (str | None)
    """
    logger.info("Starting hallucination check — audit_id=%s", audit_id)

    # --- Build prompt ---
    extracted_json_str = json.dumps(extracted_json, indent=2, ensure_ascii=False)

    prompt = HALLUCINATION_CHECK_PROMPT_TEMPLATE.format(
        raw_text=raw_text,
        extracted_json=extracted_json_str,
    )

    # --- Call Gemini ---
    llm_result = gemini.send_prompt_with_text(
        prompt=prompt,
        prompt_version=VALIDATION_PROMPT_VERSION,
    )

    if not llm_result["success"]:
        logger.error(
            "Hallucination check API call failed — audit_id=%s, error=%s",
            audit_id, llm_result["error"],
        )
        db.update_audit_record(
            audit_id,
            hallucination_check_status="ERROR",
            hallucination_check_time_ms=llm_result["time_ms"],
            fallback_triggered=0,
            reprocessing_attempt_count=0,
            overall_pipeline_status="FAILED",
            failure_stage="ValidationLayer2",
        )
        return {
            "passed": False,
            "non_traceable_fields": [],
            "summary": f"Gemini API error: {llm_result['error']}",
            "failure_reason": f"Gemini API error: {llm_result['error']}",
        }

    # --- Parse JSON response ---
    parsed = gemini.parse_json_response(llm_result["response_text"])

    if "error" in parsed and parsed["error"] == "parse_failed":
        logger.error(
            "Hallucination check JSON parse failed — audit_id=%s, "
            "raw response (first 500 chars): %s",
            audit_id, (llm_result["response_text"] or "")[:500],
        )
        db.update_audit_record(
            audit_id,
            hallucination_check_status="ERROR",
            hallucination_check_time_ms=llm_result["time_ms"],
            fallback_triggered=0,
            reprocessing_attempt_count=0,
            overall_pipeline_status="FAILED",
            failure_stage="ValidationLayer2",
        )
        return {
            "passed": False,
            "non_traceable_fields": [],
            "summary": "Failed to parse hallucination check response as JSON",
            "failure_reason": "Failed to parse hallucination check response as JSON",
        }

    # --- Extract verdict fields (with safe defaults) ---
    verdict = parsed.get("verdict", "FAIL").upper()
    non_traceable = parsed.get("non_traceable_fields", [])
    summary = parsed.get("summary", "No summary provided")

    # Normalize: if non_traceable is not a list, wrap it
    if not isinstance(non_traceable, list):
        non_traceable = [str(non_traceable)]

    # Ensure verdict is consistent with non_traceable_fields
    if non_traceable and verdict == "PASS":
        logger.warning(
            "Gemini returned verdict=PASS but non_traceable_fields is "
            "non-empty. Overriding verdict to FAIL — audit_id=%s",
            audit_id,
        )
        verdict = "FAIL"

    passed = verdict == "PASS"

    logger.info(
        "Hallucination check complete — audit_id=%s, verdict=%s, "
        "non_traceable=%d fields, time=%dms",
        audit_id, verdict, len(non_traceable), llm_result["time_ms"],
    )

    # --- Update audit log ---
    audit_kwargs = {
        "hallucination_check_status": verdict,
        "non_traceable_fields": json.dumps(non_traceable),
        "fallback_triggered": 0,
        "reprocessing_attempt_count": 0,
        "hallucination_check_time_ms": llm_result["time_ms"],
    }

    if not passed:
        audit_kwargs["overall_pipeline_status"] = "FAILED"
        audit_kwargs["failure_stage"] = "ValidationLayer2"
        logger.warning(
            "Hallucination check FAILED — audit_id=%s, "
            "non_traceable_fields=%s, summary=%s",
            audit_id, non_traceable, summary,
        )

    db.update_audit_record(audit_id, **audit_kwargs)

    # --- Build result ---
    failure_reason = None
    if not passed:
        failure_reason = (
            f"Hallucinated fields detected: {non_traceable}. {summary}"
        )

    return {
        "passed": passed,
        "non_traceable_fields": non_traceable,
        "summary": summary,
        "failure_reason": failure_reason,
    }
