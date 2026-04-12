"""
Step 7 — Structure validation (Validation Layer 1).

Checks whether the Reasoning LLM filled enough ontology fields to meet the
acceptance threshold. This is a pure ratio check — no LLM calls, no file I/O
beyond database writes.

The acceptance threshold is configured via ``ACCEPTANCE_THRESHOLD`` in the
``.env`` file (default 0.6 = 60 %).
"""

import json
import logging

from src.config import ACCEPTANCE_THRESHOLD
from src import db

logger = logging.getLogger(__name__)


# ============================================================================
# 1. Structure validation
# ============================================================================

def run_structure_validation(
    extracted_json: dict,
    fields_total: int,
    fields_filled: int,
    empty_fields: list,
    audit_id: str,
) -> dict:
    """
    Validate that the extracted JSON meets the minimum field-fill ratio.

    Args:
        extracted_json: The JSON object produced by the Reasoning LLM.
        fields_total:   Total countable fields in the ontology.
        fields_filled:  Number of non-null/non-empty fields in the extraction.
        empty_fields:   List of dot-path field names that are missing/empty.
        audit_id:       The audit record ID to update in the database.

    Returns:
        dict with keys:
            - passed (bool)
            - ratio (float): filled / total
            - threshold (float): configured acceptance threshold
            - fields_filled (int)
            - fields_total (int)
            - empty_fields (list[str])
            - failure_reason (str | None)
    """
    threshold = ACCEPTANCE_THRESHOLD

    # Calculate ratio (guard against division by zero)
    if fields_total > 0:
        ratio = round(fields_filled / fields_total, 4)
    else:
        ratio = 0.0

    passed = ratio >= threshold

    logger.info(
        "Structure validation — audit_id=%s, filled=%d/%d, ratio=%.2f, "
        "threshold=%.2f, result=%s",
        audit_id, fields_filled, fields_total, ratio, threshold,
        "PASS" if passed else "FAIL",
    )

    # --- Update audit log ---
    audit_kwargs = {
        "structure_validation_status": "PASS" if passed else "FAIL",
        "filled_to_total_ratio": ratio,
        "acceptance_threshold": threshold,
        "missing_critical_sections": json.dumps(empty_fields),
    }

    if not passed:
        audit_kwargs["overall_pipeline_status"] = "FAILED"
        audit_kwargs["failure_stage"] = "ValidationLayer1"

    db.update_audit_record(audit_id, **audit_kwargs)

    # --- Build result ---
    failure_reason = None
    if not passed:
        failure_reason = (
            f"Field-fill ratio {ratio:.2%} is below the acceptance "
            f"threshold {threshold:.0%} ({fields_filled}/{fields_total} "
            f"fields filled)"
        )
        logger.warning(
            "Structure validation FAILED — audit_id=%s, reason=%s",
            audit_id, failure_reason,
        )

    return {
        "passed": passed,
        "ratio": ratio,
        "threshold": threshold,
        "fields_filled": fields_filled,
        "fields_total": fields_total,
        "empty_fields": empty_fields,
        "failure_reason": failure_reason,
    }


# ============================================================================
# 2. Human-readable validation report
# ============================================================================

def format_validation_report(result: dict) -> str:
    """
    Return a human-readable string summarising the structure validation result.

    Includes pass/fail status, ratio, threshold, field counts, and up to
    5 missing field names.

    Args:
        result: The dict returned by :func:`run_structure_validation`.

    Returns:
        Multi-line report string suitable for logging or demo output.
    """
    status = "✅ PASS" if result["passed"] else "❌ FAIL"
    lines = [
        "═" * 50,
        f"  STRUCTURE VALIDATION:  {status}",
        "═" * 50,
        f"  Fields filled : {result['fields_filled']} / {result['fields_total']}",
        f"  Fill ratio    : {result['ratio']:.2%}",
        f"  Threshold     : {result['threshold']:.0%}",
    ]

    empty = result.get("empty_fields", [])
    if empty:
        lines.append(f"  Missing fields: {len(empty)} total")
        for field in empty[:5]:
            lines.append(f"    • {field}")
        if len(empty) > 5:
            lines.append(f"    … and {len(empty) - 5} more")

    if result.get("failure_reason"):
        lines.append(f"  Reason        : {result['failure_reason']}")

    lines.append("═" * 50)
    return "\n".join(lines)
